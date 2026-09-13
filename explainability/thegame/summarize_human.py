"""Save the calculator-free strategy experiment with all screened candidates."""
import argparse
import json
from pathlib import Path

import numpy as np

from explainability.thegame.semantic_tree import score_summary


def paired_difference(scores, baseline):
    difference = np.asarray(scores) - np.asarray(baseline)
    half = 1.96*difference.std(ddof=1)/np.sqrt(len(difference))
    return {'mean':float(difference.mean()),
            'ci95':[float(difference.mean()-half),float(difference.mean()+half)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development',type=Path,nargs='+',required=True)
    parser.add_argument('--confirmation',type=Path,required=True)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    development = [json.loads(path.read_text()) for path in args.development]
    confirmation = json.loads(args.confirmation.read_text())
    for experiment in development+[confirmation]:
        baseline = experiment['lowest']['scores']
        reference = experiment.get('teacher',experiment['compact'])['scores']
        experiment['protocol']['difference_reference'] = 'teacher' if 'teacher' in experiment else 'compact'
        for name,result in experiment.items():
            if name == 'protocol':
                continue
            result['summary'] = score_summary(np.asarray(result['scores']),np.asarray(reference))
            # Replace the legacy helper's misleading teacher label in development runs.
            result['summary']['delta_vs_reference'] = result['summary'].pop('delta_vs_teacher')
            result['difference_vs_lowest'] = paired_difference(result['scores'],baseline)
            result['difference_vs_compact'] = paired_difference(result['scores'],experiment['compact']['scores'])
    args.output_dir.mkdir(parents=True,exist_ok=True)
    archived = {'development':development,'confirmation':confirmation,
                'fitted_models':json.loads(args.models.read_text())}
    (args.output_dir/'human_rules_evaluation.json').write_text(json.dumps(archived,indent=2)+'\n')
    best = confirmation['hardest_3']
    delta = best['difference_vs_lowest']
    lines = [
        '# Calculator-free play: work toward your awkward card', '',
        '## The practical rule', '',
        'Take a backwards-ten jump when one is available. Otherwise, look at your most '
        'awkward **still-playable** card: the one whose cheapest legal placement needs the '
        'largest jump. If that jump is at least **10**, advance its closest legal pile '
        'using the cheapest card for that pile, provided this costs at most **3 more** '
        'than the globally cheapest move. Otherwise play the globally cheapest move.', '',
        'No weighted score, ratios, memory vector, or lookahead simulator is used. The only '
        'new numerical thresholds are **10** and **3**. Ordinary pile/card subtraction is '
        'still needed, just as for lowest-cost play.', '',
        '```python',
        'if backwards_ten_available():',
        '    take_it()',
        'elif hardest_card_needs_at_least(10) and its_pile_costs_at_most_extra(3):',
        '    play_cheapest_card_on_that_pile()',
        'else:',
        '    play_lowest_cost()',
        '```', '',
        'This is semantic pseudocode. The actual deterministic implementation is '
        '`HumanRule(SPECS["hardest_3"])` in `../human_rules.py`. If the awkward card has '
        'several equally cheap legal piles, choose the cheapest qualifying pile move. '
        'Move ties use original legal-move order; awkward-card ties use hand order. '
        'Cards with no currently legal destination are excluded from this rule.', '',
        '### Example', '',
        'Ascending tops **10, 40**; descending tops **90, 70**; hand **11, 43, 55**. '
        'Greedy plays 11 on 10 for cost 1. But 55 currently needs a jump of at least 15. '
        'Playing 43 on 40 costs 3—only two extra—and brings that pile closer to 55. '
        'The rule therefore plays 43. This illustrates the rule, not a proof that this '
        'particular move has higher expected value in every hidden state.', '',
        '## Fresh confirmation', '',
        '| Strategy | Mean points | SD | Worst 5% mean | Win rate |',
        '|---|---:|---:|---:|---:|',
    ]
    for name in ['lowest','reserve_3','hardest_3','compact','teacher']:
        s=confirmation[name]['summary']
        lines.append(f'| {name} | {s["mean"]:.2f} | {s["std"]:.2f} | {s["worst_5_percent_mean"]:.2f} | {100*s["win_rate"]:.2f}% |')
    lines += [
        '', f'The awkward-card rule gains **{delta["mean"]:+.2f} points** over lowest-cost; '
        f'paired approximate 95% CI **[{delta["ci95"][0]:+.2f}, {delta["ci95"][1]:+.2f}]**.', '',
        'These are 1,024 fresh matched two-player Strict deals, seeds `2140000 + i`. '
        'Both seats use the same strategy. The neural reference is YOLO `step-900.pth`, '
        'deterministic argmax. This does not yet establish transfer to other player counts, '
        'human communication, or optional extra plays. It is practical heuristic discovery, '
        'not a claim about the neural network’s internal algorithm.', '',
        '## Screening and discarded ideas', '',
        'All development candidates used the same 256 deals, seeds `2040000 + i`. '
        'Candidate selection happened before opening the fresh confirmation results. '
        'Only the standalone awkward-card and reserve rules were selected for confirmation.', '',
        '| Development candidate | Mean points | Gain over greedy |',
        '|---|---:|---:|',
    ]
    for experiment in development:
        for name,result in experiment.items():
            if name in ['protocol','lowest','compact','teacher']:
                continue
            lines.append(f'| {name} | {result["summary"]["mean"]:.2f} | {result["difference_vs_lowest"]["mean"]:+.2f} |')
    lines += [
        '', '- `reserve_N`: pay at most N extra to use an already-more-advanced pile instead '
        'of an untouched/lagging reserve pile. All such rules take an available reverse-ten first.',
        '- `run_3`: pay at most three extra to start a same-pile run with a follow-up costing at most three.',
        '- `setup_N`: pay at most N extra to set up a reverse-ten with another card already in hand.',
        '- `endpoint_2`: pay at most two extra to save a pile with at most ten spaces left '
        'when another has at least twenty.',
        '- `constrained_2`: pay at most two extra for a card with only one legal destination.',
        '- Combined rule names apply the named priorities in order; they did **not** '
        'reliably improve on the standalone tactics.',
        '- `compact_3/4/6_w1`: three-, four-, and six-leaf Boolean replacement trees '
        'trained to imitate the previous four-term formula. Their high offline agreement '
        'did not preserve the formula’s full-game strength. More neural-target trees were '
        'fit but not deployed because they used fractions or were too complex for this task.', '',
        '## Interpretation', '',
        'The useful idea is not “always play the awkward card now.” It is **use an affordable '
        'move to make progress toward it**, instead of repeatedly taking cheap plays elsewhere '
        'and leaving the awkward card expensive. This is testable guidance backed by full '
        'games; the weighted formula remains a stronger but less table-friendly reference.', '',
        'The thresholds are heuristics, not proven optima. The confirmation set was used once; '
        'further tuning should use new development and confirmation deals. Full scores, '
        'paired differences, every screened candidate, and fitted Boolean trees are in '
        '`human_rules_evaluation.json`. Reproduction commands are in `../README.md`.', '',
    ]
    (args.output_dir/'HUMAN_RULES.md').write_text('\n'.join(lines))
    print(json.dumps({'hardest_rule':best['summary'],'difference_vs_lowest':delta},indent=2))


if __name__ == '__main__':
    main()
