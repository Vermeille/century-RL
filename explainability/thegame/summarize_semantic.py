"""Archive semantic-rule experiments and regenerate their human-readable report."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from explainability.thegame.semantic_tree import score_summary


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluations', type=Path, nargs='+', required=True)
    parser.add_argument('--models', type=Path, nargs='+', required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    combined = {}
    for path in args.evaluations:
        current = json.loads(path.read_text())
        if 'teacher' in combined:
            assert current['teacher']['scores'] == combined['teacher']['scores']
            assert current['lowest']['scores'] == combined['lowest']['scores']
            assert current['protocol']['seed'] == combined['protocol']['seed']
        for name, result in current.items():
            if name == 'protocol' and name in combined:
                combined[name]['selected'].extend(result['selected'])
            else:
                combined[name] = result
    teacher = np.asarray(combined['teacher']['scores'])
    lowest = np.asarray(combined['lowest']['scores'])
    data = np.load(args.data)
    test = data['episode'] >= 640
    validation = (data['episode'] >= 512) & (data['episode'] < 640)
    combined['provenance'] = {
        'source_tag':'run-source/YOLO-20260901T125727Z-c6bb16ffa32e',
        'evaluation_checkout':'45e221d676ed9a4e1211803d7a6f1763fd338413',
        'source_check':'BoardRL tracked sources equal source tag except unused reservoir utility; game, model, and rollout sources unchanged.',
        'checkpoint_sha256':fingerprint(Path(combined['protocol']['checkpoint'])),
        'corpus_sha256':fingerprint(args.data),
        'corpus_games':len(data['scores']), 'corpus_states':len(data['episode']),
        'corpus_seed':1840000, 'train_games':512, 'validation_games':128, 'test_games':256,
        'cheapest_per_pile_coverage_validation':float(data['exact'][validation,7:11].any(axis=1).mean()),
        'cheapest_per_pile_coverage_test':float(data['exact'][test,7:11].any(axis=1).mean()),
        'test_teacher_nonlowest_rate':float(data['nonlowest'][test].mean()),
        'fit_dependencies':{'scikit-learn':'1.7.2','scipy':'1.16.2','joblib':'1.6.0','threadpoolctl':'3.6.0','cloudpickle':'3.1.2'},
    }
    rules = {}
    for path in args.models:
        rules.update(json.loads(path.read_text())['models'])
    for name, result in combined.items():
        if 'scores' not in result:
            continue
        scores = np.asarray(result['scores'])
        result['summary'] = score_summary(scores, teacher)
        difference = scores - lowest
        half = 1.96 * difference.std(ddof=1) / np.sqrt(len(scores))
        result['summary']['delta_vs_lowest'] = float(difference.mean())
        result['summary']['delta_vs_lowest_ci95'] = [float(difference.mean()-half),float(difference.mean()+half)]
    baseline_correct = data['exact'][test,0]
    combined['lowest']['test'] = {
        'states':int(test.sum()), 'exact':float(baseline_correct.mean()),
        'symmetric':float(data['symmetric'][test,0].mean()), 'nongreedy':0.0,
    }
    compact = combined['rounded_4']['summary']
    gain_fraction = compact['delta_vs_lowest'] / (teacher.mean() - lowest.mean())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'semantic_evaluation.json').write_text(json.dumps(combined,indent=2)+'\n')
    (args.output_dir/'semantic_rules.json').write_text(json.dumps({'models':rules},indent=2)+'\n')
    lines = [
        '# Strict: a compact semantic strategy', '',
        '## Outcome', '',
        f'A rounded four-term rule averages **{compact["mean"]:.2f}**, versus '
        f'**{lowest.mean():.2f}** for lowest-cost and **{teacher.mean():.2f}** for YOLO '
        f'on the same {len(teacher):,} fresh deals. It recovers **{100*gain_fraction:.1f}%** '
        'of the observed score advantage, but **does not recover the neural policy almost exactly**.', '',
        '## The strategy', '',
        'Consider only the cheapest legal card for each pile. Among those at most four moves, minimize:', '',
        '```python',
        'cost + hardest_remaining / 4 - same_direction_gap_gain / 5 + 8 * cost / space_left',
        '```', '',
        'Break ties by immediate cost, then original legal-move order. The complete standalone '
        'implementation is in `../compact_strategy.py`; it calls no neural network or decision tree.', '',
        '- `cost`: signed pile movement; a reverse-10 move costs −10.',
        '- `hardest_remaining`: after playing the candidate, take the cheapest legal placement '
        'cost of every still-playable card in your hand, then their maximum.',
        '- `same_direction_gap_gain`: change in absolute distance between this pile and the other '
        'ascending/descending pile. Positive values preserve or increase separation.',
        '- `space_left`: distance from this pile’s current top to its forward endpoint, before '
        'playing the candidate, floored at one.', '',
        'In English: **play cheaply, but sometimes pay a little more to leave an easier hand, '
        'preserve a reserve pile, or avoid consuming a large fraction of a nearly exhausted pile.**', '',
        'The remaining-hand feature omits already-unplayable cards. If no retained card is playable, '
        'it is 100; an empty retained hand gives zero. This is an empirical fitted convention, '
        'not a recommendation to ignore stranded cards. No opponent hand, deck order, or `Mem:` '
        'content is used. Memory omission here is not a new memory-ablation experiment.', '',
        '## Matched gameplay', '',
        '| Strategy | Mean | SD | Worst 5% mean | 5th percentile | Win rate | Exact action match | Non-lowest match |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name in ['lowest','tree_8','tree_64','fraction','sparse_2','sparse_4','rounded_4','sparse_5','rounded_5','teacher']:
        result = combined[name]
        s = result['summary']
        match = result.get('test')
        exact = f'{100*match["exact"]:.2f}%' if match else '100%'
        nongreedy = f'{100*match["nongreedy"]:.2f}%' if match else '100%'
        lines.append(f'| {name} | {s["mean"]:.2f} | {s["std"]:.2f} | {s["worst_5_percent_mean"]:.2f} | {s["q05"]:.0f} | {100*s["win_rate"]:.2f}% | {exact} | {nongreedy} |')
    ci = compact['delta_ci95']
    gain_ci = compact['delta_vs_lowest_ci95']
    coverage = combined['provenance']['cheapest_per_pile_coverage_test']
    lines.extend([
        '', f'Rounded four-term rule versus Strict: **{compact["delta_vs_teacher"]:+.2f} points**, '
        f'paired 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]. Versus lowest-cost: '
        f'**{compact["delta_vs_lowest"]:+.2f}**, CI [{gain_ci[0]:+.2f}, {gain_ci[1]:+.2f}]. '
        'These are approximate normal paired-deal intervals, not multiplicity-adjusted comparisons. '
        'The rule is still detectably below Strict on this sample. Similar win rates do not establish equivalence.', '',
        '## What was learned', '',
        f'1. **The action space is almost solved:** {coverage*100:.2f}% of Strict’s final-test actions '
        'are the cheapest card for its chosen pile. The hard part is ranking piles, not choosing '
        'arbitrary cards on a pile.',
        '2. **Tiny ordinary trees mostly learn the default.** The eight-leaf scorer makes no '
        'successful non-lowest predictions here. Sixty-four leaves improve fidelity, but yield '
        'a worse game-playing strategy than the four-term formula.',
        '3. **Small formulas transfer to full games.** The rounded four-term formula outperforms '
        'greedy over complete trajectories, not merely on the teacher’s recorded states.',
        '4. **Score recovery is not algorithm recovery.** About one action in five differs, '
        'and most of Strict’s deliberately non-lowest moves remain unexplained. Symmetry-aware '
        'agreement accepts only the same card on equal-top piles of the same direction; it '
        'does not close this gap.',
        '5. **More agreement need not mean more points.** The extra constrained-card term and '
        'the larger tree do not improve gameplay. These features describe useful surrogate '
        'preferences; they do not prove corresponding computations inside the neural network.', '',
        '## Method and limits', '',
        '- Teacher: Strict YOLO `step-900.pth`, historical source tag '
        '`run-source/YOLO-20260901T125727Z-c6bb16ffa32e`, deterministic argmax. This is not '
        'the stochastic training policy or exactly the training evaluator’s low-temperature protocol.',
        '- Corpus: 896 teacher games / 77,049 states, seeds `1840000 + i`; whole-game split '
        '512 train / 128 validation / 256 untouched test. Test agreement uses 22,163 decisions. '
        'Action agreement is decision-weighted; gameplay statistics are game-weighted.',
        '- Features: 414 visible-state/candidate summaries. Named action vocabularies contain '
        '7, 11, or 15 generators; all generators producing the teacher’s concrete action are '
        'accepted. Shared pile scorers use 35 features. Sparse score search uses 21 semantic '
        'features, up to six terms, and joint-choice logistic loss with duplicate-action masking.',
        '- Selection: whole-game validation, then frozen candidates. Four/five-term coefficients '
        'were rounded before inspecting their final test/gameplay results. All evaluated candidates '
        'are reported; calling the four-term rule preferable after evaluation is exploratory selection, '
        'so a fresh confirmation set would be appropriate before further tuning or strong equivalence claims.',
        '- Gameplay: 1,024 initial deals, seeds `1940000 + i`, independently constructed before '
        'rollouts. Every strategy receives the same initial deck/hands. Both seats use that strategy. '
        'Rule-generated trajectories may differ; teacher labels are not consulted during rule play.',
        '- Worst-5% mean uses the bottom 5% of scores with fractional boundary weighting. '
        'Wins mean all cards played (`points == 100`), not self-play seat win-rate.',
        '- No new claim about Omni versus Strict is made by this Strict-only distillation.', '',
        '## Remaining work toward near-exact recovery', '',
        'The next targeted experiment should label the residual non-greedy decisions with teacher '
        'action margins and fit a small exception rule over richer hand-sequencing features. '
        'It should report non-greedy precision/recall and validate on a new split. Simply growing '
        'the present trees does not meet the few-lines objective. Current evidence neither proves '
        'nor disproves that a near-exact compact rule exists.', '',
        'Full rule exports, per-deal scores, confidence intervals, checkpoint/corpus fingerprints, '
        'and provenance are in the adjacent JSON files. Reproduction commands are in `../README.md`.', '',
    ])
    (args.output_dir/'SEMANTIC_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'compact_summary':compact,'score_advantage_recovered':gain_fraction,'report':str(args.output_dir/'SEMANTIC_REPORT.md')},indent=2))


if __name__ == '__main__':
    main()
