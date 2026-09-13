"""Archive screening, validation, and fresh confirmation of awkward-card rules."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen',type=Path,nargs='+',required=True)
    parser.add_argument('--validation',type=Path,required=True)
    parser.add_argument('--confirmation',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    screens=[json.loads(path.read_text()) for path in args.screen]
    validation=json.loads(args.validation.read_text())
    confirmation=json.loads(args.confirmation.read_text())
    results=confirmation['results']
    previous=np.asarray(results['previous']['scores'])
    for name in ['worst_t0_b3','affordable_t20_b3']:
        delta=np.asarray(results[name]['scores'])-previous
        half=2.2414027276*delta.std(ddof=1)/np.sqrt(len(delta))
        results[name]['vs_previous']['two_comparison_familywise_ci95']=[float(delta.mean()-half),float(delta.mean()+half)]
    archive={'screens':screens,'validation':validation,'confirmation':confirmation,
             'selection':'Frozen after 256-game screening and independent 1024-game validation; two candidates tested on 4096 new paired deals.',
             'source_tag':'run-source/YOLO-20260901T125727Z-c6bb16ffa32e',
             'neural_inference_used':False}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'awkward_refinement.json').write_text(json.dumps(archive,indent=2)+'\n')
    lines=[
        '# Refining the awkward-card rule', '',
        'The target remains a rule you can use without a calculator. These policies use '
        'only card/pile differences, ordering, and small integer thresholds—no weighted '
        'move score, hidden state, memory vector, or learned inference.', '',
        '## Two refinements', '',
        '**Simpler version:** take an available backwards-ten jump. Otherwise, find your '
        'most awkward still-playable card. Advance its closest legal pile with that pile’s '
        'cheapest card if doing so costs at most **3 extra** relative to the globally '
        'cheapest move. Otherwise play globally cheapest. This removes the old “at least '
        '10” condition.', '',
        '**Affordable-awkward version:** take an available backwards-ten jump. Otherwise, '
        'consider cards from most awkward downward, but only those needing a jump of '
        '**at least 20**. Play the cheapest starter on the closest legal pile of the first '
        'such card you can help for **at most 3 extra**. If none qualifies, play globally '
        'cheapest. This allows helping the second-worst card when the worst is unaffordable.', '',
        '“Awkwardness” is a card’s cheapest currently legal placement cost. A card with no '
        'legal destination is excluded. Tied closest piles are all considered. Starter cards '
        'need not themselves prefer the pile they are used on. All ties otherwise retain '
        'hand/legal-move order; no additional human tie-break is claimed to be necessary.', '',
        '## Fresh confirmation', '',
        '| Rule | Mean points | SD | Worst 5% mean | Wins | Gain vs old rule |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    names={'lowest':'Lowest-cost','previous':'Original 10 / 3 rule','worst_t0_b3':'Simpler: no minimum jump',
           'affordable_t20_b3':'Affordable-awkward: 20 / 3','formula':'Previous weighted formula'}
    for key,label in names.items():
        row=results[key];s=row['summary']
        lines.append(f'| {label} | {s["mean"]:.3f} | {s["std"]:.2f} | {s["worst_5_percent_mean"]:.2f} | {100*s["win_rate"]:.2f}% | {row["vs_previous"]["mean"]:+.3f} |')
    for name in ['worst_t0_b3','affordable_t20_b3']:
        result=results[name]
        interval=result['vs_previous']['ci95']
        family=result['vs_previous']['two_comparison_familywise_ci95']
        lines += ['',f'{names[name]} versus the old rule: **{result["vs_previous"]["mean"]:+.3f}** '
                  f'points, paired approximate 95% CI **[{interval[0]:+.3f}, {interval[1]:+.3f}]**. '
                  f'Conservative two-comparison familywise 95% interval: '
                  f'[{family[0]:+.3f}, {family[1]:+.3f}].']
    gain=results['affordable_t20_b3']['vs_lowest']
    lines += ['',f'Affordable-awkward versus lowest-cost: **{gain["mean"]:+.3f}** points, '
              f'95% CI [{gain["ci95"][0]:+.3f}, {gain["ci95"][1]:+.3f}].', '',
              'These are **4,096 fresh matched two-player Strict deals**, seeds `2640000 + i`. '
              'Both players use the same rule. All variants receive identical initial decks and '
              'hands. The weighted formula is evaluated afresh as a reference, not used inside '
              'the new rules. No new neural-checkpoint evaluation was needed.', '']
    interval=results['affordable_t20_b3']['vs_previous']['two_comparison_familywise_ci95']
    if interval[0] > 0:
        lines += ['The affordable-awkward refinement improves on the previous rule in this '
                  'confirmation sample, including the conservative two-comparison interval. '
                  'This is an empirical heuristic improvement, not a proof of optimal thresholds.', '']
    else:
        lines += ['The affordable-awkward variant is **not a statistically established upgrade** '
                  'over the old rule under the conservative comparison. Do not treat the best '
                  'screening result as a confirmed improvement. The simpler variant may still '
                  'be useful as a lower-complexity alternative; near-equal means alone do not '
                  'prove exact equivalence.', '']
    lines += [
        '## How candidates were chosen', '',
        '1. Screen 50 combinations: original worst-card targeting versus hardest-affordable '
        'targeting; minimum jumps 0/5/10/15/20; extra-cost budgets 1/2/3/5/8. '
        'All use 256 shared development deals, seeds `2240000 + i`.',
        '2. Screen 12 variants requiring starters to belong to the target pile, plus six '
        'reverse-jump tie-break variants, on the same development deals. These are exploratory '
        'tests, not independent confirmations.',
        '3. Validate seven chosen variants on 1,024 separate deals, seeds `2440000 + i`. '
        'Retain the simplest near-original variant and the best simpler-budget affordable variant.',
        '4. Freeze those two candidates before evaluating the 4,096-deal confirmation set. '
        'Report both, not just the higher score. The JSON includes all per-deal scores, '
        'specifications, means, tails, and paired intervals.', '',
        '## What did not help', '',
        '- Larger extra-cost budgets were not consistently better. The confirmed candidates '
        'both keep the small **3-point** allowance.',
        '- Requiring the starter card itself to prefer the target pile did not improve screening. '
        'That restriction can remove useful preparation moves.',
        '- Elaborate reverse-jump tie-breaking—reviving the most advanced pile or starting a '
        'reverse chain—did not produce a clear validation improvement worth another rule.', '',
        '### Why a starter need not belong to the target pile', '',
        'One observed state had ascending tops **79, 46**, descending tops **78, 63**, and '
        'hand **95, 27, 89, 13, 10, 54**. Card 10 is most awkward, with minimum cost 53 '
        'on descending 63. Card 54 itself prefers ascending 46 (cost 8), but using it on '
        'descending 63 costs only 9 and moves that pile toward 10. Restricting starters to '
        'their own assigned pile would rule out this useful kind of preparation. This '
        'example explains the rule difference; it is not an isolated causal-value proof.', '',
        '## Reproduction and limits', '',
        'Implementation: `../awkward_refinement.py`. Original rules are preserved. The direct '
        'deterministic evaluator uses the existing game engine, with parity tests against '
        '`RolloutRunner` for both lowest-cost and the refined rule. It does not inspect '
        'opponent cards or deck order when choosing moves.', '',
        'Results remain specific to two-player Strict, with minimum required plays and no '
        'communication. More players, human approximations, communication, and optional extra '
        'plays have not been validated. Thresholds are empirically selected, not universal '
        'constants. Further tuning needs another untouched confirmation set.', '',
        'Commands are in `../README.md`; complete experiment data are in '
        '`awkward_refinement.json`.', '',
    ]
    (args.output_dir/'AWKWARD_REFINEMENT.md').write_text('\n'.join(lines))
    print(json.dumps({name:{'mean':results[name]['summary']['mean'],'vs_previous':results[name]['vs_previous']} for name in names},indent=2))


if __name__ == '__main__':
    main()
