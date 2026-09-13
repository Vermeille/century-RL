"""Render configuration screening and independent confirmations without refitting."""
import argparse
import json
from pathlib import Path

import numpy as np

from explainability.thegame.config_gap import summary


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    error = 1.96*values.std(ddof=1)/np.sqrt(len(values))
    return {'mean': mean, 'ci95': [float(mean-error), float(mean+error)]}


def describe(value):
    low, high = value['ci95']
    return f"{value['mean']:+.2f} [{low:+.2f}, {high:+.2f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('explainability/thegame/results'))
    args = parser.parse_args()
    load = lambda name: json.loads((args.results / name).read_text())
    screen = load('config_gap_screen.json')
    strict = load('config_gap_confirmation.json')
    supplement = load('config_gap_supplement_confirmation.json')
    strict['configurations'].update(supplement['configurations'])
    free = load('config_gap_free_confirmation.json')
    final = load('config_gap_final_confirmation.json')
    default = strict['configurations']['strict-v100-p4-m2-n2']
    default_gap = (np.asarray(default['results']['formula']['scores'])
                   - np.asarray(default['results']['greedy']['scores']))*100/98
    derived = {'strict': {}, 'free': {}, 'final': {}}
    lines = [
        '# Which configurations separate greedy and smarter play?', '',
        '## Scope', '',
        'This is a configuration search using **frozen, explicit policy probes**, not newly trained neural policies or an optimal solver. '
        'The policies use only their current hand, pile tops, legal moves, and public rule constants. '
        'A small gap here does not prove a configuration lacks deeper strategic headroom.', '',
        '- **Greedy:** lowest immediate cost, including reverse-ten; stop as soon as permitted.',
        '- **Awkward:** our fixed calculator-free rule (awkwardness at least 20; help an affordable best pile within +3 of greedy).',
        '- **Formula:** the stronger four-term surrogate, not the original neural policy. Fixed coefficients; no per-configuration fitting. '
        'For other pile counts, paired-pile separation becomes the spread of all same-direction piles; for other maxima, endpoint space changes accordingly.', '',
        'Only `max_value` and `num_players` are production constructor knobs. '
        '`ExperimentalGame` subclasses the production engine to vary the minimum and even pile counts without changing production rules. '
        'Default settings reproduce production trajectories, and both advanced policies reproduce their earlier implementations in parity tests.', '',
        '## Search and confirmation', '',
        f'Screened {len(screen["configurations"])} configurations with {screen["protocol"]["games"]} paired deals each: '
        'maximum 50/100/150; 2/4/6 piles, equally split ascending/descending; minimum 1/2/3/4; players 1–5. '
        'The reverse jump remains **10**, not a percentage of the maximum. '
        'Hands remain 7 cards for 1–3 players and 6 for 4–5; the minimum drops to 1 after the deck empties.', '',
        'The first 11 confirmations covered the leading maximum-100/minimum-three settings (2, 3, and 5 players), '
        'the best maximum-50 setting, a maximum-150/six-pile/four-minimum setting, the default, and one-knob diagnostic controls. '
        'A supplemental batch covered four additional high-ranked maximum-150/six-pile settings from the completed screen '
        '(minimum 3 with 3/4/5 players; minimum 1 with 2 players), including the overall screening leader. '
        'Each batch froze its configurations before running those confirmations; the supplemental choices use screen ranks, not new policy fitting. '
        'Together these provide '
        f'for {len(strict["configurations"])} independent confirmations of {strict["protocol"]["games"]:,} games each. '
        'No policy thresholds or formula coefficients were tuned in this search.', '',
        'Scores start at 2 and cap at `max_value`. Therefore percentage of cards played is '
        '`100 * (points - 2) / (max_value - 2)`, not simply points/max. '
        'A raw point gap increases mechanically when the deck grows; normalized gaps are the primary cross-configuration comparison.', '',
        '## Strict: card-selection differences, no optional stopping', '',
        '| Max / piles / minimum / players | Greedy points | Awkward points | Formula points | Formula gain in % of deck | Formula gap versus default gap, 95% CI |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for key, row in strict['configurations'].items():
        c, r = row['configuration'], row['results']
        baseline = np.asarray(r['greedy']['scores'])
        gap = (np.asarray(r['formula']['scores'])-baseline)*100/(c['max_value']-2)
        comparison = interval(gap-default_gap)
        derived['strict'][key] = {'normalized_gap_vs_default': comparison}
        label = f"{c['max_value']} / {c['num_piles']} / {c['minimum']} / {c['num_players']}"
        values = [r[name]['summary']['mean'] for name in ['greedy', 'awkward', 'formula']]
        lines.append(f'| {label} | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} | {gap.mean():+.2f} pp | {describe(comparison)} |')
    lines += ['', 'Intervals are approximate paired-seed normal intervals; no multiplicity adjustment. '
              'Deals are paired **between policies within a configuration**. Across configurations the same seed index '
              'need not produce the same initial deal, especially with changed deck size or player count. '
              'The paired-seed difference-of-differences measures whether the policy gap widens beyond the default.', '',
              '### Completion and remaining-card metrics', '',
              '| Max / piles / minimum / players | Greedy wins | Awkward wins | Formula wins | Formula reduction in unplayed cards |',
              '|---|---:|---:|---:|---:|']
    for row in strict['configurations'].values():
        c, r = row['configuration'], row['results']
        label = f"{c['max_value']} / {c['num_piles']} / {c['minimum']} / {c['num_players']}"
        rates = [100*r[name]['summary']['win_rate'] for name in ['greedy','awkward','formula']]
        reduction = r['formula']['summary']['leftover_reduction']*100
        lines.append(f'| {label} | {rates[0]:.2f}% | {rates[1]:.2f}% | {rates[2]:.2f}% | {reduction:.1f}% |')
    lines += ['', '## Free: control for cheap stopping-threshold improvements', '',
              'For the default and minimum-three configurations, calibrated each policy independently on '
              '256 separate games over stop-immediately or continue at cost ≤0/1/2/3/5/8. '
              'Freeze each best calibration threshold, then evaluate on 2,048 new paired games. '
              'This avoids counting a calibrated stop threshold as evidence of sophisticated card selection. '
              'All seats use the same strategy. These own-information policies receive no Omni-only information.', '',
              '| Minimum | Greedy always stop | Calibrated greedy | Calibrated awkward | Calibrated formula | Formula minus calibrated greedy, 95% CI |',
              '|---|---:|---:|---:|---:|---:|']
    for key, row in free['configurations'].items():
        r = row['results']
        baseline = r['greedy_calibrated']['scores']
        comparisons = {name: summary(value['scores'], baseline, row['configuration']['max_value'])
                       for name, value in r.items()}
        derived['free'][key] = comparisons
        delta = interval(np.asarray(r['formula_calibrated']['scores'])-np.asarray(baseline))
        values = [r[name]['summary']['mean'] for name in ['greedy','greedy_calibrated','awkward_calibrated','formula_calibrated']]
        lines.append(f"| {row['configuration']['minimum']} | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} | {values[3]:.2f} | {describe(delta)} |")
    lines += ['', 'Selected stopping limits:', '']
    for key, variants in free['protocol']['variants'].items():
        lines.append(f'- `{key}`: '+', '.join(f'{label}: {stop}' for label, name, stop in variants if label.endswith('calibrated'))+'.')
    free_gaps = {}
    for row in free['configurations'].values():
        r = row['results']
        free_gaps[row['configuration']['minimum']] = (np.asarray(r['formula_calibrated']['scores'])
                                                     - np.asarray(r['greedy_calibrated']['scores']))
    widening = interval(free_gaps[3]-free_gaps[2])
    derived['free_gap_widening'] = widening
    lines += ['', f'Increasing the minimum from two to three changes the calibrated formula-minus-greedy gap by '
              f'**{describe(widening)} points**. The interval includes zero: suggestive, not a confirmed widening.', '']
    lines += ['', '## How to interpret “83 to 86”', '',
              'At maximum 100, that is **17 to 14 cards left**, a **17.6% reduction in leftovers**, '
              'or **3.06 percentage points more of the 98-card deck played**. '
              'That can be a useful improvement while still coming from a small heuristic. '
              'Score measures effectiveness, not the complexity, generality, or human difficulty of the strategy.', '',
              'A more discriminating benchmark would report:', '',
              '1. Paired points and fraction of deck played, win probability, lower-tail scores, and leftover reduction.',
              '2. Improvement over a **calibrated simple baseline**, not just the weakest greedy implementation.',
              '3. A complexity ladder: greedy; threshold tweaks; several short semantic tactics; lookahead; trained policy.',
              '4. Held-out configurations and targeted situations requiring reverse-jump setup, avoiding stranded cards, '
              'preserving pile flexibility, or coordinating with another hand. These are proposed diagnostics, not results of this sweep.', '',
              '## Limits and reproducibility', '',
              '- The search is finite and probes transfer of default-trained/default-designed heuristics. '
              'It cannot identify the configuration with the largest **optimal-policy** gap. Training or solving each configuration could change the ranking.',
              '- Neither YOLO checkpoint was evaluated out of distribution or retrained here. '
              'In particular, the formula column must not be labeled neural performance.',
              '- Changing player count also changes cards simultaneously held, and crossing from three to four players changes hand size. '
              'Changing maximum value changes the reverse-ten jump relative to deck size. These are real rule interactions, not isolated difficulty sliders.',
              '- The Free confirmation checks two chosen settings, not the full 180-setting grid. '
              'It is not a full search over stopping policies, and 256-game calibration can choose a suboptimal threshold.',
              '- Initial games use deterministic `random.seed(seed+i)`, with full terminal play and no rollout truncation. '
              'Raw scores, selected configurations, seeds, and source hashes are in the adjacent JSON files.',
              '- Screening seeds: 3240000+i; Strict confirmation: 3340000+i; Free calibration: 3440000+i; Free confirmation: 3540000+i.', '',
              'Scripts and commands: `../README.md`.', '']
    final_lines = ['## Fresh finalist test', '',
                   'After the 180-setting screen and 15 larger validation comparisons, froze one finalist '
                   '(150 maximum, six piles, three required plays, three players) against the default '
                   'for **4,096 entirely new paired games per configuration**, seed 3640000+i. '
                   'Both are Strict; the jump stays ten. No threshold or formula changes were made.', '',
                   '| Max / piles / minimum / players | Greedy | Awkward | Formula | Formula raw gap | Formula gap as % of deck |',
                   '|---|---:|---:|---:|---:|---:|']
    final_gaps = {}
    for key, row in final['configurations'].items():
        c, r = row['configuration'], row['results']
        values = [r[name]['summary']['mean'] for name in ['greedy', 'awkward', 'formula']]
        gap = np.asarray(r['formula']['scores'])-np.asarray(r['greedy']['scores'])
        normalized = gap*100/(c['max_value']-2)
        final_gaps[key] = normalized
        derived['final'][key] = {'raw_gap': interval(gap), 'normalized_gap': interval(normalized),
                                'paired_effect': float(gap.mean()/gap.std(ddof=1))}
        label = f"{c['max_value']} / {c['num_piles']} / {c['minimum']} / {c['num_players']}"
        final_lines.append(f'| {label} | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} | {gap.mean():+.2f} | {normalized.mean():+.2f} pp |')
    widening = interval(final_gaps['strict-v150-p6-m3-n3']-final_gaps['strict-v100-p4-m2-n2'])
    derived['final_gap_widening'] = widening
    final_lines += ['', f'Finalist-minus-default **normalized gap widening**: {describe(widening)} percentage points '
                    '(paired-seed approximate 95% interval). This is the primary held-out comparison; '
                    'do not use the earlier winning validation estimate as the final effect estimate.', '',
                    'The useful distinction is that greedy plays about **81% of the deck in both configurations**. '
                    'The short awkward-card tactic also improves both by essentially the same **2.7 percentage points**. '
                    'The stronger formula gains **4.91 points of deck percentage on the default versus 5.78 on the finalist**: '
                    'about **18% greater normalized separation**. This is not merely making greedy fail sooner or inflating raw points with a longer deck. '
                    'The formula removes 25.8% of greedy leftovers in the default and 30.8% in the finalist. '
                    'Its paired standardized effect is 0.37 versus 0.48, respectively.', '',
                    'This is a better candidate for separating these two strategy levels, **not** proof of greater general intelligence. '
                    'A fixed formula may itself embody a few effective heuristics. '
                    'The mechanism creating the wider gap has not been causally isolated; no claim about neural reasoning follows.', '',
                    'These are fixed heuristic probes, not retrained neural policies. '
                    'This result concerns performance separation, not an objective measure of intelligence.', '']
    lines[2:2] = final_lines
    (args.results/'CONFIG_GAP.md').write_text('\n'.join(lines))
    (args.results/'config_gap_comparisons.json').write_text(json.dumps(derived, indent=2, allow_nan=False)+'\n')
    # Earlier screen versions used NaN for undefined leftover reduction at a perfect baseline.
    # Preserve every observation; standardize that reporting-only value to JSON null.
    for row in screen['configurations'].values():
        base = row['results']['greedy']['scores']
        for value in row['results'].values():
            value['summary'] = summary(value['scores'], base, row['configuration']['max_value'])
    (args.results/'config_gap_screen.json').write_text(json.dumps(screen, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
