# Which configurations separate greedy and smarter play?

## Fresh finalist test

After the 180-setting screen and 15 larger validation comparisons, froze one finalist (150 maximum, six piles, three required plays, three players) against the default for **4,096 entirely new paired games per configuration**, seed 3640000+i. Both are Strict; the jump stays ten. No threshold or formula changes were made.

| Max / piles / minimum / players | Greedy | Awkward | Formula | Formula raw gap | Formula gap as % of deck |
|---|---:|---:|---:|---:|---:|
| 100 / 4 / 2 / 2 | 81.34 | 84.04 | 86.15 | +4.81 | +4.91 pp |
| 150 / 6 / 3 / 3 | 122.29 | 126.33 | 130.83 | +8.55 | +5.78 pp |

Finalist-minus-default **normalized gap widening**: +0.87 [+0.31, +1.42] percentage points (paired-seed approximate 95% interval). This is the primary held-out comparison; do not use the earlier winning validation estimate as the final effect estimate.

The useful distinction is that greedy plays about **81% of the deck in both configurations**. The short awkward-card tactic also improves both by essentially the same **2.7 percentage points**. The stronger formula gains **4.91 points of deck percentage on the default versus 5.78 on the finalist**: about **18% greater normalized separation**. This is not merely making greedy fail sooner or inflating raw points with a longer deck. The formula removes 25.8% of greedy leftovers in the default and 30.8% in the finalist. Its paired standardized effect is 0.37 versus 0.48, respectively.

This is a better candidate for separating these two strategy levels, **not** proof of greater general intelligence. A fixed formula may itself embody a few effective heuristics. The mechanism creating the wider gap has not been causally isolated; no claim about neural reasoning follows.

These are fixed heuristic probes, not retrained neural policies. This result concerns performance separation, not an objective measure of intelligence.

## Scope

This is a configuration search using **frozen, explicit policy probes**, not newly trained neural policies or an optimal solver. The policies use only their current hand, pile tops, legal moves, and public rule constants. A small gap here does not prove a configuration lacks deeper strategic headroom.

- **Greedy:** lowest immediate cost, including reverse-ten; stop as soon as permitted.
- **Awkward:** our fixed calculator-free rule (awkwardness at least 20; help an affordable best pile within +3 of greedy).
- **Formula:** the stronger four-term surrogate, not the original neural policy. Fixed coefficients; no per-configuration fitting. For other pile counts, paired-pile separation becomes the spread of all same-direction piles; for other maxima, endpoint space changes accordingly.

Only `max_value` and `num_players` are production constructor knobs. `ExperimentalGame` subclasses the production engine to vary the minimum and even pile counts without changing production rules. Default settings reproduce production trajectories, and both advanced policies reproduce their earlier implementations in parity tests.

## Search and confirmation

Screened 180 configurations with 128 paired deals each: maximum 50/100/150; 2/4/6 piles, equally split ascending/descending; minimum 1/2/3/4; players 1–5. The reverse jump remains **10**, not a percentage of the maximum. Hands remain 7 cards for 1–3 players and 6 for 4–5; the minimum drops to 1 after the deck empties.

The first 11 confirmations covered the leading maximum-100/minimum-three settings (2, 3, and 5 players), the best maximum-50 setting, a maximum-150/six-pile/four-minimum setting, the default, and one-knob diagnostic controls. A supplemental batch covered four additional high-ranked maximum-150/six-pile settings from the completed screen (minimum 3 with 3/4/5 players; minimum 1 with 2 players), including the overall screening leader. Each batch froze its configurations before running those confirmations; the supplemental choices use screen ranks, not new policy fitting. Together these provide for 15 independent confirmations of 2,048 games each. No policy thresholds or formula coefficients were tuned in this search.

Scores start at 2 and cap at `max_value`. Therefore percentage of cards played is `100 * (points - 2) / (max_value - 2)`, not simply points/max. A raw point gap increases mechanically when the deck grows; normalized gaps are the primary cross-configuration comparison.

## Strict: card-selection differences, no optional stopping

| Max / piles / minimum / players | Greedy points | Awkward points | Formula points | Formula gain in % of deck | Formula gap versus default gap, 95% CI |
|---|---:|---:|---:|---:|---:|
| 100 / 4 / 2 / 2 | 81.37 | 84.06 | 86.28 | +5.01 pp | +0.00 [+0.00, +0.00] |
| 100 / 4 / 3 / 2 | 72.21 | 74.49 | 76.83 | +4.71 pp | -0.30 [-1.11, +0.51] |
| 100 / 4 / 3 / 3 | 78.64 | 80.99 | 82.93 | +4.37 pp | -0.64 [-1.47, +0.19] |
| 100 / 4 / 3 / 5 | 68.84 | 70.63 | 72.00 | +3.22 pp | -1.79 [-2.69, -0.88] |
| 50 / 2 / 2 / 2 | 40.34 | 41.06 | 41.91 | +3.26 pp | -1.76 [-2.67, -0.84] |
| 150 / 6 / 4 / 2 | 96.90 | 100.13 | 103.84 | +4.69 pp | -0.32 [-1.04, +0.40] |
| 150 / 4 / 2 / 2 | 71.55 | 73.52 | 76.11 | +3.08 pp | -1.94 [-2.59, -1.28] |
| 100 / 2 / 2 / 2 | 31.38 | 31.82 | 32.29 | +0.92 pp | -4.09 [-4.71, -3.47] |
| 100 / 4 / 4 / 2 | 62.28 | 63.51 | 65.89 | +3.69 pp | -1.33 [-2.08, -0.58] |
| 100 / 4 / 2 / 3 | 88.63 | 91.05 | 92.58 | +4.03 pp | -0.98 [-1.71, -0.25] |
| 100 / 4 / 2 / 5 | 84.77 | 86.53 | 88.21 | +3.51 pp | -1.50 [-2.27, -0.74] |
| 150 / 6 / 3 / 3 | 121.87 | 126.00 | 130.71 | +5.98 pp | +0.96 [+0.18, +1.75] |
| 150 / 6 / 3 / 5 | 111.37 | 114.86 | 119.47 | +5.47 pp | +0.46 [-0.35, +1.26] |
| 150 / 6 / 3 / 4 | 106.80 | 109.30 | 114.48 | +5.19 pp | +0.18 [-0.60, +0.96] |
| 150 / 6 / 1 / 2 | 136.39 | 140.33 | 143.10 | +4.53 pp | -0.48 [-1.17, +0.20] |

Intervals are approximate paired-seed normal intervals; no multiplicity adjustment. Deals are paired **between policies within a configuration**. Across configurations the same seed index need not produce the same initial deal, especially with changed deck size or player count. The paired-seed difference-of-differences measures whether the policy gap widens beyond the default.

### Completion and remaining-card metrics

| Max / piles / minimum / players | Greedy wins | Awkward wins | Formula wins | Formula reduction in unplayed cards |
|---|---:|---:|---:|---:|
| 100 / 4 / 2 / 2 | 1.42% | 2.78% | 3.22% | 26.4% |
| 100 / 4 / 3 / 2 | 0.29% | 0.73% | 0.59% | 16.6% |
| 100 / 4 / 3 / 3 | 0.39% | 1.27% | 1.76% | 20.1% |
| 100 / 4 / 3 / 5 | 0.05% | 0.05% | 0.10% | 10.1% |
| 50 / 2 / 2 / 2 | 5.57% | 5.03% | 8.35% | 16.2% |
| 150 / 6 / 4 / 2 | 0.00% | 0.00% | 0.05% | 13.1% |
| 150 / 4 / 2 / 2 | 0.00% | 0.00% | 0.00% | 5.8% |
| 100 / 2 / 2 / 2 | 0.00% | 0.00% | 0.00% | 1.3% |
| 100 / 4 / 4 / 2 | 0.05% | 0.05% | 0.00% | 9.6% |
| 100 / 4 / 2 / 3 | 3.86% | 6.45% | 8.64% | 34.8% |
| 100 / 4 / 2 / 5 | 1.27% | 1.86% | 2.34% | 22.6% |
| 150 / 6 / 3 / 3 | 0.78% | 1.07% | 2.05% | 31.4% |
| 150 / 6 / 3 / 5 | 0.10% | 0.05% | 0.20% | 21.0% |
| 150 / 6 / 3 / 4 | 0.05% | 0.00% | 0.15% | 17.8% |
| 150 / 6 / 1 / 2 | 5.86% | 10.79% | 15.97% | 49.3% |

## Free: control for cheap stopping-threshold improvements

For the default and minimum-three configurations, calibrated each policy independently on 256 separate games over stop-immediately or continue at cost ≤0/1/2/3/5/8. Freeze each best calibration threshold, then evaluate on 2,048 new paired games. This avoids counting a calibrated stop threshold as evidence of sophisticated card selection. All seats use the same strategy. These own-information policies receive no Omni-only information.

| Minimum | Greedy always stop | Calibrated greedy | Calibrated awkward | Calibrated formula | Formula minus calibrated greedy, 95% CI |
|---|---:|---:|---:|---:|---:|
| 2 | 81.52 | 86.06 | 88.11 | 90.39 | +4.33 [+3.81, +4.84] |
| 3 | 72.26 | 74.86 | 77.44 | 79.93 | +5.07 [+4.47, +5.67] |

Selected stopping limits:

- `free-v100-p4-m2-n2`: greedy_calibrated: 2, awkward_calibrated: 3, formula_calibrated: 3.
- `free-v100-p4-m3-n2`: greedy_calibrated: 2, awkward_calibrated: 3, formula_calibrated: 3.

Increasing the minimum from two to three changes the calibrated formula-minus-greedy gap by **+0.75 [-0.04, +1.53] points**. The interval includes zero: suggestive, not a confirmed widening.


## How to interpret “83 to 86”

At maximum 100, that is **17 to 14 cards left**, a **17.6% reduction in leftovers**, or **3.06 percentage points more of the 98-card deck played**. That can be a useful improvement while still coming from a small heuristic. Score measures effectiveness, not the complexity, generality, or human difficulty of the strategy.

A more discriminating benchmark would report:

1. Paired points and fraction of deck played, win probability, lower-tail scores, and leftover reduction.
2. Improvement over a **calibrated simple baseline**, not just the weakest greedy implementation.
3. A complexity ladder: greedy; threshold tweaks; several short semantic tactics; lookahead; trained policy.
4. Held-out configurations and targeted situations requiring reverse-jump setup, avoiding stranded cards, preserving pile flexibility, or coordinating with another hand. These are proposed diagnostics, not results of this sweep.

## Limits and reproducibility

- The search is finite and probes transfer of default-trained/default-designed heuristics. It cannot identify the configuration with the largest **optimal-policy** gap. Training or solving each configuration could change the ranking.
- Neither YOLO checkpoint was evaluated out of distribution or retrained here. In particular, the formula column must not be labeled neural performance.
- Changing player count also changes cards simultaneously held, and crossing from three to four players changes hand size. Changing maximum value changes the reverse-ten jump relative to deck size. These are real rule interactions, not isolated difficulty sliders.
- The Free confirmation checks two chosen settings, not the full 180-setting grid. It is not a full search over stopping policies, and 256-game calibration can choose a suboptimal threshold.
- Initial games use deterministic `random.seed(seed+i)`, with full terminal play and no rollout truncation. Raw scores, selected configurations, seeds, and source hashes are in the adjacent JSON files.
- Screening seeds: 3240000+i; Strict confirmation: 3340000+i; Free calibration: 3440000+i; Free confirmation: 3540000+i.

Scripts and commands: `../README.md`.
