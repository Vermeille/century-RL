# Refining the awkward-card rule

The target remains a rule you can use without a calculator. These policies use only card/pile differences, ordering, and small integer thresholds—no weighted move score, hidden state, memory vector, or learned inference.

## Two refinements

**Simpler version:** take an available backwards-ten jump. Otherwise, find your most awkward still-playable card. Advance its closest legal pile with that pile’s cheapest card if doing so costs at most **3 extra** relative to the globally cheapest move. Otherwise play globally cheapest. This removes the old “at least 10” condition.

**Affordable-awkward version:** take an available backwards-ten jump. Otherwise, consider cards from most awkward downward, but only those needing a jump of **at least 20**. Play the cheapest starter on the closest legal pile of the first such card you can help for **at most 3 extra**. If none qualifies, play globally cheapest. This allows helping the second-worst card when the worst is unaffordable.

“Awkwardness” is a card’s cheapest currently legal placement cost. A card with no legal destination is excluded. Tied closest piles are all considered. Starter cards need not themselves prefer the pile they are used on. All ties otherwise retain hand/legal-move order; no additional human tie-break is claimed to be necessary.

## Fresh confirmation

| Rule | Mean points | SD | Worst 5% mean | Wins | Gain vs old rule |
|---|---:|---:|---:|---:|---:|
| Lowest-cost | 81.072 | 11.64 | 56.77 | 1.68% | -2.557 |
| Original 10 / 3 rule | 83.629 | 10.92 | 59.56 | 2.32% | +0.000 |
| Simpler: no minimum jump | 83.579 | 10.92 | 59.54 | 2.25% | -0.050 |
| Affordable-awkward: 20 / 3 | 84.118 | 10.87 | 59.60 | 2.34% | +0.489 |
| Previous weighted formula | 86.270 | 10.01 | 63.26 | 3.71% | +2.641 |

Simpler: no minimum jump versus the old rule: **-0.050** points, paired approximate 95% CI **[-0.103, +0.003]**. Conservative two-comparison familywise 95% interval: [-0.111, +0.010].

Affordable-awkward: 20 / 3 versus the old rule: **+0.489** points, paired approximate 95% CI **[+0.184, +0.794]**. Conservative two-comparison familywise 95% interval: [+0.140, +0.838].

Affordable-awkward versus lowest-cost: **+3.046** points, 95% CI [+2.634, +3.458].

These are **4,096 fresh matched two-player Strict deals**, seeds `2640000 + i`. Both players use the same rule. All variants receive identical initial decks and hands. The weighted formula is evaluated afresh as a reference, not used inside the new rules. No new neural-checkpoint evaluation was needed.

The affordable-awkward refinement improves on the previous rule in this confirmation sample, including the conservative two-comparison interval. This is an empirical heuristic improvement, not a proof of optimal thresholds.

## How candidates were chosen

1. Screen 50 combinations: original worst-card targeting versus hardest-affordable targeting; minimum jumps 0/5/10/15/20; extra-cost budgets 1/2/3/5/8. All use 256 shared development deals, seeds `2240000 + i`.
2. Screen 12 variants requiring starters to belong to the target pile, plus six reverse-jump tie-break variants, on the same development deals. These are exploratory tests, not independent confirmations.
3. Validate seven chosen variants on 1,024 separate deals, seeds `2440000 + i`. Retain the simplest near-original variant and the best simpler-budget affordable variant.
4. Freeze those two candidates before evaluating the 4,096-deal confirmation set. Report both, not just the higher score. The JSON includes all per-deal scores, specifications, means, tails, and paired intervals.

## What did not help

- Larger extra-cost budgets were not consistently better. The confirmed candidates both keep the small **3-point** allowance.
- Requiring the starter card itself to prefer the target pile did not improve screening. That restriction can remove useful preparation moves.
- Elaborate reverse-jump tie-breaking—reviving the most advanced pile or starting a reverse chain—did not produce a clear validation improvement worth another rule.

### Why a starter need not belong to the target pile

One observed state had ascending tops **79, 46**, descending tops **78, 63**, and hand **95, 27, 89, 13, 10, 54**. Card 10 is most awkward, with minimum cost 53 on descending 63. Card 54 itself prefers ascending 46 (cost 8), but using it on descending 63 costs only 9 and moves that pile toward 10. Restricting starters to their own assigned pile would rule out this useful kind of preparation. This example explains the rule difference; it is not an isolated causal-value proof.

## Reproduction and limits

Implementation: `../awkward_refinement.py`. Original rules are preserved. The direct deterministic evaluator uses the existing game engine, with parity tests against `RolloutRunner` for both lowest-cost and the refined rule. It does not inspect opponent cards or deck order when choosing moves.

Results remain specific to two-player Strict, with minimum required plays and no communication. More players, human approximations, communication, and optional extra plays have not been validated. Thresholds are empirically selected, not universal constants. Further tuning needs another untouched confirmation set.

Commands are in `../README.md`; complete experiment data are in `awkward_refinement.json`.
