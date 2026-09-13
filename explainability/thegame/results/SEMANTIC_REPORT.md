# Strict: a compact semantic strategy

## Outcome

A rounded four-term rule averages **86.47**, versus **80.96** for lowest-cost and **87.50** for YOLO on the same 1,024 fresh deals. It recovers **84.2%** of the observed score advantage, but **does not recover the neural policy almost exactly**.

## The strategy

Consider only the cheapest legal card for each pile. Among those at most four moves, minimize:

```python
cost + hardest_remaining / 4 - same_direction_gap_gain / 5 + 8 * cost / space_left
```

Break ties by immediate cost, then original legal-move order. The complete standalone implementation is in `../compact_strategy.py`; it calls no neural network or decision tree.

- `cost`: signed pile movement; a reverse-10 move costs −10.
- `hardest_remaining`: after playing the candidate, take the cheapest legal placement cost of every still-playable card in your hand, then their maximum.
- `same_direction_gap_gain`: change in absolute distance between this pile and the other ascending/descending pile. Positive values preserve or increase separation.
- `space_left`: distance from this pile’s current top to its forward endpoint, before playing the candidate, floored at one.

In English: **play cheaply, but sometimes pay a little more to leave an easier hand, preserve a reserve pile, or avoid consuming a large fraction of a nearly exhausted pile.**

The remaining-hand feature omits already-unplayable cards. If no retained card is playable, it is 100; an empty retained hand gives zero. This is an empirical fitted convention, not a recommendation to ignore stranded cards. No opponent hand, deck order, or `Mem:` content is used. Memory omission here is not a new memory-ablation experiment.

## Matched gameplay

| Strategy | Mean | SD | Worst 5% mean | 5th percentile | Win rate | Exact action match | Non-lowest match |
|---|---:|---:|---:|---:|---:|---:|---:|
| lowest | 80.96 | 11.58 | 57.62 | 62 | 1.46% | 74.71% | 0.00% |
| tree_8 | 80.62 | 11.52 | 57.11 | 61 | 1.37% | 76.01% | 0.00% |
| tree_64 | 83.67 | 10.96 | 59.80 | 64 | 2.34% | 80.04% | 26.03% |
| fraction | 82.24 | 11.21 | 58.45 | 63 | 1.95% | 75.87% | 9.75% |
| sparse_2 | 83.70 | 11.05 | 58.72 | 63 | 2.34% | 75.69% | 6.52% |
| sparse_4 | 86.20 | 9.97 | 63.12 | 68 | 3.61% | 78.99% | 19.86% |
| rounded_4 | 86.47 | 9.76 | 63.88 | 69 | 4.30% | 79.06% | 20.36% |
| sparse_5 | 85.32 | 10.33 | 62.10 | 67 | 3.81% | 78.98% | 21.70% |
| rounded_5 | 85.47 | 10.16 | 62.62 | 67 | 4.00% | 78.99% | 22.41% |
| teacher | 87.50 | 9.05 | 66.68 | 71 | 4.10% | 100% | 100% |

Rounded four-term rule versus Strict: **-1.03 points**, paired 95% CI [-1.78, -0.28]. Versus lowest-cost: **+5.51**, CI [+4.70, +6.32]. These are approximate normal paired-deal intervals, not multiplicity-adjusted comparisons. The rule is still detectably below Strict on this sample. Similar win rates do not establish equivalence.

## What was learned

1. **The action space is almost solved:** 99.85% of Strict’s final-test actions are the cheapest card for its chosen pile. The hard part is ranking piles, not choosing arbitrary cards on a pile.
2. **Tiny ordinary trees mostly learn the default.** The eight-leaf scorer makes no successful non-lowest predictions here. Sixty-four leaves improve fidelity, but yield a worse game-playing strategy than the four-term formula.
3. **Small formulas transfer to full games.** The rounded four-term formula outperforms greedy over complete trajectories, not merely on the teacher’s recorded states.
4. **Score recovery is not algorithm recovery.** About one action in five differs, and most of Strict’s deliberately non-lowest moves remain unexplained. Symmetry-aware agreement accepts only the same card on equal-top piles of the same direction; it does not close this gap.
5. **More agreement need not mean more points.** The extra constrained-card term and the larger tree do not improve gameplay. These features describe useful surrogate preferences; they do not prove corresponding computations inside the neural network.

## Method and limits

- Teacher: Strict YOLO `step-900.pth`, historical source tag `run-source/YOLO-20260901T125727Z-c6bb16ffa32e`, deterministic argmax. This is not the stochastic training policy or exactly the training evaluator’s low-temperature protocol.
- Corpus: 896 teacher games / 77,049 states, seeds `1840000 + i`; whole-game split 512 train / 128 validation / 256 untouched test. Test agreement uses 22,163 decisions. Action agreement is decision-weighted; gameplay statistics are game-weighted.
- Features: 414 visible-state/candidate summaries. Named action vocabularies contain 7, 11, or 15 generators; all generators producing the teacher’s concrete action are accepted. Shared pile scorers use 35 features. Sparse score search uses 21 semantic features, up to six terms, and joint-choice logistic loss with duplicate-action masking.
- Selection: whole-game validation, then frozen candidates. Four/five-term coefficients were rounded before inspecting their final test/gameplay results. All evaluated candidates are reported; calling the four-term rule preferable after evaluation is exploratory selection, so a fresh confirmation set would be appropriate before further tuning or strong equivalence claims.
- Gameplay: 1,024 initial deals, seeds `1940000 + i`, independently constructed before rollouts. Every strategy receives the same initial deck/hands. Both seats use that strategy. Rule-generated trajectories may differ; teacher labels are not consulted during rule play.
- Worst-5% mean uses the bottom 5% of scores with fractional boundary weighting. Wins mean all cards played (`points == 100`), not self-play seat win-rate.
- No new claim about Omni versus Strict is made by this Strict-only distillation.

## Remaining work toward near-exact recovery

The next targeted experiment should label the residual non-greedy decisions with teacher action margins and fit a small exception rule over richer hand-sequencing features. It should report non-greedy precision/recall and validate on a new split. Simply growing the present trees does not meet the few-lines objective. Current evidence neither proves nor disproves that a near-exact compact rule exists.

Full rule exports, per-deal scores, confidence intervals, checkpoint/corpus fingerprints, and provenance are in the adjacent JSON files. Reproduction commands are in `../README.md`.
