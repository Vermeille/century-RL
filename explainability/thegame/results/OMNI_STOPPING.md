# Omni: when to play another card

## Short practical rule

**After meeting the turn minimum, keep playing while you have a legal move costing at most 3. Otherwise end the turn. Reconsider after every card.** A backwards-ten jump has cost −10, so it qualifies automatically.

This is a small empirical surrogate, not an exact reconstruction of the checkpoint. The minimum is two cards while the draw pile is nonempty and one after it empties. Ending early is never allowed. The rule can produce several extra plays, not just one.

## What the fitted tree found

An eight-leaf classifier reduces exactly to:

| Cards left in draw pile | Continue if cheapest legal move costs at most |
|---|---:|
| 62 or more | 2 |
| 4–61 | 4 |
| 0–3 | 7 |

The early/middle/end pattern describes the fitted classifier, not a proven reason inside the network. The exact boundaries 61 and 3 were learned. A rounded version uses more than 60 / 1–60 / empty instead. Counting the deck precisely may be less convenient than the constant-three rule.

Trees of up to eight leaves given the other hand and upcoming cards still chose only own minimum cost and deck size. This does **not** show that the neural policy ignores the other hand, draws, or memory; unmodeled exceptions remain.

## Isolated stopping-rule test

All rows below use the checkpoint’s card ranking. Only the decision to play a card versus end the turn is replaced. If continuing is forced when the model would stop, select its highest-logit legal card. Thus a minimum-cost gate does not necessarily force the model to choose that cheapest card.

| Stopping decision | Mean points | SD | Worst 5% mean | Wins | Held-out stop/play agreement |
|---|---:|---:|---:|---:|---:|
| Original Omni policy | 90.32 | 8.82 | 68.00 | 8.59% | 100.00% |
| Always stop after minimum | 86.29 | 9.98 | 62.74 | 2.93% | 64.21% |
| Cheapest cost ≤3 | 90.00 | 9.12 | 66.80 | 8.98% | 80.71% |
| Cheapest cost ≤4 | 89.40 | 9.13 | 66.96 | 9.28% | 79.45% |
| Fitted 2 / 4 / 7 phases | 89.42 | 9.08 | 67.31 | 6.45% | 82.28% |
| Rounded phases | 89.59 | 9.01 | 67.68 | 6.74% | 82.18% |

These are 1,024 fresh matched deals, seeds `2940000 + i`. Both seats use the same policy. Checkpoint: Omni YOLO-yay2 `step-900.pth`, argmax, using historical source `run-source/YOLO-yay2-20260901T062612Z-604c1b8083e4`.

- Cheapest cost ≤3 versus the original policy: -0.32 points, paired 95% CI [-0.94, +0.31]. Versus forced stopping: +3.71, CI [+3.02, +4.40].
- Fitted 2 / 4 / 7 phases versus the original policy: -0.90 points, paired 95% CI [-1.55, -0.25]. Versus forced stopping: +3.13, CI [+2.42, +3.84].

The phase rule has higher held-out decision agreement, but did not produce a higher mean gameplay score than the simpler constant-three rule in this batch. This is another reason to distinguish faithful imitation from useful advice.

## Fully non-neural check

Here both card selection and stopping are explicit rules: choose the cheapest card, and use the stopping rule below. These rules work without the other hand or upcoming draws; tests also check identical Free/Omni trajectories for the own-information rule. They were evaluated on a separate 2,048-deal batch, seeds `3040000 + i`; compare rows within this table, not raw scores across tables.

| Stopping rule with lowest-cost cards | Mean points | Gain versus forced stopping |
|---|---:|---:|
| Always stop after minimum | 81.04 | +0.00 |
| Cheapest cost ≤3 | 86.40 | +5.36 |
| Cheapest cost ≤4 | 86.40 | +5.36 |
| Fitted 2 / 4 / 7 phases | 86.06 | +5.02 |
| Rounded phases | 86.08 | +5.04 |

Constant-three improvement over greedy forced stopping: **+5.36** points, paired approximate 95% CI **[+4.78, +5.94]**.

This is the directly tested calculator-free advice. It does not establish the best stopping rule when card selection follows the separate awkward-card tactic; that combination would need its own evaluation.

## Observed neural continuation frequencies

Percent of held-out optional decisions where the original policy played another card, conditioned on minimum available cost and deck phase. These are conditional observations, not interventions.

| Deck phase | Reverse-ten | Cost 1–2 | Cost 3–4 | Cost 5–7 | Cost ≥8 |
|---|---:|---:|---:|---:|---:|
| early | 98.5% (n=67) | 56.9% (n=708) | 30.6% (n=640) | 13.7% (n=861) | 3.1% (n=1190) |
| middle | 98.8% (n=241) | 84.9% (n=1817) | 53.2% (n=1526) | 25.8% (n=1801) | 3.0% (n=3285) |
| near_empty | 100.0% (n=69) | 98.4% (n=311) | 90.1% (n=233) | 78.2% (n=243) | 19.2% (n=608) |

## Method and limits

- Collected 54,542 optional decisions from 1,024 teacher games. Whole-game split: 512 train / 256 validation / 256 test; 13,600 test decisions. Compulsory card plays and states without a choice between card and stop are excluded.
- Fit Boolean decision trees with 2/3/4/6/8 leaves, plus simple cost thresholds. Own-only, full-Omni, and model-preferred-card feature groups were compared. The tested threshold and phase candidates were frozen before opening final test and gameplay results. All evaluated candidates are reported.
- Features are parsed from displayed text. Memory is not a surrogate feature. Model inference stays on CUDA with no silent CPU fallback.
- Frozen rules are evaluated over complete trajectories, not just teacher states. Matched initial games are constructed with independent `random.seed(seed + i)` before rollouts, preserving deal pairing across strategies.
- Uncertainty intervals are approximate paired-game normal intervals and are not adjusted for multiple comparisons. Similar means are not proof of equivalence.
- Results are for the repository’s two-player rules. More players, human communication, and human implementation errors have not been evaluated.

Scripts and commands are in `../README.md`. Full per-game scores, rule exports, classification metrics, conditional rates, and provenance are in `omni_stopping.json`.
