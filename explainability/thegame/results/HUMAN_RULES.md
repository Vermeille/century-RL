# Calculator-free play: work toward your awkward card

## The practical rule

Take a backwards-ten jump when one is available. Otherwise, look at your most awkward **still-playable** card: the one whose cheapest legal placement needs the largest jump. If that jump is at least **10**, advance its closest legal pile using the cheapest card for that pile, provided this costs at most **3 more** than the globally cheapest move. Otherwise play the globally cheapest move.

No weighted score, ratios, memory vector, or lookahead simulator is used. The only new numerical thresholds are **10** and **3**. Ordinary pile/card subtraction is still needed, just as for lowest-cost play.

```python
if backwards_ten_available():
    take_it()
elif hardest_card_needs_at_least(10) and its_pile_costs_at_most_extra(3):
    play_cheapest_card_on_that_pile()
else:
    play_lowest_cost()
```

This is semantic pseudocode. The actual deterministic implementation is `HumanRule(SPECS["hardest_3"])` in `../human_rules.py`. If the awkward card has several equally cheap legal piles, choose the cheapest qualifying pile move. Move ties use original legal-move order; awkward-card ties use hand order. Cards with no currently legal destination are excluded from this rule.

### Example

Ascending tops **10, 40**; descending tops **90, 70**; hand **11, 43, 55**. Greedy plays 11 on 10 for cost 1. But 55 currently needs a jump of at least 15. Playing 43 on 40 costs 3—only two extra—and brings that pile closer to 55. The rule therefore plays 43. This illustrates the rule, not a proof that this particular move has higher expected value in every hidden state.

## Fresh confirmation

| Strategy | Mean points | SD | Worst 5% mean | Win rate |
|---|---:|---:|---:|---:|
| lowest | 81.06 | 11.59 | 55.94 | 0.88% |
| reserve_3 | 81.22 | 11.70 | 55.59 | 1.95% |
| hardest_3 | 83.55 | 11.16 | 59.41 | 3.22% |
| compact | 85.26 | 9.94 | 63.72 | 3.32% |
| teacher | 87.27 | 9.22 | 65.02 | 3.03% |

The awkward-card rule gains **+2.49 points** over lowest-cost; paired approximate 95% CI **[+1.67, +3.31]**.

These are 1,024 fresh matched two-player Strict deals, seeds `2140000 + i`. Both seats use the same strategy. The neural reference is YOLO `step-900.pth`, deterministic argmax. This does not yet establish transfer to other player counts, human communication, or optional extra plays. It is practical heuristic discovery, not a claim about the neural network’s internal algorithm.

## Screening and discarded ideas

All development candidates used the same 256 deals, seeds `2040000 + i`. Candidate selection happened before opening the fresh confirmation results. Only the standalone awkward-card and reserve rules were selected for confirmation.

| Development candidate | Mean points | Gain over greedy |
|---|---:|---:|
| compact_3_w1 | 82.27 | +1.82 |
| compact_4_w1 | 83.28 | +2.83 |
| compact_6_w1 | 83.28 | +2.83 |
| reserve_1 | 80.93 | +0.48 |
| reserve_2 | 82.16 | +1.71 |
| reserve_3 | 82.29 | +1.84 |
| endpoint_2 | 80.48 | +0.04 |
| endpoint_reserve_2 | 82.26 | +1.81 |
| constrained_2 | 80.59 | +0.14 |
| run_3 | 82.09 | +1.64 |
| setup_5 | 81.30 | +0.86 |
| setup_10 | 81.35 | +0.90 |
| hardest_3 | 83.85 | +3.40 |
| run_reserve_3 | 81.74 | +1.30 |
| setup_reserve_3 | 82.16 | +1.71 |
| hardest_reserve_3 | 82.57 | +2.12 |

- `reserve_N`: pay at most N extra to use an already-more-advanced pile instead of an untouched/lagging reserve pile. All such rules take an available reverse-ten first.
- `run_3`: pay at most three extra to start a same-pile run with a follow-up costing at most three.
- `setup_N`: pay at most N extra to set up a reverse-ten with another card already in hand.
- `endpoint_2`: pay at most two extra to save a pile with at most ten spaces left when another has at least twenty.
- `constrained_2`: pay at most two extra for a card with only one legal destination.
- Combined rule names apply the named priorities in order; they did **not** reliably improve on the standalone tactics.
- `compact_3/4/6_w1`: three-, four-, and six-leaf Boolean replacement trees trained to imitate the previous four-term formula. Their high offline agreement did not preserve the formula’s full-game strength. More neural-target trees were fit but not deployed because they used fractions or were too complex for this task.

## Interpretation

The useful idea is not “always play the awkward card now.” It is **use an affordable move to make progress toward it**, instead of repeatedly taking cheap plays elsewhere and leaving the awkward card expensive. This is testable guidance backed by full games; the weighted formula remains a stronger but less table-friendly reference.

The thresholds are heuristics, not proven optima. The confirmation set was used once; further tuning should use new development and confirmation deals. Full scores, paired differences, every screened candidate, and fitted Boolean trees are in `human_rules_evaluation.json`. Reproduction commands are in `../README.md`.
