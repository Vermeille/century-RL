"""Metrics for the simplified *The Game* implementation.

The metrics mirror the behaviour of :class:`LowestCostStrategy` defined in
``strategies.py``.  Each record contains the full list of legal moves along with
the index of the chosen action.  Using the pile configuration encoded in the
``state`` string we can recompute the cost of every move and derive a few simple
statistics:

``avg_cost``
    Average cost of the executed moves.

``ratio_lowest_cost``
    Fraction of moves that played the minimum-cost option available.

``ten_rule_moves``
    How often the special "10 rule" was used.  In ``The Game`` you may play a
    card exactly ten higher (descending piles) or ten lower (ascending piles)
    than the current pile value; such a move has an effective cost of ``-10``.

These metrics are aggregated across all players and all games.
"""

from __future__ import annotations

from typing import List, Tuple


def _parse_piles(state: str) -> List[int]:
    """Extract the four pile values from a ``display_with_moves`` string."""

    for line in state.splitlines():
        if line.startswith("Piles:"):
            # Format: ``Piles: asc:1, asc:1, desc:100, desc:100``
            parts = line[len("Piles:") :].split(",")
            return [int(p.split(":")[1]) for p in parts]
    raise ValueError("Could not find pile information in state string")


def _cost(move: str, piles: List[int]) -> Tuple[int, bool]:
    """Return ``(cost, ten_rule_used)`` for a move."""

    card_str, pile_str = move.split("->")
    card, pile = int(card_str), int(pile_str)
    top = piles[pile]
    if pile < 2:  # ascending piles
        return card - top, card - top == -10
    else:  # descending piles
        return top - card, top - card == -10


class Metrics:
    def __init__(self, data):
        self.data = data

    def print_short_history(self):
        for game in self.data:
            for player in game:
                print([r.moves[r.action_idx] for r in player if not getattr(r, "final", False)])
            print("--")

    def metrics_to_visdom(self, viz, epoch):
        avg_points = sum(game[0][-1].my_points for game in self.data) / len(self.data)
        viz.push("avg_points", avg_points, epoch)

        total_cost = 0.0
        total_moves = 0
        lowest_cost_moves = 0
        ten_rule_moves = 0

        for game in self.data:
            for player in game:
                for rec in player:
                    if getattr(rec, "final", False):
                        continue

                    piles = _parse_piles(rec.state)
                    costs = []
                    ten_flags = []
                    for m in rec.moves:
                        c, t = _cost(m, piles)
                        costs.append(c)
                        ten_flags.append(t)

                    chosen_cost = costs[rec.action_idx]
                    total_cost += chosen_cost
                    total_moves += 1
                    if chosen_cost == min(costs):
                        lowest_cost_moves += 1
                    if ten_flags[rec.action_idx]:
                        ten_rule_moves += 1

        avg_cost = total_cost / total_moves if total_moves else 0.0
        ratio_lowest = lowest_cost_moves / total_moves if total_moves else 0.0

        viz.push("avg_cost", avg_cost, epoch)
        viz.push("ratio_lowest_cost", ratio_lowest, epoch)
        viz.push(
            "ten_rule_moves",
            ten_rule_moves / (len(self.data) * game.num_players()),
            epoch,
        )
