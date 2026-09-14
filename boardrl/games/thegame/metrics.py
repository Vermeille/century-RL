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
    Whether each played card used the special "10 rule". The mean is therefore
    the fraction of cards played with the rule. In ``The Game`` you may play a
    card exactly ten higher (descending piles) or ten lower (ascending piles)
    than the current pile value; such a move has an effective cost of ``-10``.

``plays_before_x``
    The distribution of optional cards played on turns explicitly ended with
    ``x``. Two cards are mandatory while the deck is nonempty, and one is
    mandatory after it empties.

``message_information``
    How strongly the policy over message symbols depends on the state. It is
    zero for both a constant one-hot message and state-independent uniform
    noise, and approaches one for a balanced, state-specific protocol.

These metrics are aggregated across all players and all games. Metrics that
do not apply to the selected game mode are omitted.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch

from boardrl.games.thegame.game import MESSAGE_MOVES
from boardrl.metrics import GameMetrics, Range


def _parse_piles(state: str) -> List[int]:
    """Extract the four pile values from a ``display_with_moves`` string."""

    for line in state.splitlines():
        if line.startswith("Piles:"):
            # Format: ``Piles: asc:1, asc:1, desc:100, desc:100``
            parts = line[len("Piles: ") :].split(" ")
            return [int(p) for p in parts]
    raise ValueError("Could not find pile information in state string")


def _parse_action(state: str) -> int:
    """Extract the number of cards already played on the current turn."""

    for line in state.splitlines():
        if line.startswith("Round:"):
            return int(line.rsplit("Action:", 1)[1].strip())
    raise ValueError("Could not find action information in state string")


def _parse_cards(state: str) -> int:
    """Extract the number of cards remaining in the drawing deck."""

    for line in state.splitlines():
        if line.startswith("Cards:"):
            return int(line.removeprefix("Cards:").strip())
    raise ValueError("Could not find drawing deck size in state string")


def _optional_plays_before_x(state: str) -> int:
    mandatory_plays = 2 if _parse_cards(state) else 1
    return _parse_action(state) - mandatory_plays


def _cost(move: str, piles: List[int]) -> Tuple[int, bool]:
    """Return ``(cost, ten_rule_used)`` for a move."""

    if "->" not in move:
        return 0, False
    card_str, pile_str = move.split("->")
    card, pile = int(card_str), int(pile_str)
    top = piles[pile]
    if pile < 2:  # ascending piles
        return card - top, card - top == -10
    else:  # descending piles
        return top - card, top - card == -10


def _message_information(logits: list[torch.Tensor]) -> float:
    """Summarize how message policies vary across observed states.

    The normalized information is the Jensen-Shannon divergence of the
    per-state policies divided by the maximum entropy of the vocabulary. It is
    zero when every state has the same distribution, whether that distribution
    is one-hot or uniform, and approaches one for balanced, state-specific
    one-hot messages.
    """

    policies = torch.softmax(torch.stack(logits), dim=1)
    log_policies = policies.clamp_min(torch.finfo(policies.dtype).tiny).log()
    conditional_entropy = -(policies * log_policies).sum(dim=1).mean()

    marginal = policies.mean(dim=0)
    marginal_entropy = -(
        marginal * marginal.clamp_min(torch.finfo(marginal.dtype).tiny).log()
    ).sum()

    information = (marginal_entropy - conditional_entropy).clamp_min(0.0)
    max_information = math.log(policies.shape[1])
    return (information / max_information).item()


class Metrics(GameMetrics):
    def __init__(self, data):
        self.data = data

    def print_short_history(self):
        for game in self.data:
            for player in game:
                print(
                    [
                        f"{r.moves[r.action_idx]} ({_cost(r.moves[r.action_idx], _parse_piles(r.state))[0]})"
                        for r in player[:-1]
                    ]
                )
            print("--")

    def metrics(self):
        # self.data.my_points(0) is discounted so it's not good for logging
        points = [game[0][-1].my_points for game in self.data]
        total_cost = 0.0
        total_moves = 0
        lowest_cost_moves = 0
        ten_rule_moves = []
        x_skipped = []
        message_logits = []

        for game in self.data:
            for player in game:
                for rec in player[:-1]:
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

                    chosen_move = rec.moves[rec.action_idx]
                    if "->" in chosen_move:
                        ten_rule_moves.append(int(ten_flags[rec.action_idx]))
                    if chosen_move == "x":
                        x_skipped.append(_optional_plays_before_x(rec.state))

                    message_indices = [
                        i for i, move in enumerate(rec.moves) if move in MESSAGE_MOVES
                    ]
                    if message_indices:
                        # Compare message content distributions conditional on
                        # choosing a message. This avoids conflating the learned
                        # vocabulary with the decision to keep playing cards.
                        distribution = torch.as_tensor(rec.action_distribution).detach()
                        message_logits.append(distribution[message_indices].float())

        avg_cost = total_cost / total_moves if total_moves else 0.0
        ratio_lowest = lowest_cost_moves / total_moves if total_moves else 0.0

        metrics = {
            "points": Range(points),
            "avg_cost": avg_cost,
            "ratio_lowest_cost": ratio_lowest,
            "ten_rule_moves": Range(ten_rule_moves),
            "sensitivity": self.data.sensitivity(),
        }
        if x_skipped:
            metrics["plays_before_x"] = Range(x_skipped)
        if message_logits:
            metrics["message_information"] = _message_information(message_logits)
        return metrics
