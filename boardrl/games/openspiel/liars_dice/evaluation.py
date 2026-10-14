"""Exact Liar's Dice best responses on the public bid tree."""

from collections import Counter
from itertools import combinations_with_replacement
from math import factorial

import numpy as np

from boardrl.games.openspiel.liars_dice.game import LiarsDice
from boardrl.games.openspiel.result import NashConvResult


class LiarsDiceEvaluator:
    """Exact best responses on the public bid tree, with private-hand beliefs.

    Increasing bids identify each public history with a bit mask. Forward
    propagation computes each player's chance-weighted realization reach for
    every private hand. Backward induction maximizes separately for the
    responder's hand, AFTER summing over the opponent's hidden hand. Opponent
    nodes sum counterfactual values. This enforces information-set consistency.
    """

    def __init__(self, game):
        parameters = game.get_parameters()
        self.dice = parameters["numdice"]
        self.faces = parameters["dice_sides"]
        if (
            parameters["players"] != 2
            or parameters["bidding_rule"] != "reset-face"
            or self.dice < 1
            or not 2 <= self.faces <= 9
            or any(
                parameters.get(f"numdice{player}", self.dice) != self.dice
                for player in (0, 1)
            )
        ):
            raise ValueError(
                "exact dice evaluation requires symmetric two-player reset-face dice"
            )
        self.bids = 2 * self.dice * self.faces
        self.hands = tuple(
            combinations_with_replacement(range(1, self.faces + 1), self.dice)
        )
        self.prior = np.array(
            [
                factorial(self.dice)
                / np.prod([factorial(n) for n in Counter(hand).values()])
                / self.faces**self.dice
                for hand in self.hands
            ]
        )
        self.hand_text = ["".join(map(str, hand)) for hand in self.hands]
        state = game.new_initial_state()
        self.labels = tuple(
            state.action_to_string(0, action) for action in range(self.bids + 1)
        )
        self._view = LiarsDice(spiel_game=game, state=state, resolve_chance=False)
        # Payoff to the bidder, including the highest face as wild.
        counts = np.array(
            [
                [
                    sum(die == face or die == self.faces for die in hand)
                    for face in range(1, self.faces + 1)
                ]
                for hand in self.hands
            ]
        )
        self.payoffs = np.array(
            [
                np.where(
                    counts[:, bid % self.faces, None]
                    + counts[None, :, bid % self.faces]
                    >= bid // self.faces + 1,
                    1.0,
                    -1.0,
                )
                for bid in range(self.bids)
            ]
        )

    def _policies(self, masks, policy, batch_size):
        texts, action_counts, tables = [], [], []
        for mask in masks:
            last = mask.bit_length() - 1
            legal = list(range(last + 1, self.bids)) + ([self.bids] if mask else [])
            if len(legal) == 1:
                tables.append(np.ones((len(self.hands), 1)))
                continue
            history = "".join(
                f" {self.labels[bid]}" for bid in range(self.bids) if mask & (1 << bid)
            )
            moves = "\n" + "\n".join(f"@{self.labels[action]}" for action in legal)
            player = mask.bit_count() % 2
            texts.extend(
                self._view._display_player_view(hand + history, player) + moves
                for hand in self.hand_text
            )
            action_counts.extend([len(legal)] * len(self.hands))
            tables.append(None)
        predictions = []
        for start in range(0, len(texts), batch_size):
            predictions.extend(
                policy(
                    texts[start : start + batch_size],
                    action_counts[start : start + batch_size],
                )
            )
        offset = 0
        for index, table in enumerate(tables):
            if table is None:
                tables[index] = np.array(predictions[offset : offset + len(self.hands)])
                offset += len(self.hands)
        return tables

    def evaluate(self, policy, *, batch_size):
        import psutil

        nodes, hands = 1 << self.bids, len(self.hands)
        required = nodes * 2 * hands * 8 * 2
        if required > psutil.virtual_memory().available:
            raise RuntimeError(
                f"Exact dice evaluation needs at least {required / 2**30:.1f} GiB RAM"
            )
        reach = np.zeros((nodes, 2, hands), dtype=np.float64)
        values = np.zeros_like(reach)
        reach[0] = self.prior
        values[0, 0] = -np.inf  # The initial state does not allow calling Liar.
        expected_value = 0.0
        masks_per_batch = max(1, batch_size // hands)
        for start in range(0, nodes, masks_per_batch):
            masks = range(start, min(start + masks_per_batch, nodes))
            tables = self._policies(masks, policy, batch_size)
            for mask, probabilities in zip(masks, tables, strict=True):
                player = mask.bit_count() % 2
                other = 1 - player
                last = mask.bit_length() - 1
                if mask:
                    payoff = self.payoffs[last] * (1 if player == 1 else -1)
                    caller_reach = reach[mask].copy()
                    caller_reach[player] *= probabilities[:, -1]
                    # The responder chooses Liar freely, without multiplying
                    # by its own policy; only its opponent's reach is weighted.
                    values[mask, 0] = payoff @ (
                        caller_reach[1] if player == 1 else reach[mask, 1]
                    )
                    values[mask, 1] = -payoff.T @ (
                        caller_reach[0] if player == 0 else reach[mask, 0]
                    )
                    expected_value += caller_reach[0] @ payoff @ caller_reach[1]
                for column, bid in enumerate(range(last + 1, self.bids)):
                    child = mask | (1 << bid)
                    reach[child, player] = (
                        reach[mask, player] * probabilities[:, column]
                    )
                    reach[child, other] = reach[mask, other]
        del reach
        for mask in range(nodes - 1, 0, -1):
            parent = mask ^ (1 << (mask.bit_length() - 1))
            player = 1 - mask.bit_count() % 2
            values[parent, player] = np.maximum(
                values[parent, player], values[mask, player]
            )
            values[parent, 1 - player] += values[mask, 1 - player]
        improvements = (
            float(self.prior @ values[0, 0] - expected_value),
            float(self.prior @ values[0, 1] + expected_value),
        )
        total = sum(improvements)
        return NashConvResult(total, total / 2, improvements, nodes * hands)
