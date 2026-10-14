"""Shared evaluation for the 3x3 hidden board games using exp-a-spiel.

The optional library is pinned to revision
9412882c24e521fad7cbd96fdd68311d9bf6f203; see docs/benchmarking.md.
"""

from abc import ABC, abstractmethod

import os
import sys
from typing import ClassVar

import numpy as np

from boardrl.games.openspiel.representation import PlayerStateText
from boardrl.games.openspiel.result import NashConvResult


class BoardInformationState:
    """Render an exp-a-spiel private history as the adapter's model input."""

    def __init__(self, game, *, text_type=PlayerStateText):
        initial = game.new_initial_state()
        self.text = text_type(game, information_state=True)
        self.labels = tuple(
            tuple(initial.action_to_string(player, action) for action in range(9))
            for player in range(2)
        )

    def input(self, player, description):
        board = ["."] * 9
        history = []
        for offset in range(0, len(description), 2):
            action = int(description[offset])
            placed = description[offset + 1] == "*"
            owner = player if placed else 1 - player
            board[action] = "xo"[owner]
            history.append(f"{player},{action} ")
        text = self.text.private_board(
            ["".join(board[start : start + 3]) for start in (0, 3, 6)],
            "".join(history),
            player,
        )
        actions = tuple(index for index, cell in enumerate(board) if cell == ".")
        text += "\n" + "\n".join(
            f"@{self.labels[player][action]}" for action in actions
        )
        return text, actions


class ExpASpielEvaluator(ABC):
    """Shared policy conversion and cached sequence-form traversal."""

    text_type = PlayerStateText

    game_key: ClassVar[tuple[str, str]]
    traverser_name: ClassVar[str]
    # Paper Table 1 counts, rounded up for the allocation guard.
    infosets: ClassVar[int]

    def __init__(self, game, *, traverser=None):
        parameters = game.get_parameters()
        key = (game.get_type().short_name, parameters["gameversion"])
        if (
            key != self.game_key
            or parameters.get("num_rows", 3) != 3
            or parameters.get("num_cols", 3) != 3
            or parameters["obstype"] != "reveal-nothing"
        ):
            raise ValueError(
                "exp-a-spiel evaluation requires the matching 3x3 reveal-nothing game"
            )
        self.text = BoardInformationState(game, text_type=self.text_type)
        self._traverser = traverser

    def _make_traverser(self):
        try:
            import eas
        except ImportError as exc:
            raise RuntimeError(
                "Board exploitability requires exp-a-spiel. See "
                "docs/benchmarking.md for the pinned optional dependency."
            ) from exc
        import psutil

        required = self.infosets * (16 * 9 * 8 + 64)
        available = psutil.virtual_memory().available
        if required > available:
            raise RuntimeError(
                f"Exact {self.traverser_name} evaluation needs approximately "
                f"{required / 2**30:.1f} GiB of available RAM; "
                f"{available / 2**30:.1f} GiB available. No sampled fallback is used."
            )
        return self._create_traverser(eas)

    @abstractmethod
    def _create_traverser(self, eas):
        """Construct the native traverser selected by the concrete backend."""
        raise NotImplementedError

    def _strategy(self, traverser, player, rows, policy, batch_size):
        strategy = np.zeros((rows, 9), dtype=np.float64)
        for start in range(0, rows, batch_size):
            texts, actions, indices = [], [], []
            for row in range(start, min(start + batch_size, rows)):
                text, legal = self.text.input(
                    player, traverser.infoset_desc(player, row)
                )
                if len(legal) == 1:
                    strategy[row, legal[0]] = 1.0
                else:
                    texts.append(text)
                    actions.append(legal)
                    indices.append(row)
            if texts:
                predictions = policy(texts, [len(legal) for legal in actions])
                for row, legal, probabilities in zip(
                    indices, actions, predictions, strict=True
                ):
                    strategy[row, list(legal)] = probabilities
        return strategy

    def evaluate(self, policy, *, batch_size):
        if self._traverser is None:
            self._report("building infosets (cache miss)")
            self._traverser = self._make_traverser()
        else:
            self._report("reusing infosets (cache hit)")
        traverser = self._traverser
        counts = (traverser.NUM_INFOS_PL0, traverser.NUM_INFOS_PL1)
        self._report(f"evaluating current policy (batch size {batch_size})")
        strategies = [
            self._strategy(traverser, player, rows, policy, batch_size)
            for player, rows in enumerate(counts)
        ]
        self._report("computing exact best responses")
        result = traverser.ev_and_exploitability(*strategies)
        # eas.expl[p] is the gain of the OTHER player's best response.
        improvements = tuple(float(value) for value in reversed(result.expl))
        total = sum(improvements)
        return NashConvResult(total, total / 2, improvements, sum(counts))

    def _report(self, message):
        print(
            f"[exact evaluation pid={os.getpid()} {self.traverser_name}] {message}",
            file=sys.stderr,
            flush=True,
        )
