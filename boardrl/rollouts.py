"""High-level rollout API.

Algorithms provide concrete player objects. This module only knows how to
seat those players and execute games; opponent selection deliberately lives in
the experiment code.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from functools import singledispatch

from boardrl.games.strategies import PolicySamplingStrategy
from boardrl.rl.eval.selfplay import SelfPlayResults, Strategy, self_play2
from boardrl.utils import BatchProcessor


Lineup = Sequence[Strategy]


@singledispatch
def _lineups(lineup, games):
    return (lineup(game) for game in range(games))


@_lineups.register(Sequence)
def _(lineup, games):
    return (lineup for _ in range(games))


class Inference:
    """Turn a model into batched player strategies."""

    def __init__(self, model, *, batch_size: int, timeout: float = 0.001, name="model"):
        self.model = model
        self.processor = BatchProcessor(
            batch_size, model, timeout=timeout, model_name=name
        )

    def policy(
        self,
        *,
        temperature: float = 1.0,
        epsilon: float = 0.0,
        dirichlet_alpha: float = 0.3,
        record_moves: bool = False,
    ) -> PolicySamplingStrategy:
        return PolicySamplingStrategy(
            self.processor,
            temperature=temperature,
            epsilon=epsilon,
            dirichlet_alpha=dirichlet_alpha,
            include_moves=record_moves,
        )

    @contextmanager
    def evaluating(self):
        was_training = self.model.training
        self.model.eval()
        try:
            yield self
        finally:
            self.model.train(was_training)


class RolloutRunner:
    """Play batches of games from explicit Python player lineups."""

    def __init__(self, make_game, *, progress: bool = True):
        self.make_game = make_game
        self.progress = progress

    def play(
        self,
        lineup,
        *,
        games: int,
        max_steps: int,
        rotate: bool = True,
        description: str = "rollouts",
    ) -> SelfPlayResults:
        """Play a fixed lineup or call ``lineup(game_index)`` for each game."""
        concrete = [list(players) for players in _lineups(lineup, games)]
        if not concrete:
            return SelfPlayResults([])
        return self_play2(
            self.make_game,
            concrete,
            max_steps,
            rotate=rotate,
            desc=description if self.progress else None,
        )
