"""Reusable exact evaluation of a policy in both seats."""

from __future__ import annotations

import math
from functools import partial
from threading import RLock

from boardrl.games.openspiel.game import OpenSpielGame, _require_pyspiel
from boardrl.games.openspiel.policy import (
    OpenSpielPolicy,
    _model_probabilities,
    _require_policy_modules,
)
from boardrl.games.openspiel.result import NashConvResult


def nash_conv(
    model,
    game_name: str = "leduc",
    *,
    temperature: float = 1.0,
    batch_size: int = 64,
    spiel_game=None,
    backend=None,
) -> NashConvResult:
    """Evaluate once; retain ExactOpenSpielEvaluator to reuse board traversers."""
    if spiel_game is None:
        from boardrl.games import games_library

        descriptor = games_library(game_name)
        if descriptor.exact_evaluation is None:
            raise ValueError(f"{game_name} has no registered exact evaluator")
        spiel_game = descriptor.make_game(resolve_chance=False)._game
        backend = descriptor.exact_evaluation(spiel_game)
    elif backend is None:
        raise ValueError("a native game requires an explicit exact backend")
    evaluator = ExactOpenSpielEvaluator(spiel_game, backend=backend)
    return evaluator.evaluate(model, temperature=temperature, batch_size=batch_size)


class ExactOpenSpielEvaluator:
    """Own one game's reusable backend, never a model or its probabilities."""

    def __init__(self, game, *, backend):
        pyspiel = _require_pyspiel()
        kind = game.get_type()
        if (
            game.num_players() != 2
            or kind.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL
            or kind.utility != pyspiel.GameType.Utility.ZERO_SUM
        ):
            raise ValueError(
                "exact exploitability requires a two-player sequential zero-sum game"
            )
        self.backend = backend
        # exp-a-spiel reuses mutable working buffers as well as the infosets.
        self._lock = RLock()

    def evaluate(self, model, *, temperature=1.0, batch_size=64) -> NashConvResult:
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        predict = partial(_model_probabilities, model, temperature=temperature)
        with self._lock:
            was_training = model.training
            model.eval()
            try:
                return self.backend.evaluate(predict, batch_size=batch_size)
            finally:
                model.train(was_training)


class OpenSpielTreeEvaluator:
    def __init__(self, game, *, use_cpp_br=True):
        self.game = game
        self.use_cpp_br = use_cpp_br

    def _create_game(self):
        pyspiel = _require_pyspiel()
        information_state = (
            self.game.get_type().information
            == pyspiel.GameType.Information.IMPERFECT_INFORMATION
        )
        return OpenSpielGame.from_state(
            self.game.new_initial_state(), information_state=information_state
        )

    def evaluate(self, predict, *, batch_size):
        policy_lib, exploitability_module = _require_policy_modules()
        policy = OpenSpielPolicy(self._create_game(), predict)
        policy.precompute(
            policy_lib.TabularPolicy(self.game).states, batch_size=batch_size
        )
        result = exploitability_module.nash_conv(
            self.game,
            policy,
            return_only_nash_conv=False,
            use_cpp_br=self.use_cpp_br,
        )
        value = float(result.nash_conv)
        return NashConvResult(
            nash_conv=value,
            exploitability=value / self.game.num_players(),
            player_improvements=tuple(float(x) for x in result.player_improvements),
            evaluated_states=policy.evaluated_states,
        )
