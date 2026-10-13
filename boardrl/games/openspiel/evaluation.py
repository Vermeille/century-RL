"""Backend selection and reusable exact evaluation for OpenSpiel games."""

from __future__ import annotations

from threading import RLock

from boardrl.evaluation import ModelEvaluation, ModelEvaluator
from boardrl.games.openspiel.game import _require_pyspiel
from boardrl.games.openspiel.liars_dice.evaluation import LiarsDiceEvaluator
from boardrl.games.openspiel.policy import (
    ModelPolicy,
    OpenSpielPolicy,
    _require_policy_modules,
)
from boardrl.games.openspiel.result import NashConvResult


def exact_backend(game, *, backend="auto", use_cpp_br=True):
    if backend not in ("auto", "openspiel"):
        raise ValueError("backend must be 'auto' or 'openspiel'")
    name = game.get_type().short_name
    parameters = game.get_parameters()
    if name == "dark_hex" and parameters["obstype"] == "reveal-nothing":
        raise ValueError(
            "OpenSpiel's Dark Hex string gives best responses an extra global clock; "
            "use the exp-a-spiel backend for reveal-nothing evaluation"
        )
    if backend == "auto" and name == "liars_dice":
        return LiarsDiceEvaluator(game)
    return OpenSpielTreeEvaluator(game, use_cpp_br=use_cpp_br)


def nash_conv(
    model,
    game_name: str = "leduc_poker",
    *,
    temperature: float = 1.0,
    use_cpp_br: bool = True,
    batch_size: int = 64,
    spiel_game=None,
    backend: str = "auto",
) -> NashConvResult:
    """Evaluate once; retain ExactOpenSpielEvaluator to reuse board traversers."""
    pyspiel = _require_pyspiel()
    game = spiel_game if spiel_game is not None else pyspiel.load_game(game_name)
    evaluator = ExactOpenSpielEvaluator(game, backend=backend, use_cpp_br=use_cpp_br)
    return evaluator.evaluate(model, temperature=temperature, batch_size=batch_size)


class ExactOpenSpielEvaluator:
    """Own one game's reusable backend, never a model or its probabilities."""

    def __init__(self, game, *, backend="auto", use_cpp_br=True):
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
        self.backend = exact_backend(game, backend=backend, use_cpp_br=use_cpp_br)
        # exp-a-spiel reuses mutable working buffers as well as the infosets.
        self._lock = RLock()

    def evaluate(self, model, *, temperature=1.0, batch_size=64) -> NashConvResult:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        policy = ModelPolicy(model, temperature)
        with self._lock:
            training = model.training
            try:
                model.eval()
                return self.backend.evaluate(policy, batch_size=batch_size)
            finally:
                model.train(training)


class OpenSpielModelEvaluator(ModelEvaluator):
    """Exact exploitability of one model played in both seats."""

    metric_name = "exploitability"

    def __init__(self, game, *, spiel_game=None, progress=True):
        if spiel_game is None:
            # Inspect the registered factory without drawing chance outcomes.
            spiel_game = game.make_game(resolve_chance=False)._game
        self.spiel_game = spiel_game
        self.exact_evaluator = ExactOpenSpielEvaluator(spiel_game)

    def _evaluate(
        self, inference, *, temperature=1.0, **sampled_options
    ) -> ModelEvaluation:
        result = self.exact_evaluator.evaluate(
            inference.model,
            temperature=temperature,
            batch_size=inference.batch_size,
        )
        return ModelEvaluation(
            {"exploitability": result.exploitability, "nash_conv": result.nash_conv},
            self.metric_name,
            result.exploitability,
            -result.exploitability,
            temperature,
        )


class OpenSpielTreeEvaluator:
    def __init__(self, game, *, use_cpp_br=True):
        self.game = game
        self.use_cpp_br = use_cpp_br

    def evaluate(self, model_policy, *, batch_size):
        return _tree_nash_conv(
            model_policy.model,
            self.game,
            temperature=model_policy.temperature,
            use_cpp_br=self.use_cpp_br,
            batch_size=batch_size,
        )


def _tree_nash_conv(model, game, *, temperature, use_cpp_br, batch_size):
    pyspiel = _require_pyspiel()
    policy_lib, exploitability_module = _require_policy_modules()
    information_state = (
        game.get_type().information
        == pyspiel.GameType.Information.IMPERFECT_INFORMATION
    )
    policy = OpenSpielPolicy(
        game,
        model,
        information_state=information_state,
        temperature=temperature,
    )
    policy.precompute(policy_lib.TabularPolicy(game).states, batch_size=batch_size)
    result = exploitability_module.nash_conv(
        game,
        policy,
        return_only_nash_conv=False,
        use_cpp_br=use_cpp_br,
    )
    value = float(result.nash_conv)
    return NashConvResult(
        nash_conv=value,
        exploitability=value / game.num_players(),
        player_improvements=tuple(float(x) for x in result.player_improvements),
        evaluated_states=policy.evaluated_states,
    )
