import math
import random
from functools import partial
from types import SimpleNamespace

import pytest
import torch

pyspiel = pytest.importorskip("pyspiel")

from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability

from boardrl.games import games_library, register_game
from boardrl.games.openspiel.evaluation import (
    ExactOpenSpielEvaluator,
    OpenSpielTreeEvaluator,
    nash_conv,
)
from boardrl.games.openspiel.game import OpenSpiel
from boardrl.games.openspiel.leduc.game import Leduc
from boardrl.games.openspiel.policy import OpenSpielPolicy, _model_probabilities
from boardrl.games.semantics import CompetitiveOutcome, PointDeltaRewards, PointScores


def predict(model, temperature=1.0):
    return partial(_model_probabilities, model, temperature=temperature)


class TestPolicyModel(torch.nn.Module):
    __test__ = False

    def __init__(self, logits=(0.0, 0.0, 0.0)):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor(logits))
        self.batches = []
        self.modes = []

    def forward(self, texts):
        self.batches.append(list(texts))
        self.modes.append(self.training)
        actions = {"Fold": 0, "Call": 1, "Raise": 2}
        return SimpleNamespace(
            policy=[
                self.logits[
                    [
                        actions[line[1:]]
                        for line in text.splitlines()
                        if line.startswith("@")
                    ]
                ]
                for text in texts
            ]
        )


def test_batched_uniform_evaluation_matches_openspiel():
    game = pyspiel.load_game("leduc_poker")
    model = TestPolicyModel()
    result = nash_conv(model)
    reference = exploitability.nash_conv(game, policy.UniformRandomPolicy(game))
    assert result.nash_conv == pytest.approx(reference, abs=1e-6)
    assert result.exploitability == pytest.approx(reference / 2, abs=1e-6)
    assert result.evaluated_states == 936
    assert len(model.batches) == math.ceil(936 / 64)
    assert max(map(len, model.batches)) == 64
    assert all("[Round1:" in text for batch in model.batches for text in batch)
    inputs = [text for batch in model.batches for text in batch]
    for state, text in zip(policy.TabularPolicy(game).states, inputs, strict=True):
        training_view = games_library("leduc").make_game(
            spiel_game=game, state=state, resolve_chance=False
        )
        assert text == training_view.display_with_moves()


@pytest.mark.parametrize("temperature", [0.0, 0.5, 2.0])
def test_batching_preserves_action_order_and_temperature(temperature):
    game = pyspiel.load_game("leduc_poker")
    model = TestPolicyModel((0.1, -0.4, 1.2))
    lazy = OpenSpielPolicy(
        Leduc(spiel_game=game, resolve_chance=False), predict(model, temperature)
    )
    batched = OpenSpielPolicy(
        Leduc(spiel_game=game, resolve_chance=False), predict(model, temperature)
    )
    states = policy.TabularPolicy(game).states
    batched.precompute(states)
    for state in states:
        expected = lazy.action_probabilities(state)
        actual = batched.action_probabilities(state)
        assert list(actual) == state.legal_actions()
        assert actual == pytest.approx(expected)
    reference = exploitability.nash_conv(game, lazy)
    assert nash_conv(model, temperature=temperature).nash_conv == pytest.approx(
        reference, abs=1e-6
    )


def test_game_selected_exact_evaluation_and_fresh_cache():
    game = games_library("leduc")
    model = TestPolicyModel()
    native = game.make_game(resolve_chance=False)._game
    evaluator = ExactOpenSpielEvaluator(native, backend=game.exact_evaluation(native))
    first = evaluator.evaluate(model, batch_size=8)
    assert model.training and not any(model.modes)
    with torch.no_grad():
        model.logits[2] = 4.0
    second = evaluator.evaluate(model, batch_size=8)
    assert second.exploitability != pytest.approx(first.exploitability)
    assert len(model.batches) == 2 * math.ceil(936 / 8)
    assert max(map(len, model.batches)) == 8


@pytest.mark.parametrize("training", [True, False])
def test_exact_evaluation_restores_model_mode_after_failure(monkeypatch, training):
    model = TestPolicyModel()
    model.train(training)
    native = pyspiel.load_game("leduc_poker")
    evaluator = ExactOpenSpielEvaluator(native, backend=OpenSpielTreeEvaluator(native))

    def fail(*args, **kwargs):
        assert not model.training
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(evaluator.backend, "evaluate", fail)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        evaluator.evaluate(model, batch_size=1)
    assert model.training is training


def test_hidden_opponent_cards_have_identical_policy_inputs():
    game = pyspiel.load_game("leduc_poker")
    model = TestPolicyModel((0.0, 0.2, 0.5))
    adapter = OpenSpielPolicy(
        Leduc(spiel_game=game, resolve_chance=False), predict(model)
    )
    states = []
    for opponent_card in (1, 4):
        state = game.new_initial_state()
        state.apply_action(0)
        state.apply_action(opponent_card)
        states.append(state)
    adapter.precompute(states)
    assert model.batches == [
        [states[0].information_state_string(0) + "\n@Call\n@Raise"]
    ]
    assert adapter.action_probabilities(states[0]) == adapter.action_probabilities(
        states[1]
    )


def test_invalid_batch_size_and_action_dimensions_are_rejected():
    model = TestPolicyModel()
    with pytest.raises(ValueError, match="batch_size"):
        nash_conv(model, batch_size=0)
    with pytest.raises(ValueError, match="legal actions"):
        _model_probabilities(model, ["@Call"], [2])


def test_exact_evaluation_uses_the_registered_game_parameters(monkeypatch):
    from boardrl import games

    class VariantLeduc(OpenSpiel):
        def __init__(self, starting_player: int = 0, **kwargs):
            super().__init__(
                f"leduc_poker(suit_isomorphism=true,starting_player={starting_player})",
                information_state=True,
                **kwargs,
            )

    monkeypatch.setattr(games, "games_library", games_library.copy())
    register_game(
        "variant_leduc",
        VariantLeduc,
        scores=PointScores(),
        outcome=CompetitiveOutcome(),
        rewards=PointDeltaRewards(scale=0.1),
        args_from=VariantLeduc,
        metrics_from=OpenSpiel,
        exact_evaluation=OpenSpielTreeEvaluator,
    )
    game = games.games_library("variant_leduc,starting_player=1")
    random_state = random.getstate()
    native = game.make_game(resolve_chance=False)._game
    evaluator = ExactOpenSpielEvaluator(native, backend=game.exact_evaluation(native))
    assert random.getstate() == random_state
    assert native.get_parameters() == game.make_game()._game.get_parameters()
    assert native.get_parameters()["suit_isomorphism"] is True
    assert native.get_parameters()["starting_player"] == 1
    model = TestPolicyModel()
    result = evaluator.evaluate(model, batch_size=2)
    expected = exploitability.nash_conv(native, policy.UniformRandomPolicy(native))
    assert result.nash_conv == pytest.approx(expected, abs=1e-6)
    assert sum(map(len, model.batches)) == len(policy.TabularPolicy(native).states)
    assert sum(map(len, model.batches)) != 936


@pytest.mark.parametrize("name", ["matrix_rps", "leduc_poker(players=3)"])
def test_exact_backend_rejects_unsupported_game_dynamics(name):
    with pytest.raises(ValueError, match="two-player sequential zero-sum"):
        native = pyspiel.load_game(name)
        ExactOpenSpielEvaluator(native, backend=OpenSpielTreeEvaluator(native))
