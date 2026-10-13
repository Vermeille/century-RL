import random
from types import SimpleNamespace

import pytest
import torch

from boardrl.checkpoints import Checkpoints
from boardrl.evaluation import ModelEvaluation, ModelEvaluator, SampledModelEvaluator
from boardrl.games import games_library
from boardrl.models import make_for_game
from boardrl.rollouts import Inference


@pytest.mark.parametrize("spec,opponent", [("tictactoe", "random"), ("thegame", None)])
def test_sampled_backend_preserves_lineups_metrics_and_model_mode(spec, opponent, monkeypatch):
    game = games_library(spec)
    evaluator = game.make_model_evaluator(progress=False)
    assert type(evaluator) is SampledModelEvaluator
    model = make_for_game("toy", game)
    inference = Inference(model, batch_size=2)
    captured = {}
    score = 3.0 if game.rewards.points_value_head else 0.75
    result = SimpleNamespace(
        win_rate=lambda: 0.75, avg_points=lambda: 3.0,
        rollouts=SimpleNamespace(my_points=lambda player: [1.0, 5.0]),
    )

    def compare(players, **kwargs):
        assert not model.training
        captured.update(kwargs)
        captured["self_play"] = players[0] is players[1]
        return result

    monkeypatch.setattr(evaluator.evaluator, "compare", compare)
    actual = evaluator.evaluate(
        inference, games=7, max_steps=23, names=["a", "b"],
        opponent=opponent, temperature=0.8, rotate=False,
    )
    assert model.training
    assert captured == {
        "games": 7, "max_steps": 23, "names": ["a", "b"],
        "rotate": False, "self_play": opponent is None,
    }
    assert actual.metrics == {"win_rate": 0.75, **game.scores.evaluation_metrics(result)}
    assert actual.selection_score == score


class CheckpointEvaluator(ModelEvaluator):
    metric_name = "exploitability"

    def __init__(self):
        self.calls = []
        self.batch_sizes = []

    def evaluate(self, inference, *, temperature=1.0):
        self.calls.append(inference.model.weight.item())
        self.batch_sizes.append(inference.batch_size)
        random.random()
        torch.rand(1)
        value = inference.model.weight.item()
        return ModelEvaluation({"exploitability": value}, self.metric_name, value, -value, temperature)


def _saved_best(tmp_path, metadata):
    checkpoints = Checkpoints(tmp_path, prefix="best", keep=1)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.55)
    # The restore path loads weights into a clone; no model construction needed.
    model.spec = lambda: {"points_based": False}
    checkpoints.save(10, {"agent": model}, metadata=metadata)
    return checkpoints


@pytest.mark.parametrize("metadata", [
    {"evaluation_points": 99.0},
    {"evaluation_metric": "points", "evaluation_score": 99.0, "evaluation_temperature": 1.0},
])
def test_legacy_best_is_reevaluated_without_changing_live_model_or_rng(tmp_path, metadata):
    checkpoints = _saved_best(tmp_path, metadata)
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(0.2)
    evaluator = CheckpointEvaluator()
    python_rng = random.getstate()
    torch_rng = torch.get_rng_state().clone()
    score = evaluator.restore_best_score(checkpoints, Inference(model, batch_size=1), model_name="agent")
    assert score == pytest.approx(-0.55)
    assert evaluator.calls == pytest.approx([0.55])
    assert evaluator.batch_sizes == [1]
    assert model.weight.item() == pytest.approx(0.2)
    assert random.getstate() == python_rng
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert -0.2 > score  # A less exploitable policy wins.


def test_matching_best_metadata_is_reused_and_changed_temperature_reevaluated(tmp_path):
    metadata = ModelEvaluation({"exploitability": 0.4}, "exploitability", 0.4, -0.4, 1.0).checkpoint_metadata()
    checkpoints = _saved_best(tmp_path, metadata)
    evaluator = CheckpointEvaluator()
    inference = Inference(torch.nn.Linear(1, 1, bias=False), batch_size=1)
    assert evaluator.restore_best_score(checkpoints, inference, model_name="agent") == -0.4
    assert evaluator.calls == []
    assert evaluator.restore_best_score(checkpoints, inference, model_name="agent", temperature=0.5) == pytest.approx(-0.55)
    assert evaluator.calls == pytest.approx([0.55])


def test_no_best_checkpoint_has_no_comparison_score(tmp_path):
    evaluator = CheckpointEvaluator()
    assert evaluator.restore_best_score(Checkpoints(tmp_path), None, model_name="agent") == float("-inf")


def test_legacy_sampled_score_is_preserved_without_extra_sampling(tmp_path):
    checkpoints = _saved_best(tmp_path, {"evaluation_points": 0.75})
    evaluator = games_library("tictactoe").make_model_evaluator()
    assert evaluator.restore_best_score(checkpoints, None, model_name="agent") == 0.75


def test_changed_legacy_score_family_is_reevaluated(tmp_path, monkeypatch):
    checkpoints = _saved_best(tmp_path, {"evaluation_points": 0.75})
    evaluator = games_library("thegame").make_model_evaluator()
    inference = Inference(torch.nn.Linear(1, 1, bias=False), batch_size=1)
    calls = []

    def evaluate(saved_inference, **kwargs):
        calls.append(saved_inference.model.weight.item())
        return ModelEvaluation({"points": 3.0}, "points", 3.0, 3.0, 1.0)

    monkeypatch.setattr(evaluator, "evaluate", evaluate)
    assert evaluator.restore_best_score(checkpoints, inference, model_name="agent") == 3.0
    assert calls == pytest.approx([0.55])


def test_sampled_evaluation_failure_restores_model_mode(monkeypatch):
    game = games_library("tictactoe")
    model = make_for_game("toy", game)
    evaluator = game.make_model_evaluator()

    def fail(*args, **kwargs):
        assert not model.training
        raise RuntimeError("sampled evaluation failed")

    monkeypatch.setattr(evaluator.evaluator, "compare", fail)
    with pytest.raises(RuntimeError, match="sampled evaluation failed"):
        evaluator.evaluate(Inference(model, batch_size=2), games=1, max_steps=10)
    assert model.training


def test_sampled_evaluation_keeps_a_fixed_opponent(monkeypatch):
    game = games_library("tictactoe")
    evaluator = game.make_model_evaluator()
    inference = Inference(make_for_game("toy", game), batch_size=2)
    created = []
    players = []

    def strategy(spec):
        created.append(spec)
        return object()

    def compare(lineup, **kwargs):
        players.append(lineup[1])
        return SimpleNamespace(win_rate=lambda: 0.5)

    monkeypatch.setattr(game, "strategy_from_string", strategy)
    monkeypatch.setattr(evaluator.evaluator, "compare", compare)
    for _ in range(2):
        evaluator.evaluate(inference, opponent="random", games=1, max_steps=10)
    assert created == ["random"]
    assert players[0] is players[1]
