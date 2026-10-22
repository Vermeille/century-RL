"""Execute complete documentation examples against the current public APIs."""

import ast
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from boardrl import RolloutRunner
from boardrl.games import games_library, register_game
from boardrl.games.augmentations import shuffle_actions
from boardrl.games.semantics import CompetitiveOutcome, OutcomeScores, TerminalOutcomeRewards
from boardrl.models import make_for_game
from boardrl.rl.model import load_model
from boardrl.training import Pipeline, ReferenceTargets, ToSamples
from examples.train_minimal import train


DOCS = Path(__file__).resolve().parents[1] / "docs"


def blocks(page):
    return re.findall(r"^```python\n(.*?)^```", page.read_text(), re.MULTILINE | re.DOTALL)


def snippet(page, name):
    return next(code for code in blocks(DOCS / page) if code.startswith(f"# docs-test: {name}\n"))


def test_python_documentation_is_valid_syntax():
    for page in DOCS.rglob("*.md"):
        for code in blocks(page):
            ast.parse(code, filename=str(page))


def test_countdown_tutorial_registers_and_produces_bootstrap_targets(monkeypatch, tmp_path):
    # Emulate the module layout the tutorial asks the reader to create.
    game_module = ModuleType("boardrl.games.countdown.game")
    metrics_module = ModuleType("boardrl.games.countdown.metrics")
    monkeypatch.setitem(sys.modules, game_module.__name__, game_module)
    monkeypatch.setitem(sys.modules, metrics_module.__name__, metrics_module)
    exec(snippet("adding-games.md", "countdown"), game_module.__dict__)
    exec(snippet("adding-games.md", "countdown-metrics"), metrics_module.__dict__)
    monkeypatch.setattr(games_library, "registry", games_library.registry.copy())
    monkeypatch.setattr(games_library, "required", games_library.required.copy())
    exec(snippet("adding-games.md", "countdown-registration"), {
        "register_game": register_game, "OutcomeScores": OutcomeScores,
        "CompetitiveOutcome": CompetitiveOutcome,
        "TerminalOutcomeRewards": TerminalOutcomeRewards, "shuffle_actions": shuffle_actions,
    })
    desc = games_library("countdown,counters=7")
    initial = desc.make_game()
    copy = initial.copy()
    copy.play_idx(1)
    assert initial.counters == 7 and copy.counters == 5
    copy.moves.clear()
    assert initial.moves == ["1", "2"]
    with pytest.raises(ValueError):
        initial.play_str("3")
    for first in initial.moves:
        game = initial.copy()
        game.play_str(first)
        game.simulate_to_end()
        assert game.ended() and game.moves == []
        assert sorted(game.points_for(p) for p in range(2)) == [-1, 1]
        with pytest.raises(ValueError):
            game.play_str("1")

    bot = desc.strategy_from_string("random")
    runner = RolloutRunner(desc.make_game, outcome=desc.outcome, progress=False)
    model = make_for_game("toy", desc).eval()
    prepare = Pipeline(
        desc.rewards.make_returns(1.0), ToSamples(),
        ReferenceTargets(model, batch_size=16, discount=1.0, gae_lambda=1.0, value_lambda=1.0),
    )
    for budget in (10, 1):
        data = runner.play([bot, bot], games=4, max_steps=budget)
        assert all(trace[-1].terminal == (budget == 10) for game in data for trace in game)
        assert desc.make_metrics(data).metrics()
        samples = prepare(data)
        assert samples
        assert all(torch.isfinite(torch.tensor(s.td_lambda)) for s in samples)

    # Exercise the tutorial game through an optimizer update and the shared API.
    path = train(game_spec="countdown,counters=7", steps=1, games=2, directory=tmp_path)
    assert load_model(path, device="cpu").spec()["points_based"] is False
    from fastapi.testclient import TestClient
    from boardrl.serve import serve

    monkeypatch.setattr(serve, "sessions", {})
    client = TestClient(serve.app)
    query = {"game": "countdown,counters=7"}
    state = client.get("/state", params=query).json()
    assert state["moves"] == ["1", "2"]
    response = client.post("/do-one", params=query, json={"action": "1"})
    assert response.status_code == 200
    assert "Counters: 6" in response.json()["state"]["board"]


def test_custom_loss_tutorial_backpropagates():
    namespace = {}
    exec(snippet("internals/optimization.md", "custom-loss"), namespace)
    prediction = torch.tensor([1.0, 3.0], requires_grad=True)
    loss = namespace["ReturnMAE"]()
    result = loss([], SimpleNamespace(mean=prediction), SimpleNamespace(returns=torch.zeros(2)), {})
    result.objective.backward()
    assert prediction.grad.tolist() == [0.5, 0.5]


@pytest.mark.parametrize("game_spec", ["tictactoe", "thegame,max_value=20,mode=strict"])
def test_minimal_training_example_saves_loadable_policy(tmp_path, game_spec):
    path = train(game_spec=game_spec, steps=1, games=2, directory=tmp_path)
    payload = torch.load(path, weights_only=False, map_location="cpu")
    assert payload["step"] == 1
    assert payload["states"]["learner"]["losses"]
    assert payload["optimizers"]["current"]
    model = load_model(path, name="current", device="cpu")
    game = games_library(game_spec).make_game()
    output = model([game.display_with_moves()])
    assert output.policy[0].shape == (len(game.moves),)
    assert torch.isfinite(output.policy[0]).all()
    assert torch.isfinite(output.value.mean).all()
