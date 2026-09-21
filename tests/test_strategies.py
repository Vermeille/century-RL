import asyncio
import importlib
import sys
from types import SimpleNamespace

import pyximport  # type: ignore[import-untyped]
import pytest
import torch

from boardrl.games import games_library
from boardrl.games.semantics import (
    OutcomeScores,
    PointDeltaRewards,
    PointScores,
    TerminalOutcomeRewards,
)
from boardrl.games.strategies import PolicySamplingStrategy
from boardrl.rl.model.model import NormalValueDistribution, PolicyValue
from boardrl.utils import BatchProcessor


def toy_process(games: list[str]) -> PolicyValue:
    policies = [torch.randn(g.count("@")) for g in games]
    value = NormalValueDistribution(torch.randn(len(games)), torch.rand(len(games)))
    return PolicyValue(policies, value)


def instantiate(name: str, registry, predictor):
    _, arg_info = registry.registry[name]
    params = []
    provided = {}
    for arg, (typ, default) in arg_info.items():
        if arg == "model":
            provided["model"] = predictor
        elif default is None:
            if typ is float:
                params.append(f"{arg}=1.0")
            elif typ is int:
                params.append(f"{arg}=1")
    spec = ",".join([name] + params)
    return registry(spec, **provided)


# ensure fast_sample is loaded from cyutils implementation
if "boardrl.cyutils" in sys.modules:
    del sys.modules["boardrl.cyutils"]
pyximport.install()
fast_sample = importlib.import_module("boardrl.cyutils").fast_sample


PAIRS = []
IDS = []
for game_name in games_library.registry:
    desc = games_library(game_name)
    registry = desc.strategy_from_string
    for strat_name in registry.registry:
        PAIRS.append((game_name, strat_name))
        IDS.append(f"{game_name}:{strat_name}")


@pytest.mark.parametrize("game_name,strat_name", PAIRS, ids=IDS)
def test_strategy_game_smoke(game_name, strat_name):
    desc = games_library(game_name)
    registry = desc.strategy_from_string
    predictor = BatchProcessor(1, toy_process)
    strat = instantiate(strat_name, registry, predictor)
    g = desc.make_game()

    async def play_all():
        for _ in range(512):
            if g.ended():
                break
            dist, _ = await strat(g)
            action = fast_sample(torch.softmax(dist, dim=0))
            g.play_idx(action)

    asyncio.run(play_all())
    # No assertion needed: the test passes if no exceptions are raised.


def test_policy_sampling_epsilon_uses_dirichlet_noise(monkeypatch):
    class DummyDirichlet:
        def __init__(self, concentration):
            self.concentration = concentration

        def sample(self):
            assert torch.allclose(self.concentration, torch.full((3,), 0.7))
            return torch.tensor([0.1, 0.2, 0.7])

    async def model(_state):
        return SimpleNamespace(
            policy=[torch.tensor([0.0, 0.0, 0.0])],
            value=SimpleNamespace(
                mean=torch.tensor([0.0]),
                stddev=torch.tensor([1.25]),
            ),
        )

    game = SimpleNamespace(
        moves=["a", "b", "c"],
        display_with_moves=lambda: "state",
    )
    monkeypatch.setattr(torch.distributions, "Dirichlet", DummyDirichlet)

    strat = PolicySamplingStrategy(
        model, epsilon=1.0, dirichlet_alpha=0.7, include_moves=False
    )
    policy, info = asyncio.run(strat(game))

    assert torch.allclose(policy.exp(), torch.tensor([0.1, 0.2, 0.7]))
    assert info["reference_value_stddev"] == pytest.approx(1.25)


def test_connectfour_keeps_default_and_game_specific_strategies():
    registry = games_library("connectfour").strategy_from_string

    assert "policy_sampling" in registry
    assert "tactical_random" in registry


def test_thegame_point_delta_reward_scale():
    expected_scale = 0.1
    game_name = "thegame"
    rewards = games_library(game_name).rewards

    assert type(rewards) is PointDeltaRewards
    assert rewards.scale == expected_scale


@pytest.mark.parametrize(
    "game_name,expected_coop",
    [("thegame", True), ("guessnumber", True), ("tictactoe", False)],
)
def test_game_descriptor_coop(game_name, expected_coop):
    assert games_library(game_name).coop is expected_coop


@pytest.mark.parametrize(
    "game_name,expected_points_based",
    [
        ("century", True),
        ("thegame", True),
        ("take5", True),
        ("skullking", True),
        ("regicide", True),
        ("hanabi", True),
        ("guessnumber", True),
        ("sum", True),
        ("tictactoe", False),
        ("connectfour", False),
        ("rps", False),
        ("nim", False),
    ],
)
def test_game_descriptor_value_semantics(game_name, expected_points_based):
    expected_type = PointScores if expected_points_based else OutcomeScores
    assert type(games_library(game_name).scores) is expected_type


@pytest.mark.parametrize(
    "game_name,expected_type",
    [
        ("thegame", PointDeltaRewards),
        ("century", TerminalOutcomeRewards),
        ("take5", TerminalOutcomeRewards),
        ("tictactoe", TerminalOutcomeRewards),
    ],
)
def test_game_descriptor_reward_semantics(game_name, expected_type):
    assert type(games_library(game_name).rewards) is expected_type
