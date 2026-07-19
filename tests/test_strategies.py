import asyncio
import importlib
import sys
from types import SimpleNamespace

import pyximport
import pytest
import torch

from boardrl.games import games_library
from boardrl.games.strategies import PolicySamplingStrategy
from boardrl.rl.model.model import PolicyValue
from boardrl.utils import BatchProcessor, ModelPool


def toy_process(games: list[str]) -> PolicyValue:
    policies = [torch.randn(g.count("@")) for g in games]
    value = torch.distributions.Normal(torch.randn(len(games)), torch.rand(len(games)))
    return PolicyValue(policies, value)


def instantiate(name: str, registry, pool):
    _, arg_info = registry.registry[name]
    params = []
    provided = {}
    for arg, (typ, default) in arg_info.items():
        if arg == "model":
            provided["model"] = pool
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
    pool = ModelPool(BatchProcessor(1, toy_process), 1, 0)
    strat = instantiate(strat_name, registry, pool)
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
            value=SimpleNamespace(mean=torch.tensor([0.0])),
        )

    game = SimpleNamespace(
        moves=["a", "b", "c"],
        display_with_moves=lambda: "state",
    )
    monkeypatch.setattr(torch.distributions, "Dirichlet", DummyDirichlet)

    strat = PolicySamplingStrategy(
        model, epsilon=1.0, dirichlet_alpha=0.7, include_moves=False
    )
    policy, _ = asyncio.run(strat(game))

    assert torch.allclose(policy.exp(), torch.tensor([0.1, 0.2, 0.7]))


@pytest.mark.parametrize(
    "game_name,expected_scale",
    [
        ("tictactoe", 1.0),
        ("connectfour", 1.0),
        ("thegame", 0.01),
        ("century", 0.01),
        ("guessnumber", 0.1),
    ],
)
def test_game_descriptor_reward_rescale(game_name, expected_scale):
    assert games_library(game_name).reward_rescale == expected_scale
