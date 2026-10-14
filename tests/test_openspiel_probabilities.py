import asyncio
from types import SimpleNamespace

import pytest
import torch

from boardrl.games.openspiel.policy import _model_probabilities


class FixedPolicy(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor(logits, dtype=torch.float64)

    def forward(self, texts):
        return SimpleNamespace(policy=[self.logits for _ in texts])


@pytest.mark.parametrize("temperature", [0.0, 0.7, 1.0, 2.0, 1e-300])
def test_exact_probabilities_match_rollout_strategy(temperature):
    from boardrl.games.strategies import PolicySamplingStrategy

    logits = [0.0, -float("inf"), 2.0, 2.0]
    model = FixedPolicy(logits)
    actual = _model_probabilities(model, ["state"], [4], temperature=temperature)[0]

    async def predict(_):
        return SimpleNamespace(
            policy=[model.logits],
            value=SimpleNamespace(mean=torch.zeros(1), stddev=torch.ones(1)),
        )

    game = SimpleNamespace(moves=list("abcd"), display_with_moves=lambda: "state")
    strategy = PolicySamplingStrategy(predict, temperature=temperature)
    policy, _ = asyncio.run(strategy(game))
    torch.testing.assert_close(torch.from_numpy(actual), policy.softmax(0))
    assert actual[1] == 0
    assert sum(actual) == pytest.approx(1)
    if temperature == 0:
        assert actual.tolist() == [0, 0, 1, 0]


@pytest.mark.parametrize(
    "logits", [[float("nan"), 0], [float("inf"), 0], [-float("inf"), -float("inf")]]
)
def test_exact_probabilities_reject_invalid_logits(logits):
    with pytest.raises(ValueError, match="finite probability"):
        _model_probabilities(FixedPolicy(logits), ["state"], [2])


@pytest.mark.parametrize("temperature", [-1.0, float("nan"), float("inf")])
def test_exact_probabilities_reject_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        _model_probabilities(
            FixedPolicy([0, 1]), ["state"], [2], temperature=temperature
        )
