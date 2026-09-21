from types import SimpleNamespace

import pytest
import torch

from boardrl.rollouts import GameTrace, PlayerTrace, Record, Rollouts
from boardrl.training import ToSamples


def test_to_samples_converts_rollout_metadata_into_training_fields():
    game = SimpleNamespace(
        moves=["a", "b"],
        diff_points=lambda: 3,
        points=lambda: 5,
        current_player=lambda: 0,
        round=lambda: 7,
    )
    record = Record(
        game,
        torch.tensor([0.1, 0.2]),
        1,
        {
            "state": "position\n@a\n@b",
            "reference_policy": torch.tensor([0.3, 0.4]),
            "reference_value": 2.0,
            "reference_value_stddev": 1.25,
            "reference_max_q": 2.05,
            "diagnostic_only": "do not leak into training samples",
        },
    )
    record.score = 3.0
    record.reward = 1.0
    record.returns = 1.0

    end = SimpleNamespace(state="terminal", terminal=True, truncated=False)
    trace = PlayerTrace(seat_id=0, strategy_id=0)
    trace.extend([record, end])

    samples = ToSamples()(Rollouts([GameTrace([trace])]))

    assert len(samples) == 1
    sample = samples[0]
    assert sample.state == record.state
    assert sample.action_idx == 1
    assert torch.equal(sample.action_distribution, record.action_distribution)
    assert sample.score == pytest.approx(3.0)
    assert sample.reward == pytest.approx(1.0)
    assert sample.returns == pytest.approx(1.0)
    assert torch.equal(sample.reference_policy, torch.tensor([0.3, 0.4]))
    assert sample.reference_value == pytest.approx(2.0)
    assert sample.reference_value_stddev == pytest.approx(1.25)
    assert sample.reference_max_q == pytest.approx(2.05)
    assert not hasattr(sample, "diagnostic_only")
    assert sample.next is end
