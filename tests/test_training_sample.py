import pytest
import torch

from boardrl.training import TrainingSample


def test_collate_basic():
    s1 = TrainingSample(
        action_idx=1,
        reference_policy=torch.tensor([1, 2]),
    )
    s2 = TrainingSample(
        action_idx=3,
        reference_policy=torch.tensor([3, 4]),
    )
    batch = TrainingSample.collate([s1, s2])

    assert torch.equal(batch.action_idx, torch.tensor([1, 3]))
    assert torch.equal(batch.reference_policy, torch.tensor([[1, 2], [3, 4]]))


def test_collate_mismatched_tensor_list():
    t1 = torch.tensor([1, 2])
    t2 = torch.tensor([3])
    s1 = TrainingSample(action_distribution=t1)
    s2 = TrainingSample(action_distribution=t2)

    batch = TrainingSample.collate([s1, s2])

    assert isinstance(batch.action_distribution, list)
    assert torch.equal(batch.action_distribution[0], t1)
    assert torch.equal(batch.action_distribution[1], t2)


def test_optional_fields_default_to_none():
    sample = TrainingSample(state="position")

    assert sample.state == "position"
    assert sample.reference_value is None
    assert sample.gae is None
    assert sample.terminal is False
    assert sample.truncated is False

    sample.reference_value = 1.5

    assert sample.reference_value == pytest.approx(1.5)


def test_collate_rejects_partially_populated_fields():
    first = TrainingSample(reference_value=1.0)
    second = TrainingSample(reference_value=None)

    with pytest.raises(ValueError, match="reference_value"):
        TrainingSample.collate([first, second])


def test_unknown_fields_are_rejected():
    with pytest.raises(TypeError):
        TrainingSample(typoed_advantage=1.0)

    sample = TrainingSample(state="position")
    with pytest.raises(AttributeError):
        sample.typoed_advantage = 1.0


def test_to_moves_tensor_fields_without_touching_python_values_or_none():
    sample = TrainingSample(
        state=["a", "b"],
        action_idx=torch.tensor([0, 1]),
        action_distribution=[torch.tensor([1.0]), torch.tensor([2.0])],
    )

    returned = sample.to("cpu")

    assert returned is sample
    assert sample.state == ["a", "b"]
    assert sample.reference_value is None
    assert sample.action_idx.device.type == "cpu"
    assert all(tensor.device.type == "cpu" for tensor in sample.action_distribution)
