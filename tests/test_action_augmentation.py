import torch

import main
from boardrl.training import TrainingSample


def test_shuffle_actions_keeps_action_metadata_aligned(monkeypatch):
    def reverse(values):
        values.reverse()

    monkeypatch.setattr(main.random, "shuffle", reverse)
    sample = TrainingSample(
        state="position\n@first\nother\n@second\n@third",
        moves=["first", "second", "third"],
        action_idx=0,
        action_distribution=torch.tensor([10.0, 20.0, 30.0]),
        reference_policy=torch.tensor([1.0, 2.0, 3.0]),
    )

    augmented = main.shuffle_actions([sample])[0]

    assert augmented is not sample
    assert augmented.state == "position\n@third\nother\n@second\n@first"
    assert augmented.moves == ["third", "second", "first"]
    assert augmented.action_idx == 2
    assert torch.equal(augmented.action_distribution, torch.tensor([30.0, 20.0, 10.0]))
    assert torch.equal(augmented.reference_policy, torch.tensor([3.0, 2.0, 1.0]))
    assert sample.state == "position\n@first\nother\n@second\n@third"
    assert sample.moves == ["first", "second", "third"]
    assert sample.action_idx == 0


def test_shuffle_actions_preserves_non_action_lines():
    sample = TrainingSample(
        state="@a\nboard @ marker\n",
        action_idx=0,
        action_distribution=[0.1],
    )

    # A one-action input is left alone; this also verifies that only lines
    # beginning with '@' are considered actions.
    augmented = main.shuffle_actions([sample])[0]

    assert augmented is not sample
    assert augmented.state == "@a\nboard @ marker\n"
    assert augmented.action_idx == 0
    assert augmented.action_distribution == [0.1]
