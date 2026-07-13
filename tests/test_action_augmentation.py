import torch

from boardrl.games import games_library
from boardrl.games.augmentations import shuffle_actions
from boardrl.games.thegame.augmentations import shuffle_hand
from boardrl.training import TrainingSample


def test_shuffle_actions_keeps_action_metadata_aligned(monkeypatch):
    def reverse(values):
        values.reverse()

    monkeypatch.setattr("boardrl.games.augmentations.random.shuffle", reverse)
    sample = TrainingSample(
        state="position\n@first\nother\n@second\n@third",
        moves=["first", "second", "third"],
        action_idx=0,
        action_distribution=torch.tensor([10.0, 20.0, 30.0]),
        reference_policy=torch.tensor([1.0, 2.0, 3.0]),
    )

    augmented = shuffle_actions([sample])[0]

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
    augmented = shuffle_actions([sample])[0]

    assert augmented is not sample
    assert augmented.state == "@a\nboard @ marker\n"
    assert augmented.action_idx == 0
    assert augmented.action_distribution == [0.1]


def test_thegame_shuffle_hand_keeps_actions_and_metadata(monkeypatch):
    def reverse(values):
        values.reverse()

    monkeypatch.setattr(
        "boardrl.games.thegame.augmentations.random.shuffle", reverse
    )
    sample = TrainingSample(
        state="Piles: 1 1 100 100\nHand: 12 45 78\n@12->0\n@45->0",
        moves=["12->0", "45->0"],
        action_idx=1,
        action_distribution=torch.tensor([10.0, 20.0]),
    )

    augmented = shuffle_hand([sample])[0]

    assert augmented is not sample
    assert augmented.state == (
        "Piles: 1 1 100 100\nHand: 78 45 12\n@12->0\n@45->0"
    )
    assert augmented.moves == sample.moves
    assert augmented.action_idx == sample.action_idx
    assert torch.equal(augmented.action_distribution, sample.action_distribution)
    assert sample.state == "Piles: 1 1 100 100\nHand: 12 45 78\n@12->0\n@45->0"


def test_thegame_shuffle_hand_preserves_hand_line_ending(monkeypatch):
    monkeypatch.setattr(
        "boardrl.games.thegame.augmentations.random.shuffle",
        lambda values: values.reverse(),
    )
    sample = TrainingSample(state="Hand: 12 45\r\n", action_idx=0)

    assert shuffle_hand([sample])[0].state == "Hand: 45 12\r\n"


def test_thegame_registers_hand_augmentation():
    assert games_library("thegame").augmentations == (shuffle_actions, shuffle_hand)
