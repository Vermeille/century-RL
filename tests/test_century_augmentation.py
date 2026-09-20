import torch

from boardrl.games import games_library
from boardrl.games.augmentations import shuffle_actions
from boardrl.games.century.augmentations import shuffle_discard, shuffle_hand
from boardrl.training import TrainingSample


def _reverse_shuffle(monkeypatch):
    monkeypatch.setattr(
        "boardrl.games.century.augmentations.random.shuffle",
        lambda values: values.reverse(),
    )


def test_shuffle_hand_remaps_hand_moves_without_touching_discard(monkeypatch):
    _reverse_shuffle(monkeypatch)
    sample = TrainingSample(
        state=(
            "0    4\n"
            "_Board\n"
            "V0 YRGB>12\n"
            "_Him 1 0\n"
            "S YYYY\n"
            "H0 SHOULD_STAY\n"
            "_Me 0\n"
            "V 1\n"
            "S YYY\n"
            "H0 >YY\n"
            "H1 XX\n"
            "H2 G>RR\n"
            "D0 YY>R\n"
            "D1 >B\n"
            "_Moves\n"
            "@R\n"
            "@H0 >YY\n"
            "@H2 G>RR"
        ),
        moves=["R", "H0 >YY", "H2 G>RR"],
        action_idx=2,
        action_distribution=torch.tensor([0.1, 0.2, 0.7]),
        reference_policy=torch.tensor([0.3, 0.4, 0.3]),
    )

    augmented = shuffle_hand([sample])[0]

    assert augmented is not sample
    assert augmented.state == (
        "0    4\n"
        "_Board\n"
        "V0 YRGB>12\n"
        "_Him 1 0\n"
        "S YYYY\n"
        "H0 SHOULD_STAY\n"
        "_Me 0\n"
        "V 1\n"
        "S YYY\n"
        "H0 G>RR\n"
        "H1 XX\n"
        "H2 >YY\n"
        "D0 YY>R\n"
        "D1 >B\n"
        "_Moves\n"
        "@R\n"
        "@H2 >YY\n"
        "@H0 G>RR"
    )
    assert augmented.moves == ["R", "H2 >YY", "H0 G>RR"]
    assert augmented.action_idx == 2
    assert torch.equal(augmented.action_distribution, sample.action_distribution)
    assert torch.equal(augmented.reference_policy, sample.reference_policy)

    assert sample.moves == ["R", "H0 >YY", "H2 G>RR"]
    assert "H0 >YY\nH1 XX\nH2 G>RR" in sample.state
    assert "D0 YY>R\nD1 >B" in sample.state


def test_shuffle_discard_does_not_touch_hand_or_moves(monkeypatch):
    _reverse_shuffle(monkeypatch)
    sample = TrainingSample(
        state=(
            "_Me 0\n"
            "H0 >YY\n"
            "H1 G>RR\n"
            "D0 YY>R\n"
            "D1 >B\n"
            "_Moves\n"
            "@H0 >YY\n"
            "@H1 G>RR"
        ),
        moves=["H0 >YY", "H1 G>RR"],
        action_idx=1,
        action_distribution=torch.tensor([0.4, 0.6]),
    )

    augmented = shuffle_discard([sample])[0]

    assert augmented.state == (
        "_Me 0\n"
        "H0 >YY\n"
        "H1 G>RR\n"
        "D0 >B\n"
        "D1 YY>R\n"
        "_Moves\n"
        "@H0 >YY\n"
        "@H1 G>RR"
    )
    assert augmented.moves == sample.moves
    assert augmented.action_idx == sample.action_idx
    assert torch.equal(augmented.action_distribution, sample.action_distribution)


def test_shuffle_hand_preserves_line_endings(monkeypatch):
    _reverse_shuffle(monkeypatch)
    sample = TrainingSample(
        state=(
            "_Me 0\r\n"
            "H0 >YY\r\n"
            "H1 G>RR\r\n"
            "_Moves\r\n"
            "@H0 >YY\r\n"
            "@H1 G>RR\r\n"
        ),
        moves=("H0 >YY", "H1 G>RR"),
        action_idx=0,
    )

    augmented = shuffle_hand([sample])[0]

    assert augmented.state == (
        "_Me 0\r\n"
        "H0 G>RR\r\n"
        "H1 >YY\r\n"
        "_Moves\r\n"
        "@H1 >YY\r\n"
        "@H0 G>RR\r\n"
    )
    assert augmented.moves == ("H1 >YY", "H0 G>RR")


def test_shuffle_discard_preserves_line_endings(monkeypatch):
    _reverse_shuffle(monkeypatch)
    sample = TrainingSample(
        state=(
            "_Me 0\r\n"
            "D0 >R\r\n"
            "D1 >G\r\n"
            "_Moves\r\n"
            "@R\r\n"
        ),
        action_idx=0,
    )

    augmented = shuffle_discard([sample])[0]

    assert augmented.state == (
        "_Me 0\r\n"
        "D0 >G\r\n"
        "D1 >R\r\n"
        "_Moves\r\n"
        "@R\r\n"
    )


def test_century_card_shuffles_without_me_section_are_noops(monkeypatch):
    _reverse_shuffle(monkeypatch)
    sample = TrainingSample(
        state="H0 >YY\nH1 G>RR\nD0 >R\nD1 >G\n@H0 >YY\n@H1 G>RR",
        moves=["H0 >YY", "H1 G>RR"],
        action_idx=1,
    )

    hand_augmented = shuffle_hand([sample])[0]
    discard_augmented = shuffle_discard([sample])[0]

    for augmented in (hand_augmented, discard_augmented):
        assert augmented is not sample
        assert augmented.state == sample.state
        assert augmented.moves == sample.moves
        assert augmented.action_idx == sample.action_idx


def test_century_registers_hand_and_discard_augmentations_separately():
    assert games_library("century").augmentations == (
        shuffle_actions,
        shuffle_hand,
        shuffle_discard,
    )
