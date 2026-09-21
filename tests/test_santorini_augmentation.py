import torch

from boardrl.games import games_library
from boardrl.games.augmentations import shuffle_actions
from boardrl.games.santorini.augmentations import (
    horizontal_symmetry,
    rotation_symmetry,
    vertical_symmetry,
)
from boardrl.games.santorini.game import Santorini
from boardrl.training import TrainingSample


def setup_sample():
    game = Santorini()
    game.play_str("P:a1")
    return TrainingSample(
        state=game.display_with_moves(),
        moves=list(game.moves),
        action_idx=game.moves.index("P:a2"),
        action_distribution=torch.arange(len(game.moves), dtype=torch.float32),
        reference_policy=torch.arange(len(game.moves), dtype=torch.float32) + 100,
    )


def board_rows(state):
    return [
        line
        for line in state.splitlines()
        if len(line) == 16 and line[:1] in "12345" and line[1] == " "
    ]


def test_horizontal_symmetry_mirrors_board_and_setup_moves(monkeypatch):
    monkeypatch.setattr(
        "boardrl.games.santorini.augmentations.random.random", lambda: 0.0
    )
    sample = setup_sample()

    augmented = horizontal_symmetry([sample])[0]

    assert board_rows(augmented.state)[0] == "1 0. 0. 0. 0. 0O"
    assert "@P:e1" not in augmented.state  # occupied transformed worker square
    index = sample.moves.index("P:a2")
    assert augmented.moves[index] == "P:e2"
    assert augmented.action_idx == sample.action_idx
    assert torch.equal(augmented.action_distribution, sample.action_distribution)
    assert torch.equal(augmented.reference_policy, sample.reference_policy)
    assert board_rows(sample.state)[0] == "1 0O 0. 0. 0. 0."


def test_vertical_symmetry_mirrors_board_and_setup_moves(monkeypatch):
    monkeypatch.setattr(
        "boardrl.games.santorini.augmentations.random.random", lambda: 0.0
    )
    sample = setup_sample()

    augmented = vertical_symmetry([sample])[0]

    assert board_rows(augmented.state)[4] == "5 0O 0. 0. 0. 0."
    index = sample.moves.index("P:a2")
    assert augmented.moves[index] == "P:a4"
    assert augmented.action_idx == sample.action_idx


def test_rotation_symmetry_rotates_board_and_all_move_coordinates(monkeypatch):
    monkeypatch.setattr(
        "boardrl.games.santorini.augmentations.random.randrange", lambda _n: 1
    )
    sample = setup_sample()

    augmented = rotation_symmetry([sample])[0]

    assert board_rows(augmented.state)[0] == "1 0. 0. 0. 0. 0O"
    index = sample.moves.index("P:a2")
    assert augmented.moves[index] == "P:d1"


def test_symmetries_transform_source_and_relative_directions(monkeypatch):
    game = Santorini()
    for move in ("P:a1", "P:e5", "P:e1", "P:a5"):
        game.play_str(move)

    game.heights[game._index("a1")] = 2
    game.heights[game._index("b2")] = 3
    game._refresh_moves()
    assert "a1>dr" in game.moves
    assert "e5>ul+ul" in game.moves

    sample = TrainingSample(
        state=game.display_with_moves(),
        moves=list(game.moves),
        action_idx=game.moves.index("a1>dr"),
    )
    monkeypatch.setattr(
        "boardrl.games.santorini.augmentations.random.randrange", lambda _n: 1
    )

    augmented = rotation_symmetry([sample])[0]

    assert "e1>dl" in augmented.moves
    assert "a5>ur+ur" in augmented.moves
    assert "@e1>dl" in augmented.state
    assert "@a5>ur+ur" in augmented.state
    assert augmented.moves[augmented.action_idx] == "e1>dl"


def test_horizontal_and_vertical_flips_transform_directions(monkeypatch):
    game = Santorini()
    for move in ("P:a1", "P:e5", "P:e1", "P:a5"):
        game.play_str(move)
    assert "a1>r+l" in game.moves
    assert "e5>ul+ul" in game.moves
    sample = TrainingSample(state=game.display_with_moves(), moves=list(game.moves))

    monkeypatch.setattr(
        "boardrl.games.santorini.augmentations.random.random", lambda: 0.0
    )
    horizontal = horizontal_symmetry([sample])[0]
    vertical = vertical_symmetry([sample])[0]

    assert "e1>l+r" in horizontal.moves
    assert "a5>ur+ur" in horizontal.moves
    assert "a5>r+l" in vertical.moves
    assert "e1>dl+dl" in vertical.moves


def test_symmetry_augmentations_can_be_identity(monkeypatch):
    sample = setup_sample()
    monkeypatch.setattr(
        "boardrl.games.santorini.augmentations.random.random", lambda: 1.0
    )
    monkeypatch.setattr(
        "boardrl.games.santorini.augmentations.random.randrange", lambda _n: 0
    )

    horizontal = horizontal_symmetry([sample])[0]
    vertical = vertical_symmetry([sample])[0]
    rotated = rotation_symmetry([sample])[0]

    for augmented in (horizontal, vertical, rotated):
        assert augmented is not sample
        assert augmented.state == sample.state
        assert augmented.moves == sample.moves
        assert augmented.action_idx == sample.action_idx


def test_santorini_registers_full_square_symmetry_augmentations():
    assert games_library("santorini").augmentations == (
        shuffle_actions,
        horizontal_symmetry,
        vertical_symmetry,
        rotation_symmetry,
    )
