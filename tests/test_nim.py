import pytest

from boardrl.games import games_library
from boardrl.games.nim.game import Nim


def test_nim_registered_with_arguments():
    desc = games_library("nim,num_stones=5,max_pick=2")
    game = desc.make_game()

    assert game.num_stones == 5
    assert game.max_pick == 2
    assert game.moves == ["1", "2"]


def test_nim_taking_last_stone_wins():
    game = Nim(num_stones=4, max_pick=3)

    game.play_str("1")
    game.play_str("3")

    assert game.ended()
    assert game.winner() == 1
    assert game.points_for(0) == -1
    assert game.points_for(1) == 1


def test_nim_legal_moves_follow_remaining_stones():
    game = Nim(num_stones=2, max_pick=3)

    assert game.moves == ["1", "2"]
    with pytest.raises(AssertionError):
        game.play_str("3")

    game.play_str("1")

    assert game.moves == ["1"]
    with pytest.raises(AssertionError):
        game.play_str("2")


def test_nim_copy_is_independent():
    game = Nim(num_stones=3, max_pick=3)
    game.play_str("1")

    copied = game.copy()
    copied.play_str("2")

    assert copied.ended()
    assert not game.ended()
    assert game.num_stones == 2
    assert game.moves == ["1", "2"]
