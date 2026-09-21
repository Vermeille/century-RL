import pytest

from boardrl.games import games_library
from boardrl.games.santorini.game import Santorini
from boardrl.games.semantics import (
    CompetitiveOutcome,
    OutcomeScores,
    TerminalOutcomeRewards,
)


def setup_game():
    game = Santorini()
    game.play_str("P:a1")
    game.play_str("P:e5")
    game.play_str("P:e1")
    game.play_str("P:a5")
    return game


def test_registration_and_initial_setup():
    desc = games_library("santorini")
    game = desc.make_game()

    assert isinstance(game, Santorini)
    assert type(desc.scores) is OutcomeScores
    assert type(desc.outcome) is CompetitiveOutcome
    assert type(desc.rewards) is TerminalOutcomeRewards
    assert desc.coop is False

    assert game.current_player() == 0
    assert game.round() == 0
    assert len(game.moves) == 25
    assert game.moves[0] == "P:a1"
    assert game.display_with_moves().count("@") == 25
    assert len(game.display_with_moves()) < 2048


def test_setup_places_two_workers_for_each_player_in_order():
    game = Santorini()

    game.play_str("P:a1")
    assert game.current_player() == 0
    assert len(game.moves) == 24

    game.play_str("P:e5")
    assert game.current_player() == 1
    assert game.workers[0] == [game._index("a1"), game._index("e5")]
    assert len(game.moves) == 23

    game.play_str("P:e1")
    assert game.current_player() == 1
    assert len(game.moves) == 22

    game.play_str("P:a5")
    assert game.current_player() == 0
    assert game.workers[1] == [game._index("e1"), game._index("a5")]
    assert game.round() == 0
    assert all(not move.startswith("P:") for move in game.moves)


def test_normal_actions_use_relative_move_and_build_directions():
    game = setup_game()

    assert "a1>r+l" in game.moves
    assert "a1>d+u" in game.moves
    assert "e5>ul+ul" in game.moves
    assert all(
        move.split(">", 1)[1].split("+", 1)[0] in Santorini.DIRECTIONS
        for move in game.moves
    )


def test_move_and_build_can_use_vacated_space():
    game = setup_game()

    assert "a1>r+l" in game.moves
    game.play_str("a1>r+l")

    assert game._index("b1") in game.workers[0]
    assert game.heights[game._index("a1")] == 1
    assert game.current_player() == 1


def test_cannot_climb_more_than_one_level_or_move_onto_dome():
    game = Santorini()
    for move in ("P:b2", "P:e5", "P:e1", "P:a5"):
        game.play_str(move)

    game.heights[game._index("c3")] = 2
    game.heights[game._index("a2")] = game.DOME
    game._refresh_moves()

    assert not any(move.startswith("b2>dr") for move in game.moves)
    assert not any(move.startswith("b2>l") for move in game.moves)


def test_moving_up_to_level_three_wins_without_build():
    game = Santorini()
    for move in ("P:b2", "P:e5", "P:e1", "P:a5"):
        game.play_str(move)

    game.heights[game._index("b2")] = 2
    game.heights[game._index("c3")] = 3
    game._refresh_moves()

    assert "b2>dr" in game.moves
    assert not any(move.startswith("b2>dr+") for move in game.moves)

    game.play_str("b2>dr")

    assert game.ended()
    assert game.winner() == 0
    assert game.points_for(0) == 1
    assert game.points_for(1) == -1
    assert game.moves == []


def test_player_with_no_complete_move_and_build_loses():
    game = setup_game()

    game.heights = [game.DOME for _ in game.heights]
    for workers in game.workers:
        for position in workers:
            game.heights[position] = 0
    game._winner = None
    game._refresh_moves()

    assert game.ended()
    assert game.winner() == 1
    assert game.moves == []


def test_copy_is_independent():
    game = setup_game()
    copied = game.copy()

    copied.play_str("a1>r+l")

    assert game.workers != copied.workers
    assert game.heights != copied.heights
    assert game.turn == 4
    assert copied.turn == 5


def test_illegal_move_is_rejected():
    game = setup_game()

    with pytest.raises(AssertionError):
        game.play_str("a1>ul+r")


def test_random_playouts_terminate():
    for _ in range(8):
        game = Santorini()
        game.simulate_to_end()

        assert game.ended()
        assert game.winner() in (0, 1)
        assert game.turn <= 130
