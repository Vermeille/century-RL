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
    game.play_str("P:a1,e5")
    game.play_str("P:e1,a5")
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
    assert len(game.moves) == 300
    assert game.moves[0] == "P:a1,b1"
    assert game.display_with_moves().count("@") == 300


def test_setup_places_both_workers_in_one_action():
    game = Santorini()
    game.play_str("P:a1,e5")

    assert game.current_player() == 1
    assert game.workers[0] == [game._index("a1"), game._index("e5")]
    assert len(game.moves) == 253

    game.play_str("P:e1,a5")
    assert game.current_player() == 0
    assert game.round() == 0
    assert all(not move.startswith("P:") for move in game.moves)


def test_move_and_build_can_use_vacated_space():
    game = setup_game()

    assert "a1>b1+a1" in game.moves
    game.play_str("a1>b1+a1")

    assert game._index("b1") in game.workers[0]
    assert game.heights[game._index("a1")] == 1
    assert game.current_player() == 1


def test_cannot_climb_more_than_one_level_or_move_onto_dome():
    game = Santorini()
    game.play_str("P:b2,e5")
    game.play_str("P:e1,a5")

    game.heights[game._index("c3")] = 2
    game.heights[game._index("a2")] = game.DOME
    game._refresh_moves()

    assert not any(move.startswith("b2>c3") for move in game.moves)
    assert not any(move.startswith("b2>a2") for move in game.moves)


def test_moving_up_to_level_three_wins_without_build():
    game = Santorini()
    game.play_str("P:b2,e5")
    game.play_str("P:e1,a5")

    game.heights[game._index("b2")] = 2
    game.heights[game._index("c3")] = 3
    game._refresh_moves()

    assert "b2>c3" in game.moves
    assert not any(move.startswith("b2>c3+") for move in game.moves)

    game.play_str("b2>c3")

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

    copied.play_str("a1>b1+a1")

    assert game.workers != copied.workers
    assert game.heights != copied.heights
    assert game.turn == 2
    assert copied.turn == 3


def test_illegal_move_is_rejected():
    game = setup_game()

    with pytest.raises(AssertionError):
        game.play_str("a1>e5+a2")


def test_random_playouts_terminate():
    for _ in range(8):
        game = Santorini()
        game.simulate_to_end()

        assert game.ended()
        assert game.winner() in (0, 1)
        assert game.turn <= 128
