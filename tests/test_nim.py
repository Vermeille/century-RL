import pytest
import torch
from types import SimpleNamespace

from boardrl.games import games_library
from boardrl.games.nim.game import Nim
from boardrl.games.nim.metrics import Metrics
from boardrl.rl.eval.selfplay import GameTrace, PlayerTrace, SelfPlayResults


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


def test_nim_optimal_strategy_plays_winning_move():
    desc = games_library("nim,num_stones=7,max_pick=3")
    strat = desc.strategy_from_string("optimal")
    game = desc.make_game()

    import asyncio

    dist, info = asyncio.run(strat(game))

    assert info["moves"] == {"1": 0.0, "2": 0.0, "3": 1.0}
    assert dist.argmax().item() == 2


def test_nim_optimal_strategy_falls_back_on_losing_position():
    desc = games_library("nim,num_stones=4,max_pick=3")
    strat = desc.strategy_from_string("optimal")
    game = desc.make_game()

    import asyncio

    dist, info = asyncio.run(strat(game))

    assert dist.argmax().item() == 0
    assert info["moves"] == {"1": 1.0, "2": 0.0, "3": 0.0}


def _record(moves, action_distribution, action_idx):
    return SimpleNamespace(
        moves=moves,
        action_distribution=action_distribution,
        action_idx=action_idx,
        final=False,
    )


def _end(player, points=0):
    return SimpleNamespace(
        final=True,
        player=player,
        round=0,
        state="",
        my_points=points,
        current_diff_points=points,
    )


def test_nim_metrics_reports_per_match_choice_probability():
    moves = ["1", "2", "3"]

    game1 = GameTrace(
        [
            PlayerTrace(0, 0),
            PlayerTrace(1, 1),
        ]
    )
    game1[0].extend(
        [
            _record(moves, torch.log(torch.tensor([0.2, 0.3, 0.5])), 2),
            _record(moves, torch.log(torch.tensor([0.9, 0.05, 0.05])), 0),
            _end(0),
        ]
    )
    game1[1].extend([
        _record(moves, torch.log(torch.tensor([0.4, 0.4, 0.2])), 1),
        _end(1),
    ])

    game2 = GameTrace(
        [
            PlayerTrace(0, 0),
            PlayerTrace(1, 1),
        ]
    )
    game2[0].extend(
        [
            _record(moves, torch.log(torch.tensor([0.3, 0.2, 0.5])), 2),
            _record(moves, torch.log(torch.tensor([0.1, 0.1, 0.8])), 2),
            _end(0),
        ]
    )
    game2[1].extend([
        _record(moves, torch.log(torch.tensor([0.1, 0.7, 0.2])), 1),
        _end(1),
    ])

    metrics = Metrics(SelfPlayResults([game1, game2]))

    assert metrics.metrics()["chosen_move_probability_by_match"] == [
        0.5,
        pytest.approx(0.85),
    ]
