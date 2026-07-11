import math

from types import SimpleNamespace

import pytest

from boardrl.games import games_library
from boardrl.rl.eval.matchmaker import MatchMaker
from boardrl.rl.eval.selfplay import GameTrace, PlayerTrace, SelfPlayResults, pit


def test_selfplay_oop_indexing_and_filters():
    game_desc = games_library("tictactoe")
    strategies = [
        lambda: game_desc.strategy_from_string("random"),
        lambda: game_desc.strategy_from_string("random"),
    ]

    results = pit(game_desc.make_game, strategies, n_games=2, max_len=2)
    assert isinstance(results, SelfPlayResults)

    game0 = results[0]
    # seat indexing
    assert game0[0].seat_id == 0
    # strategy indexing
    assert game0.by_strategy[0].strategy_id == 0

    # filtering helpers
    assert len(results.only_player([0])[0]) == 1
    assert len(results.only_strategy([0])[0]) == 1

    # metrics: strategy-indexed
    win_rate = results.win_rate(0)
    assert 0.0 <= win_rate <= 1.0 and not math.isnan(win_rate)
    # metrics: seat-indexed
    seat_win_rate = results.win_rate(0, by="seat")
    assert 0.0 <= seat_win_rate <= 1.0 and not math.isnan(seat_win_rate)


def test_pit_rotate_flag():
    game_desc = games_library("tictactoe")
    strategies = [
        lambda: game_desc.strategy_from_string("random"),
        lambda: game_desc.strategy_from_string("random"),
    ]

    rotated = pit(game_desc.make_game, strategies, n_games=2, max_len=2, rotate=True)
    static = pit(game_desc.make_game, strategies, n_games=2, max_len=2, rotate=False)

    assert rotated[1][0].strategy_id == 1
    assert static[1][0].strategy_id == 0


def test_max_len_is_truncation_not_terminal():
    game_desc = games_library("tictactoe")
    strategies = [
        lambda: game_desc.strategy_from_string("random"),
        lambda: game_desc.strategy_from_string("random"),
    ]

    results = pit(game_desc.make_game, strategies, n_games=1, max_len=1)

    for trace in results[0]:
        end = trace[-1]
        assert end.cause == "toolong"
        assert not end.terminal
        assert end.truncated


def _make_game_trace(scores: list[tuple[int, float]]):
    traces = []
    for seat_id, (strategy_id, score) in enumerate(scores):
        trace = PlayerTrace(seat_id=seat_id, strategy_id=strategy_id)
        trace.append(
            SimpleNamespace(current_diff_points=score, my_points=score, final=True)
        )
        traces.append(trace)
    return GameTrace(traces)


def test_matchmaker_tracks_win_rates():
    game_desc = games_library("tictactoe")
    maker = MatchMaker(game_desc, model_pool=None, discount_factor=1.0)
    strategies = ["alpha", "beta"]

    first_game = SelfPlayResults([
        _make_game_trace([(0, 1.0), (1, -1.0)])
    ])
    maker._record_outcomes(strategies, first_game)

    matrix = maker.win_matrix
    assert matrix["alpha"]["beta"] == pytest.approx(1.0)
    assert matrix["beta"]["alpha"] == pytest.approx(0.0)

    second_game = SelfPlayResults([
        _make_game_trace([(0, 0.0), (1, 0.0)])
    ])
    maker._record_outcomes(strategies, second_game)

    matrix = maker.win_matrix
    assert matrix["alpha"]["beta"] == pytest.approx(0.75)
    assert matrix["beta"]["alpha"] == pytest.approx(0.25)

    stats = maker.head_to_head("alpha", "beta")
    assert stats.games == 2
    assert stats.win_rate == pytest.approx(0.75)

    # Elo tracking is no longer supported; only win-rates and head-to-head remain
