import math

from types import SimpleNamespace

import pytest

from boardrl.games import games_library
from boardrl.evaluation import Evaluation, Scoreboard
from boardrl.metrics import rollout_metrics
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
            SimpleNamespace(
                current_diff_points=score,
                my_points=score,
                reward=0.0,
                final=True,
            )
        )
        traces.append(trace)
    return GameTrace(traces)


def _make_outcome_trace(scores, won):
    traces = []
    for seat_id, (strategy_id, score) in enumerate(scores):
        trace = PlayerTrace(seat_id=seat_id, strategy_id=strategy_id)
        trace.append(
            SimpleNamespace(
                current_diff_points=score,
                my_points=score,
                reward=0.0,
                final=True,
                won=won,
            )
        )
        traces.append(trace)
    return GameTrace(traces)


def test_adversarial_win_rate_is_for_first_strategy_and_draws_are_half() -> None:
    results = SelfPlayResults(
        [
            _make_game_trace([(0, 1.0), (1, -1.0)]),
            _make_game_trace([(0, 0.0), (1, 0.0)]),
            _make_game_trace([(0, -1.0), (1, 1.0)]),
        ]
    )

    evaluation = Evaluation(("first", "second"), results)

    assert evaluation.win_rate() == pytest.approx(0.5)
    assert rollout_metrics(results)["win_rate"] == pytest.approx(0.5)


def test_cooperative_win_rate_is_shared_objective_success() -> None:
    results = SelfPlayResults(
        [
            _make_outcome_trace([(0, 98.0), (1, 98.0)], won=True),
            _make_outcome_trace([(0, 50.0), (1, 50.0)], won=False),
        ]
    )

    evaluation = Evaluation(("first", "first"), results, coop=True)

    assert evaluation.win_rate() == pytest.approx(0.5)
    assert evaluation.win_rate(1) == pytest.approx(0.5)
    assert rollout_metrics(results, coop=True)["win_rate"] == pytest.approx(0.5)


def test_thegame_won_requires_empty_deck_and_hands() -> None:
    game = games_library("thegame").make_game(num_players=2, max_value=20)

    game.deck.clear()
    game.hands = [[], []]
    assert game.won()

    game.hands[0].append(2)
    assert not game.won()


def test_scoreboard_tracks_explicit_evaluations():
    scoreboard = Scoreboard()
    scoreboard.record(
        Evaluation(
            ("alpha", "beta"),
            SelfPlayResults([_make_game_trace([(0, 1.0), (1, -1.0)])]),
        )
    )
    scoreboard.record(
        Evaluation(
            ("alpha", "beta"),
            SelfPlayResults([_make_game_trace([(0, 0.0), (1, 0.0)])]),
        )
    )

    assert scoreboard.win_rate("alpha", "beta") == pytest.approx(0.75)
    assert scoreboard.win_rate("beta", "alpha") == pytest.approx(0.25)
