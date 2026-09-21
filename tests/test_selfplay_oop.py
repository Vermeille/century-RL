import math

from types import SimpleNamespace

import pytest
import torch

from boardrl.games import games_library
from boardrl.games.semantics import CooperativeOutcome, OutcomeScores, PointScores
from boardrl.evaluation import Evaluation, Scoreboard
from boardrl.metrics import rollout_metrics
from boardrl.rl.eval.selfplay import (
    EndState,
    GameTrace,
    PlayerTrace,
    Record,
    SelfPlayResults,
    pit,
)


class _TerminalGame:
    def __init__(self, points, *, won=False, ended=True):
        self._points = points if isinstance(points, list) else [points, 0]
        self._won = won
        self._ended = ended
        self.num_players = len(self._points)

    def display(self, force=-1):
        return "terminal"

    def ended(self):
        return self._ended

    def points_for(self, player):
        return self._points[player]

    def diff_points_for(self, player):
        opponents = self._points[:player] + self._points[player + 1 :]
        return self._points[player] - max(opponents)

    def won(self):
        return self._won

    def round(self):
        return 3


def test_record_preserves_complete_rollout_reference_values():
    game = SimpleNamespace(
        moves=["a", "b"],
        diff_points=lambda: 3,
        points=lambda: 5,
        current_player=lambda: 0,
        round=lambda: 7,
    )
    record = Record(
        game,
        torch.tensor([0.1, 0.2]),
        1,
        {
            "state": "position\n@a\n@b",
            "reference_policy": torch.tensor([0.3, 0.4]),
            "reference_value": 2.0,
            "reference_value_stddev": 1.25,
            "reference_max_q": 2.05,
        },
    )
    record.score = 3.0
    record.reward = 1.0
    record.returns = 1.0

    sample = record.training_sample()

    assert sample.reference_value_stddev == pytest.approx(1.25)


def test_end_state_keeps_points_separate_from_competitive_utility():
    end = EndState(_TerminalGame([17, 3]), 0)

    assert end.my_points == 17
    assert end.current_diff_points == 14
    assert end.episodic_utility == 1.0


def test_end_state_uses_relative_rank_when_all_points_are_negative():
    winner = EndState(_TerminalGame([-3, -8]), 0)
    loser = EndState(_TerminalGame([-3, -8]), 1)

    assert winner.episodic_utility == 1.0
    assert loser.episodic_utility == -1.0


def test_end_state_uses_cooperative_win_condition_for_utility():
    end = EndState(
        _TerminalGame([17, 17], won=False), 0, outcome=CooperativeOutcome()
    )

    assert end.my_points == 17
    assert end.episodic_utility == -1.0


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


def _make_action_trace(seat_id, strategy_id, action, score):
    trace = PlayerTrace(seat_id=seat_id, strategy_id=strategy_id)
    trace.append(
        SimpleNamespace(
            action_idx=0,
            action_distribution=torch.zeros(1),
            moves=[action],
            reward=score,
        )
    )
    trace.append(
        SimpleNamespace(
            current_diff_points=score,
            my_points=score,
            reward=0.0,
        )
    )
    return trace


def test_trace_groups_distinguish_strategy_identity_from_rotated_seats():
    results = SelfPlayResults(
        [
            GameTrace(
                [
                    _make_action_trace(0, 0, "agent", 1.0),
                    _make_action_trace(1, 1, "bot", -1.0),
                ]
            ),
            GameTrace(
                [
                    _make_action_trace(0, 1, "bot", 1.0),
                    _make_action_trace(1, 0, "agent", -1.0),
                ]
            ),
        ]
    )

    agent = results.by_strategy.group(0)
    first_seat = results.by_seat.group(0)

    assert agent.points() == [1.0, -1.0]
    assert agent.win_rate() == pytest.approx(0.5)
    assert agent.sensitivity() == pytest.approx(0.0)
    assert first_seat.points() == [1.0, 1.0]
    assert first_seat.win_rate() == pytest.approx(1.0)
    assert first_seat.sensitivity() == pytest.approx(1.0)
    assert {str(group.identity): group.win_rate() for group in results.by_strategy} == {
        "0": pytest.approx(0.5),
        "1": pytest.approx(0.5),
    }


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
    assert rollout_metrics(results, scores=OutcomeScores())["win_rate"] == pytest.approx(
        0.5
    )
    assert "points" not in rollout_metrics(results, scores=OutcomeScores())


def test_cooperative_win_rate_is_shared_objective_success() -> None:
    results = SelfPlayResults(
        [
            _make_outcome_trace([(0, 98.0), (1, 98.0)], won=True),
            _make_outcome_trace([(0, 50.0), (1, 50.0)], won=False),
        ]
    )

    evaluation = Evaluation(
        ("first", "first"), results, outcome=CooperativeOutcome()
    )

    assert evaluation.win_rate() == pytest.approx(0.5)
    assert evaluation.win_rate(1) == pytest.approx(0.5)
    assert rollout_metrics(
        results, outcome=CooperativeOutcome(), scores=PointScores()
    )[
        "win_rate"
    ] == pytest.approx(0.5)


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
