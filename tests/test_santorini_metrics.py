from types import SimpleNamespace

import pytest

from boardrl.games.santorini.metrics import SantoriniTraceMetrics
from boardrl.games.semantics import OutcomeScores
from boardrl.rollouts import PlayerTrace, TraceGroup


def _record(state, moves, action_idx=0):
    return SimpleNamespace(state=state, moves=moves, action_idx=action_idx)


def _trace(records, *, score=1, seat_id=0):
    trace = PlayerTrace(seat_id=seat_id, strategy_id=0)
    trace.extend(records)
    trace.append(SimpleNamespace(state="", my_points=score))
    return trace


def _state(rows):
    return "\n".join(
        [">0 play", "   a  b  c  d  e", *rows]
    ) + "\n"


def test_santorini_behavior_metrics_track_strategy_shape():
    first = _record(
        _state(
            [
                "1 0. 0. 0. 0. 0.",
                "2 0. 0O 0. 0. 0.",
                "3 0. 0. 1. 0. 0.",
                "4 0. 0. 0. 1X 0.",
                "5 0O 0. 0. 0. 0.",
            ]
        ),
        ["b2>c3+b2", "b2>a2+b2"],
    )
    second = _record(
        _state(
            [
                "1 0. 0. 0. 0. 0.",
                "2 0. 0. 0. 0. 0.",
                "3 0. 0. 2O 0. 0.",
                "4 0. 0. 3. 0. 0.",
                "5 0O 0. 0. 0. 0X",
            ]
        ),
        ["c3>c4", "c3>b3+b2"],
    )
    third = _record(
        _state(
            [
                "1 0. 0. 0. 0. 0.",
                "2 0. 0O 0. 0. 0.",
                "3 0. 0. 0. 1. 0X",
                "4 0. 0. 0. 0. 0.",
                "5 0O 0. 0. 0. 0.",
            ]
        ),
        ["b2>c2+d3"],
    )
    group = TraceGroup(0, [_trace([first, second, third])])

    metrics = SantoriniTraceMetrics(
        group, scores=OutcomeScores()
    ).behavior_metrics()

    assert metrics["mobility"]["avg_move_options"] == pytest.approx(5 / 3)
    assert metrics["mobility"]["winning_threat_state_rate"] == pytest.approx(1 / 3)
    assert metrics["mobility"]["winning_threat_conversion_rate"] == pytest.approx(1.0)

    assert metrics["position"]["avg_worker_height"] == pytest.approx(1 / 3)
    assert metrics["position"]["inner_board_destination_share"] == pytest.approx(1.0)

    assert metrics["movement"]["height_mix"] == pytest.approx(
        {"up": 2 / 3, "flat": 1 / 3, "down": 0.0}
    )
    assert metrics["building"]["result_level_share"] == pytest.approx(
        {"1": 0.5, "2": 0.5, "3": 0.0, "dome": 0.0}
    )
    assert metrics["building"]["vacated_square_share"] == pytest.approx(0.5)
    assert metrics["building"]["opponent_blocking_share"] == pytest.approx(0.5)


def test_santorini_metrics_accept_relative_actions():
    record = _record(
        _state(
            [
                "1 0. 0. 0. 0. 0.",
                "2 0. 0O 0. 0. 0.",
                "3 0. 0. 1. 0. 0.",
                "4 0. 0. 0. 0X 0.",
                "5 0O 0. 0. 0. 0.",
            ]
        ),
        ["b2>dr+ul"],
    )
    group = TraceGroup(0, [_trace([record])])

    metrics = SantoriniTraceMetrics(
        group, scores=OutcomeScores()
    ).behavior_metrics()

    assert metrics["movement"]["height_mix"]["up"] == pytest.approx(1.0)
    assert metrics["building"]["vacated_square_share"] == pytest.approx(1.0)
    assert metrics["building"]["result_level_share"]["1"] == pytest.approx(1.0)


def test_santorini_win_cause_distinguishes_climb_and_immobilization():
    empty = _state(
        [
            "1 0. 0. 0. 0. 0.",
            "2 0. 0O 0. 0. 0.",
            "3 0. 0. 3. 0. 0.",
            "4 0. 0. 0. 0X 0.",
            "5 0O 0. 0. 0. 0.",
        ]
    )
    climb = _trace([_record(empty, ["b2>c3"])], score=1)
    immobilization = _trace(
        [_record(empty, ["b2>c2+d3"])], score=1
    )
    loss = _trace([_record(empty, ["b2>c2+d3"])], score=-1)
    group = TraceGroup(0, [climb, immobilization, loss])

    metrics = SantoriniTraceMetrics(
        group, scores=OutcomeScores()
    ).win_cause_metrics()

    assert metrics == pytest.approx(
        {"climb_share": 0.5, "immobilization_share": 0.5}
    )


def test_santorini_behavior_metrics_are_zero_safe():
    trace = _trace([], score=0)
    group = TraceGroup(0, [trace])

    metrics = SantoriniTraceMetrics(
        group, scores=OutcomeScores()
    ).behavior_metrics()

    assert metrics["mobility"]["avg_move_options"] == 0.0
    assert metrics["mobility"]["winning_threat_state_rate"] == 0.0
    assert metrics["mobility"]["winning_threat_conversion_rate"] == 0.0
    assert metrics["position"]["avg_worker_height"] == 0.0
    assert metrics["movement"]["height_mix"] == {
        "up": 0.0,
        "flat": 0.0,
        "down": 0.0,
    }
    assert metrics["building"]["result_level_share"] == {
        "1": 0.0,
        "2": 0.0,
        "3": 0.0,
        "dome": 0.0,
    }
