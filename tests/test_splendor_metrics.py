from types import SimpleNamespace

import pytest

from boardrl.games.semantics import PointScores
from boardrl.games.splendor.metrics import SplendorTraceMetrics
from boardrl.rollouts import PlayerTrace, TraceGroup


def _record(move):
    return SimpleNamespace(moves=[move], action_idx=0)


def _trace(moves, *, score=15, developments=8, reserved=2):
    trace = PlayerTrace(seat_id=0, strategy_id=0)
    trace.extend(_record(move) for move in moves)
    hand = "-" if reserved == 0 else " ".join(f"{i}=?" for i in range(reserved))
    trace.append(
        SimpleNamespace(
            state=(
                f"P0 S{score} D{developments} TW0B0G0R0K0Y0 "
                f"CW1B2G2R2K1 H{hand}\n"
            ),
            my_points=score,
        )
    )
    return trace


def test_splendor_behavior_metrics_track_strategy_shape():
    trace = _trace(
        [
            "T:WBG",
            "T:WW",
            "R:1.D",
            "R:2.0",
            "B:1.0",
            "B:2.1~WB",
            "B:H0~G",
            "D:W",
        ]
    )
    group = TraceGroup(0, [trace])

    metrics = SplendorTraceMetrics(group, scores=PointScores()).behavior_metrics()

    assert metrics["main_action_mix"] == pytest.approx(
        {"take": 2 / 7, "reserve": 2 / 7, "buy": 3 / 7}
    )
    assert metrics["take_double_same_share"] == pytest.approx(0.5)
    assert metrics["reserve_blind_share"] == pytest.approx(0.5)
    assert metrics["buy_from_reserve_share"] == pytest.approx(1 / 3)
    assert metrics["buy_with_gold_share"] == pytest.approx(2 / 3)
    assert metrics["gold_per_buy"] == pytest.approx(1.0)
    assert metrics["discard_per_main_action"] == pytest.approx(1 / 7)
    assert metrics["market_buy_tier_share"] == pytest.approx(
        {"1": 0.5, "2": 0.5, "3": 0.0}
    )


def test_splendor_endgame_metrics_track_efficiency_and_leftovers():
    first = _trace(
        ["T:WBG", "R:1.D", "B:1.0", "B:H0"],
        score=15,
        developments=8,
        reserved=1,
    )
    second = _trace(
        ["T:WBG", "T:WW", "B:1.0", "B:2.0", "B:3.0", "R:2.D"],
        score=12,
        developments=10,
        reserved=2,
    )
    group = TraceGroup(0, [first, second])

    metrics = SplendorTraceMetrics(group, scores=PointScores()).endgame_metrics()

    assert metrics["avg_final_developments"] == pytest.approx(9.0)
    assert metrics["avg_final_reserved"] == pytest.approx(1.5)
    assert metrics["points_per_main_action"] == pytest.approx((15 / 4 + 12 / 6) / 2)


def test_splendor_behavior_metrics_are_zero_safe():
    trace = _trace(["N:2"], score=0, developments=0, reserved=0)
    group = TraceGroup(0, [trace])

    metrics = SplendorTraceMetrics(group, scores=PointScores()).behavior_metrics()

    assert metrics["main_action_mix"] == {"take": 0.0, "reserve": 0.0, "buy": 0.0}
    assert metrics["take_double_same_share"] == 0.0
    assert metrics["reserve_blind_share"] == 0.0
    assert metrics["buy_from_reserve_share"] == 0.0
    assert metrics["buy_with_gold_share"] == 0.0
    assert metrics["gold_per_buy"] == 0.0
    assert metrics["market_buy_tier_share"] == {"1": 0.0, "2": 0.0, "3": 0.0}
