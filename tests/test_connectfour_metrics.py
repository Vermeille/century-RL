import math
from types import SimpleNamespace

import pytest
import torch

from boardrl.games.connectfour.metrics import Metrics
from boardrl.metrics import Range, Trackio
from boardrl.rl.eval.selfplay import GameTrace, PlayerTrace, SelfPlayResults


def _trace(seat, strategy, move, probabilities, score):
    moves = ["a", "b"]
    trace = PlayerTrace(seat_id=seat, strategy_id=strategy)
    trace.append(
        SimpleNamespace(
            action_distribution=torch.tensor(probabilities).log(),
            action_idx=moves.index(move),
            moves=moves,
        )
    )
    trace.append(
        SimpleNamespace(
            current_diff_points=score,
            my_points=score,
            terminal=True,
            state="terminal",
        )
    )
    return trace


def _rotated_results():
    return SelfPlayResults(
        [
            GameTrace(
                [
                    _trace(0, 0, "b", [0.25, 0.75], 1.0),
                    _trace(1, 1, "a", [1.0, 0.0], -1.0),
                ]
            ),
            GameTrace(
                [
                    _trace(0, 1, "a", [1.0, 0.0], 1.0),
                    _trace(1, 0, "b", [0.25, 0.75], -1.0),
                ]
            ),
        ]
    )


def test_connectfour_metrics_separate_strategies():
    metrics = Metrics(_rotated_results()).metrics()

    assert metrics["terminal_rate"] == 1.0
    assert metrics["draw_rate"] == 0.0
    assert metrics["avg_game_actions"] == 2.0

    agent = metrics["strategy"]["0"]
    bot = metrics["strategy"]["1"]
    assert agent["win_rate"] == pytest.approx(0.5)
    assert agent["points"] == Range([1.0, -1.0])
    assert agent["winning_games"] == 1
    assert agent["avg_winning_move_probability"] == pytest.approx(0.75)
    assert agent["sensitivity"] == pytest.approx(0.0)
    assert bot["avg_winning_move_probability"] == pytest.approx(1.0)
    assert bot["sensitivity"] == pytest.approx(0.0)


def test_connectfour_winning_probability_is_nan_without_wins():
    results = _rotated_results().only_strategy([0])
    for game in results:
        game[0][-1].current_diff_points = -1.0

    probability = Metrics(results).metrics()["strategy"]["0"][
        "avg_winning_move_probability"
    ]

    assert math.isnan(probability)


def test_connectfour_winning_probability_ignores_unrecorded_opening_win():
    results = _rotated_results()
    opening_win = PlayerTrace(seat_id=0, strategy_id=0)
    opening_win.append(
        SimpleNamespace(
            current_diff_points=1.0,
            my_points=1.0,
            terminal=True,
            state="terminal",
        )
    )
    results.append(
        GameTrace(
            [
                opening_win,
                _trace(1, 1, "a", [1.0, 0.0], -1.0),
            ]
        )
    )

    strategy = Metrics(results).metrics()["strategy"]["0"]

    assert strategy["winning_games"] == 2
    assert strategy["avg_winning_move_probability"] == pytest.approx(0.75)

    opening_only = SelfPlayResults([results[-1]])
    opening_probability = Metrics(opening_only).metrics()["strategy"]["0"][
        "avg_winning_move_probability"
    ]
    assert math.isnan(opening_probability)


def test_connectfour_metrics_flatten_to_separate_trackio_paths():
    class RunRecorder:
        def log(self, values, *, step):
            self.values = values
            self.step = step

    run = RunRecorder()
    Trackio(run).log(7, {"game": Metrics(_rotated_results()).metrics()})

    assert run.step == 7
    assert run.values["game/strategy/0/avg_winning_move_probability"] == pytest.approx(
        0.75
    )
    assert run.values["game/strategy/1/avg_winning_move_probability"] == pytest.approx(
        1.0
    )
    assert not any(path.startswith("game/seat/") for path in run.values)
    assert "game/avg_winning_move_probability" not in run.values
