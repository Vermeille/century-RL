import math

import pytest
import torch

from boardrl import RolloutRunner
from boardrl.games import games_library
from boardrl.games.game2048.game import Game2048
from boardrl.games.game2048.metrics import Metrics
from boardrl.games.game2048.versus import PLACING, Versus2048
from boardrl.metrics import Trackio
from boardrl.rollouts import EndState, GameTrace, PlayerTrace, Record, Rollouts


def trace_for(rows, moves=()):
    game = Game2048(seed=7)
    game.board = [cell for row in rows for cell in row]
    game._refresh_moves()
    trace = PlayerTrace(0, 0)
    for move in moves:
        trace.append(Record(game, torch.zeros(len(game.moves)), game.moves.index(move), {}))
        game.play_str(move)
    trace.append(EndState(game, 0, outcome=games_library("2048").outcome))
    return GameTrace([trace])


def test_metrics_track_wins_losses_truncation_progress_and_merge_behavior():
    win = trace_for([[1024, 1024, 0, 0], [0] * 4, [0] * 4, [0] * 4], ["left"])
    loss = trace_for([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    partial = trace_for([[2, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4], ["right"])
    data = Rollouts([win, loss, partial])
    metrics = Metrics(data).metrics()
    assert metrics["score"].values == [2048, 0, 0]
    assert metrics["moves"].values == [1, 0, 1]
    assert metrics["max_tile"].values == [2048, 4, 2]
    assert metrics["empty_cells"].values == [14, 0, 14]
    assert metrics["win_rate"] == metrics["stuck_rate"] == metrics["truncated_rate"] == 1 / 3
    assert metrics["terminal_rate"] == 2 / 3
    assert metrics["tile_reached"] == {
        str(tile): 1 / 3 if tile <= 2048 else 0
        for tile in (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
    }
    assert metrics["direction_share"] == {"up": 0, "right": 0.5, "down": 0, "left": 0.5}
    assert metrics["merge_move_rate"] == 0.5
    assert metrics["score_per_move"] == 1024
    # Action indices remain relative to each record's own action ordering.
    record = win[0][0]
    record.action_idx = len(record.moves) - 1 - record.action_idx
    record.moves.reverse()
    assert Metrics(data).metrics() == metrics


def test_metrics_are_finite_when_empty_or_without_actions():
    for data in (Rollouts([]), Rollouts([trace_for([[2, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4])])):
        class RunRecorder:
            def log(self, values, *, step):
                self.values = values

        recorder = RunRecorder()
        metrics = Metrics(data).metrics()
        assert metrics["merge_move_rate"] == metrics["score_per_move"] == 0
        assert sum(metrics["direction_share"].values()) == 0
        Trackio(recorder).log(1, {"2048": metrics})
        assert all(math.isfinite(value) for value in recorder.values.values())


@pytest.mark.parametrize("budget", [1, 10000])
def test_metrics_on_real_single_player_rollouts(budget):
    desc = games_library("2048,seed=42")
    data = RolloutRunner(desc.make_game, outcome=desc.outcome, progress=False).play(
        [desc.strategy_from_string("random")], games=3, max_steps=budget,
    )
    metrics = desc.make_metrics(data).metrics()
    assert sum(metrics["direction_share"].values()) == pytest.approx(1)
    assert metrics["truncated_rate"] == float(budget == 1)
    assert metrics["terminal_rate"] + metrics["truncated_rate"] == 1
    assert metrics["win_rate"] + metrics["stuck_rate"] == metrics["terminal_rate"]
    assert all(value >= 2 for value in metrics["max_tile"].values)


def versus_rollout(game, moves, strategies=(0, 1)):
    traces = [PlayerTrace(seat, strategies[seat]) for seat in (0, 1)]
    for move in moves:
        traces[game.current_player()].append(
            Record(game, torch.zeros(len(game.moves)), game.moves.index(move), {}),
        )
        game.play_str(move)
    for seat, trace in enumerate(traces):
        trace.append(EndState(game, seat, outcome=games_library("2048versus").outcome))
    return GameTrace(traces)


def test_versus_metrics_count_each_game_once_and_separate_placements_from_slides():
    winner = Versus2048(seed=0)
    winner.board = [1024, 1024] + [0] * 14
    winner._refresh_moves()
    loser = Versus2048(seed=0)
    loser.board = [2, 4, 2, 0, 4, 2, 4, 2, 2, 4, 2, 4, 4, 2, 4, 2]
    loser.score = 16
    loser.phase, loser.pending_tile = PLACING, 4
    loser._refresh_moves()
    data = Rollouts([
        versus_rollout(winner, ["left"]),
        versus_rollout(loser, ["spawn d1"], strategies=(1, 0)),
    ])
    metrics = Metrics(data).metrics()
    assert metrics["score"].values == [2048, 16]
    assert metrics["max_tile"].values == [2048, 4]
    assert metrics["empty_cells"].values == [15, 0]
    assert metrics["win_rate"] == metrics["stuck_rate"] == 0.5
    assert metrics["terminal_rate"] == 1 and metrics["truncated_rate"] == 0
    assert metrics["tile_reached"]["2048"] == 0.5
    assert metrics["direction_share"] == {"up": 0, "right": 0, "down": 0, "left": 1}
    assert metrics["merge_move_rate"] == 1 and metrics["score_per_move"] == 2048
    assert metrics["spawn_actions"].values == [0, 1]
    assert metrics["spawn_position_share"]["d1"] == 1
    assert sum(metrics["spawn_position_share"].values()) == 1


@pytest.mark.parametrize("budget", [1, 2, 10000])
def test_versus_metrics_on_rotated_rollouts_and_both_truncation_phases(budget):
    desc = games_library("2048versus,seed=7")
    bot = desc.strategy_from_string("random")
    data = RolloutRunner(desc.make_game, outcome=desc.outcome, progress=False).play(
        [bot, bot], games=4, max_steps=budget, rotate=True,
    )
    metrics = desc.make_metrics(data).metrics()
    assert len(metrics["score"].values) == 4
    assert all(score >= 0 for score in metrics["score"].values)
    assert sum(metrics["direction_share"].values()) == pytest.approx(1)
    assert sum(metrics["spawn_position_share"].values()) == pytest.approx(float(budget != 1))
    assert metrics["truncated_rate"] == float(budget != 10000)
    assert metrics["win_rate"] + metrics["stuck_rate"] == metrics["terminal_rate"]


@pytest.mark.parametrize("name,game_class", [("2048", Game2048), ("2048versus", Versus2048)])
def test_nolimits_metrics_distinguish_reached_milestones_truncation_and_blockage(name, game_class):
    desc = games_library(f"{name},nolimits=True")
    game = game_class(seed=0, nolimits=True)
    data = []
    for board in ([4096] + [0] * 15, [4096, 2, 4, 8, 2, 4, 8, 16, 4, 8, 16, 32, 8, 16, 32, 64]):
        game.board = board
        game._refresh_moves()
        traces = [PlayerTrace(player, player) for player in range(game.num_players)]
        for player, trace in enumerate(traces):
            trace.append(EndState(game, player, outcome=desc.outcome))
        data.append(GameTrace(traces))
    metrics = desc.make_metrics(Rollouts(data)).metrics()
    assert metrics["tile_reached"]["2048"] == metrics["tile_reached"]["4096"] == 1
    assert metrics["tile_reached"]["8192"] == 0
    assert metrics["max_tile"].values == [4096, 4096]
    assert metrics["win_rate"] == metrics["terminal_rate"] == metrics["truncated_rate"] == metrics["stuck_rate"] == 0.5
