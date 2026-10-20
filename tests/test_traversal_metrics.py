import math

import pytest
import torch

from boardrl.games import games_library
from boardrl.games.semantics import PointScores
from boardrl.games.traversal.game import EXPLORER, SENTINEL, VANGUARD, Piece, Traversal
from boardrl.games.traversal.metrics import Metrics, Observation, TraversalTraceMetrics
from boardrl.games.traversal.semantics import TraversalOutcome
from boardrl.metrics import Trackio
from boardrl.rollouts import EndState, GameTrace, PlayerTrace, Record, Rollouts, TraceGroup


def position(pieces, heights=None, player=0):
    game = Traversal(terrain="flat")
    game.board = pieces.copy()
    game.turn = player
    if heights is not None:
        game.heights = tuple(tuple(row) for row in heights)
    game._refresh_moves()
    return game


def chosen(game, move):
    return Record(game, torch.zeros(len(game.moves)), game.moves.index(move), {})


def rollout(game, move=None, strategies=(0, 1)):
    traces = [PlayerTrace(seat, strategies[seat]) for seat in range(2)]
    if move is not None:
        traces[game.current_player()].append(chosen(game, move))
        game.play_str(move)
    for seat, trace in enumerate(traces):
        trace.append(EndState(game, seat, outcome=TraversalOutcome()))
    return GameTrace(traces)


def behavior(records):
    trace = PlayerTrace(0, 0)
    trace.extend(records)
    trace.append(EndState(Traversal(terrain="flat"), 0))
    return TraversalTraceMetrics(TraceGroup(0, [trace]), scores=PointScores()).behavior_metrics()


def test_metrics_describe_piece_choice_progress_support_and_captures():
    heights = [[0] * 10 for _ in range(10)]
    heights[5][5] = 1
    downhill = position({
        (5, 5): Piece(0, VANGUARD), (3, 5): Piece(1, SENTINEL),
        (3, 4): Piece(0, EXPLORER), (2, 5): Piece(1, SENTINEL),
    }, heights)
    heights = [[0] * 10 for _ in range(10)]
    heights[4][5] = 1
    uphill = position({(5, 5): Piece(0, SENTINEL), (4, 5): Piece(1, EXPLORER)}, heights)
    mobile = position({(5, 5): Piece(0, EXPLORER), (0, 0): Piece(1, SENTINEL)})
    finish = position({
        (0, 4): Piece(0, EXPLORER), (5, 0): Piece(0, SENTINEL), (9, 9): Piece(1, SENTINEL),
    })
    metrics = behavior([
        chosen(downhill, "f6>f4"), chosen(uphill, "f6>f5"),
        chosen(mobile, "f6>f7"), chosen(mobile, "f6>g6"), chosen(finish, "e1>off"),
    ])
    assert metrics["piece_share"] == pytest.approx({"explorer": 3 / 5, "vanguard": 1 / 5, "sentinel": 1 / 5})
    assert metrics["direction_share"] == pytest.approx({"forward": 3 / 5, "backward": 1 / 5, "lateral": 1 / 5})
    assert metrics["avg_forward_rows"] == pytest.approx(3 / 5)
    assert metrics["capture_share"] == pytest.approx(2 / 5)
    assert metrics["finish_share"] == pytest.approx(1 / 5)
    assert metrics["combat"] == pytest.approx({
        "downhill_capture_share": 0.5, "uphill_capture_share": 0.5,
        "supported_capture_share": 0.5, "reinforced_target_share": 0.5,
    })
    assert metrics["opportunity"]["capture_conversion_rate"] == 1
    # The supporting Explorer in the downhill position can also finish.
    assert metrics["opportunity"]["finish_conversion_rate"] == 0.5
    assert metrics["finish_type_share"] == {"explorer": 1, "vanguard": 0, "sentinel": 0}


def test_opportunities_distinguish_available_and_chosen_finishes():
    game = position({
        (0, 4): Piece(0, SENTINEL), (5, 5): Piece(0, EXPLORER), (9, 9): Piece(1, SENTINEL),
    })
    metrics = behavior([chosen(game, "f6>f5"), chosen(game, "e1>off")])
    assert metrics["opportunity"]["finish_available_rate"] == 1
    assert metrics["opportunity"]["finish_conversion_rate"] == 0.5
    assert metrics["opportunity"]["sentinel_win_available_rate"] == 1
    assert metrics["opportunity"]["sentinel_win_conversion_rate"] == 0.5

    game.finished = [Piece(1, EXPLORER)]
    game.scores = [0, 1]
    after_first = behavior([chosen(game, "e1>off")])
    assert after_first["opportunity"]["finish_conversion_rate"] == 1
    assert after_first["opportunity"]["sentinel_win_available_rate"] == 0
    assert after_first["opportunity"]["sentinel_win_conversion_rate"] == 0


def test_strategy_grouping_and_forward_direction_survive_seat_rotation():
    first = position({(0, 4): Piece(0, SENTINEL), (5, 5): Piece(1, EXPLORER)})
    second = position({(9, 4): Piece(1, SENTINEL), (5, 5): Piece(0, EXPLORER)}, player=1)
    data = Rollouts([rollout(first, "e1>off"), rollout(second, "e10>off", strategies=(1, 0))])
    metrics = Metrics(data).metrics()
    assert metrics["avg_game_actions"] == 1
    assert metrics["terminal_rate"] == 1 and metrics["draw_rate"] == 0
    assert metrics["end_reason"]["sentinel"] == 1
    agent, opponent = metrics["strategy"]["0"], metrics["strategy"]["1"]
    assert agent["win_rate"] == 1 and opponent["win_rate"] == 0
    assert agent["avg_points"] == 3 and opponent["avg_points"] == 0
    assert agent["win_cause"]["sentinel"] == 1
    assert agent["behavior"]["direction_share"]["forward"] == 1
    assert agent["behavior"]["piece_share"]["sentinel"] == 1
    assert opponent["behavior"]["finish_share"] == 0
    assert opponent["behavior"]["opportunity"]["sentinel_win_conversion_rate"] == 0


def test_winners_and_draws_follow_utility_instead_of_raw_scores():
    tiebreak = Traversal(terrain="flat")
    tiebreak._finish("move_limit_advancement", 0)
    tiebreak._refresh_moves()
    repetition = Traversal(terrain="flat")
    repetition.finished = [Piece(0, EXPLORER)]
    repetition.scores = [1, 0]
    repetition._finish("repetition", None)
    repetition._refresh_moves()
    metrics = Metrics(Rollouts([rollout(tiebreak), rollout(repetition)])).metrics()
    assert metrics["draw_rate"] == 0.5
    assert metrics["end_reason"]["move_limit_advancement"] == 0.5
    assert metrics["end_reason"]["repetition"] == 0.5
    assert metrics["strategy"]["0"]["win_cause"]["move_limit_advancement"] == 1
    assert metrics["strategy"]["0"]["win_cause"]["repetition"] == 0
    assert metrics["strategy"]["0"]["avg_points"] == 0.5


def test_terrain_metrics_and_truncation_use_one_map_per_game():
    gradient = Traversal(terrain="flat")
    gradient.terrain_name = "mountains"
    gradient.heights = tuple((row,) * 10 for row in range(10))
    flat = Traversal(terrain="flat")
    metrics = Metrics(Rollouts([rollout(gradient), rollout(flat)])).metrics()
    assert metrics["terminal_rate"] == metrics["draw_rate"] == 0
    assert metrics["end_reason"]["truncated"] == 1
    terrain = metrics["terrain"]
    assert terrain["preset_share"]["mountains"] == terrain["preset_share"]["flat"] == 0.5
    assert sum(terrain["preset_share"].values()) == 1
    assert terrain["unique_map_share"] == 1
    assert terrain["avg_height_range"] == 4.5
    assert terrain["avg_adjacent_height_difference"] == 0.25
    duplicate = Metrics(Rollouts([rollout(flat), rollout(flat.copy())])).metrics()
    assert duplicate["terrain"]["unique_map_share"] == 0.5


def test_resolved_random_preset_is_recorded_and_survives_copy():
    for seed in range(5):
        game = Traversal(seed=seed)
        assert game.terrain == "random" and game.terrain_name != "random"
        observation = Observation(game.display_with_moves())
        assert observation.preset == game.terrain_name
        assert observation.heights == game.heights and observation.board == game.board
        assert game.copy().terrain_name == game.terrain_name
        assert game.copy().display() == game.display()


def test_legacy_unknown_preset_and_malformed_observations():
    game = Traversal(terrain="flat")
    legacy = game.display().replace("Terrain: flat\n", "")
    assert Observation(legacy).preset == "unknown"
    for state in ("", "garbage", game.display().replace(" 1 0 0", " 2 0 0", 1)):
        assert Observation.parse(state) is None
    record = chosen(game, game.moves[0])
    record.state = "garbage"
    assert behavior([record])["observation_coverage"] == 0
    assert behavior([record])["capture_share"] == 0
    assert behavior([record])["avg_forward_rows"] == 0


def test_action_order_does_not_change_behavior_metrics():
    game = Traversal(terrain="flat")
    record = chosen(game, "e10>e5")
    original = behavior([record])
    record.action_idx = len(record.moves) - 1 - record.action_idx
    record.moves.reverse()
    assert behavior([record]) == original


def test_missing_endpoint_observations_report_coverage_and_unknown_cause():
    game = position({(0, 4): Piece(0, SENTINEL), (5, 5): Piece(1, EXPLORER)})
    data = Rollouts([rollout(game, "e1>off")])
    data[0].by_seat[0][-1].state = "legacy terminal"
    metrics = Metrics(data).metrics()
    assert metrics["terrain"]["observation_coverage"] == 0
    assert metrics["end_reason"]["unknown"] == 1
    assert metrics["strategy"]["0"]["win_cause"]["unknown"] == 1


def test_metrics_are_zero_safe_for_empty_data_and_terminal_only_traces():
    empty = Metrics(Rollouts([])).metrics()
    assert empty["terminal_rate"] == empty["avg_game_actions"] == 0
    assert empty["strategy"] == {}
    assert empty["terrain"]["unique_map_share"] == 0
    game = Traversal(terrain="flat")
    game._finish("stalemate", None)
    game._refresh_moves()
    metrics = Metrics(Rollouts([rollout(game)])).metrics()
    assert metrics["draw_rate"] == metrics["end_reason"]["stalemate"] == 1
    assert metrics["strategy"]["0"]["sensitivity"] == 0
    assert metrics["strategy"]["0"]["behavior"]["finish_share"] == 0


def test_metrics_flatten_to_numeric_strategy_specific_trackio_paths():
    class RunRecorder:
        def log(self, values, *, step):
            self.values = values
            self.step = step

    game = position({(0, 4): Piece(0, SENTINEL), (5, 5): Piece(1, EXPLORER)})
    metrics = Metrics(Rollouts([rollout(game, "e1>off")])).metrics()
    run = RunRecorder()
    Trackio(run).log(7, {"game": metrics})
    assert run.step == 7
    assert run.values["game/strategy/0/behavior/finish_share"] == 1
    assert run.values["game/end_reason/sentinel"] == 1
    assert run.values["game/terrain/preset_share/flat"] == 1
    assert all(math.isfinite(value) for value in run.values.values())
    assert not any("/seat/" in path for path in run.values)


@pytest.mark.parametrize("budget", [1, 200])
def test_metrics_on_real_rollouts_include_behavior_and_resolved_terrain(budget):
    from boardrl import RolloutRunner

    desc = games_library("traversal")
    bot = desc.strategy_from_string("random")
    data = RolloutRunner(desc.make_game, outcome=desc.outcome, progress=False).play(
        [bot, bot], games=3, max_steps=budget,
    )
    metrics = desc.make_metrics(data).metrics()
    assert sum(metrics["terrain"]["preset_share"].values()) == pytest.approx(1)
    assert metrics["terrain"]["preset_share"]["unknown"] == 0
    assert sum(metrics["end_reason"].values()) == pytest.approx(1)
    assert metrics["end_reason"]["truncated"] == float(budget == 1)
    for strategy in metrics["strategy"].values():
        assert sum(strategy["behavior"]["piece_share"].values()) == pytest.approx(1)
        assert strategy["behavior"]["observation_coverage"] == 1
