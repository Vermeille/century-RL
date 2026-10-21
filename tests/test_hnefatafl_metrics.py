import math

import pytest
import torch

from boardrl import RolloutRunner
from boardrl.games import games_library
from boardrl.games.hnefatafl.game import Hnefatafl
from boardrl.games.hnefatafl.metrics import Metrics, Observation
from boardrl.metrics import Trackio
from boardrl.models import make_for_game
from boardrl.rollouts import EndState, GameTrace, PlayerTrace, Record, Rollouts
from boardrl.training import Pipeline, ReferenceTargets, ToSamples

from tests.test_hnefatafl import position


def rollout(game, moves=(), strategies=(0, 1)):
    traces = [PlayerTrace(seat, strategies[seat]) for seat in range(2)]
    for move in moves:
        traces[game.current_player()].append(
            Record(game, torch.zeros(len(game.moves)), game.moves.index(move), {}),
        )
        game.play_str(move)
    for seat, trace in enumerate(traces):
        trace.append(EndState(game, seat))
    return GameTrace(traces)


def test_observation_roundtrip_and_initial_diagnostics():
    game = Hnefatafl()
    observation = Observation(game.display_with_moves())
    assert observation.board == {square: piece.symbol for square, piece in game.board.items()}
    assert observation.king == game.king_position() and observation.captured == (0, 0)
    metrics = Metrics(Rollouts([rollout(game)])).metrics()
    assert metrics["terminal_rate"] == metrics["draw_rate"] == 0
    assert metrics["end_reason"]["truncated"] == 1
    assert metrics["material"] == {
        "avg_attackers_remaining": 24, "avg_defenders_remaining": 12,
        "avg_attackers_captured": 0, "avg_defenders_captured": 0,
    }
    assert metrics["king"] == {"on_board_rate": 1, "avg_corner_distance": 10, "avg_adjacent_attackers": 0}
    assert metrics["strategy"]["0"]["behavior"]["king_move_share"] == 0


def test_role_outcomes_count_once_and_strategy_identity_survives_rotation():
    attack = rollout(position({"d4": "K", "d3": "A", "c4": "A", "e4": "A", "d6": "A"}), ["d65"])
    defend = rollout(position({"a4": "K", "f3": "A"}, player=1), ["a41"], strategies=(1, 0))
    draw = rollout(Hnefatafl(max_plies=1), [Hnefatafl().moves[0]])
    incomplete = rollout(Hnefatafl())
    metrics = Metrics(Rollouts([attack, defend, draw, incomplete])).metrics()
    assert metrics["terminal_rate"] == 0.75
    assert metrics["attacker_win_rate"] == metrics["defender_win_rate"] == metrics["draw_rate"] == 0.25
    for reason in ("king_captured", "king_escaped", "move_limit", "truncated"):
        assert metrics["end_reason"][reason] == 0.25
    winners = Metrics(Rollouts([attack, defend])).metrics()
    assert winners["strategy"]["0"]["win_rate"] == 1
    assert winners["strategy"]["1"]["win_rate"] == 0
    assert winners["strategy"]["0"]["behavior"]["king_move_share"] == 0.5
    assert winners["king"]["on_board_rate"] == 0.5
    assert winners["king"]["avg_corner_distance"] == 0
    # Selecting a strategy can leave only the defender endpoint of a game.
    selected = Metrics(Rollouts([defend]).only_strategy([0])).metrics()
    assert selected["defender_win_rate"] == 1 and selected["attacker_win_rate"] == 0


def test_capture_metrics_include_last_action_and_exclude_king_from_soldier_counts():
    game = position({
        "e8": "A", "e4": "D", "e3": "A", "d5": "D", "c5": "A",
        "f5": "D", "g5": "A", "j10": "K",
    })
    metrics = Metrics(Rollouts([rollout(game, ["e85", "jA9"])])).metrics()
    behavior = metrics["strategy"]["0"]["behavior"]
    assert behavior["capture_share"] == behavior["multi_capture_share"] == 1
    assert behavior["avg_captures"] == 3 and behavior["avg_move_distance"] == 3
    assert behavior["observation_coverage"] == behavior["capture_observation_coverage"] == 1
    assert metrics["strategy"]["1"]["behavior"]["capture_share"] == 0
    assert metrics["strategy"]["1"]["behavior"]["king_move_share"] == 1
    assert metrics["material"]["avg_defenders_captured"] == 3
    assert metrics["material"]["avg_attackers_captured"] == 0


def test_opponent_capture_is_not_attributed_to_same_seat_trace():
    game = position({"b2": "A", "e4": "A", "f4": "D", "c4": "D", "j10": "K"})
    data = Rollouts([rollout(game, ["b23", "c4d", "b32"])])
    metrics = Metrics(data).metrics()
    assert metrics["strategy"]["0"]["behavior"]["avg_captures"] == 0
    assert metrics["strategy"]["1"]["behavior"]["avg_captures"] == 1
    assert metrics["material"]["avg_attackers_captured"] == 1


def test_metrics_handle_empty_terminal_only_and_missing_observations():
    empty = Metrics(Rollouts([])).metrics()
    assert empty["terminal_rate"] == empty["observation_coverage"] == 0
    assert empty["strategy"] == {} and empty["king"]["avg_corner_distance"] == 0
    terminal = rollout(position({"f6": "K"}))
    metrics = Metrics(Rollouts([terminal])).metrics()
    assert metrics["draw_rate"] == metrics["end_reason"]["stalemate"] == 1
    assert metrics["strategy"]["0"]["behavior"]["avg_move_distance"] == 0
    terminal.by_seat[0][-1].state = "legacy state"
    missing = Metrics(Rollouts([terminal])).metrics()
    assert missing["observation_coverage"] == 0 and missing["end_reason"]["unknown"] == 1
    game = Hnefatafl()
    for state in (
        "", "garbage", game.display().replace("X..AAAAA..X", "X..AAAAA.X"),
        game.display().replace("C:0 0", "C:-1 0"),
    ):
        assert Observation.parse(state) is None
    data = Rollouts([rollout(Hnefatafl(max_plies=1), [game.moves[0]])])
    data[0].by_seat[0][-1].state = "unavailable"
    behavior = Metrics(data).metrics()["strategy"]["0"]["behavior"]
    assert behavior["observation_coverage"] == 1 and behavior["capture_observation_coverage"] == 0
    data[0].by_seat[0][0].state = "unavailable"
    assert Metrics(data).metrics()["strategy"]["0"]["behavior"]["observation_coverage"] == 0


def test_behavior_uses_chosen_move_not_action_order():
    game = position({"a4": "K", "f3": "A"}, player=1)
    data = Rollouts([rollout(game, ["a41"])])
    original = Metrics(data).metrics()
    record = data[0].by_seat[1][0]
    record.action_idx = len(record.moves) - 1 - record.action_idx
    record.moves.reverse()
    assert Metrics(data).metrics() == original


@pytest.mark.parametrize("budget", [1, 12])
def test_rollout_pipeline_rewards_and_metrics_flattening(budget):
    desc = games_library("hnefatafl,max_plies=12")
    bot = desc.strategy_from_string("random")
    data = RolloutRunner(desc.make_game, outcome=desc.outcome, progress=False).play(
        [bot, bot], games=2, max_steps=budget, rotate=True,
    )
    for game in data:
        for trace in game:
            assert trace[-1].terminal == (budget == 12)
            assert trace[-1].truncated == (budget == 1)
    metrics = desc.make_metrics(data).metrics()
    assert metrics["terminal_rate"] == float(budget == 12)
    assert metrics["draw_rate"] == float(budget == 12)
    assert sum(metrics["end_reason"].values()) == pytest.approx(1)
    samples = Pipeline(desc.rewards.make_returns(1.0), ToSamples())(data)
    assert samples
    if budget == 12:
        assert all(sample.returns == 0 for sample in samples)
    else:
        assert all(math.isnan(sample.returns) for sample in samples)
    model = make_for_game("toy", desc).eval()
    ReferenceTargets(
        model, batch_size=16, discount=1.0, gae_lambda=1.0, value_lambda=1.0,
    )(samples)
    assert all(math.isfinite(sample.td_lambda) for sample in samples)

    class RunRecorder:
        def log(self, values, *, step):
            self.values = values
            self.step = step

    run = RunRecorder()
    Trackio(run).log(7, {"game": metrics})
    assert run.step == 7
    assert "game/strategy/0/behavior/avg_move_distance" in run.values
    assert all(math.isfinite(value) for value in run.values.values())


def test_terminal_outcome_returns_reward_escape_instead_of_material():
    game = position({"a4": "K", "f3": "A"}, player=1)
    data = Rollouts([rollout(game, ["a41"])])
    desc = games_library("hnefatafl")
    samples = Pipeline(desc.rewards.make_returns(1.0), ToSamples())(data)
    assert len(samples) == 1 and samples[0].returns == 1
