import random
from unittest.mock import Mock

import pytest
import torch

from boardrl import RolloutRunner
from boardrl.games import games_library
from boardrl.games.augmentations import shuffle_actions
from boardrl.games.game2048.game import Game2048, SLIDES
from boardrl.games.game2048.versus import Versus2048
from boardrl.models import make_for_game
from boardrl.training import Pipeline, ToSamples


def position(rows, *, score=0, turn=0):
    game = Game2048(seed=0)
    game.board = [cell for row in rows for cell in row]
    game.score = score
    game.turn = turn
    game._refresh_moves()
    return game


@pytest.mark.parametrize("seed", range(5))
def test_initial_state_and_serialization(seed):
    game = Game2048(seed=seed)
    assert len(game.board) == 16
    assert sum(value != 0 for value in game.board) == 2
    assert set(game.board) <= {0, 2, 4}
    assert game.current_player() == 0 and game.num_players == 1
    assert game.round() == game.points() == game.diff_points_for(0) == 0
    assert not game.ended() and not game.won() and game.winner() is None
    assert game.display() == game.display(force=0)
    assert [line[1:] for line in game.display_with_moves().splitlines() if line.startswith("@")] == game.moves
    assert game.moves == [name for name, slide in SLIDES.items() if slide.apply(game.board)[0] != game.board]
    assert game.display() == Game2048(seed=seed).display()


@pytest.mark.parametrize("tiles,expected,score", [
    ([0, 2, 0, 2], [4, 0, 0, 0], 4),
    ([2, 2, 2, 0], [4, 2, 0, 0], 4),
    ([2, 2, 2, 2], [4, 4, 0, 0], 8),
    ([2, 2, 4, 0], [4, 4, 0, 0], 4),
    ([4, 4, 8, 8], [8, 16, 0, 0], 24),
    ([2, 4, 2, 4], [2, 4, 2, 4], 0),
])
def test_merge_order_no_chain_merges_and_score(tiles, expected, score):
    board = tiles + [0] * 12
    result, gained = SLIDES["left"].apply(board)
    assert result == expected + [0] * 12
    assert gained == score
    assert board == tiles + [0] * 12
    assert sum(result) == sum(board)


@pytest.mark.parametrize("move,indices", [
    ("left", [0, 1, 2, 3]), ("right", [3, 2, 1, 0]),
    ("up", [0, 4, 8, 12]), ("down", [12, 8, 4, 0]),
])
def test_every_direction_merges_from_destination_edge(move, indices):
    board = [0] * 16
    for index, value in zip(indices, [2, 2, 4, 4]):
        board[index] = value
    result, gained = SLIDES[move].apply(board)
    expected = [0] * 16
    expected[indices[0]], expected[indices[1]] = 4, 8
    assert result == expected and gained == 12


@pytest.mark.parametrize("roll,value", [(0.0, 2), (0.89999, 2), (0.9, 4), (0.99999, 4)])
def test_one_spawn_after_legal_move_and_exact_probability_boundary(roll, value):
    game = position([[2, 2, 0, 0], [0] * 4, [0] * 4, [0] * 4], score=8, turn=3)
    game.rng = Mock()
    game.rng.choice.return_value = 15
    game.rng.random.return_value = roll
    game.play_str("left")
    assert game.board == [4] + [0] * 14 + [value]
    assert game.score == game.points_for(0) == game.diff_points() == 12
    assert game.round() == 4
    game.rng.choice.assert_called_once_with(list(range(1, 16)))
    game.rng.random.assert_called_once_with()


def test_illegal_and_no_op_moves_do_not_mutate_or_consume_randomness():
    game = position([[2, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4])
    assert game.moves == ["right", "down"]
    before = game.display_with_moves(), game.rng.getstate()
    for move in ("left", "up", "diagonal", "0"):
        with pytest.raises(ValueError, match="Illegal"):
            game.play_str(move)
        assert (game.display_with_moves(), game.rng.getstate()) == before
    with pytest.raises(IndexError):
        game.play_idx(len(game.moves))


def test_win_ends_game_with_raw_merge_points_and_a_final_spawn():
    game = position([[1024, 1024, 0, 0], [0] * 4, [0] * 4, [0] * 4], score=100)
    game.play_str("left")
    assert game.ended() and game.won() and game.winner() == 0
    assert game.moves == [] and game.points() == 2148
    assert game.board[0] == 2048 and sum(value > 0 for value in game.board) == 2
    assert "Status: won" in game.display()
    assert "@" not in game.display_with_moves()
    with pytest.raises(ValueError):
        game.play_str("right")


def test_full_board_loss_and_full_board_with_a_merge():
    game = position([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    assert game.ended() and not game.won() and game.moves == []
    assert game.winner() is None and "Status: stuck" in game.display()
    with pytest.raises(ValueError):
        game.play_str("left")
    game.board[1] = 2
    game._refresh_moves()
    assert game.moves == list(SLIDES) and not game.ended()


def test_last_slide_can_lose_after_spawning():
    game = position([[0, 2, 4, 2], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    game.rng = Mock()
    game.rng.choice.return_value = 3
    game.rng.random.return_value = 0.9
    game.play_str("left")
    assert game.ended() and not game.won() and game.moves == []


def test_copy_preserves_random_stream_and_owns_all_mutable_state():
    game = Game2048(seed=17)
    cloned = game.copy()
    assert cloned.board is not game.board and cloned.moves is not game.moves
    assert cloned.rng is not game.rng
    original = game.display_with_moves(), game.rng.getstate()
    move = game.moves[0]
    cloned.play_str(move)
    assert (game.display_with_moves(), game.rng.getstate()) == original
    game.play_str(move)
    assert game.display_with_moves() == cloned.display_with_moves()
    cloned.moves.clear()
    assert game.moves


def test_seed_zero_does_not_reseed_global_random():
    state = random.getstate()
    Game2048(seed=0)
    assert random.getstate() == state


def test_randomized_search_copy_preserves_observation_with_new_spawn_stream():
    game = Game2048(seed=0)
    cloned = game.copy(randomize=True)
    assert cloned.display_with_moves() == game.display_with_moves()
    assert cloned.rng.getstate() != game.rng.getstate()


@pytest.mark.parametrize("kwargs", [{"num_players": 2}, {"num_players": 0}])
def test_rejects_invalid_player_counts(kwargs):
    with pytest.raises(ValueError, match="requires exactly"):
        Game2048(**kwargs)


def test_rejects_invalid_views_and_scores():
    game = Game2048()
    with pytest.raises(ValueError):
        game.display(force=1)
    with pytest.raises(ValueError):
        game.points_for(1)


@pytest.mark.parametrize("seed", range(3))
def test_random_simulation_terminates(seed):
    game = Game2048(seed=seed)
    game.simulate_to_end()
    assert game.ended() and game.moves == []
    assert 0 < game.round() < 8192
    assert game.score > 0


@pytest.mark.parametrize("budget", [1, 10000])
def test_registry_single_player_rollouts_and_point_reward_pipeline(budget):
    desc = games_library("2048,seed=7")
    assert desc.coop and desc.rewards.points_value_head
    assert desc.rewards.scale == 1 / 2048
    assert shuffle_actions in desc.augmentations
    game = desc.make_game(num_players=1)
    assert game.num_players == 1
    data = RolloutRunner(desc.make_game, outcome=desc.outcome, progress=False).play(
        [desc.strategy_from_string("random")], games=3, max_steps=budget,
    )
    for trace in data.all_traces():
        assert trace.seat_id == trace.strategy_id == 0
        end = trace[-1]
        assert end.terminal == (budget == 10000)
        assert end.truncated == (budget == 1)
        assert end.won == (end.terminal and "Status: won" in end.state)
        assert all(record.moves[record.action_idx] in SLIDES for record in trace[:-1])
    samples = Pipeline(desc.rewards.make_returns(1.0), ToSamples())(data)
    assert samples
    if budget == 1:
        # Truncated returns require a value bootstrap in ReferenceTargets.
        assert all(torch.isnan(torch.tensor(sample.returns)) for sample in samples)
    else:
        assert all(torch.isfinite(torch.tensor(sample.returns)) for sample in samples)
    for trace in data.all_traces():
        assert sum(record.reward for record in trace[:-1]) == trace[-1].my_points / 2048


def test_model_factory_uses_point_value_head_and_ordered_legal_actions():
    desc = games_library("2048,seed=0")
    game = desc.make_game()
    model = make_for_game("toy", desc).eval()
    prediction = model([game.display_with_moves()])
    assert prediction.policy[0].shape == (len(game.moves),)
    assert torch.isfinite(prediction.policy[0]).all()
    assert torch.isfinite(prediction.value.mean).all()
    assert model.spec()["points_based"] is True


@pytest.mark.parametrize("game_class", [Game2048, Versus2048])
def test_nolimits_continues_through_2048_and_4096_with_normal_spawns(game_class):
    game = game_class(seed=0, nolimits=True)
    for tile in (1024, 2048):
        game.board = [tile, tile] + [0] * 14
        game._refresh_moves()
        game.play_str("left")
        assert game.board[0] == tile * 2
        assert game.won() and not game.ended() and game.winner() is None
        assert "Nolimits: True" in game.display() and "Status: playing" in game.display()
        if game.num_players == 2:
            assert game.current_player() == 1
            assert game.pending_tile in (2, 4)
            assert game.moves == [f"spawn {chr(97 + col)}{row + 1}" for row in range(4) for col in range(4) if row or col]
            game.play_str("spawn d4")
            assert game.points_for(1) == -game.points_for(0)
        assert sum(value != 0 for value in game.board) == 2
        assert game.current_player() == 0 and not game.ended()
    assert game.score == 6144
    cloned = game.copy()
    assert cloned.nolimits and cloned.display_with_moves() == game.display_with_moves()
    assert game.copy(randomize=True).nolimits


@pytest.mark.parametrize("game_class", [Game2048, Versus2048])
def test_nolimits_still_ends_when_stuck_above_2048(game_class):
    game = game_class(seed=0, nolimits=True)
    game.board = [4096, 2, 4, 8, 2, 4, 8, 16, 4, 8, 16, 32, 8, 16, 32, 64]
    game._refresh_moves()
    assert game.ended() and game.moves == [] and game.won()
    assert game.winner() == 0 and "Status: stuck" in game.display()
    with pytest.raises(ValueError):
        game.play_str("left")


@pytest.mark.parametrize("name", ["2048", "2048versus"])
@pytest.mark.parametrize("flag", [True, False])
def test_nolimits_boolean_registry_option(name, flag):
    desc = games_library(f"{name},nolimits={flag},seed=0")
    game = desc.make_game()
    assert game.nolimits is flag
    game.board = [2048] + [0] * 15
    game._refresh_moves()
    assert game.ended() is not flag
    assert desc.rewards.scale == 1 / 2048
    assert games_library(name).make_game().nolimits is False
    with pytest.raises(ValueError, match="Invalid value for nolimits"):
        games_library(f"{name},nolimits=yes")
