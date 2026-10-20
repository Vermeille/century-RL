from unittest.mock import Mock

import pytest
import torch

from boardrl import RolloutRunner
from boardrl.games import games_library
from boardrl.games.game2048.game import Game2048, SLIDES
from boardrl.games.game2048.versus import PLACING, SLIDING, SPAWN_CELLS, Versus2048
from boardrl.rollouts import EndState
from boardrl.training import Pipeline, ToSamples


def position(rows, *, pending=0, score=0):
    game = Versus2048(seed=0)
    game.board = [value for row in rows for value in row]
    game.score = score
    game.pending_tile = pending
    game.phase = PLACING if pending else SLIDING
    game._refresh_moves()
    return game


def test_initial_state_matches_solo_and_exposes_both_roles():
    game = Versus2048(seed=0)
    assert game.board == Game2048(seed=0).board
    assert game.num_players == 2 and game.current_player() == 0
    assert game.phase is SLIDING and game.pending_tile == 0
    assert game.moves == Game2048(seed=0).moves
    assert game.round() == 0 and game.winner() is None
    assert "Mode: versus" in game.display()
    assert "Viewer: 1" in game.display(force=1)
    assert "Phase: slide" in game.display()


@pytest.mark.parametrize("roll,value", [(0, 2), (0.8999, 2), (0.9, 4), (0.9999, 4)])
def test_slider_draws_value_without_choosing_location_and_spawner_places_it(roll, value):
    game = position([[2, 2, 0, 0], [0] * 4, [0] * 4, [0] * 4])
    game.rng = Mock()
    game.rng.random.return_value = roll
    game.play_str("left")
    assert game.board == [4] + [0] * 15
    assert game.score == 4 and game.points() == game.diff_points() == -4
    assert game.points_for(0) == -game.points_for(1) == 4
    assert game.current_player() == 1 and game.phase is PLACING
    assert game.pending_tile == value and game.round() == 1 and not game.ended()
    assert game.moves == list(SPAWN_CELLS)[1:]
    assert f"Next tile: {value}" in game.display_with_moves()
    game.rng.random.assert_called_once_with()
    game.rng.choice.assert_not_called()
    game.play_str("spawn d4")
    assert game.board == [4] + [0] * 14 + [value]
    assert game.current_player() == 0 and game.phase is SLIDING
    assert game.pending_tile == 0 and game.round() == 1 and game.points() == 4
    assert set(game.moves) <= SLIDES.keys()
    game.rng.random.assert_called_once_with()


def test_only_empty_cells_and_current_phase_actions_are_legal():
    game = position([[2, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4])
    with pytest.raises(ValueError):
        game.play_str("spawn b1")
    game.play_str("right")
    before = game.display_with_moves(), game.rng.getstate(), game.board.copy()
    for move in ("right", "spawn d1", "spawn a0", "spawn e1", "spawn a5", "spawn 0"):
        with pytest.raises(ValueError):
            game.play_str(move)
        assert (game.display_with_moves(), game.rng.getstate(), game.board) == before
    assert [line[1:] for line in game.display_with_moves().splitlines() if line.startswith("@")] == game.moves
    index = game.moves.index("spawn c2")
    game.play_idx(index)
    assert game.board[6] > 0 and game.current_player() == 0


@pytest.mark.parametrize("placing", [False, True])
def test_copy_independence_preserves_phase_and_pending_tile(placing):
    game = Versus2048(seed=4)
    if placing:
        game.play_idx(0)
    cloned = game.copy()
    assert cloned.phase is game.phase and cloned.pending_tile == game.pending_tile
    assert cloned.num_players == 2
    assert cloned.display_with_moves() == game.display_with_moves()
    before = game.display_with_moves(), game.rng.getstate()
    cloned.play_idx(0)
    assert (game.display_with_moves(), game.rng.getstate()) == before
    game.play_idx(0)
    assert game.display_with_moves() == cloned.display_with_moves()
    randomized = game.copy(randomize=True)
    assert randomized.display_with_moves() == game.display_with_moves()
    assert randomized.rng.getstate() != game.rng.getstate()


def test_reaching_2048_wins_immediately_without_an_extra_placement():
    game = position([[1024, 1024, 0, 0], [0] * 4, [0] * 4, [0] * 4], score=100)
    rng_before = game.rng.getstate()
    game.play_str("left")
    assert game.won() and game.ended() and game.winner() == 0
    assert game.moves == [] and game.pending_tile == 0
    assert game.rng.getstate() == rng_before
    assert game.points_for(0) == 2148 and game.points_for(1) == -2148
    outcome = games_library("2048versus").outcome
    assert [EndState(game, player, outcome=outcome).episodic_utility for player in (0, 1)] == [1, -1]
    with pytest.raises(ValueError):
        game.play_str("spawn d4")


def test_placement_can_block_slider_and_win_despite_positive_slider_score():
    game = position([
        [0, 2, 4, 2], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2],
    ], score=120)
    game.rng = Mock()
    game.rng.random.return_value = 0.9
    game.play_str("left")
    assert game.moves == ["spawn d1"] and not game.ended()
    game.play_str("spawn d1")
    assert game.ended() and not game.won() and game.winner() == 1
    assert game.moves == [] and "Winner: 1" in game.display()
    outcome = games_library("2048versus").outcome
    assert game.points_for(0) > game.points_for(1)
    assert [EndState(game, player, outcome=outcome).episodic_utility for player in (0, 1)] == [-1, 1]


def test_same_board_can_allow_or_block_a_move_depending_on_placement():
    game = position([[2, 4, 2, 0], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]], pending=2)
    game.play_str("spawn d1")
    assert not game.ended() and game.winner() is None
    assert "left" in game.moves


@pytest.mark.parametrize("budget", [1, 2, 10000])
def test_two_player_rotated_rollouts_zero_sum_rewards_and_phase_truncation(budget):
    desc = games_library("2048versus,seed=7")
    assert not desc.coop and desc.rewards.points_value_head
    assert desc.rewards.scale == 1 / 2048
    bot = desc.strategy_from_string("random")
    data = RolloutRunner(desc.make_game, outcome=desc.outcome, progress=False).play(
        [bot, bot], games=4, max_steps=budget, rotate=True,
    )
    samples = Pipeline(desc.rewards.make_returns(1), ToSamples())(data)
    assert samples and all(torch.isfinite(torch.tensor(sample.reward)) for sample in samples)
    for index, game in enumerate(data):
        assert game.by_seat[0].strategy_id == index % 2
        ends = [trace[-1] for trace in game]
        assert ends[0].my_points == -ends[1].my_points
        assert ends[0].episodic_utility == -ends[1].episodic_utility
        assert all(end.terminal == (budget == 10000) for end in ends)
        for seat, trace in enumerate(game):
            assert all(record.player == seat for record in trace[:-1])
            assert all((record.moves[record.action_idx] in SLIDES) == (seat == 0) for record in trace[:-1])
        if budget < 10000:
            phase = "spawn" if budget == 1 else "slide"
            assert f"Phase: {phase}" in ends[0].state
            assert all(end.truncated and end.episodic_utility == 0 for end in ends)


def test_random_simulation_terminates():
    game = Versus2048(seed=2)
    game.simulate_to_end()
    assert game.ended() and game.moves == [] and game.winner() in (0, 1)


@pytest.mark.parametrize("players", [0, 1, 3])
def test_invalid_player_count(players):
    with pytest.raises(ValueError, match="requires exactly 2"):
        Versus2048(num_players=players)


def test_invalid_score_and_observation_views():
    game = Versus2048()
    with pytest.raises(ValueError):
        game.points_for(2)
    with pytest.raises(ValueError):
        game.display(force=2)
