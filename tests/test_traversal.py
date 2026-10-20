import random

import pytest

from boardrl.games import games_library
from boardrl.games.traversal.game import (
    EXPLORER,
    MOVE_LIMIT,
    SENTINEL,
    SIZE,
    VANGUARD,
    Piece,
    Traversal,
    neighbors,
)
from boardrl.games.traversal.terrain import TERRAIN_TYPES, TRANSFORMS, terrain_library


def position(pieces, heights=None, turn=0, finished=()):
    game = Traversal(terrain="flat")
    game.board = pieces.copy()
    if heights is not None:
        game.heights = tuple(tuple(row) for row in heights)
    game.turn = turn
    game.finished = list(finished)
    game.scores = [sum(p.kind.strength for p in finished if p.player == player) for player in range(2)]
    game._refresh_moves()
    return game


def test_initial_state_and_registry():
    desc = games_library("traversal,terrain=flat,seed=42")
    game = desc.make_game(num_players=2)
    assert game.num_players == 2
    assert len(game.board) == 12
    assert game.current_player() == game.round() == 0
    assert game.board[(9, 2)] == Piece(0, SENTINEL)
    assert game.board[(0, 7)] == Piece(1, SENTINEL)
    for player in range(2):
        for kind in (EXPLORER, VANGUARD, SENTINEL):
            assert sum(p.player == player and p.kind == kind for p in game.board.values()) == 2
    assert len(set(game.moves)) == len(game.moves) > 0
    assert "c10>c7" in game.moves
    assert "e10>e5" in game.moves
    assert game.points_for(0) == game.points_for(1) == 0
    assert game.winner() is None and not game.ended()
    assert desc.make_metrics.__module__ == "boardrl.games.traversal.metrics"
    assert not desc.coop and not desc.rewards.points_value_head
    assert "random" in desc.strategy_from_string.registry
    assert desc.outcome.utility(game, 0, False) == 0


@pytest.mark.parametrize("kwargs", [{"num_players": 1}, {"num_players": 3}, {"terrain": "unknown"}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        Traversal(**kwargs)


@pytest.mark.parametrize("terrain", ("random", "flat", *TERRAIN_TYPES))
def test_seeded_terrain_is_reproducible_and_traversable(terrain):
    for seed in range(5):
        game = Traversal(terrain=terrain, seed=seed)
        assert game.heights == Traversal(terrain=terrain, seed=seed).heights
        for row in range(SIZE):
            for col in range(SIZE):
                height = game.heights[row][col]
                assert height >= 0
                assert all(abs(height - game.heights[r][c]) <= 2 for r, c in neighbors((row, col)))


def test_sampled_maps_and_orientations_are_reachable_without_modifying_the_pool():
    library = terrain_library()
    pool = library.boards["mountains"]
    allowed = {transform.apply(board) for board in pool for transform in TRANSFORMS}
    sampled = {Traversal(terrain="mountains", seed=seed).heights for seed in range(100)}
    assert sampled <= allowed
    assert len(sampled) > len(pool)  # sampling varies both map and orientation
    assert library.boards["mountains"] == pool
    assert {piece.player for piece in Traversal(seed=0).board.values()} == {0, 1}


def test_all_terrain_symmetries_preserve_geometry_and_height_values():
    from collections import Counter

    board = tuple(tuple(row * SIZE + col for col in range(SIZE)) for row in range(SIZE))
    original_neighbors = sorted(
        abs(board[row][col] - board[r][c])
        for row in range(SIZE) for col in range(SIZE)
        for r, c in ((row + 1, col), (row, col + 1)) if r < SIZE and c < SIZE
    )
    variants = {transform.apply(board) for transform in TRANSFORMS}
    assert len(variants) == 8
    for variant in variants:
        assert Counter(value for row in variant for value in row) == Counter(range(SIZE ** 2))
        assert sorted(
            abs(variant[row][col] - variant[r][c])
            for row in range(SIZE) for col in range(SIZE)
            for r, c in ((row + 1, col), (row, col + 1)) if r < SIZE and c < SIZE
        ) == original_neighbors


def test_packaged_terrain_pool_has_twenty_authentic_samples_per_type():
    from importlib.resources import files
    import json

    data = json.loads(files("boardrl.games.traversal").joinpath("terrain_samples.json").read_text())
    assert data["source"] == "https://api.traversalgame.com/api/boards/generate"
    assert data["samples_per_type"] == 20
    assert set(data["boards"]) == set(TERRAIN_TYPES)
    for boards in data["boards"].values():
        assert len(boards) == 20
        for board in boards:
            assert len(board) == SIZE and all(len(row) == SIZE for row in board)
            for row in range(SIZE):
                for col in range(SIZE):
                    assert board[row][col] >= 0
                    assert all(abs(board[row][col] - board[r][c]) <= 2 for r, c in neighbors((row, col)))


def test_random_mode_samples_the_presets_and_flat_stays_available():
    library = terrain_library()
    allowed = {
        transform.apply(board)
        for name in TERRAIN_TYPES for board in library.boards[name] for transform in TRANSFORMS
    }
    samples = [Traversal(seed=seed).heights for seed in range(30)]
    assert set(samples) <= allowed
    assert len(set(samples)) > 1
    assert Traversal(terrain="flat").heights == tuple((0,) * SIZE for _ in range(SIZE))
    desc = games_library("traversal,terrain=mountains,seed=7")
    assert desc.make_game().heights == Traversal(terrain="mountains", seed=7).heights


def test_normal_movement_can_turn_but_cannot_cross_pieces_or_return():
    game = position({(5, 5): Piece(0, SENTINEL), (4, 5): Piece(0, EXPLORER), (0, 0): Piece(1, SENTINEL)})
    assert "f6>e4" in game.moves  # three steps around the allied blocker
    assert "f6>f4" not in game.moves  # four steps around it
    assert "f6>f5" not in game.moves
    assert "f6>f6" not in game.moves
    game.play_str("f6>e4")
    assert game.board[(3, 4)] == Piece(0, SENTINEL)
    assert (5, 5) not in game.board
    assert game.current_player() == 1 and game.round() == 0
    game.play_idx(0)
    assert game.current_player() == 0 and game.round() == 1


def test_height_costs_apply_in_both_directions():
    heights = [[0] * SIZE for _ in range(SIZE)]
    heights[5][5] = 2
    game = position({(5, 5): Piece(0, SENTINEL), (0, 0): Piece(1, SENTINEL)}, heights)
    assert game.movement_cost((5, 5), (4, 5)) == 3
    assert game.movement_cost((4, 5), (5, 5)) == 3
    assert "f6>f5" in game.moves
    assert "f6>f4" not in game.moves
    assert "f6>e5" not in game.moves


def test_explorer_specials_ignore_height_and_are_standalone():
    # Every normal first step costs 3, making diagonal and leap costs 6.
    heights = [[0] * SIZE for _ in range(SIZE)]
    for row, col in neighbors((5, 5)):
        heights[row][col] = 2
    game = position({(5, 5): Piece(0, EXPLORER), (0, 0): Piece(1, SENTINEL)}, heights)
    assert "f6>e5" in game.moves
    assert "f6>f4" in game.moves
    assert "f6>f3" not in game.moves
    assert "f6>d4" not in game.moves


@pytest.mark.parametrize("owner", [0, 1])
def test_explorer_cannot_leap_over_pieces(owner):
    heights = [[2] * SIZE for _ in range(SIZE)]
    heights[5][5] = heights[3][5] = 0
    game = position({(5, 5): Piece(0, EXPLORER), (4, 5): Piece(owner, SENTINEL), (0, 0): Piece(1, SENTINEL)}, heights)
    assert "f6>f4" not in game.moves


def test_explorer_special_can_capture_but_not_land_on_an_ally():
    heights = [[0] * SIZE for _ in range(SIZE)]
    for row, col in neighbors((5, 5)):
        heights[row][col] = 2
    game = position({(5, 5): Piece(0, EXPLORER), (4, 4): Piece(1, EXPLORER), (4, 6): Piece(0, SENTINEL), (0, 0): Piece(1, SENTINEL)}, heights)
    assert "f6>e5" in game.moves
    assert "f6>g5" not in game.moves
    game.play_str("f6>e5")
    assert game.board[(4, 4)] == Piece(0, EXPLORER)
    assert game.captured == [Piece(1, EXPLORER)]
    assert game.current_player() == 1


def test_failed_attack_is_illegal_and_enemies_block_paths():
    game = position({(5, 5): Piece(0, EXPLORER), (4, 5): Piece(1, SENTINEL), (4, 4): Piece(1, SENTINEL), (4, 6): Piece(1, SENTINEL)})
    assert "f6>f5" not in game.moves
    before = game.display_with_moves()
    with pytest.raises(ValueError, match="Illegal"):
        game.play_str("f6>f5")
    assert game.display_with_moves() == before


def test_attack_uses_starting_elevation_and_excludes_attacker_from_support():
    heights = [[0] * SIZE for _ in range(SIZE)]
    heights[5][5] = 1
    game = position({(5, 5): Piece(0, EXPLORER), (3, 5): Piece(1, VANGUARD), (0, 0): Piece(1, SENTINEL)}, heights)
    assert "f6>f4" in game.moves  # origin is higher, attack is 1+1 >= 2
    assert game.attack_strengths((5, 5), (3, 5), 3) == (2, 2)
    game.play_str("f6>f4")
    assert game.current_player() == 1
    assert game.captured == [Piece(1, VANGUARD)]

    game = position({(5, 5): Piece(0, EXPLORER), (4, 5): Piece(1, VANGUARD)})
    assert game.attack_strengths((5, 5), (4, 5), 1) == (1, 2)
    assert "f6>f5" not in game.moves


def test_allies_add_base_strength_and_defenders_add_one_each():
    game = position({(5, 5): Piece(0, EXPLORER), (3, 5): Piece(1, SENTINEL), (3, 4): Piece(0, VANGUARD), (2, 5): Piece(1, EXPLORER)})
    assert game.attack_strengths((5, 5), (3, 5), 2) == (3, 4)
    assert "f6>f4" not in game.moves
    game.board[(3, 6)] = Piece(0, EXPLORER)
    game._refresh_moves()
    assert game.attack_strengths((5, 5), (3, 5), 2) == (4, 4)
    assert "f6>f4" in game.moves


def test_uphill_is_one_strength_penalty_regardless_of_height_difference():
    heights = [[0] * SIZE for _ in range(SIZE)]
    heights[4][5] = 2
    game = position({(5, 5): Piece(0, SENTINEL), (4, 5): Piece(1, SENTINEL)}, heights)
    assert game.attack_strengths((5, 5), (4, 5), 3) == (2, 3)
    assert "f6>f5" not in game.moves


def test_vanguard_charge_uses_cheapest_route_including_attack_step():
    game = position({(6, 5): Piece(0, VANGUARD), (2, 5): Piece(1, SENTINEL), (0, 0): Piece(1, SENTINEL)})
    assert "f7>f3" in game.moves
    assert game._actions["f7>f3"].cost == 4
    game.play_str("f7>f3")
    assert game.captured == [Piece(1, SENTINEL)]

    game = position({(6, 5): Piece(0, VANGUARD), (4, 5): Piece(1, SENTINEL)})
    assert "f7>f5" not in game.moves  # cost 2; a cost-4 detour earns no charge


def test_vanguard_charge_can_be_created_by_terrain():
    heights = [[0] * SIZE for _ in range(SIZE)]
    heights[4][5] = 1
    game = position({(5, 5): Piece(0, VANGUARD), (3, 5): Piece(1, SENTINEL)}, heights)
    assert "f6>f4" in game.moves  # up one, down one: 2+2 = 4
    assert game._actions["f6>f4"].cost == 4


@pytest.mark.parametrize("player,source,move", [(0, (0, 4), "e1>off"), (1, (9, 4), "e10>off")])
def test_sentinel_first_finish_wins_immediately(player, source, move):
    game = position({source: Piece(player, SENTINEL), (4, 4): Piece(1 - player, EXPLORER)}, turn=player)
    game.play_str(move)
    assert game.ended() and game.winner() == player
    assert game.end_reason == "sentinel"
    assert game.points_for(player) == 3
    assert game.points_for(1 - player) == 0
    assert game.moves == []
    with pytest.raises(ValueError):
        game.play_str(move)


def test_finishing_requires_one_extra_movement():
    game = position({(3, 4): Piece(0, SENTINEL), (9, 9): Piece(1, SENTINEL)})
    assert "e4>e1" in game.moves and "e4>off" not in game.moves
    game.play_str("e4>e1")
    assert not game.ended() and game.finished == []
    assert game.scores == [0, 0]
    game.play_idx(0)
    assert "e1>off" in game.moves


def test_explorer_leap_can_finish_from_penultimate_row():
    heights = [[0] * SIZE for _ in range(SIZE)]
    heights[0][4] = 2
    heights[1][4] = 0
    game = position({(1, 4): Piece(0, EXPLORER), (5, 0): Piece(0, SENTINEL), (9, 9): Piece(1, SENTINEL)}, heights)
    assert "e2>off" in game.moves
    game.play_str("e2>off")
    assert game.finished == [Piece(0, EXPLORER)] and game.scores == [1, 0]
    assert not game.ended()


def test_pieces_cannot_exit_home_or_side_edges():
    game = position({(9, 0): Piece(0, EXPLORER), (0, 9): Piece(1, SENTINEL)})
    assert "a10>off" not in game.moves
    assert all("a0" not in move and "k" not in move for move in game.moves)


@pytest.mark.parametrize("first,second,winner,scores", [
    (Piece(0, EXPLORER), Piece(1, EXPLORER), None, [1, 1]),
    (Piece(0, EXPLORER), Piece(1, VANGUARD), 1, [1, 2]),
    (Piece(1, EXPLORER), Piece(0, SENTINEL), 0, [3, 1]),
    (Piece(0, VANGUARD), Piece(0, EXPLORER), 0, [3, 0]),
])
def test_second_finish_ends_game_for_any_combination(first, second, winner, scores):
    source = (Traversal.goal_row(second.player), 4)
    game = position({source: second, (4, 4): Piece(1 - second.player, SENTINEL)}, turn=second.player, finished=[first])
    game.play_str(f"e{source[0] + 1}>off")
    assert game.ended() and game.moves == []
    assert game.end_reason == "two_finished" and game.winner() == winner
    assert game.scores == scores


def test_elimination_credits_strongest_survivor_in_addition_to_finishes():
    game = position({(5, 5): Piece(0, SENTINEL), (4, 5): Piece(1, EXPLORER)}, finished=[Piece(1, VANGUARD)])
    game.play_str("f6>f5")
    assert game.ended() and game.end_reason == "elimination"
    assert game.scores == [3, 2] and game.winner() == 0
    assert game.moves == []


@pytest.mark.parametrize("pieces,finished,winner,reason", [
    ({(4, 4): Piece(0, SENTINEL), (8, 8): Piece(1, SENTINEL)}, [], 1, "move_limit_advancement"),
    ({(4, 4): Piece(0, VANGUARD), (5, 8): Piece(1, SENTINEL)}, [], 1, "move_limit_advancement"),
    ({(4, 4): Piece(0, SENTINEL), (5, 8): Piece(1, SENTINEL)}, [], None, "move_limit_draw"),
    ({(1, 4): Piece(0, SENTINEL), (5, 8): Piece(1, SENTINEL)}, [Piece(1, EXPLORER)], 1, "move_limit_score"),
])
def test_200_move_adjudication(pieces, finished, winner, reason):
    # Move player 1 horizontally to preserve distance/strength for comparison.
    game = position(pieces, turn=MOVE_LIMIT - 1, finished=finished)
    row = next(row for (row, col), piece in pieces.items() if piece.player == 1)
    game.play_str(f"i{row + 1}>j{row + 1}")
    assert game.turn == MOVE_LIMIT and game.ended() and game.moves == []
    assert game.winner() == winner and game.end_reason == reason
    desc = games_library("traversal")
    utility = 0 if winner is None else (1 if winner == 0 else -1)
    assert desc.outcome.utility(game, 0, True) == utility
    assert game.diff_points_for(0) == -game.diff_points_for(1)
    if not finished and winner is not None:
        assert game.points_for(0) == game.points_for(1) == 0
        assert game.diff_points_for(winner) == 0.01


def test_repetition_draw_overrides_a_score_lead():
    game = position({(5, 2): Piece(0, SENTINEL), (4, 7): Piece(1, SENTINEL)}, finished=[Piece(0, EXPLORER)])
    cycle = ("c6>d6", "h5>i5", "d6>c6", "i5>h5")
    for move in cycle * 3:
        game.play_str(move)
    assert game.ended() and game.end_reason == "repetition"
    assert game.winner() is None and game.moves == []
    assert game.points_for(0) == 1
    assert game.diff_points_for(0) == game.diff_points_for(1) == 0
    assert games_library("traversal").outcome.utility(game, 0, True) == 0


def test_stuck_player_is_a_stalemate_draw():
    # Player 0 on its home edge has no affordable attack or escape.
    game = position({
        (9, 0): Piece(0, EXPLORER),
        (9, 1): Piece(1, SENTINEL),
        (8, 0): Piece(1, SENTINEL),
        (8, 1): Piece(1, VANGUARD),
    })
    assert game.ended() and game.end_reason == "stalemate"
    assert game.winner() is None and game.moves == []


def test_display_action_order_forced_view_and_illegal_actions():
    game = Traversal(seed=0)
    assert game.display().isascii()
    assert [line[1:] for line in game.display_with_moves().splitlines() if line.startswith("@")] == game.moves
    assert game.display_with_moves().count("@") == len(game.moves)
    assert "Viewer: 1" in game.display(force=1)
    assert "0S" in game.display() and "1S" in game.display(force=0)
    for redundant in ("Traversal", "Player:", "Goals:", "Types:", "Scores:", "Captured:", "History:", "Status:"):
        assert redundant not in game.display()
    assert f"Turn: 0/{MOVE_LIMIT} Viewer: 0" in game.display()
    assert "Finished: -" in game.display() and "Repetition: -" in game.display()
    with pytest.raises(ValueError):
        game.display(force=2)
    for move in ("invalid", "c1>c2", "c10>c10", "c10>off"):
        with pytest.raises(ValueError):
            game.play_str(move)
    for move in game.moves:
        clone = game.copy()
        clone.play_str(move)
        assert clone.turn == 1
        assert clone.moves == list(clone._actions)


def test_copy_keeps_all_mutable_state_independent():
    game = Traversal(seed=4)
    game.play_idx(0)
    original = game.display_with_moves()
    clone = game.copy()
    clone.play_idx(0)
    clone.scores[0] = 99
    clone.finished.append(Piece(0, EXPLORER))
    clone.captured.append(Piece(1, EXPLORER))
    clone.moves.clear()
    clone._actions.clear()
    clone._positions.clear()
    assert game.display_with_moves() == original
    assert len(game.history) == len(game._positions) == 1


def repetition_positions_from_observation(state):
    """Reverse quiet moves to recover the ordered, relevant public positions."""
    from boardrl.games.traversal.metrics import Observation

    observation = Observation(state)
    board = observation.board.copy()
    turn = int(state.splitlines()[0].split()[1].split("/")[0])
    context = next(line[12:] for line in state.splitlines() if line.startswith("Repetition: "))
    moves = [] if context == "-" else context.split()
    positions = []

    def key():
        return (
            turn % 2,
            tuple((position, piece.player, piece.kind.symbol) for position, piece in sorted(board.items())),
            tuple((piece.player, piece.kind.symbol) for piece in observation.finished),
        )

    for move in reversed(moves):
        positions.append(key())
        source, target = move.split(">")
        source = int(source[1:]) - 1, ord(source[0]) - ord("a")
        target = int(target[1:]) - 1, ord(target[0]) - ord("a")
        board[source] = board.pop(target)
        turn -= 1
    if turn:
        positions.append(key())  # the last capture/finish position was counted
    return positions[::-1]


@pytest.mark.parametrize("boundary", ["capture", "finish"])
def test_irreversible_history_is_omitted_without_losing_repetition_context(boundary):
    pieces = {(5, 2): Piece(0, SENTINEL), (4, 7): Piece(1, SENTINEL)}
    if boundary == "capture":
        pieces[(5, 3)] = Piece(1, EXPLORER)
        irreversible = "c6>d6"
        cycle = ("h5>i5", "d6>c6", "i5>h5", "c6>d6")
    else:
        pieces[(0, 4)] = Piece(0, EXPLORER)
        irreversible = "e1>off"
        cycle = ("h5>i5", "c6>d6", "i5>h5", "d6>c6")
    game = position(pieces)
    prefix = ("c6>b6", "h5>i5", "b6>c6", "i5>h5")
    for move in prefix:
        game.play_str(move)
    assert repetition_positions_from_observation(game.display()) == game._positions
    game.play_str(irreversible)
    assert "Repetition: -" in game.display()
    assert game.history == [*prefix, irreversible]
    assert repetition_positions_from_observation(game.display()) == game._positions[-1:]
    restored = game.copy()
    restored._positions = repetition_positions_from_observation(game.display())
    assert restored._repetition_start == game._repetition_start == 5
    for step, move in enumerate((cycle * 3)[:11]):
        # The compact text recovers the same repetition automaton state at
        # each step, despite omitting positions before the material change.
        assert repetition_positions_from_observation(game.display()) == game._positions[-(step + 1):]
        for candidate in game.moves:
            original_next, restored_next = game.copy(), restored.copy()
            original_next.play_str(candidate)
            restored_next.play_str(candidate)
            assert original_next.ended() == restored_next.ended()
            assert original_next.end_reason == restored_next.end_reason
            assert original_next.winner() == restored_next.winner()
        game.play_str(move)
        restored.play_str(move)
    assert game.ended() and game.end_reason == restored.end_reason == "repetition"
    assert "Repetition:" not in game.display()
    assert "Scores:" in game.display() and "Status: repetition" in game.display()


def test_same_board_and_turn_can_have_different_repetition_outcomes():
    pieces = {(5, 2): Piece(0, SENTINEL), (4, 7): Piece(1, SENTINEL)}
    near_draw, different_path = position(pieces), position(pieces)
    cycle = ("c6>d6", "h5>i5", "d6>c6", "i5>h5")
    excursion = ("c6>c7", "h5>h6", "c7>c6", "h6>h5")
    for move in (cycle * 3)[:11]:
        near_draw.play_str(move)
    for move in (*excursion, *cycle, *cycle[:3]):
        different_path.play_str(move)
    assert near_draw.turn == different_path.turn == 11
    assert near_draw.board == different_path.board
    assert near_draw.moves == different_path.moves
    assert near_draw.display().split("Repetition:")[0] == different_path.display().split("Repetition:")[0]
    assert near_draw.display() != different_path.display()
    for game in (near_draw, different_path):
        game.play_str("i5>h5")
    assert near_draw.ended() and near_draw.end_reason == "repetition"
    assert not different_path.ended()


def test_observation_retains_move_limit_and_forced_value_perspective():
    pieces = {(5, 2): Piece(0, SENTINEL), (4, 7): Piece(1, SENTINEL)}
    last_move = position(pieces, turn=MOVE_LIMIT - 1)
    earlier = position(pieces, turn=MOVE_LIMIT - 3)
    assert last_move.moves == earlier.moves
    assert last_move.display() != earlier.display()
    assert last_move.display(force=0) != last_move.display(force=1)
    for game in (last_move, earlier):
        game.play_str("h5>h6")
    assert last_move.ended() and not earlier.ended()


@pytest.mark.parametrize("seed", range(5))
def test_random_playouts_are_bounded_and_conserve_pieces(seed):
    rng = random.Random(seed)
    game = Traversal(seed=seed)
    while not game.ended():
        assert game.turn < MOVE_LIMIT and game.moves
        game.play_idx(rng.randrange(len(game.moves)))
        assert len(game.board) + len(game.finished) + len(game.captured) == 12
    assert game.turn <= MOVE_LIMIT and game.moves == []
    assert game.diff_points_for(0) == -game.diff_points_for(1)


def test_simulate_to_end():
    game = Traversal(seed=10)
    game.simulate_to_end()
    assert game.ended() and game.turn <= MOVE_LIMIT and not game.moves


@pytest.mark.parametrize("budget", [1, MOVE_LIMIT])
def test_rollouts_rewards_and_metrics(budget):
    import torch

    from boardrl import RolloutRunner
    from boardrl.training import ToSamples

    desc = games_library("traversal,seed=12")
    bot = desc.strategy_from_string("random")
    data = RolloutRunner(desc.make_game, outcome=desc.outcome, progress=False).play(
        [bot, bot], games=2, max_steps=budget,
    )
    for game in data:
        for trace in game:
            assert trace[-1].terminal == (budget == MOVE_LIMIT)
            assert trace[-1].truncated == (budget == 1)
            assert trace[-1].episodic_utility == -game.by_seat[1 - trace.seat_id][-1].episodic_utility
            for record in trace[:-1]:
                assert record.moves == [line[1:] for line in record.state.splitlines() if line.startswith("@")]
                assert len(record.action_distribution) == len(record.moves)
    metrics = desc.make_metrics(data).metrics()
    assert metrics["terminal_rate"] == float(budget == MOVE_LIMIT)
    assert 0 <= metrics["draw_rate"] <= 1
    assert set(metrics["strategy"]) == {"0", "1"}
    samples = ToSamples()(desc.rewards.make_returns(1.0)(data))
    assert samples
    assert all(bool(torch.isfinite(torch.tensor(sample.returns))) == (budget == MOVE_LIMIT) for sample in samples)
    assert all(-1 <= sample.reward <= 1 for sample in samples)


def test_http_state_actions_and_undo_redo(monkeypatch):
    from fastapi.testclient import TestClient
    from boardrl.serve import serve

    monkeypatch.setattr(serve, "sessions", {})
    client = TestClient(serve.app)
    params = {"game": "traversal,terrain=flat,seed=7"}
    state = client.get("/state", params=params).json()
    assert state["name"] == "traversal" and state["current_player"] == 0
    assert "traversal" in client.get("/games").json()["games"]
    assert state["moves"] and not state["ended"]
    assert client.post("/do-one", params=params, json={"action": "invalid"}).status_code == 400
    moved = client.post("/do-one", params=params, json={"action": "e10>e5"})
    assert moved.status_code == 200
    updated = moved.json()["state"]
    assert updated["current_player"] == 1 and updated["can_undo"]
    assert updated["board"] != state["board"]
    undone = client.post("/undo", params=params).json()
    assert undone["board"] == state["board"]
    redone = client.post("/redo", params=params).json()
    assert redone["board"] == updated["board"]
