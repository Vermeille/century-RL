from collections import Counter
import random

import pytest

from boardrl.games import games_library
from boardrl.games.hnefatafl.game import (
    ATTACKER, DEFENDER, KING, CORNERS, THRONE, Hnefatafl,
)
from boardrl.games.semantics import OutcomeScores, TerminalOutcomeRewards


def position(pieces, player=0, max_plies=500):
    game = Hnefatafl(max_plies=max_plies)
    kinds = {"A": ATTACKER, "D": DEFENDER, "K": KING}
    game.board = {
        (int(square[1:]) - 1, ord(square[0]) - ord("a")): kinds[symbol]
        for square, symbol in pieces.items()
    }
    game.turn = player
    game._positions = Counter({game._position_key(): 1})
    game._refresh_moves()
    return game


def assert_outcome(game, reason, winner):
    assert game.ended() and game.moves == []
    assert game.end_reason == reason and game.winner() == winner
    expected = [0, 0] if winner is None else [1 if player == winner else -1 for player in range(2)]
    assert [game.points_for(player) for player in range(2)] == expected
    assert game.diff_points() == game.points()
    assert game.display().splitlines()[-1] == f"End:{reason}"
    assert game.display_with_moves() == game.display()
    with pytest.raises(ValueError):
        game.play_str("a1>a2")


def test_initial_setup_registration_and_observation_contract():
    desc = games_library("hnefatafl")
    game = desc.make_game()
    assert game.num_players == 2 and not desc.coop
    assert type(desc.scores) is OutcomeScores
    assert type(desc.rewards) is TerminalOutcomeRewards
    assert desc.augmentations and "random" in desc.strategy_from_string
    assert Counter(piece.symbol for piece in game.board.values()) == {"A": 24, "D": 12, "K": 1}
    assert game.board[THRONE] == KING and game.king_position() == THRONE
    assert not CORNERS.intersection(game.board)
    assert game.current_player() == game.round() == 0
    assert len(game.moves) == len(set(game.moves)) == 116
    assert [line[1:] for line in game.display_with_moves().splitlines() if line.startswith("@")] == game.moves
    assert game.display().splitlines()[0] == "V:0 P:0 Ply:0/500"
    assert game.display(force=1).splitlines()[0] == "V:1 P:0 Ply:0/500"
    assert game.display(force=1).splitlines()[1:] == game.display().splitlines()[1:]
    assert "End:" not in game.display()
    assert game.points_for(0) == game.points_for(1) == 0
    first = game.moves[0]
    game.play_idx(0)
    assert game.current_player() == 1 and game.round() == 0
    assert game.history == [first]
    game.play_idx(0)
    assert game.current_player() == 0 and game.round() == 1
    assert games_library("hnefatafl,max_plies=7").make_game().max_plies == 7


@pytest.mark.parametrize("kwargs", [{"num_players": 1}, {"num_players": 3}, {"max_plies": 0}, {"max_plies": -1}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        Hnefatafl(**kwargs)


def test_moves_are_rook_slides_and_cannot_cross_pieces_or_land_on_occupied_squares():
    game = position({"c3": "A", "c5": "D", "g3": "A", "f6": "K"})
    assert {"c3>c1", "c3>c4", "c3>a3", "c3>f3"} <= set(game.moves)
    for move in ("c3>c5", "c3>c6", "c3>g3", "c3>h3", "c3>d4", "c3>c3", "c5>c4", "garbage"):
        before = game.display_with_moves()
        with pytest.raises(ValueError):
            game.play_str(move)
        assert game.display_with_moves() == before


def test_warriors_cross_empty_throne_but_cannot_stop_on_throne_or_corners():
    game = position({"d6": "A", "f4": "A", "b1": "A", "j10": "K"})
    assert {"d6>e6", "d6>g6", "d6>k6", "f4>f7", "f4>f11"} <= set(game.moves)
    assert "d6>f6" not in game.moves and "f4>f6" not in game.moves
    assert "b1>a1" not in game.moves and "b1>k1" not in game.moves
    blocked = position({"d6": "A", "f6": "K"})
    assert "d6>g6" not in blocked.moves


def test_king_can_return_to_throne():
    game = position({"f4": "K", "b2": "A"}, player=1)
    game.play_str("f4>f6")
    assert game.board[THRONE] == KING and not game.ended()


@pytest.mark.parametrize("pieces,move,victim", [
    ({"b4": "A", "d4": "D", "e4": "A", "f6": "K"}, "b4>c4", (3, 3)),
    ({"e4": "A", "c4": "D", "b4": "A", "f6": "K"}, "e4>d4", (3, 2)),
    ({"d2": "A", "d4": "D", "d5": "A", "f6": "K"}, "d2>d3", (3, 3)),
    ({"d5": "A", "d3": "D", "d2": "A", "f6": "K"}, "d5>d4", (2, 3)),
])
def test_sandwich_capture_in_every_direction(pieces, move, victim):
    game = position(pieces)
    game.play_str(move)
    assert victim not in game.board
    assert game.captured == [0, 1] and not game.ended()


def test_three_simultaneous_captures_and_no_chain_capture():
    game = position({
        "e8": "A", "e4": "D", "e3": "A", "d5": "D", "c5": "A",
        "f5": "D", "g5": "A", "g4": "D", "g3": "A", "j10": "K",
    })
    game.play_str("e8>e5")
    assert game.captured == [0, 3]
    assert all(square not in game.board for square in ((3, 4), (4, 3), (4, 5)))
    assert game.board[3, 6] == DEFENDER  # No cascading capture from the anvil.


def test_moving_into_a_sandwich_is_safe_and_diagonals_and_edges_do_not_capture():
    game = position({"d4": "A", "f4": "A", "e2": "D", "j10": "K"}, player=1)
    game.play_str("e2>e4")
    assert game.board[3, 4] == DEFENDER and game.captured == [0, 0]
    diagonal = position({"c2": "A", "d4": "D", "e5": "A", "j10": "K"})
    diagonal.play_str("c2>c3")
    assert diagonal.board[3, 3] == DEFENDER
    edge = position({"b4": "A", "a2": "D", "j10": "K"})
    edge.play_str("b4>b2")
    assert edge.board[1, 0] == DEFENDER


@pytest.mark.parametrize("player,mover,victim", [(0, "A", "D"), (1, "D", "A")])
def test_empty_throne_and_corner_are_hostile_to_both_sides(player, mover, victim):
    throne = position({"c6": mover, "e6": victim, "j10": "K"}, player)
    throne.play_str("c6>d6")
    assert (5, 4) not in throne.board and throne.captured[1 - player] == 1
    corner = position({"c3": mover, "b1": victim, "j10": "K"}, player)
    corner.play_str("c3>c1")
    assert (0, 1) not in corner.board and corner.captured[1 - player] == 1


def test_occupied_throne_protects_defenders_and_king_can_capture_actively_or_as_anvil():
    game = position({"c6": "A", "e6": "D", "f6": "K"})
    game.play_str("c6>d6")
    assert game.board[5, 4] == DEFENDER and game.captured == [0, 0]
    passive = position({"c6": "D", "e6": "A", "f6": "K"}, player=1)
    passive.play_str("c6>d6")
    assert (5, 4) not in passive.board and passive.captured == [1, 0]
    active = position({"d3": "K", "e4": "A", "f4": "D"}, player=1)
    active.play_str("d3>d4")
    assert (3, 4) not in active.board and active.captured == [1, 0]


@pytest.mark.parametrize("square,source,move", [
    ("f6", "f8", "f8>f7"), ("d4", "d6", "d6>d5"),
])
def test_king_requires_four_attackers_on_throne_and_in_open_board(square, source, move):
    row, col = int(square[1:]) - 1, ord(square[0]) - ord("a")
    pieces = {
        square: "K", source: "A", f"{chr(97 + col)}{row}": "A",
        f"{chr(96 + col)}{row + 1}": "A", f"{chr(98 + col)}{row + 1}": "A",
    }
    game = position(pieces)
    game.play_str(move)
    assert game.king_position() is None and game.captured == [0, 0]
    assert_outcome(game, "king_captured", 0)


def test_throne_does_not_replace_a_fourth_attacker_under_2018_panel_rules():
    game = position({"f5": "K", "e5": "A", "g5": "A", "f2": "A"})
    game.play_str("f2>f4")
    assert game.king_position() == (4, 5) and not game.ended()


def test_king_cannot_be_sandwiched_or_captured_against_board_edge():
    ordinary = position({"d4": "K", "e4": "A", "b4": "A"})
    ordinary.play_str("b4>c4")
    assert ordinary.king_position() == (3, 3) and not ordinary.ended()
    edge = position({"f1": "K", "e1": "A", "g1": "A", "f4": "A", "j10": "D"})
    edge.play_str("f4>f2")
    assert edge.king_position() == (0, 5) and not edge.ended()


@pytest.mark.parametrize("source,corner", [("a4", "a1"), ("k4", "k1"), ("a8", "a11"), ("k8", "k11")])
def test_all_four_escape_corners_and_win_takes_precedence_over_move_cap(source, corner):
    game = position({source: "K", "f3": "A"}, player=1, max_plies=2)
    game.play_str(f"{source}>{corner}")
    assert_outcome(game, "king_escaped", 1)


def test_threefold_repetition_includes_initial_position_and_side_to_move():
    game = position({"b2": "A", "j10": "K"})
    initial_display = game.display()
    initial = game._position_key()
    opposite = game.copy()
    opposite.turn = 1
    assert opposite._position_key() != initial
    cycle = ("b2>b3", "j10>j9", "b3>b2", "j9>j10")
    for move in cycle:
        game.play_str(move)
    assert not game.ended() and game._positions[initial] == 2
    assert game.display().splitlines()[2:13] == initial_display.splitlines()[2:13]
    assert game.display().splitlines()[0] == "V:0 P:0 Ply:4/500"
    assert game.display().splitlines()[-1] == "H:" + " ".join(cycle)
    copy = game.copy()
    for move in cycle:
        copy.play_str(move)
    assert_outcome(copy, "repetition", None)
    assert not game.ended() and game._positions[initial] == 2


def test_stalemate_is_draw_and_not_attacker_victory_by_encirclement():
    game = position({"f6": "K", "f5": "D", "e6": "D", "g6": "D", "f7": "D"})
    assert_outcome(game, "stalemate", None)  # No attackers to move.
    trapped = position({
        "b2": "K", "b1": "D", "a2": "D", "c2": "D", "b3": "D",
        "a3": "A", "b4": "A", "c3": "A", "d2": "A",
        "c1": "A", "k10": "A",
    })
    trapped.play_str("k10>j10")
    assert_outcome(trapped, "stalemate", None)


def test_cap_copy_independence_and_bounded_playouts():
    game = Hnefatafl(max_plies=1)
    copy = game.copy()
    assert copy.display_with_moves() == game.display_with_moves()
    copy.play_idx(0)
    assert_outcome(copy, "move_limit", None)
    assert game.turn == 0 and game.captured == [0, 0] and game.history == []
    copy.board.clear()
    copy.captured[0] = 99
    copy.history.clear()
    copy.moves.append("bad")
    copy._legal_moves.clear()
    assert len(game.board) == 37 and len(game.moves) == len(game._legal_moves) == 116
    with pytest.raises(ValueError):
        game.display(force=2)
    with pytest.raises(ValueError):
        game.points_for(2)
    rng = random.Random(20261020)
    for _ in range(3):
        game = Hnefatafl(max_plies=100)
        while not game.ended():
            game.play_str(rng.choice(game.moves))
            assert len(game.board) == 37 - sum(game.captured) - int(game.king_position() is None)
        assert game.turn <= 100 and game.moves == []
    game = Hnefatafl(max_plies=10)
    game.simulate_to_end()
    assert game.ended() and game.turn <= 10


def test_shared_web_state_action_and_undo_redo(monkeypatch):
    from fastapi.testclient import TestClient
    from boardrl.serve import serve

    monkeypatch.setattr(serve, "sessions", {})
    client = TestClient(serve.app)
    query = {"game": "hnefatafl,max_plies=2"}
    initial = client.get("/state", params=query).json()
    assert initial["name"] == "hnefatafl" and len(initial["moves"]) == 116
    response = client.post("/do-one", params=query, json={"action": initial["moves"][0]})
    assert response.status_code == 200
    state = response.json()["state"]
    assert state["current_player"] == 1 and state["can_undo"]
    undone = client.post("/undo", params=query).json()
    assert undone["board"] == initial["board"]
    redone = client.post("/redo", params=query).json()
    assert redone["board"] == state["board"]
    ended = client.post("/do-one", params=query, json={"action": redone["moves"][0]}).json()["state"]
    assert ended["ended"] and ended["moves"] == [] and ended["points"] == 0
