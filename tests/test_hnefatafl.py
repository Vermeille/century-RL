from collections import Counter
import random

import pytest

from boardrl.games import games_library
from boardrl.games.hnefatafl.game import (
    ATTACKER, DEFENDER, KING, CORNERS, THRONE, Hnefatafl, Move, coordinate,
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
        game.play_str("a12")


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
    game.play_idx(0)
    assert game.current_player() == 1 and game.round() == 0
    game.play_idx(0)
    assert game.current_player() == 0 and game.round() == 1
    assert games_library("hnefatafl,max_plies=7").make_game().max_plies == 7


@pytest.mark.parametrize("kwargs", [{"num_players": 1}, {"num_players": 3}, {"max_plies": 0}, {"max_plies": -1}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        Hnefatafl(**kwargs)


def test_compact_moves_roundtrip_all_source_squares_and_both_axes():
    game = Hnefatafl()
    for row in range(11):
        for col in range(11):
            source = row, col
            if source in CORNERS or source == THRONE:
                continue
            game.board = {source: ATTACKER, THRONE: KING}
            game._refresh_moves()
            for text, move in game._legal_moves.items():
                assert len(text) == 3 and text.isascii()
                assert text[:2] == coordinate(source)
                assert Move.from_text(text) == move
    assert Move.from_text("f9A") == Move((8, 5), (9, 5))
    assert Move.from_text("kBg") == Move((10, 10), (10, 6))
    for text in ("", "d1>d3", "z13", "d0e", "d1?", "d1x"):
        with pytest.raises(ValueError):
            Move.from_text(text)


def test_high_mobility_observation_fits_context_with_every_action():
    # 287 legal slides: the former full-coordinate notation exceeded 2050
    # characters here even without history.
    sources = [
        (0, 3), (0, 4), (0, 9), (1, 0), (1, 2), (1, 8), (2, 1), (2, 10),
        (3, 5), (4, 0), (4, 7), (5, 4), (5, 6), (6, 6), (7, 5), (7, 10),
        (8, 0), (8, 10), (9, 1), (9, 9), (10, 2), (10, 3), (10, 4), (10, 8),
    ]
    game = Hnefatafl()
    game.board = dict.fromkeys(sources, ATTACKER) | {THRONE: KING}
    game.turn = 498
    game.captured = [0, 12]
    game._refresh_moves()
    text = game.display_with_moves()
    assert len(game.moves) == 287
    assert len(text) <= 2048
    assert [line[1:] for line in text.splitlines() if line.startswith("@")] == game.moves
    for index, move in enumerate(game.moves):
        copied = game.copy()
        copied.play_idx(index)
        assert copied.board[Move.from_text(move).destination] == ATTACKER

    import torch
    from boardrl.models import make_for_game

    model = make_for_game("toy", games_library("hnefatafl")).eval()
    with torch.no_grad():
        output = model([text, Hnefatafl().display_with_moves()])
    assert output.policy[0].shape == (287,)
    assert output.policy[1].shape == (116,)
    assert all(torch.isfinite(policy).all() for policy in output.policy)
    assert torch.isfinite(output.value.mean).all()


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_full_default_playouts_fit_context_including_final_state(seed):
    rng = random.Random(seed)
    game = Hnefatafl()
    while not game.ended():
        assert len(game.display_with_moves()) <= 2048
        assert "H:" not in game.display()
        game.play_str(rng.choice(game.moves))
    assert len(game.display_with_moves()) <= 2048
    assert game.turn <= game.max_plies


def test_moves_are_rook_slides_and_cannot_cross_pieces_or_land_on_occupied_squares():
    game = position({"c3": "A", "c5": "D", "g3": "A", "f6": "K"})
    assert {"c31", "c34", "c3a", "c3f"} <= set(game.moves)
    for move in ("c35", "c36", "c3g", "c3h", "c3>d4", "c3c", "c54", "garbage"):
        before = game.display_with_moves()
        with pytest.raises(ValueError):
            game.play_str(move)
        assert game.display_with_moves() == before


def test_warriors_cross_empty_throne_but_cannot_stop_on_throne_or_corners():
    game = position({"d6": "A", "f4": "A", "b1": "A", "j10": "K"})
    assert {"d6e", "d6g", "d6k", "f47", "f4B"} <= set(game.moves)
    assert "d6f" not in game.moves and "f46" not in game.moves
    assert "b1a" not in game.moves and "b1k" not in game.moves
    blocked = position({"d6": "A", "f6": "K"})
    assert "d6g" not in blocked.moves


def test_king_can_return_to_throne():
    game = position({"f4": "K", "b2": "A"}, player=1)
    game.play_str("f46")
    assert game.board[THRONE] == KING and not game.ended()


@pytest.mark.parametrize("pieces,move,victim", [
    ({"b4": "A", "d4": "D", "e4": "A", "f6": "K"}, "b4c", (3, 3)),
    ({"e4": "A", "c4": "D", "b4": "A", "f6": "K"}, "e4d", (3, 2)),
    ({"d2": "A", "d4": "D", "d5": "A", "f6": "K"}, "d23", (3, 3)),
    ({"d5": "A", "d3": "D", "d2": "A", "f6": "K"}, "d54", (2, 3)),
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
    game.play_str("e85")
    assert game.captured == [0, 3]
    assert all(square not in game.board for square in ((3, 4), (4, 3), (4, 5)))
    assert game.board[3, 6] == DEFENDER  # No cascading capture from the anvil.


def test_moving_into_a_sandwich_is_safe_and_diagonals_and_edges_do_not_capture():
    game = position({"d4": "A", "f4": "A", "e2": "D", "j10": "K"}, player=1)
    game.play_str("e24")
    assert game.board[3, 4] == DEFENDER and game.captured == [0, 0]
    diagonal = position({"c2": "A", "d4": "D", "e5": "A", "j10": "K"})
    diagonal.play_str("c23")
    assert diagonal.board[3, 3] == DEFENDER
    edge = position({"b4": "A", "a2": "D", "j10": "K"})
    edge.play_str("b42")
    assert edge.board[1, 0] == DEFENDER


@pytest.mark.parametrize("player,mover,victim", [(0, "A", "D"), (1, "D", "A")])
def test_empty_throne_and_corner_are_hostile_to_both_sides(player, mover, victim):
    throne = position({"c6": mover, "e6": victim, "j10": "K"}, player)
    throne.play_str("c6d")
    assert (5, 4) not in throne.board and throne.captured[1 - player] == 1
    corner = position({"c3": mover, "b1": victim, "j10": "K"}, player)
    corner.play_str("c31")
    assert (0, 1) not in corner.board and corner.captured[1 - player] == 1


def test_occupied_throne_protects_defenders_and_king_can_capture_actively_or_as_anvil():
    game = position({"c6": "A", "e6": "D", "f6": "K"})
    game.play_str("c6d")
    assert game.board[5, 4] == DEFENDER and game.captured == [0, 0]
    passive = position({"c6": "D", "e6": "A", "f6": "K"}, player=1)
    passive.play_str("c6d")
    assert (5, 4) not in passive.board and passive.captured == [1, 0]
    active = position({"d3": "K", "e4": "A", "f4": "D"}, player=1)
    active.play_str("d34")
    assert (3, 4) not in active.board and active.captured == [1, 0]


@pytest.mark.parametrize("square,source,move", [
    ("f6", "f8", "f87"), ("d4", "d6", "d65"),
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
    game.play_str("f24")
    assert game.king_position() == (4, 5) and not game.ended()


def test_king_cannot_be_sandwiched_or_captured_against_board_edge():
    ordinary = position({"d4": "K", "e4": "A", "b4": "A"})
    ordinary.play_str("b4c")
    assert ordinary.king_position() == (3, 3) and not ordinary.ended()
    edge = position({"f1": "K", "e1": "A", "g1": "A", "f4": "A", "j10": "D"})
    edge.play_str("f42")
    assert edge.king_position() == (0, 5) and not edge.ended()


@pytest.mark.parametrize("source,move", [("a4", "a41"), ("k4", "k41"), ("a8", "a8B"), ("k8", "k8B")])
def test_all_four_escape_corners_and_win_takes_precedence_over_move_cap(source, move):
    game = position({source: "K", "f3": "A"}, player=1, max_plies=2)
    game.play_str(move)
    assert game.king_position() in CORNERS
    assert_outcome(game, "king_escaped", 1)


def test_repetitions_continue_until_move_cap_with_no_history_in_observation():
    game = position({"b2": "A", "j10": "K"}, max_plies=12)
    initial_display = game.display()
    cycle = ("b23", "jA9", "b32", "j9A")
    for move in cycle * 2:
        game.play_str(move)
    assert not game.ended()
    assert game.display().splitlines()[2:13] == initial_display.splitlines()[2:13]
    assert game.display().splitlines()[0] == "V:0 P:0 Ply:8/12"
    assert "H:" not in game.display()
    assert len(game.display()) == len(initial_display)
    copy = game.copy()
    for move in cycle:
        copy.play_str(move)
    assert_outcome(copy, "move_limit", None)
    assert not game.ended() and game.turn == 8


def test_stalemate_is_draw_and_not_attacker_victory_by_encirclement():
    game = position({"f6": "K", "f5": "D", "e6": "D", "g6": "D", "f7": "D"})
    assert_outcome(game, "stalemate", None)  # No attackers to move.
    trapped = position({
        "b2": "K", "b1": "D", "a2": "D", "c2": "D", "b3": "D",
        "a3": "A", "b4": "A", "c3": "A", "d2": "A",
        "c1": "A", "k10": "A",
    })
    trapped.play_str("kAj")
    assert_outcome(trapped, "stalemate", None)


def test_cap_copy_independence_and_bounded_playouts():
    game = Hnefatafl(max_plies=1)
    copy = game.copy()
    assert copy.display_with_moves() == game.display_with_moves()
    copy.play_idx(0)
    assert_outcome(copy, "move_limit", None)
    assert game.turn == 0 and game.captured == [0, 0]
    copy.board.clear()
    copy.captured[0] = 99
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
