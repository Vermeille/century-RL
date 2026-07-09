import pytest

from boardrl.games.tictactoe.game import TicTacToe
from boardrl.games.connectfour.game import ConnectFour
from boardrl.games.sum.game import Sum, RockPaperScissors
from boardrl.games.thegame.game import TheGame


def test_tictactoe_basic_win():
    g = TicTacToe()
    g.play_str("0")
    g.play_str("3")
    g.play_str("1")
    g.play_str("4")
    g.play_str("2")
    assert g.winner() == 0
    assert g.ended()
    assert g.points_for(0) == 1
    assert g.points_for(1) == -1


def test_tictactoe_illegal_move():
    g = TicTacToe()
    g.play_str("0")
    with pytest.raises(AssertionError):
        g.play_str("0")


def test_tictactoe_draw():
    g = TicTacToe()
    moves = ["0", "1", "2", "4", "3", "5", "7", "6", "8"]
    for m in moves:
        g.play_str(m)
    assert g.winner() is None
    assert g.ended()
    assert g.points_for(0) == 0
    assert g.points_for(1) == 0


def test_connectfour_vertical_win():
    g = ConnectFour()
    for _ in range(3):
        g.play_str("0")
        g.play_str("1")
    g.play_str("0")
    assert g.winner() == 0
    assert g.ended()


def test_connectfour_full_column():
    g = ConnectFour()
    for _ in range(g.height // 2):
        g.play_str("0")
        g.play_str("0")
    assert "0" not in g.moves


def test_connectfour_winner_detection():
    g = ConnectFour()
    for i in range(4):
        g.board[i][0] = 0
    assert g._check_winner() == 0

    g = ConnectFour()
    for i in range(4):
        g.board[0][i] = 1
    assert g._check_winner() == 1

    g = ConnectFour()
    g.board[0][0] = 0
    g.board[1][1] = 0
    g.board[2][2] = 0
    g.board[3][3] = 0
    assert g._check_winner() == 0

    g = ConnectFour()
    g.board[3][0] = 1
    g.board[2][1] = 1
    g.board[1][2] = 1
    g.board[0][3] = 1
    assert g._check_winner() == 1


def test_sum_correct_guess_increases_score():
    g = Sum()
    g.a = 2
    g.b = 4
    g.play_str("3")  # correct average
    assert g.scores[0] == 1


def test_sum_wrong_guess_passes_turn():
    g = Sum()
    g.a = 2
    g.b = 4
    g.play_str("0")
    assert g.current_player() == 1


def test_sum_ended_on_three_points():
    g = Sum()
    g.scores[0] = 2
    g.a = 2
    g.b = 4
    g.play_str("3")
    assert g.ended()


def test_rps_win_and_end():
    g = RockPaperScissors()
    g.play_str("rock")
    g.play_str("scissors")
    assert g.scores[0] == 1
    g.scores[0] = 2
    g.play_str("rock")
    g.play_str("scissors")
    assert g.ended()


def test_thegame_gen_moves():
    g = TheGame(num_players=2)
    g.deck = []
    g.piles = [30, 1, 100, 90]
    g.hands[0] = [20]
    g.turn = 0
    g.moves = g.gen_moves()
    assert set(g.moves) == {"20->0", "20->1", "20->2", "20->3"}


def test_thegame_play_and_draw():
    g = TheGame(num_players=2)
    g.deck = [50, 51]
    g.piles = [1, 1, 100, 100]
    g.hands[0] = [20, 21]
    g.turn = 0
    g.moves = g.gen_moves()
    g.play_str("20->0")
    assert len(g.deck) == 2
    g.play_str("21->0")
    assert len(g.deck) == 0
    assert len(g.hands[0]) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "unknown"},
        {"messages": "before_draw"},
        {"more_than_two_actions": True},
    ],
)
def test_thegame_rejects_invalid_or_removed_mode_args(kwargs):
    with pytest.raises((TypeError, ValueError)):
        TheGame(**kwargs)


def _thegame_ready_for_second_play(**kwargs):
    g = TheGame(num_players=2, **kwargs)
    g.deck = [50, 51, 52]
    g.piles = [1, 1, 100, 100]
    g.hands[0] = [20, 21, 22]
    g.moves = g.gen_moves()
    g.play_str("20->0")
    return g


def test_thegame_default_passes_immediately_after_two_cards():
    g = _thegame_ready_for_second_play()

    g.play_str("21->0")

    assert g.current_player() == 1
    assert g.action == 0
    assert "x" not in g.moves
    assert not set("ABCDEFGHIJ") & set(g.moves)


def test_thegame_strict_before_draw_message_gets_message_action_only():
    g = _thegame_ready_for_second_play(mode="strict_message_before_draw")

    g.play_str("21->0")

    assert g.current_player() == 0
    assert g.action == 2
    assert g.moves == list("ABCDEFGHIJ")

    g.play_str("A")
    assert g.current_player() == 1
    assert g._last_messages[0] == "A"


def test_thegame_free_without_messages_can_continue_or_exit():
    g = _thegame_ready_for_second_play(mode="free")

    g.play_str("21->0")

    assert g.current_player() == 0
    assert "x" in g.moves
    assert any(move.startswith("22->") for move in g.moves)
    assert not set("ABCDEFGHIJ") & set(g.moves)

    g.play_str("x")
    assert g.current_player() == 1


def test_thegame_free_before_draw_message_can_continue_or_message():
    g = _thegame_ready_for_second_play(mode="free_message_before_draw")

    g.play_str("21->0")

    assert g.current_player() == 0
    assert "x" not in g.moves
    assert set("ABCDEFGHIJ").issubset(g.moves)
    assert any(move.startswith("22->") for move in g.moves)


def test_thegame_strict_after_draw_message_draws_then_gets_message_only():
    g = _thegame_ready_for_second_play(mode="strict_message_after_draw")

    g.play_str("21->0")

    assert g.current_player() == 0
    assert g.action == 2
    assert len(g.deck) == 0
    assert set(g.hands[0]) == {22, 50, 51, 52}
    assert g.moves == list("ABCDEFGHIJ")

    g.play_str("B")
    assert g.current_player() == 1
    assert g._last_messages[0] == "B"


def test_thegame_free_after_draw_message_exits_then_draws_then_messages():
    g = _thegame_ready_for_second_play(mode="free_message_after_draw")

    g.play_str("21->0")

    assert g.current_player() == 0
    assert "x" in g.moves
    assert any(move.startswith("22->") for move in g.moves)
    assert not set("ABCDEFGHIJ") & set(g.moves)

    g.play_str("x")
    assert g.current_player() == 0
    assert len(g.deck) == 0
    assert set(g.hands[0]) == {22, 50, 51, 52}
    assert g.moves == list("ABCDEFGHIJ")

    g.play_str("C")
    assert g.current_player() == 1
    assert g._last_messages[0] == "C"


def test_thegame_deck_empty_skips_empty_player_instead_of_ending():
    g = TheGame(num_players=2)
    g.deck = []
    g.piles = [1, 1, 100, 100]
    g.hands = [[], [20]]
    g.curplay = 1
    g.action = 0
    g.moves = g.gen_moves()

    g.play_str("20->0")

    assert g.moves == []
    assert g.points() == g.max_value
    assert g.ended()


def test_thegame_deck_empty_strict_player_can_play_final_cards_past_empty_players():
    g = TheGame(num_players=2)
    g.deck = []
    g.piles = [1, 1, 100, 100]
    g.hands = [[], [20, 21]]
    g.curplay = 1
    g.action = 0
    g.moves = g.gen_moves()

    g.play_str("20->0")

    assert g.current_player() == 1
    assert "21->0" in g.moves
    assert not g.ended()

    g.play_str("21->0")
    assert g.points() == g.max_value
    assert g.ended()


def test_thegame_deck_empty_skip_clears_stale_message():
    g = TheGame(num_players=2, mode="strict_message_before_draw")
    g.deck = []
    g.piles = [1, 1, 100, 100]
    g.hands = [[], [20]]
    g.curplay = 1
    g.action = 0
    g._last_messages = ["A", ""]
    g.moves = g.gen_moves()

    g.play_str("20->0")
    g.play_str("B")

    assert g._last_messages[0] == ""
    assert g.ended()


def test_thegame_deck_empty_non_empty_active_player_without_moves_loses():
    g = TheGame(num_players=2)
    g.deck = []
    g.piles = [90, 90, 10, 10]
    g.hands = [[50], []]
    g.curplay = 0
    g.action = 0
    g.moves = g.gen_moves()

    assert g.moves == []
    assert g.points() == g.max_value - 1
    assert g.ended()
