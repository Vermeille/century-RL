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
