import pytest

from boardrl.games.tictactoe.game import TicTacToe
from boardrl.games.connectfour.game import ConnectFour
from boardrl.games.sum.game import Sum
from boardrl.games.thegame.game import PLAYED_CARD_SYMBOLS, TheGame


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


def test_connectfour_display_uses_eight_character_board_lines():
    g = ConnectFour()

    lines = g.display().splitlines(keepends=True)

    assert lines[:6] == ["       \n"] * 6
    assert lines[6] == "-------\n"
    assert lines[7] == ">O\n"
    assert all(len(line) == 8 for line in lines[:7])


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


def test_thegame_played_card_memory_starts_empty_in_descending_decade_order():
    g = TheGame(num_players=2)

    assert len(PLAYED_CARD_SYMBOLS) == 32
    assert len(set(PLAYED_CARD_SYMBOLS)) == 32
    assert PLAYED_CARD_SYMBOLS.isascii()
    assert not set(PLAYED_CARD_SYMBOLS) & set("0123456789 \t\r\n")
    assert g.played_cards_memory() == " ".join(
        f"{decade}aa" for decade in range(9, -1, -1)
    )

    g._played_cards = sum(1 << card for card in range(80, 90))
    assert "8FF" in g.played_cards_memory().split()


def test_thegame_played_card_memory_orders_high_half_first_and_high_card_as_low_bit():
    g = TheGame(num_players=1)
    g.deck = []
    g.hands = [[80, 84, 85, 89]]
    g.moves = g.gen_moves()

    g.play_str("84->0")
    assert "8ab" in g.played_cards_memory().split()
    g.play_str("80->1")
    assert "8ar" in g.played_cards_memory().split()
    g.play_str("89->2")
    assert "8br" in g.played_cards_memory().split()
    g.play_str("85->3")
    assert "8rr" in g.played_cards_memory().split()


def test_thegame_played_card_memory_accumulates_and_uses_dynamic_decades():
    g = TheGame(num_players=1, max_value=30, mode="free")
    g.deck = []
    g.hands = [[4, 9, 14, 19, 24, 29]]
    g.moves = g.gen_moves()

    g.play_str("4->0")
    before_exit = g.played_cards_memory()
    g.play_str("x")
    assert g.played_cards_memory() == before_exit

    for card in (9, 14, 19, 24, 29):
        g.play_str(f"{card}->0")

    assert g.played_cards_memory() == "2bb 1bb 0bb"


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

    before_message = g.played_cards_memory()
    g.play_str("A")
    assert g.current_player() == 1
    assert g._last_messages[0] == "A"
    assert g.played_cards_memory() == before_message


def test_thegame_free_without_messages_can_continue_or_exit():
    g = _thegame_ready_for_second_play(mode="free")

    g.play_str("21->0")

    assert g.current_player() == 0
    assert "x" in g.moves
    assert any(move.startswith("22->") for move in g.moves)
    assert not set("ABCDEFGHIJ") & set(g.moves)

    g.play_str("x")
    assert g.current_player() == 1


def test_thegame_omni_matches_free_moves_and_displays_full_information():
    free = TheGame(num_players=3, mode="free")
    omni = TheGame(num_players=3, mode="omni")
    for game in (free, omni):
        game.curplay = 1
        game.deck = list(range(1, 13))
        game.piles = [1, 1, 100, 100]
        game.hands = [[20], [21], [22]]
        game.moves = game.gen_moves()

    assert omni.moves == free.moves
    assert omni.display() == (
        "Round: 0, Action: 0\n"
        "Piles: 1 1 100 100\n"
        "Cards: 12\n"
        "Mem: 9aa 8aa 7aa 6aa 5aa 4aa 3aa 2aa 1aa 0aa\n"
        "Hand: 21\n"
        "Hand: 22\n"
        "Hand: 20\n"
        "Deck: 12 11 10 9 8 7 6 5 4 3\n"
    )


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
