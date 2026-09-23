import random

from boardrl.games import games_library
from boardrl.games.augmentations import shuffle_actions
from boardrl.games.semantics import PointScores, TerminalOutcomeRewards
from boardrl.games.splendor.data import CARDS, CARDS_BY_TIER, COLORS, GOLD, NOBLES
from boardrl.games.splendor.game import (
    EMPTY_CARD_ID,
    MAX_RESERVED,
    MAX_TOKENS,
    ReservedCard,
    Splendor,
)
from boardrl.games.splendor.semantics import SplendorOutcome


def test_splendor_base_data_and_two_player_setup():
    random.seed(0)
    game = Splendor()

    assert len(CARDS) == 90
    assert {tier: len(CARDS_BY_TIER[tier]) for tier in (1, 2, 3)} == {
        1: 40,
        2: 30,
        3: 20,
    }
    assert len(NOBLES) == 10
    assert game.bank == {color: 4 for color in COLORS} | {GOLD: 5}
    assert [len(game.market[tier]) for tier in (1, 2, 3)] == [4, 4, 4]
    assert [len(game.decks[tier]) for tier in (1, 2, 3)] == [36, 26, 16]
    assert len(game.nobles) == 3
    assert len(game.display_with_moves()) < 2048


def test_take_three_and_take_two_requirements():
    game = Splendor()

    assert "T:WBG" in game.moves
    assert "T:WW" in game.moves

    game.bank["W"] = 3
    game._refresh_moves()

    assert "T:WW" not in game.moves


def test_take_fewer_different_colors_when_bank_has_fewer_available():
    game = Splendor()
    for color in COLORS:
        game.bank[color] = 0
    game.bank["W"] = 1
    game.bank["G"] = 1
    game._refresh_moves()

    take_moves = [move for move in game.moves if move.startswith("T:")]
    assert take_moves == ["T:WG"]


def test_overflow_uses_one_compact_discard_action():
    game = Splendor()
    player = game.players[0]
    player.tokens.update({"W": 2, "B": 2, "G": 2, "R": 2, "K": 1, "Y": 0})
    game.bank.update({"W": 2, "B": 2, "G": 2, "R": 2, "K": 3, "Y": 5})
    game._refresh_moves()

    game.play_str("T:WBG")

    assert game.phase == "discard"
    assert game.current_player() == 0
    assert player.token_total() == 12
    assert game.moves
    assert all(move.startswith("D:") and len(move[2:]) == 2 for move in game.moves)

    game.play_str(game.moves[0])

    assert player.token_total() == MAX_TOKENS
    assert game.current_player() == 1
    assert game.phase == "main"


def test_visible_reserve_stays_public_but_blind_reserve_is_hidden():
    visible = Splendor()
    visible_card_id = visible.market[1][0]
    visible.play_str("R:1.0")
    assert visible_card_id != EMPTY_CARD_ID
    assert visible._card_text(CARDS[visible_card_id]) in visible.display(force=1)

    blind = Splendor()
    blind_card_id = blind.decks[1][-1]
    blind.play_str("R:1.D")
    assert blind.players[0].reserved[0].card_id == blind_card_id
    assert blind._card_text(CARDS[blind_card_id]) in blind.display(force=0)
    opponent_view = blind.display(force=1)
    assert "H0=?" in opponent_view
    assert blind._card_text(CARDS[blind_card_id]) not in opponent_view


def test_reserve_is_legal_without_gold_and_stops_at_three_cards():
    game = Splendor()
    game.bank[GOLD] = 0
    game._refresh_moves()
    assert "R:1.0" in game.moves

    player = game.players[0]
    player.reserved = [
        ReservedCard(0, True),
        ReservedCard(1, True),
        ReservedCard(2, False),
    ]
    game._refresh_moves()
    assert len(player.reserved) == MAX_RESERVED
    assert not any(move.startswith("R:") for move in game.moves)


def test_buy_can_choose_gold_substitution_and_returns_spent_tokens():
    game = Splendor()
    card = CARDS[7]  # K1, costs B4
    assert card.cost == (0, 4, 0, 0, 0)
    game.market[1][0] = card.id
    player = game.players[0]
    player.tokens["B"] = 4
    player.tokens[GOLD] = 1
    game.bank["B"] = 0
    game.bank[GOLD] = 4
    game._refresh_moves()

    assert "B:1.0" in game.moves
    assert "B:1.0~B" in game.moves

    game.play_str("B:1.0~B")

    assert card.id in player.purchased
    assert player.tokens["B"] == 1
    assert player.tokens[GOLD] == 0
    assert game.bank["B"] == 3
    assert game.bank[GOLD] == 5


def test_single_noble_is_automatic_and_multiple_nobles_require_one_choice():
    game = Splendor()
    player = game.players[0]
    game.nobles = bytearray([2])  # W4 B4
    player.purchased = bytearray([16, 17, 18, 19, 8, 9, 10, 11])
    game.play_str("T:WBG")

    assert 2 in player.nobles
    assert game.current_player() == 1

    game = Splendor()
    player = game.players[0]
    game.nobles = bytearray([2, 9])  # W4B4 and W3B3G3
    player.purchased = bytearray(
        [16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26]
    )
    game.play_str("T:WBG")

    assert game.phase == "noble"
    assert game.current_player() == 0
    assert set(game.moves) == {f"N:{NOBLES[2].id}", f"N:{NOBLES[9].id}"}

    game.play_str(f"N:{NOBLES[9].id}")
    assert player.nobles == bytearray([9])
    assert game.current_player() == 1


def test_endgame_finishes_round_and_tiebreak_uses_fewest_developments():
    game = Splendor()
    game.players[0].purchased = bytearray([73, 77, 81])  # 5 + 5 + 5 prestige

    game.play_str("T:WBG")
    assert game.final_round
    assert not game.ended()
    assert game.current_player() == 1

    game.play_str(game.moves[0])
    if game.phase == "discard":
        game.play_str(game.moves[0])
    if game.phase == "noble":
        game.play_str(game.moves[0])
    assert game.ended()

    tied = Splendor()
    tied.players[0].purchased = bytearray([73, 77, 81])  # 15 points, 3 cards
    tied.players[1].purchased = bytearray([85, 87, 71, 42])  # 15 points, 4 cards
    tied._ended = True
    tied.moves = []

    assert tied.prestige_for(0) == tied.prestige_for(1) == 15
    assert tied.winners() == (0,)
    assert tied.diff_points_for(0) == 0.01
    assert tied.diff_points_for(1) == -0.01


def test_stalemate_is_terminal_draw_instead_of_infinite_pass_loop():
    game = Splendor()
    for color in COLORS:
        game.bank[color] = 0
    for player in game.players:
        player.tokens.update({"W": 2, "B": 2, "G": 2, "R": 2, "K": 2, "Y": 0})
        player.reserved = [
            ReservedCard(71, False),
            ReservedCard(75, False),
            ReservedCard(79, False),
        ]
        player.purchased = bytearray()
    for tier in (1, 2, 3):
        game.market[tier] = bytearray([71, 75, 79, 83])
    game._refresh_moves()

    assert game.ended()
    assert game.stalemate()
    assert game.moves == []
    assert game.winners() == ()
    assert game.diff_points_for(0) == game.diff_points_for(1) == 0


def test_copy_is_independent_and_random_playouts_terminate():
    random.seed(7)
    game = Splendor()
    clone = game.copy()
    clone.bank["W"] -= 1
    clone.players[0].tokens["W"] += 1
    clone.market[1][0] = EMPTY_CARD_ID

    assert game.bank["W"] == 4
    assert game.players[0].tokens["W"] == 0
    assert game.market[1][0] != EMPTY_CARD_ID

    for seed in range(20):
        random.seed(seed)
        game = Splendor()
        game.simulate_to_end(max_steps=300)
        assert game.ended()


def test_splendor_descriptor_uses_hidden_info_safe_game_and_tiebreak_outcome():
    desc = games_library("splendor")
    assert desc.make_game().num_players == 2
    assert desc.augmentations == (shuffle_actions,)
    assert isinstance(desc.scores, PointScores)
    assert isinstance(desc.outcome, SplendorOutcome)
    assert isinstance(desc.rewards, TerminalOutcomeRewards)
