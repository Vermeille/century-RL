from itertools import permutations

import pytest

from boardrl.games import games_library
from boardrl.games.skullking.game import SkullKing, _winning_play, fresh_deck


def _set_play_state(game, hands, *, starter=0):
    game.phase = "play"
    game.round_starter = starter
    game.current_player_ = starter
    game.hands = [hand[:] for hand in hands]
    game.tricks = [0] * game.num_players
    game.captured = [[] for _ in range(game.num_players)]
    game._captured_tricks = [[] for _ in range(game.num_players)]
    game._played_tricks = []
    game._trick = []
    game.current_color = "_"
    game.current_top = "_:E"
    game.current_winner = starter
    game._lead_suit_pending = True
    game.moves = game.gen_moves()


def test_skullking_is_registered_with_constructor_options():
    desc = games_library("skullking,num_players=2,num_rounds=1")
    game = desc.make_game()

    assert game.num_players == 2
    assert game.num_rounds == 1
    assert game.moves == ["0", "1"]


def test_skullking_exposes_framework_state_and_random_playout():
    game = games_library("skullking,num_players=2,num_rounds=1").make_game()

    assert "Moves" in game.display_with_moves()
    assert all(
        line.startswith("@") for line in game.display_with_moves().splitlines()[-2:]
    )

    game.simulate_to_end()

    assert game.ended()
    assert game.moves == []
    assert game.winner() in [0, 1]


@pytest.mark.parametrize("cards", list(permutations(["_:P", "_:M", "_:K"])))
def test_mermaid_wins_every_order_when_all_three_characters_are_played(cards):
    winner, card = _winning_play(list(enumerate(cards)), "_")

    assert cards[winner] == "_:M"
    assert card == "_:M"


@pytest.mark.parametrize(
    "cards,winner_card",
    [
        (["_:P", "_:M"], "_:P"),
        (["_:M", "_:P"], "_:P"),
        (["_:K", "_:P"], "_:K"),
        (["_:P", "_:K"], "_:K"),
        (["_:M", "_:K"], "_:M"),
        (["_:K", "_:M"], "_:M"),
    ],
)
def test_character_pair_hierarchy(cards, winner_card):
    _, card = _winning_play(list(enumerate(cards)), "_")

    assert card == winner_card


def test_bids_are_hidden_until_everyone_has_committed():
    game = SkullKing(num_players=3, num_rounds=1)

    game.play_str("0")
    assert "Bids: -" in game.display()

    game.play_str("1")
    assert "Bids: -" in game.display()

    game.play_str("0")
    assert game.phase == "play"
    assert "Bids: 0 1 0" in game.display()


def test_round_starter_rotates_instead_of_using_previous_trick_winner():
    game = SkullKing(num_players=2, num_rounds=2)
    game.play_str("0")
    game.play_str("0")

    # Round one consists of a single trick. Its winner is irrelevant to who
    # opens round two: the starting seat rotates with the dealer.
    while game.phase == "play":
        game.play_str(game.moves[0])

    assert game.round_ == 2
    assert game.round_starter == 1
    assert game.current_player_ == 0  # hidden bids are stored in seat order

    game.play_str("0")
    game.play_str("0")
    assert game.phase == "play"
    assert game.current_player_ == 1


def test_character_after_leading_escape_leaves_the_trick_without_a_suit():
    game = SkullKing(num_players=3, num_rounds=1)
    _set_play_state(
        game,
        [["_:E"], ["_:P"], ["r:10", "y:2"]],
    )

    game.play_str("_:E")
    assert game._lead_suit_pending

    game.play_str("_:P")
    assert not game._lead_suit_pending
    assert game.current_color == "_"
    assert set(game.moves) == {"r:10", "y:2"}


def test_escape_chain_defers_suit_until_first_numbered_card():
    game = SkullKing(num_players=4, num_rounds=1)
    _set_play_state(
        game,
        [["_:E"], ["_:E"], ["r:7"], ["r:3", "y:14"]],
    )

    game.play_str("_:E")
    game.play_str("_:E")
    assert game._lead_suit_pending

    game.play_str("r:7")
    assert game.current_color == "r"
    assert not game._lead_suit_pending
    assert game.moves == ["r:3"]


def test_observation_contains_current_trick_and_completed_public_history():
    game = SkullKing(num_players=2, num_rounds=2)
    game.round_ = 2
    game.bids = [1, 1]
    _set_play_state(game, [["r:1", "y:1"], ["r:2", "y:2"]])

    game.play_str("r:1")
    game.play_str("r:2")

    state = game.display()
    assert "History: 0:r:1 1:r:2" in state

    game.play_str("y:2")
    assert "Trick: 1:y:2" in game.display()


def test_identical_cards_do_not_create_duplicate_policy_actions():
    game = SkullKing(num_players=2, num_rounds=1)
    _set_play_state(game, [["_:P", "_:P", "r:1"], ["r:2"]])

    assert game.moves.count("_:P") == 1


def test_core_deck_contains_tigress_and_seven_players_can_receive_ten_cards():
    deck = fresh_deck()
    assert len(deck) == 70
    assert deck.count("_:T") == 1

    game = SkullKing(num_players=7, num_rounds=10)
    game.round_ = 10
    game._deal()

    assert all(len(hand) == 10 for hand in game.hands)
    assert game.deck == []


def test_tigress_is_a_single_card_with_pirate_and_escape_actions():
    game = SkullKing(num_players=2, num_rounds=1)
    _set_play_state(game, [["_:T"], ["_:M"]])

    assert game.moves == ["_:T=P", "_:T=E"]

    game.play_str("_:T=P")
    assert "_:T" not in game.hands[0]
    assert game.current_top == "_:T=P"


def test_diff_points_is_margin_to_best_opponent_for_competitive_evaluation():
    game = SkullKing(num_players=4, num_rounds=1)
    game.the_points = [200, 100, -20, -50]

    assert game.diff_points_for(0) == 100
    assert game.diff_points_for(1) == -100
    assert game.diff_points_for(2) == -220

    game.the_points = [-20, -20, -50, -80]
    assert game.diff_points_for(0) == 0
    assert game.diff_points_for(1) == 0
    assert game.diff_points_for(2) == -30


def test_copy_preserves_public_history_without_aliasing_it():
    game = SkullKing(num_players=2, num_rounds=2)
    game.round_ = 2
    game.bids = [1, 1]
    _set_play_state(game, [["r:1", "y:1"], ["r:2", "y:2"]])
    game.play_str("r:1")
    game.play_str("r:2")

    copied = game.copy()
    assert copied.display() == game.display()

    copied._played_tricks[0].append((0, "B:14"))
    assert copied._played_tricks != game._played_tricks
