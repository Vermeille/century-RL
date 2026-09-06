import random

from boardrl.games import games_library
from boardrl.games.hanabi.game import Card, CardKnowledge, Hanabi


def _unknown(game):
    return CardKnowledge.unknown(game.colors, game.ranks)


def test_hanabi_registered_as_coop():
    desc = games_library("hanabi,mode=mini,num_players=2")
    game = desc.make_game()

    assert desc.coop
    assert game.mode == "mini"
    assert game.num_players == 2


def test_full_mode_matches_standard_hanabi_setup():
    game = Hanabi(num_players=2, mode="full")

    assert game.colors == ("R", "Y", "G", "B", "W")
    assert game.hand_size == 5
    assert game.max_information_tokens == 8
    assert game.max_life_tokens == 3
    assert game.max_score == 25
    assert len(game.deck) == 40


def test_full_mode_uses_four_card_hands_for_four_and_five_players():
    assert Hanabi(num_players=4, mode="full").hand_size == 4
    assert Hanabi(num_players=5, mode="full").hand_size == 4


def test_mini_matches_deepmind_hanabi_small():
    game = Hanabi(num_players=2, mode="mini")

    assert game.colors == ("R", "Y")
    assert game.hand_size == 2
    assert game.max_information_tokens == 3
    assert game.max_life_tokens == 1
    assert game.max_score == 10
    assert len(game.deck) == 16


def test_display_is_compact_and_hides_only_own_card_identity():
    game = Hanabi(num_players=2, mode="mini")
    game.hands = [[Card("R", 1), Card("Y", 2)], [Card("R", 3), Card("Y", 4)]]
    game.knowledge = [[_unknown(game), _unknown(game)], [_unknown(game), _unknown(game)]]
    game.curplay = 0
    game.moves = game.gen_moves()

    display = game.display()

    assert "mini r0 t me" in display
    assert "s 0/10 i 3/3 l 1/1 d 16 f -" in display
    assert "me ?/RY12345/- ?/RY12345/-" in display
    assert "p1 R3/RY12345/- Y4/RY12345/-" in display
    assert "R1" not in display
    assert "Y2" not in display
    assert "[" not in display
    assert "]" not in display
    assert "|" not in display
    assert ":" not in display


def test_hint_tracks_explicit_and_inferred_knowledge_and_last_action():
    game = Hanabi(num_players=2, mode="mini")
    game.hands = [[Card("R", 3), Card("Y", 3)], [Card("R", 1), Card("Y", 2)]]
    game.knowledge = [[_unknown(game), _unknown(game)], [_unknown(game), _unknown(game)]]
    game.curplay = 0
    game.moves = game.gen_moves()

    game.play_str("h p1 cR")

    assert game.information_tokens == 2
    assert game.knowledge[1][0].colors == {"R"}
    assert game.knowledge[1][0].hinted_color == "R"
    assert game.knowledge[1][1].colors == {"Y"}
    assert game.knowledge[1][1].hinted_color is None

    display = game.display()
    assert "la p1 h me cR 0" in display
    assert "me ?/R12345/R ?/Y12345/-" in display


def test_hint_targets_are_relative_to_acting_player():
    game = Hanabi(num_players=3, mode="mini")
    game.curplay = 2
    game.hands = [
        [Card("R", 1), Card("R", 2)],
        [Card("Y", 1), Card("Y", 2)],
        [Card("R", 3), Card("Y", 3)],
    ]
    game.knowledge = [[_unknown(game), _unknown(game)] for _ in range(3)]
    game.moves = game.gen_moves()

    assert "h p1 cR" in game.moves
    assert "h p2 cY" in game.moves


def test_successful_play_advances_firework_and_draws():
    game = Hanabi(num_players=2, mode="full")
    game.hands[0][0] = Card("R", 1)
    game.fireworks["R"] = 0
    deck_before = len(game.deck)
    game.moves = game.gen_moves()

    game.play_str("p 0")

    assert game.fireworks["R"] == 1
    assert len(game.deck) == deck_before - 1
    assert game.current_player() == 1
    assert "la p1 p R1 ok" in game.display()


def test_fatal_misplay_ends_game_and_scores_zero():
    game = Hanabi(num_players=2, mode="mini")
    game.fireworks["R"] = 3
    game.fireworks["Y"] = 2
    game.hands[0][0] = Card("R", 5)
    game.moves = game.gen_moves()

    assert game.score() == 5
    game.play_str("p 0")

    assert game.life_tokens == 0
    assert game.ended()
    assert game.moves == []
    assert game.score() == 0
    assert game.points() == 0
    assert game.points_for(0) == 0
    assert sum(game.fireworks.values()) == 5
    assert Card("R", 5) in game.discard


def test_discard_is_illegal_at_max_information_and_restores_a_token_otherwise():
    game = Hanabi(num_players=2, mode="mini")
    assert not any(move.startswith("d ") for move in game.moves)

    game.information_tokens = 2
    game.moves = game.gen_moves()
    discarded = game.hands[0][0]
    game.play_str("d 0")

    assert game.information_tokens == 3
    assert discarded in game.discard


def test_playing_five_restores_information_token():
    game = Hanabi(num_players=2, mode="full")
    game.fireworks["R"] = 4
    game.information_tokens = 7
    game.hands[0][0] = Card("R", 5)
    game.moves = game.gen_moves()

    game.play_str("p 0")

    assert game.fireworks["R"] == 5
    assert game.information_tokens == 8


def test_last_draw_gives_every_player_one_final_turn_including_drawer():
    game = Hanabi(num_players=2, mode="mini")
    game.deck = [Card("R", 1)]
    game.information_tokens = 2
    game.moves = game.gen_moves()

    game.play_str("d 0")
    assert game.final_turns_left == 2
    assert not game.ended()

    game.play_str(next(move for move in game.moves if move.startswith("h ")))
    assert game.final_turns_left == 1
    assert not game.ended()

    game.play_str(next(move for move in game.moves if move.startswith("h ")))
    assert game.final_turns_left == 0
    assert game.ended()
    assert game.moves == []


def test_compact_moves_work_with_action_line_interface():
    game = Hanabi(num_players=2, mode="mini")
    state = game.display_with_moves()

    action_lines = [line[1:] for line in state.splitlines() if line.startswith("@")]
    assert action_lines == game.moves
    assert all(":" not in move for move in game.moves)
    assert all(move.split()[0] in {"p", "d", "h"} for move in game.moves)


def test_randomized_copy_preserves_every_players_visible_information():
    random.seed(1234)
    game = Hanabi(num_players=4, mode="full")
    before_hands = [hand[:] for hand in game.hands]
    before_views = [game.display(force=player) for player in range(game.num_players)]
    before_deck_multiset = sorted(game.deck)

    copied = game.copy(randomize=True)

    assert copied.hands == before_hands
    assert [copied.display(force=player) for player in range(game.num_players)] == before_views
    assert sorted(copied.deck) == before_deck_multiset
    assert copied.discard == game.discard
    assert copied.fireworks == game.fireworks


def test_copy_is_independent():
    game = Hanabi(num_players=2, mode="full")
    copied = game.copy()

    copied.fireworks["R"] = 3
    copied.hands[0].pop()
    copied.knowledge[0][0].colors.clear()

    assert game.fireworks["R"] == 0
    assert len(game.hands[0]) == game.hand_size
    assert game.knowledge[0][0].colors


def test_random_playouts_terminate_for_both_modes_and_player_counts():
    for mode in ("full", "mini"):
        for num_players in range(2, 6):
            for seed in range(5):
                random.seed(seed)
                game = Hanabi(num_players=num_players, mode=mode)
                game.simulate_to_end()
                assert game.ended()
                assert game.moves == []
