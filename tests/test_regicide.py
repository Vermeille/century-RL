import random

from boardrl.games import games_library
from boardrl.games.regicide.game import ATTACK, DEFEND, Card, JOKER, Regicide


def _refresh(game):
    game._lost = False
    game._won = False
    game.moves = game.gen_moves()


def test_regicide_registered_and_setup():
    desc = games_library("regicide,num_players=3")
    game = desc.make_game()

    assert game.num_players == 3
    assert [len(hand) for hand in game.hands] == [6, 6, 6]
    assert game.enemy.rank == "J"
    assert len(game.castle) == 11
    assert len(game.tavern) == 23
    assert desc.coop


def test_ace_association_and_combo_moves():
    game = Regicide(2)
    game.hands[0] = [
        Card("A", "C"),
        Card("8", "D"),
        Card("2", "H"),
        Card("2", "D"),
        Card("2", "C"),
    ]
    game.curplay = 0
    game.phase = ATTACK
    _refresh(game)

    assert "AC+8D" in game.moves
    assert "2H+2D" in game.moves
    assert "2H+2D+2C" in game.moves


def test_club_doubles_damage_but_enemy_suit_is_immune():
    game = Regicide(2)
    game.castle = [Card("J", "D")]
    game.hands[0] = [Card("10", "C"), Card("10", "H")]
    game.hands[1] = [Card("10", "H")]
    game.curplay = 0
    game.phase = ATTACK

    game.enemy = Card("J", "H")
    _refresh(game)
    game.play_str("10C")
    assert game.defeated_hp == 20

    game = Regicide(2)
    game.enemy = Card("J", "C")
    game.hands[0] = [Card("10", "C"), Card("10", "H")]
    game.hands[1] = [Card("10", "H")]
    game.curplay = 0
    game.phase = ATTACK
    game.enemy_damage = 0
    game.spade_shield = 0
    game.immunity_lifted = False
    game.battle_cards = []
    game.discard = []
    _refresh(game)

    game.play_str("10C")
    assert game.enemy_damage == 10
    assert game.phase == DEFEND


def test_perfect_execution_puts_enemy_on_top_of_tavern():
    game = Regicide(2)
    enemy = Card("J", "H")
    next_enemy = Card("J", "D")
    game.enemy = enemy
    game.castle = [next_enemy]
    game.hands[0] = [Card("10", "C"), Card("2", "H")]
    game.hands[1] = [Card("3", "H")]
    game.enemy_damage = 0
    game.spade_shield = 0
    game.immunity_lifted = False
    game.battle_cards = []
    game.discard = []
    game.phase = ATTACK
    _refresh(game)

    game.play_str("10C")

    assert game.enemy == next_enemy
    assert game.tavern[-1] == enemy
    assert game.known_tavern_prefix[0] == enemy


def test_spade_shields_become_retroactive_after_joker():
    game = Regicide(2)
    game.enemy = Card("J", "S")
    game.castle = [Card("J", "H")]
    game.hands[0] = [Card("5", "S"), Card("10", "H")]
    game.hands[1] = [JOKER, Card("2", "H")]
    game.enemy_damage = 0
    game.spade_shield = 0
    game.immunity_lifted = False
    game.battle_cards = []
    game.discard = []
    game.curplay = 0
    game.phase = ATTACK
    _refresh(game)

    game.play_str("5S")
    assert game.effective_enemy_attack() == 10
    game.play_str("10H")

    assert game.current_player() == 1
    game.play_str("joker:P1")

    assert game.immunity_lifted
    assert game.effective_enemy_attack() == 5


def test_defense_is_one_atomic_subset_action():
    game = Regicide(2)
    game.enemy = Card("J", "H")
    game.hands[0] = [Card("3", "D"), Card("7", "S"), Card("9", "H")]
    game.hands[1] = [Card("2", "D")]
    game.curplay = 0
    game.phase = DEFEND
    game.defense_remaining = 10
    _refresh(game)

    assert "3D+7S" in game.moves
    assert "3D+9H" in game.moves
    assert "3D" not in game.moves
    assert "7S" not in game.moves

    game.play_str("3D+7S")
    assert game.phase == ATTACK
    assert game.current_player() == 1
    assert Card("3", "D") in game.discard
    assert Card("7", "S") in game.discard


def test_pass_is_forbidden_after_every_other_player_just_passed():
    game = Regicide(2)
    game.enemy = Card("J", "H")
    game.spade_shield = 10
    game.hands[0] = [Card("2", "D")]
    game.hands[1] = [Card("3", "D")]
    game.curplay = 0
    game.phase = ATTACK
    game.consecutive_passes = 0
    _refresh(game)

    assert "pass" in game.moves
    game.play_str("pass")

    assert game.current_player() == 1
    assert "pass" not in game.moves


def test_randomized_copy_preserves_all_hands_and_public_state():
    random.seed(1234)
    game = Regicide(4)
    original_hands = [hand[:] for hand in game.hands]
    original_tavern_cards = sorted(card.code for card in game.tavern)

    copied = game.copy(randomize=True)

    assert copied is not game
    assert copied.hands == original_hands
    assert game.hands == original_hands
    assert copied.enemy == game.enemy
    assert copied.discard == game.discard
    assert copied.battle_cards == game.battle_cards
    assert sorted(card.code for card in copied.tavern) == original_tavern_cards
    assert len(copied.tavern) == len(game.tavern)


def test_randomized_copy_preserves_known_tavern_top():
    random.seed(4321)
    game = Regicide(2)
    known = Card("J", "H")
    game.tavern.append(known)
    game.known_tavern_prefix = [known]

    copied = game.copy(randomize=True)

    assert copied.tavern[-1] == known
    assert copied.known_tavern_prefix == [known]


def test_random_playouts_terminate():
    for players in range(1, 5):
        for seed in range(10):
            random.seed(seed)
            game = Regicide(players)
            game.simulate_to_end()
            assert game.ended()
