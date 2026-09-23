import pytest

from boardrl.games.thegame.game import TheGame


def test_thegame_copy_deepcopy_and_randomize(monkeypatch):
    # Create a small game to make assertions simpler
    g = TheGame(num_players=3, max_value=30, mode="strict_message_before_draw")

    # Record references and snapshots for comparison
    orig_deck = g.deck[:]
    orig_piles = g.piles[:]
    orig_hands = [h[:] for h in g.hands]
    orig_last_msgs = g._last_messages[:]
    orig_played_cards = g._played_cards

    # 1) Plain deep copy (no randomization)
    g2 = g.copy(randomize=False)

    # Content equality
    assert g2.deck == orig_deck
    assert g2.piles == orig_piles
    assert g2.hands == orig_hands
    assert g2._last_messages == orig_last_msgs
    assert g2._played_cards == orig_played_cards

    # Different objects (deep copy of containers)
    assert g2.deck is not g.deck
    assert g2.piles is not g.piles
    assert g2.hands is not g.hands
    assert all(h2 is not h for h2, h in zip(g2.hands, g.hands))
    assert g2._last_messages is not g._last_messages

    # Moves are consistent with regenerated legal moves
    assert g2.moves == g2.gen_moves()

    # Mutating original does not affect copy and vice versa. Keep mutations
    # inside the valid one-byte game-state domain and don't manually invalidate
    # the current player's cached legal moves.
    g.deck.append(29)
    g.piles[0] += 1
    g.hands[1].append(23)
    g._last_messages[0] = "A"
    g._played_cards |= 1 << 24

    assert g2.deck == orig_deck
    assert g2.piles == orig_piles
    assert g2.hands == orig_hands
    assert g2._last_messages == orig_last_msgs
    assert g2._played_cards == orig_played_cards

    # 2) Randomized copy should shuffle the deck but keep other state identical
    def fake_shuffle(xs):
        xs.reverse()  # deterministic shuffle for test stability

    monkeypatch.setattr("random.shuffle", fake_shuffle)
    g3 = g.copy(randomize=True)

    # Deck order changed deterministically
    assert list(g3.deck) == list(reversed(g.deck))

    # Other state deep-copied and equal
    assert g3.piles == g.piles
    assert g3.hands == g.hands
    assert g3._last_messages == g._last_messages
    assert g3._played_cards == g._played_cards
    assert g3.played_cards_memory() == g.played_cards_memory()

    # Containers are distinct objects
    assert g3.deck is not g.deck
    assert g3.piles is not g.piles
    assert g3.hands is not g.hands
    assert all(h3 is not h for h3, h in zip(g3.hands, g.hands))
    assert g3._last_messages is not g._last_messages

    # Moves remain consistent (length of deck unchanged)
    assert g3.moves == g3.gen_moves()


def test_thegame_copy_preserves_message_only_turn():
    g = TheGame(num_players=2, mode="strict_message_before_draw")
    g.deck = [50, 51, 52]
    g.piles = [1, 1, 100, 100]
    g.hands[0] = [20, 21, 22]
    g.moves = g.gen_moves()

    g.play_str("20->0")
    g.play_str("21->0")

    g2 = g.copy()

    assert g.moves == list("ABCDEFGHIJ")
    assert g2.moves == list("ABCDEFGHIJ")
    assert g2.current_player() == 0
    assert g2.action == 2


def test_thegame_copy_preserves_after_draw_message_turn():
    g = TheGame(num_players=2, mode="strict_message_after_draw")
    g.deck = [50, 51, 52]
    g.piles = [1, 1, 100, 100]
    g.hands[0] = [20, 21, 22]
    g.moves = g.gen_moves()

    g.play_str("20->0")
    g.play_str("21->0")

    g2 = g.copy()

    assert len(g.deck) == 0
    assert g.moves == list("ABCDEFGHIJ")
    assert g2.moves == list("ABCDEFGHIJ")
    assert g2.current_player() == 0
    assert g2.action == 2