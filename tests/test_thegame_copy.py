import pytest

from boardrl.games.thegame.game import TheGame


def test_thegame_copy_deepcopy_and_randomize(monkeypatch):
    # Create a small game to make assertions simpler
    g = TheGame(num_players=3, max_value=30, messages=True)

    # Record references and snapshots for comparison
    orig_deck = g.deck[:]
    orig_piles = g.piles[:]
    orig_hands = [h[:] for h in g.hands]
    orig_last_msgs = g._last_messages[:]

    # 1) Plain deep copy (no randomization)
    g2 = g.copy(randomize=False)

    # Content equality
    assert g2.deck == orig_deck
    assert g2.piles == orig_piles
    assert g2.hands == orig_hands
    assert g2._last_messages == orig_last_msgs

    # Different objects (deep copy of containers)
    assert g2.deck is not g.deck
    assert g2.piles is not g.piles
    assert g2.hands is not g.hands
    assert all(h2 is not h for h2, h in zip(g2.hands, g.hands))
    assert g2._last_messages is not g._last_messages

    # Moves are consistent with regenerated legal moves
    assert g2.moves == g2.gen_moves()

    # Mutating original does not affect copy and vice versa
    g.deck.append(999)
    g.piles[0] += 1
    g.hands[0].append(123)
    g._last_messages[0] = "Z"

    assert g2.deck == orig_deck
    assert g2.piles == orig_piles
    assert g2.hands == orig_hands
    assert g2._last_messages == orig_last_msgs

    # 2) Randomized copy should shuffle the deck but keep other state identical
    def fake_shuffle(xs):
        xs.reverse()  # deterministic shuffle for test stability

    monkeypatch.setattr("random.shuffle", fake_shuffle)
    g3 = g.copy(randomize=True)

    # Deck order changed deterministically
    assert g3.deck == list(reversed(g.deck))

    # Other state deep-copied and equal
    assert g3.piles == g.piles
    assert g3.hands == g.hands
    assert g3._last_messages == g._last_messages

    # Containers are distinct objects
    assert g3.deck is not g.deck
    assert g3.piles is not g.piles
    assert g3.hands is not g.hands
    assert all(h3 is not h for h3, h in zip(g3.hands, g.hands))
    assert g3._last_messages is not g._last_messages

    # Moves remain consistent (length of deck unchanged)
    assert g3.moves == g3.gen_moves()

