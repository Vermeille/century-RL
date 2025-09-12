from boardrl.games.guessnumber.game import GuessNumber
from boardrl.games import games_library


def test_guessnumber_flow():
    g = GuessNumber(num_symbols=2, secret=42)
    assert g.current_player() == 0
    assert g.moves == ["A", "B"]

    g.play_str("A")
    assert g.current_player() == 1
    assert "0" in g.moves and "100" in g.moves

    g.play_str("41")
    assert not g.ended()
    assert g.round() == 1

    g.play_str("B")
    g.play_str("42")
    assert g.ended()
    assert g.round() == 2
    assert g.points_for(0) == -2
    assert g.points_for(1) == -2


def test_guessnumber_registered():
    assert "guessnumber" in games_library.registry
