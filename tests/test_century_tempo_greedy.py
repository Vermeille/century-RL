import asyncio

from boardrl.games import games_library
from boardrl.games.century.strategies import TempoGreedyStrategy, _stock_value


def _century():
    return games_library("century").make_game()


def test_weighted_stock_value_uses_century_stock_object():
    g = _century()
    assert _stock_value(g.get_player(0).stock) == 3
    assert _stock_value(g.get_player(1).stock) == 4


def test_tempo_greedy_returns_a_deterministic_legal_move():
    g = _century()
    policy, info = asyncio.run(TempoGreedyStrategy()(g))

    probs = policy.exp()
    assert probs.sum().item() == 1.0
    assert (probs == 1).sum().item() == 1
    assert set(info["scores"]) == set(g.moves)
