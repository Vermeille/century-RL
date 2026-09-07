import asyncio

from boardrl.games import games_library
from boardrl.games.century.strategies import (
    TempoGreedyStrategy,
    _card_potential,
    _stock_value,
)


def _century():
    return games_library("century").make_game()


def test_weighted_value_and_card_potential_use_game_objects():
    g = _century()
    assert _stock_value(g.get_player(0).stock) == 3

    yy_rr = next(card for card in g.action.pile if str(card) == "YY>RR")
    yyy_rrr = next(card for card in g.action.pile if str(card) == "YYY>RRR")
    assert _card_potential(yy_rr) == 10.0
    assert _card_potential(yyy_rrr) == 9.0


def test_tempo_greedy_returns_a_deterministic_legal_move():
    g = _century()
    policy, info = asyncio.run(TempoGreedyStrategy()(g))

    probs = policy.exp()
    assert probs.sum().item() == 1.0
    assert (probs == 1).sum().item() == 1
    assert set(info["scores"]) == set(g.moves)
