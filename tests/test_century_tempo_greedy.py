import asyncio
import inspect

from boardrl.games import games_library
from boardrl.games.century.strategies import TempoGreedyStrategy


def _century():
    return games_library("century").make_game()


def test_preview_stock_does_not_mutate_game():
    g = _century()
    move = next(move for move in g.moves if move.startswith("H"))
    player = g.get_player(g.current_player())
    before_stock = player.stock.to_str()
    before_moves = list(g.moves)
    before_actions = [str(card) for card, _ in g.action.visible()]
    before_victories = [str(card) for card in g.visible_victory()]

    preview = g.preview_stock(move)

    assert preview.to_str() != before_stock
    assert player.stock.to_str() == before_stock
    assert g.moves == before_moves
    assert [str(card) for card, _ in g.action.visible()] == before_actions
    assert [str(card) for card in g.visible_victory()] == before_victories


def test_safe_accessors_only_expose_visible_or_owned_information():
    g = _century()
    player = g.get_player(g.current_player())

    assert len(g.visible_victory()) <= 5
    assert g.goal_card_count() == 6
    assert player.victory_count() == 0
    assert player.discard_count() == 0


def test_tempo_greedy_never_simulates_full_game_transition():
    source = inspect.getsource(TempoGreedyStrategy)
    assert ".play_str(" not in source
    assert ".copy(" not in source
    assert ".display(" not in source


def test_tempo_greedy_returns_a_deterministic_legal_move():
    g = _century()
    policy, info = asyncio.run(TempoGreedyStrategy()(g))

    probs = policy.exp()
    assert probs.sum().item() == 1.0
    assert (probs == 1).sum().item() == 1
    assert set(info["scores"]) == set(g.moves)
