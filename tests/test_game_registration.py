import pytest

from boardrl.games import games_library
from boardrl.games.semantics import CooperativeOutcome, PointDeltaRewards


def test_configurable_game_keeps_constructor_arguments():
    desc = games_library("nim,num_stones=5,max_pick=2")
    game = desc.make_game()

    assert game.num_stones == 5
    assert game.max_pick == 2
    assert desc.make_metrics.__module__ == "boardrl.games.nim.metrics"
    assert "optimal" in desc.strategy_from_string.registry
    assert "random" in desc.strategy_from_string.registry


def test_fixed_registration_does_not_expose_internal_constructor_arguments():
    with pytest.raises(ValueError, match="Unknown argument.*num_players"):
        games_library("tictactoe,num_players=2")


def test_custom_strategy_registry_is_fresh_for_each_descriptor():
    first = games_library("nim")
    first.strategy_from_string.registry["test-only"] = object()

    second = games_library("nim")

    assert "test-only" not in second.strategy_from_string.registry


def test_declarative_registration_preserves_game_semantics():
    desc = games_library("thegame,mode=free,max_value=30")
    game = desc.make_game(num_players=2)

    assert game.mode == "free"
    assert game.max_value == 30
    assert isinstance(desc.outcome, CooperativeOutcome)
    assert isinstance(desc.rewards, PointDeltaRewards)
    assert desc.rewards.scale == pytest.approx(0.1)
