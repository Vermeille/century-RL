import pytest

pytest.importorskip("pyspiel")

from boardrl.evaluation import SampledModelEvaluator
from boardrl.games import GameDesc, games_library
from boardrl.games.openspiel.game import OpenSpiel
from boardrl.games.openspiel.metrics import Metrics


def test_openspiel_requires_a_concrete_game():
    with pytest.raises(TypeError, match="abstract"):
        OpenSpiel("leduc_poker")
    assert "openspiel" not in games_library.registry


@pytest.mark.parametrize(
    "name",
    [
        "leduc", "liars_dice", "dark_hex", "abrupt_dark_hex",
        "phantom_ttt", "abrupt_phantom_ttt", "othello",
    ],
)
def test_registered_games_use_concrete_classes_and_keep_their_type_when_copied(name):
    descriptor = games_library(name)
    assert type(descriptor) is GameDesc
    assert descriptor.make_metrics is Metrics
    assert descriptor.make_game.__module__ == f"boardrl.games.openspiel.{name}.game"
    game = descriptor.make_game()
    original = game.display_with_moves()
    copied = game.copy()
    assert type(copied) is type(game)
    assert copied.display_with_moves() == original
    copied.play_idx(0)
    assert game.display_with_moves() == original


def test_othello_keeps_sampled_evaluation_and_public_observations():
    descriptor = games_library("othello")
    assert type(descriptor.make_model_evaluator(progress=False)) is SampledModelEvaluator
    game = descriptor.make_game()
    assert game.information_state is False
    assert game._game.get_type().short_name == "othello"
