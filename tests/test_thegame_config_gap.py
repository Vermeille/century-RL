import random

import pytest

from boardrl.games.thegame.game import TheGame
from explainability.thegame.awkward_refinement import AwkwardRule
from explainability.thegame.compact_strategy import choose_move
from explainability.thegame.config_gap import (
    Awkward, Configuration, ExperimentalGame, Formula, Greedy, Observation,
    action, candidates, play, summary,
)
from explainability.thegame.semantic_tree import VisibleState


@pytest.mark.parametrize('mode', ['strict', 'free'])
@pytest.mark.parametrize('players', [1, 2, 5])
def test_default_variant_matches_production(mode, players):
    random.seed(510)
    original = TheGame(num_players=players, mode=mode)
    random.seed(510)
    variant = ExperimentalGame(Configuration(num_players=players, mode=mode))
    while not original.ended():
        assert original.display_with_moves() == variant.display_with_moves()
        assert original.moves == variant.moves
        index = action(Observation.from_game(variant), Greedy(), 3)
        original.play_idx(index)
        variant.play_idx(index)
    assert variant.ended()
    assert original.points() == variant.points()


def test_default_policies_match_previous_implementations():
    previous = AwkwardRule({'target': 'affordable', 'threshold': 20, 'budget': 3})
    for seed in range(10):
        random.seed(seed)
        game = ExperimentalGame()
        while not game.ended():
            observation = Observation.from_game(game)
            state = VisibleState.parse(game.display_with_moves())
            assert action(observation, Formula()) == choose_move(state).index
            assert action(observation, Awkward()) == previous.action(state)
            game.play_idx(action(observation, Formula()))


@pytest.mark.parametrize('piles', [2, 4, 6])
@pytest.mark.parametrize('minimum', [1, 3, 4])
def test_variant_rules_and_termination(piles, minimum):
    configuration = Configuration(50, piles, minimum, 3)
    game = ExperimentalGame(configuration)
    assert game.piles == [1]*(piles//2)+[50]*(piles//2)
    for _ in range(minimum):
        assert game.current_player() == 0
        game.play_idx(action(Observation.from_game(game), Greedy()))
    assert game.current_player() == 1
    game.deck = []
    assert game.min_actions() == 1
    copied = game.copy()
    copied.piles[0] = 30
    assert copied.configuration == configuration
    assert copied.piles != game.piles
    assert 2 <= play(configuration, Formula(), 175) <= 50


def test_correct_normalization_and_win_semantics():
    result = summary([50, 40], [40, 30], 50)
    assert result['win_rate'] == .5
    assert result['played_fraction'] == pytest.approx(43/48)
    assert result['gap_percentage_points'] == pytest.approx(1000/48)
    assert result['leftover_reduction'] == pytest.approx(10/15)


def test_policy_observation_excludes_hidden_state():
    game = ExperimentalGame(Configuration(mode='free'))
    before = Observation.from_game(game)
    game.deck.reverse()
    game.hands[1].reverse()
    game._played_cards = 123
    assert before == Observation.from_game(game)


def test_reverse_ten_on_all_pile_directions():
    state = Observation((30, 30, 30, 70, 70, 70), (20, 80),
                        ('20->0', '20->1', '20->2', '80->3', '80->4', '80->5'), 100)
    assert all(move.cost == -10 for move in candidates(state))
