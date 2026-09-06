import random

import numpy as np

from boardrl.games import games_library
from explainability.thegame.awkward_refinement import AwkwardRule, evaluate_rule, LowestRule
from explainability.thegame.human_rules import HumanRule, SPECS
from explainability.thegame.semantic_tree import VisibleState, placements, play
from boardrl.games.thegame.strategies import LowestCostStrategy


def test_direct_engine_scores_match_rollout_evaluation():
    expected = play(LowestCostStrategy(),16,2350000)
    actual = evaluate_rule(LowestRule(),16,2350000)
    np.testing.assert_array_equal(actual,expected)


def test_refined_rule_has_direct_rollout_parity():
    rule = AwkwardRule({'target':'affordable','threshold':20,'budget':3})
    np.testing.assert_array_equal(evaluate_rule(rule,16,2351000),play(rule,16,2351000))


def test_original_parameterization_exactly_matches_previous_rule():
    previous = HumanRule(SPECS['hardest_3'])
    refined = AwkwardRule({'target':'worst','threshold':10,'budget':3})
    for index in range(8):
        random.seed(2341000+index)
        game = games_library('thegame,mode=strict').make_game(num_players=2)
        while not game.ended():
            state = VisibleState.parse(game.display_with_moves())
            assert refined.action(state) == previous.action(state)
            game.play_idx(previous.action(state))


def test_affordable_variant_falls_through_to_a_reachable_target():
    piles,hand = (10,40,90,70),(11,43,58,85)
    moves = tuple(f'{card}->{pile}' for card in hand for pile,_ in placements(card,piles))
    state = VisibleState(piles,hand,40,0,moves)
    # 58 needs a jump of 12 on pile 3, whose cheapest starter is too expensive.
    # Next is 85, also unaffordable. The affordable target is therefore 43.
    rule = AwkwardRule({'target':'affordable','threshold':0,'budget':3})
    assert state.moves[rule.action(state)] == '43->1'
    strict = AwkwardRule({'target':'worst','threshold':0,'budget':3})
    assert state.moves[strict.action(state)] == '11->0'


def test_absolute_threshold_can_disable_intervention():
    piles,hand = (10,40,90,70),(11,43,55)
    moves = tuple(f'{card}->{pile}' for card in hand for pile,_ in placements(card,piles))
    state = VisibleState(piles,hand,40,0,moves)
    active = AwkwardRule({'target':'worst','threshold':10,'budget':3})
    inactive = AwkwardRule({'target':'worst','threshold':20,'budget':3})
    assert state.moves[active.action(state)] == '43->1'
    assert state.moves[inactive.action(state)] == '11->0'


def test_starter_need_not_be_assigned_to_target_pile():
    piles,hand = (79,46,78,63),(95,27,89,13,10,54)
    moves = tuple(f'{card}->{pile}' for card in hand for pile,_ in placements(card,piles))
    state = VisibleState(piles,hand,40,0,moves)
    legal = AwkwardRule({'target':'worst','threshold':0,'budget':3})
    assigned = AwkwardRule({'target':'worst','threshold':0,'budget':3,'starter':'assigned'})
    assert state.moves[legal.action(state)] == '54->3'
    assert state.moves[assigned.action(state)] == '54->1'


def test_reverse_tie_break_rescues_closest_endpoint():
    piles,hand = (20,90,75,60),(10,80)
    moves = tuple(f'{card}->{pile}' for card in hand for pile,_ in placements(card,piles))
    state = VisibleState(piles,hand,40,0,moves)
    first = AwkwardRule({'target':'worst','threshold':0,'budget':3})
    endpoint = AwkwardRule({'target':'worst','threshold':0,'budget':3,'reverse':'endpoint'})
    assert state.moves[first.action(state)] == '10->0'
    assert state.moves[endpoint.action(state)] == '80->1'
