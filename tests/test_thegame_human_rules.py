import random

import numpy as np

from boardrl.games import games_library
from explainability.thegame.semantic_tree import CandidateSet, VisibleState, placements
from explainability.thegame.pile_tree import prepare
from explainability.thegame.discrete_rules import DiscreteRule, apply_tree
from explainability.thegame.human_rules import HumanRule, SPECS


def state(piles, hand):
    moves = tuple(f'{card}->{pile}' for card in hand for pile,_ in placements(card,piles))
    return VisibleState(tuple(piles),tuple(hand),40,0,moves)


def test_reserve_rule_uses_small_fixed_budget():
    visible = state([10,40,65,75],[11,43,55])
    assert visible.moves[HumanRule(SPECS['reserve_1']).action(visible)] == '11->0'
    assert visible.moves[HumanRule(SPECS['reserve_2']).action(visible)] == '43->1'


def test_reserve_prefers_working_pile_for_one_extra_point():
    visible = state([10,40,65,75],[11,42,55])
    # No reverse jump: costs 1 on reserve pile 0 versus 2 on working pile 1.
    rule = HumanRule(SPECS['reserve_1'])
    assert visible.moves[rule.action(visible)] == '42->1'


def test_reverse_jump_is_never_sacrificed():
    visible = state([30,60,70,90],[20,61,80])
    for spec in SPECS.values():
        action = HumanRule(spec).action(visible)
        card,pile = map(int,visible.moves[action].split('->'))
        cost = card-visible.piles[pile] if pile < 2 else visible.piles[pile]-card
        assert cost == -10


def test_awkward_card_rule_advances_its_closest_pile():
    visible = state([10,40,90,70],[11,43,55])
    # 55 currently needs at least 15; advancing 40 to 43 helps it for only
    # two extra points compared with the globally cheapest 11 -> 10.
    assert visible.moves[HumanRule(SPECS['hardest_3']).action(visible)] == '43->1'


def test_awkward_card_rule_does_not_chase_an_unaffordable_pile():
    visible = state([10,40,90,70],[11,46,55])
    assert visible.moves[HumanRule(SPECS['hardest_3']).action(visible)] == '11->0'


def test_boolean_tree_has_no_numerical_leaf_score():
    tree = {'index':0,'threshold':2.5,'left':{'replace':True},'right':{'replace':False}}
    x = np.array([[2],[3]],dtype=float)
    assert apply_tree(tree,x).tolist() == [True,False]


def test_keep_tree_matches_greedy():
    visible = state([10,40,65,75],[11,42,55])
    candidates = CandidateSet.build(visible)
    indices = candidates.indices[7:11][None,:]
    selected = DiscreteRule({'tree':{'replace':False}}).predict(prepare(candidates.x),indices)[0]
    assert indices[0,selected] == candidates.indices[0]


def test_human_rules_are_legal_on_full_games():
    previous_rng = random.getstate()
    try:
        for index,spec in enumerate(SPECS.values()):
            random.seed(90210+index)
            game = games_library('thegame,mode=strict').make_game(num_players=2)
            rule = HumanRule(spec)
            steps = 0
            while not game.ended():
                action = rule.action(VisibleState.parse(game.display_with_moves()))
                assert 0 <= action < len(game.moves)
                game.play_idx(action)
                steps += 1
                assert steps <= 98
    finally:
        random.setstate(previous_rng)
