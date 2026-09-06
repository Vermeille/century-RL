import asyncio
import random
from types import SimpleNamespace

import numpy as np
import torch

from boardrl.games import games_library
from explainability.thegame.omni_stopping import OWN_FEATURES,StopObservation,StopRule,StopHybrid,GreedyWithStop


TEXT='Round: 3, Action: 2\nPiles: 10 40 90 70\nCards: 30\nMem: abc\nHand: 11 43 55\nHand: 65 91\nDeck: 20 30 50 60\n@11->0\n@43->1\n@55->1\n@x'


def test_own_features_ignore_other_hand_draws_and_memory():
    a=StopObservation.parse(TEXT).features('43->1')
    b=StopObservation.parse(TEXT.replace('Hand: 65 91','Hand: 2 98').replace('Deck: 20 30 50 60','Deck: 3 4 5 6').replace('Mem: abc','Mem: xyz')).features('43->1')
    np.testing.assert_array_equal(a[:len(OWN_FEATURES)],b[:len(OWN_FEATURES)])
    assert not np.array_equal(a[len(OWN_FEATURES):],b[len(OWN_FEATURES):])


def test_features_use_distinct_cards_for_second_best():
    observation=StopObservation.parse(TEXT)
    row=observation.features('43->1')
    assert row[0] == 1
    assert row[1] == 3
    assert row[2] == 3
    assert row[5] == 30
    assert row[-1] == 3


def test_threshold_and_boolean_tree_decide_continue():
    features=np.array([[-10],[1],[2]],dtype=float)
    assert StopRule({'kind':'threshold','cost':1}).predict(features).tolist() == [True,True,False]
    tree={'index':0,'threshold':1,'left':{'continue':True},'right':{'continue':False}}
    assert StopRule({'kind':'tree','tree':tree}).predict(features).tolist() == [True,True,False]


def test_hybrid_only_changes_optional_stop_not_card_ranking():
    async def processor(text):
        return SimpleNamespace(policy=[torch.tensor([1.,3.,2.,4.])])
    game=SimpleNamespace(moves=['11->0','43->1','55->1','x'],display_with_moves=lambda:TEXT)
    original=StopHybrid(processor)
    play=StopHybrid(processor,StopRule({'kind':'constant','continue':True}))
    stop=StopHybrid(processor,StopRule({'kind':'constant','continue':False}))
    assert int(asyncio.run(original(game))[0].argmax()) == 3
    assert int(asyncio.run(play(game))[0].argmax()) == 1
    assert int(asyncio.run(stop(game))[0].argmax()) == 3


def test_stop_rule_cannot_skip_compulsory_cards():
    async def processor(text):
        return SimpleNamespace(policy=[torch.tensor([1.,3.,2.])])
    game=SimpleNamespace(moves=['11->0','43->1','55->1'],display_with_moves=lambda:TEXT.replace('\n@x',''))
    stop=StopHybrid(processor,StopRule({'kind':'constant','continue':False}))
    assert int(asyncio.run(stop(game))[0].argmax()) == 1


def test_phase_rule_boundaries_are_simple_integer_thresholds():
    x=np.zeros((6,18))
    x[:,0]=[3,3,5,5,2,8]
    x[:,5]=[61,62,3,4,84,0]
    assert StopRule({'kind':'phase','early_deck':61,'late_deck':3}).predict(x).tolist() == [True,False,True,False,True,False]


def test_practical_gate_works_without_omni_information():
    private=TEXT.replace('Hand: 65 91\n','').replace('Deck: 20 30 50 60\n','')
    for spec in [{'kind':'threshold','cost':3},{'kind':'phase','early_deck':60,'late_deck':0}]:
        policy=GreedyWithStop(StopRule(spec))
        assert policy.action(TEXT) == policy.action(private)


def test_greedy_gate_plays_the_cheap_extra_card():
    policy=GreedyWithStop(StopRule({'kind':'threshold','cost':3}))
    assert policy.action(TEXT) == 0


def test_own_only_gate_has_identical_free_and_omni_trajectories():
    policy=GreedyWithStop(StopRule({'kind':'threshold','cost':3}))
    previous_rng=random.getstate()
    try:
        for seed in range(3051000,3051004):
            random.seed(seed)
            omni=games_library('thegame,mode=omni').make_game(num_players=2)
            random.seed(seed)
            free=games_library('thegame,mode=free').make_game(num_players=2)
            while not omni.ended():
                assert not free.ended()
                assert omni.moves == free.moves
                a=policy.action(omni.display_with_moves())
                b=policy.action(free.display_with_moves())
                assert a == b
                omni.play_idx(a);free.play_idx(b)
            assert free.ended()
            assert omni.points() == free.points()
    finally:
        random.setstate(previous_rng)
