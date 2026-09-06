import numpy as np
import random

from explainability.thegame.semantic_tree import (
    ACTION_NAMES, FEATURE_NAMES, CandidateSet, TreePolicy, VisibleState, placements,
)
from explainability.thegame.pile_tree import PILE_FEATURES, PileRule, choose, prepare
from explainability.thegame.sparse_rule import NAMES, SparseRule, engineer
from explainability.thegame.compact_strategy import adjusted_cost, cheapest_per_pile, choose_move
from boardrl.games import games_library


def sample():
    piles = (10, 30, 90, 70)
    hand = (12, 20, 40, 68, 80)
    moves = tuple(f'{card}->{pile}' for card in hand for pile, _ in placements(card, piles))
    return VisibleState(piles, hand, 40, 0, moves)


def test_candidates_are_legal_and_greedy_is_minimum():
    candidates = CandidateSet.build(sample())
    assert len(candidates.indices) == len(ACTION_NAMES)
    assert candidates.x.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(candidates.x).all()
    assert all(0 <= i < len(candidates.state.moves) for i in candidates.indices)
    assert candidates.move_features[candidates.indices[0], 0] == candidates.move_features[:, 0].min()


def test_per_pile_generators_are_ordered_by_actual_cost():
    candidates = CandidateSet.build(sample())
    for pile in range(4):
        pool = [i for i, move in enumerate(candidates.state.moves) if move.endswith(f'->{pile}')]
        ordered = sorted(pool, key=lambda i: (candidates.move_features[i, 0], i))
        assert candidates.indices[7 + pile] == ordered[0]
        assert candidates.indices[11 + pile] == ordered[1]


def test_aliases_are_all_accepted():
    candidates = CandidateSet.build(sample())
    teacher = candidates.indices[0]
    exact, symmetric = candidates.labels(teacher)
    assert np.array_equal(exact, candidates.indices == teacher)
    assert np.all(symmetric[exact])


def test_visible_parser_ignores_memory_and_other_hands():
    text = 'Round: 3, Action: 1\nPiles: 10 30 90 70\nCards: 40\nMem: xxx\nHand: 12 20\nHand: 99 98\n@12->0\n@20->0\n'
    first = VisibleState.parse(text)
    second = VisibleState.parse(text.replace('Mem: xxx', 'Mem: yyy').replace('Hand: 99 98', 'Hand: 2 3'))
    assert first == second
    assert first.hand == (12, 20)
    assert first.action == 1
    assert np.array_equal(CandidateSet.build(first).x, CandidateSet.build(second).x)


def test_json_tree_uses_expected_branch():
    specification = {'tree': {'feature_index': 0, 'threshold': 10, 'left': {'action': 0}, 'right': {'action': 2}}}
    policy = TreePolicy(specification)
    assert policy.candidate(np.array([10])) == 0
    assert policy.candidate(np.array([11])) == 2


def test_pile_feature_dimensions_and_costs():
    candidates = CandidateSet.build(sample())
    features = prepare(candidates.x)
    assert features.shape == (1, 4, len(PILE_FEATURES))
    assert engineer(features).shape == (1, 4, len(NAMES))
    np.testing.assert_array_equal(features[0, :, 0], candidates.move_features[candidates.indices[7:11], 0])


def test_pile_ties_use_cost_then_original_move_order():
    features = np.zeros((1, 4, len(PILE_FEATURES)))
    features[0, :, 0] = [4, 3, 3, 5]
    indices = np.array([[0, 8, 2, 1]])
    assert choose(np.zeros((1, 4)), features, indices).tolist() == [2]
    assert choose(np.array([[0, 1, 0, 0]]), features, indices).tolist() == [1]


def test_pile_json_tree_and_sparse_score():
    candidates = CandidateSet.build(sample())
    features = prepare(candidates.x)
    tree = PileRule({'kind':'tree', 'tree':{'index':0,'threshold':0,'left':{'value':1},'right':{'value':0}}})
    np.testing.assert_array_equal(tree.scores(features), features[:, :, 0] <= 0)
    sparse = SparseRule({'columns':[0, 6], 'weights':[1, .25]})
    np.testing.assert_allclose(sparse.scores(features), -features[:, :, 0] - .25 * features[:, :, 7])


def test_absent_piles_fall_back_to_legal_greedy():
    state = VisibleState((95, 99, 3, 2), (96,), 0, 0, ('96->0',))
    candidates = CandidateSet.build(state)
    assert candidates.indices.tolist() == [0] * len(ACTION_NAMES)
    assert np.isfinite(prepare(candidates.x)).all()


def test_compact_formula_matches_exported_rule_on_reachable_states():
    random.seed(8141)
    rule = SparseRule({'columns':[0, 6, 12, 16], 'weights':[1, .25, -.2, 8]})
    for _ in range(4):
        game = games_library('thegame,mode=strict').make_game(num_players=2)
        while not game.ended():
            state = VisibleState.parse(game.display_with_moves())
            candidates = CandidateSet.build(state)
            features = prepare(candidates.x)
            indices = candidates.indices[7:11][None, :]
            scores = rule.scores(features)[0]
            for move in cheapest_per_pile(state):
                column = np.flatnonzero(indices[0] == move.index)[0]
                np.testing.assert_allclose(adjusted_cost(state, move), -scores[column], atol=1e-5)
            selected = choose_move(state)
            generic = indices[0, rule.predict(features, indices)[0]]
            assert selected.index == generic
            game.play_idx(selected.index)
