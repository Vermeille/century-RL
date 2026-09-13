"""Small rules for choosing which pile's cheapest card to play.

Consumes semantic_tree's corpus, fits only on training games, selects on
validation games, and exports pure JSON rules (no sklearn needed at play time).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from boardrl.games.strategies import one_hot
from explainability.thegame.semantic_tree import (
    CandidateSet, VisibleState, GLOBAL_FEATURES, MOVE_FEATURES, Teacher,
    load_teacher, play, score_summary,
)


PILE_FEATURES = tuple(MOVE_FEATURES) + tuple('delta_' + name for name in MOVE_FEATURES) + (
    'next_card_gap_same_pile', 'cost_fraction_of_open_space',
    'deck_count', 'action_in_turn', 'lowest_other_pile_cost',
    'cost_minus_lowest_other', 'cost_minus_mean_other',
)


def prepare(x):
    x = np.atleast_2d(x)
    n = len(x)
    actions = x[:, len(GLOBAL_FEATURES):len(GLOBAL_FEATURES) + 15 * len(MOVE_FEATURES)].reshape(n, 15, len(MOVE_FEATURES))
    table = actions[:, 7:11]
    relative = table - actions[:, :1]
    second_gap = actions[:, 11:15, 0] - table[:, :, 0]
    fraction = table[:, :, 0] / np.maximum(1, table[:, :, 12] + table[:, :, 0])
    other = np.stack([np.min(np.delete(table[:, :, 0], i, axis=1), axis=1) for i in range(4)], axis=1)
    other_mean = (table[:, :, 0].sum(axis=1, keepdims=True) - table[:, :, 0]) / 3
    extra = np.stack([second_gap, fraction, np.repeat(x[:, :1], 4, axis=1), np.repeat(x[:, 1:2], 4, axis=1), other, table[:, :, 0] - other, table[:, :, 0] - other_mean], axis=2)
    return np.concatenate([table, relative, extra], axis=2)


def choose(scores, features, indices):
    # Equal model scores fall back to lowest cost, then original legal order.
    return np.lexsort((indices, features[:, :, 0], -scores), axis=1)[:, 0]


def node_value(node, row):
    while 'value' not in node:
        node = node['left'] if row[node['index']] <= node['threshold'] else node['right']
    return node['value']


class PileRule:
    def __init__(self, specification):
        self.specification = specification

    def predict(self, features, indices):
        return choose(self.scores(features), features, indices)

    def scores(self, features):
        spec = self.specification
        methods = {'tree':self.tree_scores, 'linear':self.linear_scores, 'fraction':self.fraction_scores}
        return methods[spec['kind']](features)

    def tree_scores(self, features):
        return np.asarray([node_value(self.specification['tree'], row) for row in features.reshape(-1, features.shape[-1])]).reshape(features.shape[:2])

    def linear_scores(self, features):
        return -features[:, :, 0] - self.specification['weight'] * features[:, :, self.specification['feature_index']]

    def fraction_scores(self, features):
        before = np.maximum(1, features[:, :, 12] + features[:, :, 0])
        return -features[:, :, 0] / before ** self.specification['exponent']

    async def __call__(self, game):
        candidates = CandidateSet.build(VisibleState.parse(game.display_with_moves()))
        features = prepare(candidates.x)
        indices = candidates.indices[7:11][None, :]
        pile = self.predict(features, indices)[0]
        return one_hot(int(indices[0, pile]), len(game.moves)).log(), {}


def agreement(data, mask, rule, features):
    indices = data['indices'][mask, 7:11]
    pred = rule.predict(features[mask], indices)
    exact = data['exact'][mask, 7:11]
    symmetric = data['symmetric'][mask, 7:11]
    correct = exact[np.arange(len(pred)), pred]
    nongreedy = data['nonlowest'][mask]
    return {'states':len(pred), 'coverage':float(exact.any(axis=1).mean()), 'exact':float(correct.mean()), 'symmetric':float(symmetric[np.arange(len(pred)), pred].mean()), 'nongreedy':float(correct[nongreedy].mean())}


def export(tree, node=0):
    if tree.children_left[node] == -1:
        return {'value':float(tree.value[node, 0, 0]), 'samples':int(tree.n_node_samples[node])}
    index = int(tree.feature[node])
    return {'index':index, 'feature':PILE_FEATURES[index], 'threshold':float(tree.threshold[node]), 'left':export(tree,tree.children_left[node]), 'right':export(tree,tree.children_right[node])}


def tree_text(node, indent=''):
    if 'value' in node:
        return indent + f"score = {node['value']:.4f}\n"
    return indent + f"if {node['feature']} <= {node['threshold']:.4g}:\n" + tree_text(node['left'], indent+'  ') + indent + 'else:\n' + tree_text(node['right'], indent+'  ')


def fit(args):
    from sklearn.tree import DecisionTreeRegressor
    data = np.load(args.data)
    features = prepare(data['x'])
    train = data['episode'] < 512
    validation = (data['episode'] >= 512) & (data['episode'] < 640)
    models = {}
    for leaves in [4, 8, 16, 32, 64]:
        tree = DecisionTreeRegressor(max_leaf_nodes=leaves,min_samples_leaf=200,random_state=17)
        tree.fit(features[train].reshape(-1,len(PILE_FEATURES)), data['exact'][train,7:11].reshape(-1).astype(float))
        spec = {'kind':'tree','leaves':int(tree.get_n_leaves()),'tree':export(tree.tree_)}
        spec['validation'] = agreement(data,validation,PileRule(spec),features)
        spec['text'] = tree_text(spec['tree'])
        models[f'tree_{leaves}'] = spec
        print(f'tree_{leaves}', spec['validation'],flush=True)
    # Each family has just one fitted numerical parameter, selected on train.
    for feature in ['cheapest_next_card','mean_hand_cost_after','dead_hand_cards_after','legal_hand_placements_after','pile_separation_after','card_legal_piles','next_card_gap_same_pile']:
        index = PILE_FEATURES.index(feature)
        best = None
        for weight in np.linspace(-3,3,121):
            spec = {'kind':'linear','feature':feature,'feature_index':index,'weight':float(weight)}
            accuracy = agreement(data,train,PileRule(spec),features)['exact']
            if best is None or accuracy > best[0]:
                best = accuracy, spec
        spec = best[1]
        spec['train_accuracy'] = best[0]
        spec['validation'] = agreement(data,validation,PileRule(spec),features)
        models['linear_' + feature] = spec
        print('linear_' + feature,spec,flush=True)
    best = None
    for exponent in np.linspace(-2,2,161):
        spec = {'kind':'fraction','exponent':float(exponent)}
        accuracy = agreement(data,train,PileRule(spec),features)['exact']
        if best is None or accuracy > best[0]:
            best = accuracy,spec
    spec = best[1]
    spec['train_accuracy'] = best[0]
    spec['validation'] = agreement(data,validation,PileRule(spec),features)
    models['fraction'] = spec
    print('fraction',spec,flush=True)
    args.output.write_text(json.dumps({'models':models,'features':PILE_FEATURES},indent=2))


def evaluate(args):
    from boardrl.games.thegame.strategies import LowestCostStrategy
    from explainability.thegame.sparse_rule import SparseRule
    from explainability.thegame.compact_strategy import CompactStrategy
    constructors = {'tree':PileRule, 'linear':PileRule, 'fraction':PileRule, 'sparse':SparseRule}
    specs = {}
    for path in args.models:
        specs.update(json.loads(path.read_text())['models'])
    data = np.load(args.data)
    features = prepare(data['x'])
    test = data['episode'] >= 640
    if args.baseline_results:
        previous = json.loads(args.baseline_results.read_text())
        assert previous['protocol']['games'] == args.games
        assert previous['protocol']['seed'] == args.seed
        assert previous['protocol']['checkpoint'] == str(args.checkpoint.resolve())
        teacher = np.asarray(previous['teacher']['scores'])
        lowest = np.asarray(previous['lowest']['scores'])
    else:
        inference = load_teacher(args.checkpoint)
        teacher = play(Teacher(inference.processor),args.games,args.seed)
        lowest = play(LowestCostStrategy(),args.games,args.seed)
    results = {'protocol':{'checkpoint':str(args.checkpoint.resolve()), 'games':args.games, 'seed':args.seed, 'test_episode_start':640, 'selected':args.selected, 'mode':'strict', 'policy':'argmax', 'deal_seeding':'Python random.seed(seed + game_index) before constructing each initial game'}, 'teacher':{'scores':teacher.tolist(),'summary':score_summary(teacher,teacher)},'lowest':{'scores':lowest.tolist(),'summary':score_summary(lowest,teacher)}}
    print('teacher',results['teacher']['summary'],flush=True)
    print('lowest',results['lowest']['summary'],flush=True)
    args.output.write_text(json.dumps(results,indent=2))
    for name in args.selected:
        rule = constructors[specs[name]['kind']](specs[name])
        # Evaluate the small standalone implementation for the rounded rule,
        # not just the generic fitted-rule executor.
        strategy = CompactStrategy() if name == 'rounded_4' else rule
        scores = play(strategy,args.games,args.seed)
        results[name] = {'specification':specs[name], 'test':agreement(data,test,rule,features), 'scores':scores.tolist(), 'summary':score_summary(scores,teacher)}
        print(name,results[name]['test'],results[name]['summary'],flush=True)
        args.output.write_text(json.dumps(results,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['fit','evaluate'])
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--models',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--baseline-results',type=Path)
    parser.add_argument('--games',type=int,default=1024)
    parser.add_argument('--seed',type=int,default=1940000)
    parser.add_argument('--selected',nargs='+',default=['tree_8','tree_16','fraction'])
    args=parser.parse_args()
    {'fit':fit,'evaluate':evaluate}[args.stage](args)


if __name__ == '__main__':
    main()
