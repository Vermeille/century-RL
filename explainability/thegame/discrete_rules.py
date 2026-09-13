"""Boolean replacement rules, with no weighted scoring at execution time.

Start with the cheapest move and scan the remaining cheapest-per-pile moves
in increasing cost order. A small Boolean tree decides whether to replace the
current choice. Training targets may be the neural policy or the compact
formula, but exported policies contain only comparisons and Boolean leaves.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from boardrl.games.strategies import one_hot
from explainability.thegame.semantic_tree import CandidateSet, VisibleState, Teacher, load_teacher, play, score_summary
from explainability.thegame.pile_tree import prepare, agreement
from explainability.thegame.sparse_rule import SparseRule
from explainability.thegame.compact_strategy import CompactStrategy


FEATURE_NAMES = (
    'extra_cost', 'hardest_card_improvement', 'pile_gap_improvement',
    'challenger_space', 'incumbent_space', 'extra_space',
    'challenger_cost', 'incumbent_cost', 'extra_dead_cards',
    'extra_legal_piles_for_card', 'fraction_consumed_reduction',
    'challenger_hardest_card', 'incumbent_hardest_card',
)


def pair_features(challenger, incumbent):
    c, b = challenger, incumbent
    cs = c[:, 12] + c[:, 0]
    bs = b[:, 12] + b[:, 0]
    return np.column_stack([
        c[:, 0] - b[:, 0], b[:, 7] - c[:, 7], c[:, 11] - b[:, 11],
        cs, bs, cs - bs, c[:, 0], b[:, 0], c[:, 3] - b[:, 3],
        c[:, 2] - b[:, 2], b[:, 29] - c[:, 29], c[:, 7], b[:, 7],
    ])


def apply_tree(node, x):
    if 'replace' in node:
        return np.full(len(x), node['replace'], dtype=bool)
    left = x[:, node['index']] <= node['threshold']
    result = np.empty(len(x), dtype=bool)
    result[left] = apply_tree(node['left'], x[left])
    result[~left] = apply_tree(node['right'], x[~left])
    return result


def export(tree, node=0):
    if tree.children_left[node] == -1:
        return {'replace':bool(np.argmax(tree.value[node, 0]) == 1)}
    index = int(tree.feature[node])
    return {'index':index, 'feature':FEATURE_NAMES[index],
            'threshold':float(tree.threshold[node]),
            'left':export(tree, tree.children_left[node]),
            'right':export(tree, tree.children_right[node])}


def render(node, indent=''):
    if 'replace' in node:
        return indent + ('REPLACE' if node['replace'] else 'KEEP') + '\n'
    return (indent + f"if {node['feature']} <= {node['threshold']:.4g}:\n"
            + render(node['left'], indent + '    ') + indent + 'else:\n'
            + render(node['right'], indent + '    '))


class DiscreteRule:
    def __init__(self, specification):
        self.specification = specification

    def predict(self, features, indices):
        order = np.lexsort((indices, features[:, :, 0]), axis=1)
        incumbent = order[:, 0].copy()
        rows = np.arange(len(features))
        for rank in range(1, 4):
            challenger = order[:, rank]
            pairs = pair_features(features[rows, challenger], features[rows, incumbent])
            replace = apply_tree(self.specification['tree'], pairs)
            replace &= indices[rows, challenger] != indices[rows, incumbent]
            incumbent[replace] = challenger[replace]
        return incumbent

    async def __call__(self, game):
        candidates = CandidateSet.build(VisibleState.parse(game.display_with_moves()))
        features = prepare(candidates.x)
        indices = candidates.indices[7:11][None, :]
        column = self.predict(features, indices)[0]
        return one_hot(int(indices[0, column]), len(game.moves)).log(), {}


def compact_labels(features, indices):
    rule = SparseRule({'columns':[0,6,12,16], 'weights':[1,.25,-.2,8]})
    return rule.scores(features)


def fit(args):
    from sklearn.tree import DecisionTreeClassifier
    data = np.load(args.data)
    features = prepare(data['x'])
    indices = data['indices'][:, 7:11]
    train = data['episode'] < 512
    validation = (data['episode'] >= 512) & (data['episode'] < 640)
    compact = compact_labels(features, indices)
    order = np.lexsort((indices, features[:, :, 0]), axis=1)
    rows = np.arange(len(features))
    models = {}
    for target in ['compact','neural']:
        xs, ys = [], []
        for low in range(3):
            for high in range(low+1,4):
                a, b = order[:, high], order[:, low]
                valid = train & (indices[rows,a] != indices[rows,b])
                if target == 'neural':
                    valid &= data['exact'][rows,7+a] | data['exact'][rows,7+b]
                    labels = data['exact'][rows,7+a]
                else:
                    labels = compact[rows,a] > compact[rows,b]
                xs.append(pair_features(features[rows,a],features[rows,b])[valid])
                ys.append(labels[valid])
        x, y = np.concatenate(xs), np.concatenate(ys)
        for leaves in [3,4,6,8,12]:
            for positive_weight in [1,2]:
                tree = DecisionTreeClassifier(max_leaf_nodes=leaves,min_samples_leaf=200,
                                              class_weight={0:1,1:positive_weight},random_state=24)
                tree.fit(x,y)
                spec = {'kind':'boolean_tree','target':target,'leaves':int(tree.get_n_leaves()),
                        'positive_weight':positive_weight,'tree':export(tree.tree_)}
                rule = DiscreteRule(spec)
                prediction = rule.predict(features[validation],indices[validation])
                expected = indices[validation,np.argmax(compact[validation],axis=1)]
                actual = indices[validation,prediction]
                spec['compact_validation_agreement'] = float(np.mean(expected == actual))
                spec['neural_validation'] = agreement(data,validation,rule,features)
                spec['text'] = render(spec['tree'])
                name = f'{target}_{leaves}_w{positive_weight}'
                models[name] = spec
                print(name, spec['compact_validation_agreement'], spec['neural_validation'],flush=True)
        args.output.write_text(json.dumps({'models':models,'features':FEATURE_NAMES},indent=2))


def evaluate(args):
    from boardrl.games.thegame.strategies import LowestCostStrategy
    from explainability.thegame.human_rules import HumanRule, SPECS
    specs = json.loads(args.models.read_text())['models']
    specs.update(SPECS)
    constructors = {'human':HumanRule,'boolean_tree':DiscreteRule}
    results = {'protocol':{'seed':args.seed,'games':args.games,'selected':args.selected,
                          'checkpoint':str(args.checkpoint.resolve()) if args.checkpoint else None,
                          'stage':'development' if args.development else 'confirmation'}}
    lowest = play(LowestCostStrategy(),args.games,args.seed)
    compact = play(CompactStrategy(),args.games,args.seed)
    teacher = None
    if args.checkpoint:
        inference = load_teacher(args.checkpoint)
        teacher = play(Teacher(inference.processor),args.games,args.seed)
    baseline = compact if teacher is None else teacher
    results['lowest'] = {'scores':lowest.tolist(),'summary':score_summary(lowest,baseline)}
    results['compact'] = {'scores':compact.tolist(),'summary':score_summary(compact,baseline)}
    if teacher is not None:
        results['teacher'] = {'scores':teacher.tolist(),'summary':score_summary(teacher,teacher)}
    results['protocol']['difference_reference'] = 'compact' if teacher is None else 'teacher'
    # score_summary retains historical key names; mark the reference explicitly.
    print('lowest',results['lowest']['summary'],flush=True)
    print('compact',results['compact']['summary'],flush=True)
    if teacher is not None:
        print('teacher',results['teacher']['summary'],flush=True)
    for name in args.selected:
        scores = play(constructors[specs[name]['kind']](specs[name]),args.games,args.seed)
        results[name] = {'specification':specs[name],'scores':scores.tolist(), 'summary':score_summary(scores,baseline)}
        print(name,results[name]['summary'],flush=True)
        args.output.write_text(json.dumps(results,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['fit','evaluate'])
    parser.add_argument('--data',type=Path)
    parser.add_argument('--models',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--selected',nargs='+')
    parser.add_argument('--games',type=int,default=256)
    parser.add_argument('--seed',type=int,default=2040000)
    parser.add_argument('--development',action='store_true')
    args=parser.parse_args()
    {'fit':fit,'evaluate':evaluate}[args.stage](args)


if __name__ == '__main__':
    main()
