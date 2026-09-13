"""Distill Strict into a tree over named, deterministic move generators.

Only the serialized player-visible observation is used for features/actions.
Model selection uses whole-game validation; final test games stay separate.
Requires scikit-learn for fitting, not for executing exported JSON trees.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from boardrl.games import games_library
from boardrl.games.strategies import one_hot
from boardrl.models import make
from boardrl.rollouts import Inference, RolloutRunner


@dataclass(frozen=True)
class VisibleState:
    piles: tuple[int, ...]
    hand: tuple[int, ...]
    deck_count: int
    action: int
    moves: tuple[str, ...]

    @classmethod
    def parse(cls, text):
        lines = text.splitlines()
        def field(prefix):
            return next(line[len(prefix):].strip() for line in lines if line.startswith(prefix))
        return cls(
            tuple(map(int, field('Piles:').split())),
            tuple(map(int, field('Hand:').split())),
            int(field('Cards:')),
            int(field('Round:').rsplit('Action:', 1)[1]),
            tuple(line[1:] for line in lines if line.startswith('@')),
        )


def placements(card, piles):
    costs = [card - top if pile < 2 else top - card for pile, top in enumerate(piles)]
    return [(pile, cost) for pile, cost in enumerate(costs) if cost >= 0 or cost == -10]


MOVE_FEATURES = (
    'cost', 'extra_cost', 'card_legal_piles', 'dead_hand_cards_after',
    'legal_hand_placements_after', 'cheapest_next_card', 'mean_hand_cost_after',
    'worst_hand_cost_after', 'two_card_cost', 'reverse_setup',
    'pile_separation_after', 'separation_change', 'pile_open_space_after',
    'card_hand_position',
)
CORE_ACTIONS = (
    'greedy', 'fewest_dead_cards', 'best_two_card_start', 'most_constrained_card',
    'cheapest_reverse_setup', 'easiest_remaining_hand', 'separation_within_5',
)
ACTION_NAMES = CORE_ACTIONS + tuple(f'cheapest_pile_{i}' for i in range(4)) + tuple(f'second_cheapest_pile_{i}' for i in range(4))
GLOBAL_FEATURES = ('deck_count', 'action_in_turn', 'hand_size', 'legal_moves', 'greedy_cost', 'minimum_cost_ties', 'ascending_gap', 'descending_gap')
FEATURE_NAMES = GLOBAL_FEATURES + tuple(f'{action}.{feature}' for action in ACTION_NAMES for feature in MOVE_FEATURES) + tuple(f'{action}.delta_{feature}' for action in ACTION_NAMES[1:] for feature in MOVE_FEATURES)
VOCABULARIES = {'core': 7, 'piles': 11, 'expanded': 15}


@dataclass
class CandidateSet:
    state: VisibleState
    move_features: np.ndarray
    indices: np.ndarray
    x: np.ndarray

    @classmethod
    def build(cls, state):
        decoded = [tuple(map(int, move.split('->'))) for move in state.moves]
        costs = np.array([card - state.piles[pile] if pile < 2 else state.piles[pile] - card for card, pile in decoded])
        minimum = int(costs.min())
        old_gap = abs(state.piles[0] - state.piles[1]) + abs(state.piles[2] - state.piles[3])
        rows = []
        for (card, pile), cost in zip(decoded, costs):
            after = list(state.piles)
            after[pile] = card
            remaining = [other for other in state.hand if other != card]
            options = [placements(other, after) for other in remaining]
            best = [min(c for _, c in choices) for choices in options if choices]
            dead = sum(not choices for choices in options)
            # 100 is an explicit unavailable sentinel, not a hidden-state rollout.
            next_cost = min(best) if best else (0 if not remaining else 100)
            gap = abs(after[0] - after[1]) + abs(after[2] - after[3])
            rows.append([
                cost, cost - minimum, len(placements(card, state.piles)), dead,
                sum(len(choices) for choices in options), next_cost,
                float(np.mean(best)) if best else next_cost, max(best) if best else next_cost,
                cost + next_cost,
                int(any(p == pile and c == -10 for choices in options for p, c in choices)),
                gap, gap - old_gap, 100 - card if pile < 2 else card - 1,
                state.hand.index(card),
            ])
        table = np.asarray(rows, dtype=np.float32)
        all_indices = list(range(len(decoded)))
        greedy = int(np.argmin(costs))
        def choose(pool, columns):
            return min(pool, key=lambda i: tuple(float(table[i, col]) for col in columns) + (i,)) if pool else greedy
        indices = [
            greedy,
            choose(all_indices, [3, 0]),
            choose(all_indices, [8, 0]),
            choose(all_indices, [2, 0]),
            choose([i for i in all_indices if table[i, 9]], [0]),
            choose(all_indices, [3, 6, 0]),
            min([i for i in all_indices if costs[i] <= minimum + 5], key=lambda i: (-table[i, 10], costs[i], i)),
        ]
        per_pile = [sorted([i for i, (_, p) in enumerate(decoded) if p == pile], key=lambda i: (costs[i], i)) for pile in range(4)]
        indices.extend(pool[0] if pool else greedy for pool in per_pile)
        indices.extend(pool[1] if len(pool) > 1 else greedy for pool in per_pile)
        indices = np.asarray(indices, dtype=np.int32)
        selected = table[indices]
        global_values = [state.deck_count, state.action, len(state.hand), len(decoded), minimum, sum(costs == minimum), abs(state.piles[0] - state.piles[1]), abs(state.piles[2] - state.piles[3])]
        x = np.concatenate([global_values, selected.ravel(), (selected[1:] - selected[0]).ravel()]).astype(np.float32)
        return cls(state, table, indices, x)

    def labels(self, teacher):
        exact = self.indices == teacher
        card, pile = map(int, self.state.moves[teacher].split('->'))
        symmetric = []
        for index in self.indices:
            other_card, other_pile = map(int, self.state.moves[index].split('->'))
            symmetric.append(other_card == card and other_pile // 2 == pile // 2 and self.state.piles[other_pile] == self.state.piles[pile])
        return exact, np.asarray(symmetric)


def feature_columns(count):
    names = set(ACTION_NAMES[:count])
    return [i for i, name in enumerate(FEATURE_NAMES) if '.' not in name or name.split('.')[0] in names]


class TreePolicy:
    def __init__(self, specification):
        self.specification = specification

    def candidate(self, x):
        node = self.specification['tree']
        while 'action' not in node:
            node = node['left'] if x[node['feature_index']] <= node['threshold'] else node['right']
        return node['action']

    async def __call__(self, game):
        candidates = CandidateSet.build(VisibleState.parse(game.display_with_moves()))
        action = int(candidates.indices[self.candidate(candidates.x)])
        return one_hot(action, len(game.moves)).log(), {}


class Teacher:
    def __init__(self, processor, records=None, game_ids=None):
        self.processor, self.records, self.game_ids = processor, records, game_ids

    async def __call__(self, game):
        text = game.display_with_moves()
        output = await self.processor(text)
        chosen = int(output.policy[0].argmax())
        if self.records is not None:
            candidates = CandidateSet.build(VisibleState.parse(text))
            exact, symmetric = candidates.labels(chosen)
            self.records.append((candidates.x, exact, symmetric, candidates.indices, chosen, candidates.move_features[chosen, 1] > 0, self.game_ids[id(game)]))
        return one_hot(chosen, len(game.moves)).log(), {}


def play(strategy, games, seed, batch_size=128, game_ids=None, mode='strict'):
    descriptor = games_library(f'thegame,mode={mode}')
    scores = []
    for start in range(0, games, batch_size):
        initial = []
        for i in range(start, min(start + batch_size, games)):
            random.seed(seed + i)
            initial.append(descriptor.make_game(num_players=2))
        cursor = 0
        def factory(num_players):
            nonlocal cursor
            game = copy.deepcopy(initial[cursor])
            if game_ids is not None:
                game_ids[id(game)] = start + cursor
            cursor += 1
            return game
        result = RolloutRunner(factory, progress=False, coop=True).play([strategy, strategy], games=len(initial), max_steps=800, rotate=False)
        scores.extend(float(game.by_seat[0][-1].my_points) for game in result)
    return np.asarray(scores)


def load_teacher(checkpoint):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = make(payload['metadata']['architecture'])
    model.load_state_dict(payload['models']['current'])
    model.cuda().eval()
    return Inference(model, batch_size=1024)


def collect(args):
    inference = load_teacher(args.checkpoint)
    records, game_ids = [], {}
    scores = play(Teacher(inference.processor, records, game_ids), args.games, args.seed, game_ids=game_ids)
    x, exact, symmetric, indices, chosen, nonlowest, episode = zip(*records)
    np.savez_compressed(args.data, x=x, exact=exact, symmetric=symmetric, indices=indices, chosen=chosen, nonlowest=nonlowest, episode=episode, scores=scores)
    print(json.dumps({'states':len(records), 'games':args.games, 'mean_score':float(scores.mean()), 'features':len(FEATURE_NAMES)}, indent=2), flush=True)


def export_node(tree, node, columns):
    if tree.children_left[node] == -1:
        values = tree.value[node, :, 0]
        return {'action':int(np.argmax(values)), 'name':ACTION_NAMES[int(np.argmax(values))], 'samples':int(tree.n_node_samples[node])}
    index = columns[tree.feature[node]]
    return {'feature_index':int(index), 'feature':FEATURE_NAMES[index], 'threshold':float(tree.threshold[node]), 'left':export_node(tree, tree.children_left[node], columns), 'right':export_node(tree, tree.children_right[node], columns)}


def predict(specification, x):
    policy = TreePolicy(specification)
    return np.asarray([policy.candidate(row) for row in x])


def fidelity(data, mask, prediction, count):
    exact = data['exact'][mask]
    symmetric = data['symmetric'][mask]
    nonlowest = data['nonlowest'][mask]
    correct = exact[np.arange(len(prediction)), prediction]
    return {
        'states':len(prediction), 'coverage':float(exact[:, :count].any(axis=1).mean()),
        'nongreedy_coverage':float(exact[nonlowest, :count].any(axis=1).mean()) if nonlowest.any() else None,
        'exact_agreement':float(correct.mean()),
        'symmetric_agreement':float(symmetric[np.arange(len(prediction)), prediction].mean()),
        'nongreedy_agreement':float(correct[nonlowest].mean()) if nonlowest.any() else None,
    }


def render_tree(node, indent=''):
    if 'action' in node:
        return indent + 'PLAY ' + node['name'] + '\n'
    return (indent + f"IF {node['feature']} <= {node['threshold']:.4g}:\n" + render_tree(node['left'], indent + '  ') + indent + 'ELSE:\n' + render_tree(node['right'], indent + '  '))


def fit(args):
    from sklearn.tree import DecisionTreeRegressor
    data = np.load(args.data)
    train = data['episode'] < args.train_games
    validation = (data['episode'] >= args.train_games) & (data['episode'] < args.train_games + args.validation_games)
    results = {}
    for vocabulary, count in VOCABULARIES.items():
        columns = feature_columns(count)
        for leaves in args.leaves:
            estimator = DecisionTreeRegressor(max_leaf_nodes=leaves, min_samples_leaf=args.min_leaf, random_state=17)
            estimator.fit(data['x'][train][:, columns], data['exact'][train, :count].astype(float))
            spec = {'vocabulary':vocabulary, 'candidate_count':count, 'leaves':int(estimator.get_n_leaves()), 'depth':int(estimator.get_depth()), 'tree':export_node(estimator.tree_, 0, columns)}
            predictions = predict(spec, data['x'][validation])
            spec['validation'] = fidelity(data, validation, predictions, count)
            name = f'{vocabulary}_{leaves}'
            results[name] = spec
            print(name, json.dumps(spec['validation']), flush=True)
    args.output.write_text(json.dumps({'models':results, 'features':FEATURE_NAMES, 'actions':ACTION_NAMES, 'train_games':args.train_games, 'validation_games':args.validation_games}, indent=2))


def score_summary(scores, baseline):
    delta = scores - baseline
    half = 1.96 * delta.std(ddof=1) / np.sqrt(len(delta))
    ordered = np.sort(scores)
    tail_count = .05 * len(scores)
    whole = int(tail_count)
    tail_mean = (ordered[:whole].sum() + (tail_count - whole) * ordered[whole]) / tail_count
    return {'games':len(scores), 'mean':float(scores.mean()), 'std':float(scores.std(ddof=1)), 'q05':float(np.quantile(scores,.05)), 'worst_5_percent_mean':float(tail_mean), 'win_rate':float(np.mean(scores==100)), 'delta_vs_teacher':float(delta.mean()), 'delta_ci95':[float(delta.mean()-half),float(delta.mean()+half)]}


def evaluate(args):
    from boardrl.games.thegame.strategies import LowestCostStrategy
    fitted = json.loads(args.models.read_text())
    data = np.load(args.data)
    test = data['episode'] >= fitted['train_games'] + fitted['validation_games']
    inference = load_teacher(args.checkpoint)
    teacher = play(Teacher(inference.processor), args.games, args.seed)
    results = {'teacher':{'scores':teacher.tolist(), 'summary':score_summary(teacher,teacher)}}
    lowest = play(LowestCostStrategy(), args.games, args.seed)
    results['lowest'] = {'scores':lowest.tolist(), 'summary':score_summary(lowest,teacher)}
    for name in args.selected:
        spec = fitted['models'][name]
        test_result = fidelity(data, test, predict(spec,data['x'][test]), spec['candidate_count'])
        scores = play(TreePolicy(spec),args.games,args.seed)
        results[name] = {'test':test_result, 'summary':score_summary(scores,teacher), 'scores':scores.tolist(), 'tree_text':render_tree(spec['tree'])}
        print(name, json.dumps(results[name]['summary']), json.dumps(test_result), flush=True)
    args.output.write_text(json.dumps(results,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['collect','fit','evaluate'])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--models', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--games', type=int, default=896)
    parser.add_argument('--seed', type=int, default=1840000)
    parser.add_argument('--train-games', type=int, default=512)
    parser.add_argument('--validation-games', type=int, default=128)
    parser.add_argument('--leaves', type=int, nargs='+', default=[4,8,16,32,64])
    parser.add_argument('--min-leaf', type=int, default=100)
    parser.add_argument('--selected', nargs='+', default=['expanded_8','expanded_32'])
    args = parser.parse_args()
    {'collect':collect, 'fit':fit, 'evaluate':evaluate}[args.stage](args)


if __name__ == '__main__':
    main()
