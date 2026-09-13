"""Paired configuration sweep; experimental rules never change the production game.

Run screen first, freeze selected configurations, then confirm on fresh seeds.
Only current-hand/pile information enters any policy. Jump size stays ten.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from boardrl.games.thegame.game import TheGame


@dataclass(frozen=True)
class Configuration:
    max_value: int = 100
    num_piles: int = 4
    minimum: int = 2
    num_players: int = 2
    mode: str = 'strict'

    def key(self):
        return f'{self.mode}-v{self.max_value}-p{self.num_piles}-m{self.minimum}-n{self.num_players}'


class ExperimentalGame(TheGame):
    def __init__(self, configuration=Configuration()):
        self.configuration = configuration
        if configuration.num_piles < 2 or configuration.num_piles % 2:
            raise ValueError('Use an even pile count, half ascending and half descending')
        hand_size = 6 if configuration.num_players > 3 else 7
        if not 1 <= configuration.minimum <= hand_size:
            raise ValueError('Minimum must fit in a hand')
        if configuration.max_value - 2 < hand_size * configuration.num_players:
            raise ValueError('Deck must cover the initial deal')
        super().__init__(num_players=configuration.num_players,
                         max_value=configuration.max_value, mode=configuration.mode)
        half = configuration.num_piles // 2
        self.piles = [1] * half + [self.max_value] * half
        self.moves = self.gen_moves()

    def min_actions(self):
        return self.configuration.minimum if self.deck else 1

    def card_moves(self):
        half = len(self.piles) // 2
        return [f'{card}->{pile}' for card in self.hands[self.curplay]
                for pile, top in enumerate(self.piles)
                if (cost := (card-top if pile < half else top-card)) >= 0 or cost == -10]

    def copy(self, randomize=False):
        result = copy.deepcopy(self)
        if randomize:
            result.game_mode.randomize_hidden_state(result)
        return result


@dataclass(frozen=True)
class Observation:
    piles: tuple[int, ...]
    hand: tuple[int, ...]
    moves: tuple[str, ...]
    max_value: int

    @classmethod
    def from_game(cls, game):
        # Deliberately excludes every other hand, future draw, and played memory.
        return cls(tuple(game.piles), tuple(game.hands[game.current_player()]),
                   tuple(game.moves), game.max_value)

    def cost(self, card, pile):
        return card-self.piles[pile] if pile < len(self.piles)//2 else self.piles[pile]-card

    def placements(self, card):
        return [(p, c) for p in range(len(self.piles))
                if (c := self.cost(card, p)) >= 0 or c == -10]


@dataclass(frozen=True)
class Move:
    index: int
    card: int
    pile: int
    cost: int


def candidates(state):
    result = {}
    for index, text in enumerate(state.moves):
        if text == 'x':
            continue
        card, pile = map(int, text.split('->'))
        move = Move(index, card, pile, state.cost(card, pile))
        if pile not in result or move.cost < result[pile].cost:
            result[pile] = move
    return sorted(result.values(), key=lambda m: (m.cost, m.index))


class Greedy:
    def choose(self, state, moves):
        return moves[0]


class Awkward:
    def choose(self, state, moves):
        greedy = moves[0]
        if greedy.cost == -10:
            return greedy
        assigned = []
        for card in state.hand:
            options = state.placements(card)
            if options:
                cost = min(c for _, c in options)
                assigned.append((cost, {p for p, c in options if c == cost}))
        for cost, piles in sorted(assigned, key=lambda item: -item[0]):
            if cost < 20:
                break
            for move in moves:
                if move.cost <= greedy.cost+3 and move.pile in piles:
                    return move
        return greedy


class Formula:
    """Existing four-term surrogate, generalized to same-direction pile spread.

    For exactly four piles/max=100 this is the original formula, not a refit.
    Singleton orientation has zero spread; larger groups use max(top)-min(top).
    """
    def choose(self, state, moves):
        def key(move):
            piles = list(state.piles)
            piles[move.pile] = move.card
            remaining = tuple(card for card in state.hand if card != move.card)
            after = Observation(tuple(piles), remaining, (), state.max_value)
            costs = [min(c for _, c in options) for card in remaining
                     if (options := after.placements(card))]
            hardest = max(costs) if costs else (state.max_value if remaining else 0)
            half = len(piles)//2
            group = slice(0, half) if move.pile < half else slice(half, len(piles))
            before_group, after_group = state.piles[group], piles[group]
            gap_gain = max(after_group)-min(after_group)-max(before_group)+min(before_group)
            space = state.max_value-state.piles[move.pile] if move.pile < half else state.piles[move.pile]-1
            value = move.cost+hardest/4-gap_gain/5+8*move.cost/max(1, space)
            return value, move.cost, move.index
        return min(moves, key=key)


POLICIES = {'greedy': Greedy, 'awkward': Awkward, 'formula': Formula}


def action(state, policy, stop_cost=None):
    moves = candidates(state)
    if 'x' in state.moves and (not moves or stop_cost is None or moves[0].cost > stop_cost):
        return state.moves.index('x')
    return policy.choose(state, moves).index


def play(configuration, policy, seed, stop_cost=None):
    random.seed(seed)
    game = ExperimentalGame(configuration)
    steps = 0
    while not game.ended():
        game.play_idx(action(Observation.from_game(game), policy, stop_cost))
        steps += 1
        if steps > 2*configuration.max_value:
            raise RuntimeError('Game exceeded finite card/turn bound')
    return game.points()


def summary(scores, base, maximum):
    scores, base = np.asarray(scores), np.asarray(base)
    delta = scores-base
    sem = float(delta.std(ddof=1)/np.sqrt(len(delta)))
    return {'mean': float(scores.mean()), 'sd': float(scores.std(ddof=1)),
            'win_rate': float((scores == maximum).mean()),
            'played_fraction': float(((scores-2)/(maximum-2)).mean()),
            'gap_points': float(delta.mean()),
            'gap_percentage_points': float(100*delta.mean()/(maximum-2)),
            'gap_ci95': [float(delta.mean()-1.96*sem), float(delta.mean()+1.96*sem)],
            'leftover_reduction': (float(delta.mean()/(maximum-base.mean()))
                                   if base.mean() < maximum else None),
            'paired_win_fraction': float((delta > 0).mean()),
            'worst5_mean': float(np.sort(scores)[:max(1, int(np.ceil(len(scores)*.05)))].mean())}


def evaluate_job(job):
    configuration, games, seed, variants = job
    result = {'configuration': asdict(configuration), 'results': {}}
    for label, name, stop in variants:
        scores = [play(configuration, POLICIES[name](), seed+i, stop) for i in range(games)]
        result['results'][label] = {'scores': scores, 'policy': name, 'stop_cost': stop}
    base = result['results']['greedy']['scores']
    for row in result['results'].values():
        row['summary'] = summary(row['scores'], base, configuration.max_value)
    return configuration.key(), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=128)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-values', type=int, nargs='+', default=[50, 100, 150])
    parser.add_argument('--piles', type=int, nargs='+', default=[2, 4, 6])
    parser.add_argument('--minimums', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--players', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--mode', choices=['strict', 'free'], default='strict')
    parser.add_argument('--stopping', type=int, nargs='*', default=[])
    parser.add_argument('--configurations', type=Path, help='Frozen JSON list of configuration dictionaries')
    parser.add_argument('--calibrated-from', type=Path,
                        help='Freeze each policy best stopping threshold from independent calibration games')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configurations = ([Configuration(**c) for c in json.loads(args.configurations.read_text())]
                      if args.configurations else
                      [Configuration(v, p, m, n, args.mode) for v, p, m, n in
                       itertools.product(args.max_values, args.piles, args.minimums, args.players)])
    result = {'protocol': {'games': args.games, 'seed': args.seed, 'jump': 10,
                          'hand_size': '7 for 1-3 players; 6 for 4-5',
                          'endgame_minimum': 1, 'stopping': [None]+args.stopping,
                          'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                            for p in [Path(__file__), Path('boardrl/games/thegame/game.py')]}},
              'configurations': {}}
    calibration = json.loads(args.calibrated_from.read_text()) if args.calibrated_from else None
    jobs = []
    result['protocol']['variants'] = {}
    for configuration in configurations:
        variants = [(name if stop is None else f'{name}_stop{stop}', name, stop)
                    for name in POLICIES for stop in [None]+args.stopping]
        if calibration:
            rows = calibration['configurations'][configuration.key()]['results']
            variants = [('greedy', 'greedy', None)]
            for name in POLICIES:
                best = max((label for label in rows if label.split('_stop')[0] == name),
                           key=lambda label: rows[label]['summary']['mean'])
                stop = int(best.split('_stop')[1]) if '_stop' in best else None
                variants.append((name+'_calibrated', name, stop))
            result['protocol']['calibration_file'] = str(args.calibrated_from)
        result['protocol']['variants'][configuration.key()] = variants
        jobs.append((configuration, args.games, args.seed, variants))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for key, row in pool.map(evaluate_job, jobs):
            result['configurations'][key] = row
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result)+'\n')
            print(key, {n: round(r['summary']['gap_percentage_points'], 2)
                        for n, r in row['results'].items()}, flush=True)


if __name__ == '__main__':
    main()
