"""Refine the calculator-free awkward-card tactic with held-out gameplay.

No fitted move score is used. Card ordering is semantic: most awkward first;
move ordering is cheapest first. Whole-game evaluations use the same explicit
per-deal seeds as the earlier rollout-based studies.
"""
import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from boardrl.games import games_library
from boardrl.games.strategies import one_hot
from explainability.thegame.compact_strategy import Move, cheapest_per_pile, choose_move
from explainability.thegame.human_rules import HumanRule, SPECS
from explainability.thegame.semantic_tree import VisibleState, placements, score_summary
from explainability.thegame.summarize_human import paired_difference


@dataclass(frozen=True)
class AssignedCard:
    card: int
    cost: int
    piles: frozenset[int]


def assign_cards(state):
    result = []
    for card in state.hand:
        options = placements(card,state.piles)
        if options:
            cost = min(cost for _,cost in options)
            result.append(AssignedCard(card,cost,frozenset(pile for pile,value in options if value == cost)))
    return sorted(result,key=lambda card:-card.cost)


class WorstCard:
    def targets(self, cards, eligible):
        return cards[:1]


class AffordableWorstCard:
    def targets(self, cards, eligible):
        return [card for card in cards if any(move.pile in card.piles for move in eligible)][:1]


TARGETS = {'worst':WorstCard,'affordable':AffordableWorstCard}


class LegalStarter:
    def moves(self,state,cards):
        return cheapest_per_pile(state)


class AssignedStarter:
    """Use a card assigned to the target pile, not any legal card."""
    def moves(self,state,cards):
        assignment = {card.card:card for card in cards}
        by_pile = {}
        for index,text in enumerate(state.moves):
            card,pile = map(int,text.split('->'))
            if pile in assignment[card].piles:
                move = Move(index,card,pile,assignment[card].cost)
                if pile not in by_pile or move.cost < by_pile[pile].cost:
                    by_pile[pile] = move
        return list(by_pile.values())


STARTERS = {'legal':LegalStarter,'assigned':AssignedStarter}


class FirstReverse:
    def choose(self,state,moves):
        return moves[0]


class RevivePile:
    """Among reverse jumps, rescue the pile closest to its forward endpoint."""
    def key(self,state,move):
        space = 100-state.piles[move.pile] if move.pile < 2 else state.piles[move.pile]-1
        return space,move.index

    def choose(self,state,moves):
        return min(moves,key=lambda move:self.key(state,move))


class ChainReverse(RevivePile):
    def key(self,state,move):
        next_card = move.card-10 if move.pile < 2 else move.card+10
        return (next_card not in state.hand,)+super().key(state,move)


REVERSES = {'first':FirstReverse,'endpoint':RevivePile,'chain':ChainReverse}


class AwkwardRule:
    def __init__(self, specification):
        self.specification = specification
        self.target = TARGETS[specification['target']]()
        self.starter = STARTERS[specification.get('starter','legal')]()
        self.reverse = REVERSES[specification.get('reverse','first')]()

    def action(self, state):
        cards = assign_cards(state)
        moves = sorted(self.starter.moves(state,cards),key=lambda move:(move.cost,move.index))
        greedy = moves[0]
        if greedy.cost == -10:
            return self.reverse.choose(state,[move for move in moves if move.cost == -10]).index
        eligible = [move for move in moves if move.cost <= greedy.cost+self.specification['budget']]
        targets = self.target.targets(cards,eligible)
        for target in targets:
            if target.cost < self.specification['threshold']:
                return greedy.index
            for move in eligible:
                if move.pile in target.piles:
                    return move.index
        return greedy.index

    async def __call__(self, game):
        return one_hot(self.action(VisibleState.parse(game.display_with_moves())),len(game.moves)).log(), {}


class LowestRule:
    def action(self, state):
        return min(cheapest_per_pile(state),key=lambda move:(move.cost,move.index)).index


class FormulaRule:
    def action(self, state):
        return choose_move(state).index


def evaluate_rule(rule, games, seed):
    """Direct deterministic engine execution; parity-tested against RolloutRunner."""
    scores = []
    descriptor = games_library('thegame,mode=strict')
    for index in range(games):
        random.seed(seed+index)
        game = descriptor.make_game(num_players=2)
        steps = 0
        while not game.ended():
            state = VisibleState.parse(game.display_with_moves())
            game.play_idx(rule.action(state))
            steps += 1
            if steps > 98:
                raise RuntimeError('Strict card-only game exceeded its card count')
        scores.append(game.points())
    return np.asarray(scores,dtype=float)


def all_specifications():
    specs = {f'{target}_t{threshold}_b{budget}':{'target':target,'threshold':threshold,'budget':budget}
             for target in TARGETS for threshold in [0,5,10,15,20] for budget in [1,2,3,5,8]}
    specs.update({f'assigned_{target}_t{threshold}_b{budget}':
                  {'target':target,'threshold':threshold,'budget':budget,'starter':'assigned'}
                  for target in TARGETS for threshold in [0,10] for budget in [2,3,5]})
    for base in ['worst_t0_b3','worst_t10_b3','affordable_t20_b5']:
        for reverse in ['endpoint','chain']:
            specs[f'{base}_reverse_{reverse}'] = dict(specs[base],reverse=reverse)
    return specs


def evaluate(args):
    specs = all_specifications()
    groups = {
        'all':list(specs),
        'grid':[name for name in specs if not name.startswith('assigned_') and '_reverse_' not in name],
        'assigned':[name for name in specs if name.startswith('assigned_')],
        'reverse':[name for name in specs if '_reverse_' in name],
    }
    selected = groups.get(args.selected[0],args.selected) if len(args.selected) == 1 else args.selected
    lowest = evaluate_rule(LowestRule(),args.games,args.seed)
    previous = evaluate_rule(HumanRule(SPECS['hardest_3']),args.games,args.seed)
    formula = evaluate_rule(FormulaRule(),args.games,args.seed)
    result = {'protocol':{'stage':args.stage,'seed':args.seed,'games':args.games,'selected':selected,
                          'mode':'strict','players':2,'engine':'deterministic play_idx; no hidden features'},
              'results':{}}
    def save(name,scores,spec=None):
        summary = score_summary(scores,previous)
        summary['delta_vs_previous'] = summary.pop('delta_vs_teacher')
        entry = {'scores':scores.tolist(),'summary':summary,
                 'vs_lowest':paired_difference(scores,lowest),
                 'vs_previous':paired_difference(scores,previous)}
        if spec is not None:
            entry['specification'] = spec
        result['results'][name] = entry
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(name,summary['mean'],'vs previous',entry['vs_previous'],flush=True)
    save('lowest',lowest)
    save('previous',previous)
    save('formula',formula)
    for name in selected:
        scores = evaluate_rule(AwkwardRule(specs[name]),args.games,args.seed)
        save(name,scores,specs[name])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['screen','validation','confirmation'],required=True)
    parser.add_argument('--games',type=int,default=256)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--selected',nargs='+',default=['all'])
    parser.add_argument('--output',type=Path,required=True)
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
