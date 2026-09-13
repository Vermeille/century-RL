"""Table-friendly Strict tactics: small integer thresholds, no move scores."""
from dataclasses import dataclass

from boardrl.games.strategies import one_hot
from explainability.thegame.compact_strategy import Move, cheapest_per_pile
from explainability.thegame.semantic_tree import VisibleState, placements


@dataclass(frozen=True)
class TableMove:
    move: Move
    space: int
    working_pile: bool
    destinations: int
    close_followups: int
    reverse_followup: bool
    supports_hardest: bool


def table_moves(state):
    candidates = []
    legal = {card:placements(card,state.piles) for card in state.hand}
    playable = {card:min(cost for _,cost in options) for card,options in legal.items() if options}
    hardest = max(playable, key=playable.get)
    support_piles = {pile for pile,cost in legal[hardest] if cost == playable[hardest]}
    for move in cheapest_per_pile(state):
        top, partner = state.piles[move.pile], state.piles[move.pile ^ 1]
        ascending = move.pile < 2
        next_costs = [card-move.card if ascending else move.card-card for card in state.hand if card != move.card]
        candidates.append(TableMove(move, 100-top if ascending else top-1,
                                    top > partner if ascending else top < partner,
                                    len(legal[move.card]), sum(0 < cost <= 3 for cost in next_costs),
                                    -10 in next_costs,
                                    playable[hardest] >= 10 and move.pile in support_piles))
    return sorted(candidates,key=lambda c:(c.move.cost,c.move.index))


class PreserveReserve:
    """Prefer an already-more-advanced pile, paying at most the given extra cost."""
    def accepts(self, challenger, greedy):
        return challenger.working_pile and not greedy.working_pile


class AvoidEndpoint:
    """Save a pile with ten or fewer spaces if another has at least twenty."""
    def accepts(self, challenger, greedy):
        return greedy.space <= 10 and challenger.space >= 20


class PlayConstrained:
    """Prefer a card with just one destination over a card with several."""
    def accepts(self, challenger, greedy):
        return challenger.destinations == 1 and greedy.destinations > 1


class StartRun:
    """Start a run when another card can follow on that pile for at most three."""
    def accepts(self, challenger, greedy):
        return challenger.close_followups > 0 and greedy.close_followups == 0


class SetUpReverse:
    """Play one of a pair so the other can immediately reverse the pile by ten."""
    def accepts(self, challenger, greedy):
        return challenger.reverse_followup and not greedy.reverse_followup


class SupportHardest:
    """Work toward the hardest card if its cheapest placement costs at least ten."""
    def accepts(self, challenger, greedy):
        return challenger.supports_hardest and not greedy.supports_hardest


TACTICS = {'reserve':PreserveReserve,'endpoint':AvoidEndpoint,'constrained':PlayConstrained,
           'run':StartRun,'setup':SetUpReverse,'hardest':SupportHardest}


class HumanRule:
    def __init__(self, specification):
        self.specification = specification
        self.tactics = [TACTICS[name]() for name in specification['tactics']]

    def action(self, state):
        candidates = table_moves(state)
        greedy = candidates[0]
        # Take available backwards-ten jumps without applying forward-play rules.
        if greedy.move.cost == -10:
            return greedy.move.index
        eligible = [c for c in candidates if c.move.cost <= greedy.move.cost+self.specification['budget']]
        for tactic in self.tactics:
            for candidate in eligible:
                if tactic.accepts(candidate,greedy):
                    return candidate.move.index
        return greedy.move.index

    async def __call__(self, game):
        return one_hot(self.action(VisibleState.parse(game.display_with_moves())),len(game.moves)).log(), {}


SPECS = {
    **{f'reserve_{budget}':{'kind':'human','budget':budget,'tactics':['reserve']} for budget in [1,2,3]},
    'endpoint_2':{'kind':'human','budget':2,'tactics':['endpoint']},
    'endpoint_reserve_2':{'kind':'human','budget':2,'tactics':['endpoint','reserve']},
    'constrained_2':{'kind':'human','budget':2,'tactics':['constrained']},
    'run_3':{'kind':'human','budget':3,'tactics':['run']},
    'setup_5':{'kind':'human','budget':5,'tactics':['setup']},
    'setup_10':{'kind':'human','budget':10,'tactics':['setup']},
    'hardest_3':{'kind':'human','budget':3,'tactics':['hardest']},
    'run_reserve_3':{'kind':'human','budget':3,'tactics':['run','reserve']},
    'setup_reserve_3':{'kind':'human','budget':3,'tactics':['setup','reserve']},
    'hardest_reserve_3':{'kind':'human','budget':3,'tactics':['hardest','reserve']},
}
