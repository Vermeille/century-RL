"""A four-term, memory-free Strict strategy; not an exact copy of YOLO.

All features are computed from the displayed hand and pile tops. Helpers here
make the arithmetic explicit; no learned model or fitted tree is called.
"""
from dataclasses import dataclass

from boardrl.games.strategies import one_hot
from explainability.thegame.semantic_tree import VisibleState, placements


@dataclass(frozen=True)
class Move:
    index: int
    card: int
    pile: int
    cost: int


def cheapest_per_pile(state):
    moves = []
    for index, text in enumerate(state.moves):
        card, pile = map(int, text.split('->'))
        cost = card - state.piles[pile] if pile < 2 else state.piles[pile] - card
        moves.append(Move(index, card, pile, cost))
    return [min(pool, key=lambda m: (m.cost, m.index)) for pile in range(4)
            if (pool := [move for move in moves if move.pile == pile])]


def consequences(state, move):
    after = list(state.piles)
    after[move.pile] = move.card
    remaining = [card for card in state.hand if card != move.card]
    best_costs = [min(cost for _, cost in options) for card in remaining
                  if (options := placements(card, after))]
    # This matches the fitted feature: unplayable cards are omitted, not assigned
    # an artificial cost. If none is playable, use 100 (or zero for an empty hand).
    hardest = max(best_costs) if best_costs else (100 if remaining else 0)
    partner = move.pile ^ 1
    gap_gain = abs(move.card - after[partner]) - abs(state.piles[move.pile] - after[partner])
    space = 100 - state.piles[move.pile] if move.pile < 2 else state.piles[move.pile] - 1
    return hardest, gap_gain, max(1, space)


def adjusted_cost(state, move):
    hardest, gap_gain, space = consequences(state, move)
    return move.cost + hardest / 4 - gap_gain / 5 + 8 * move.cost / space


def choose_move(state):
    return min(cheapest_per_pile(state),
               key=lambda move: (adjusted_cost(state, move), move.cost, move.index))


class CompactStrategy:
    async def __call__(self, game):
        state = VisibleState.parse(game.display_with_moves())
        return one_hot(choose_move(state).index, len(state.moves)).log(), {}
