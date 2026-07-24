from boardrl.utils import RegisterByName
from boardrl.games.strategies import one_hot

strategy_from_string = RegisterByName()


@strategy_from_string.register("lowest_cost")
class LowestCostStrategy:
    def __init__(self, use_10_rule: bool = True):
        self.use_10_rule = use_10_rule

    async def __call__(self, g):
        if "x" in g.moves:
            return one_hot(g.moves.index("x"), len(g.moves)).log(), {
                "moves": list(zip(g.moves, [0] * len(g.moves)))
            }
        costs = []
        for m in g.moves:
            card, pile = int(m.split("->")[0]), int(m.split("->")[1])
            asc = pile < 2
            if asc:
                cost = card - g.piles[pile]
            else:
                cost = g.piles[pile] - card
            if not self.use_10_rule and cost == -10:
                cost = 100  # to avoid picking -10 moves, make them very costly
            costs.append(cost)
        best_idx = min(range(len(costs)), key=lambda i: costs[i])
        return one_hot(best_idx, len(g.moves)).log(), {
            "moves": list(zip(g.moves, costs))
        }
