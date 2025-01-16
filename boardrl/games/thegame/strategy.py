from boardrl.utils import RegisterByName
from boardrl.games.strategies import one_hot

strategy_from_string = RegisterByName()


@strategy_from_string.register("lowest_cost")
class LowestCostStrategy:
    def __call__(self, g):
        costs = []
        for m in g.moves:
            card, pile = int(m.split("->")[0]), int(m.split("->")[1])
            asc = pile < 2
            if asc:
                cost = card - g.piles[pile]
            else:
                cost = g.piles[pile] - card
            costs.append(cost)
        print(g.piles, list(zip(g.moves, costs)))
        best_idx = min(range(len(costs)), key=lambda i: costs[i])
        return one_hot(best_idx, len(g.moves)), {"moves": list(zip(g.moves, costs))}
