import torch

from boardrl.games.sum.game import Sum
from boardrl.utils import RegisterByName


strategy_from_string = RegisterByName()


@strategy_from_string.register("sum_exact")
class SumExactStrategy:
    async def __call__(self, g: Sum):
        mov = int((g.a + g.b) // 2)
        one_hot = torch.zeros(len(g.moves), dtype=torch.float)
        one_hot[mov] = 1
        return one_hot.log(), {"moves": dict(zip(g.moves, one_hot.tolist()))}
