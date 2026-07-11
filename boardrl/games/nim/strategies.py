from boardrl.games.nim.game import Nim
from boardrl.games.strategies import one_hot
from boardrl.utils import RegisterByName


strategy_from_string = RegisterByName()


@strategy_from_string.register("optimal")
class OptimalStrategy:
    async def __call__(self, g: Nim):
        assert g.moves, "Nim strategy requires at least one legal move"

        target = (g.max_pick + 1)
        winning_pick = g.num_stones % target
        if winning_pick == 0 or str(winning_pick) not in g.moves:
            move_idx = 0
        else:
            move_idx = g.moves.index(str(winning_pick))

        distribution = one_hot(move_idx, len(g.moves))
        return distribution.log(), {"moves": dict(zip(g.moves, distribution.tolist()))}
