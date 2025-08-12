import torch
import torch

from .mcts import MCTS, Simulate, StochasticUCT


class MCTSStrategy:
    def __init__(
        self, discount_factor: float, max_unroll: int, iterations: int, c: float = 0.0
    ):
        self.discount_factor = discount_factor
        self.max_unroll = max_unroll
        self.iterations = iterations
        self.c = c

    async def __call__(self, g):
        eval_fn = Simulate(self.max_unroll, self.discount_factor)
        searcher = MCTS(
            g.current_player(),
            self.discount_factor,
            eval_fn,
            StochasticUCT(self.c),
        )
        root_node = await searcher.search(g, self.iterations)
        visits = [c.visits for c in root_node.children]
        tvisits = torch.tensor(visits, dtype=torch.float)
        tvisits /= tvisits.sum()
        return tvisits.log(), {"moves": dict(zip(g.moves, visits))}


class MCTSValueStrategy:
    def __init__(self, model, discount_factor: float, iterations: int, c: float = 0.0):
        self.discount_factor = discount_factor
        self.iterations = iterations
        self.model = model
        self.c = c

    async def __call__(self, g):
        async def eval_fn(g):
            out = await self.model(g.display_with_moves())
            return out.value.mean.item()

        searcher = MCTS(
            g.current_player(),
            self.discount_factor,
            eval_fn,
            StochasticUCT(self.c),
        )
        root_node = await searcher.search(g, self.iterations)
        visits = [c.visits for c in root_node.children]
        tvisits = torch.tensor(visits, dtype=torch.float)
        tvisits /= tvisits.sum()
        return tvisits.log(), {"moves": dict(zip(g.moves, visits))}
