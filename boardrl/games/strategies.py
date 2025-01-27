import torch

from boardrl.utils import RegisterByName, Game
from boardrl.rl.model import load_model
import boardrl.games.mcts as mcts
import pyximport

pyximport.install()
from boardrl.cyutils import fast_sample


def _recent_models(topk):
    import os
    import psutil

    # Get the current process start time
    process_start_time = psutil.Process().create_time()

    # Recursively get all files in the current directory and subdirectories
    files_in_directory = []
    for root, _, files in os.walk("."):
        for f in files:
            if f.endswith(".pth"):
                files_in_directory.append(os.path.join(root, f))

    # Filter files based on their modification time
    recent_files = [
        f for f in files_in_directory if os.path.getmtime(f) > process_start_time
    ]

    # Convert modification times to readable format for display
    recent_files_with_times = [(f, os.path.getmtime(f)) for f in recent_files]
    recent_files_with_times.sort(key=lambda x: x[1], reverse=True)

    return [f[0] for f in recent_files_with_times[:topk]]


def get_model(arg_str, default, provided_arg):
    import random

    assert default is None
    assert arg_str is not None
    if arg_str == "this":
        assert provided_arg is not None
        return provided_arg
    if arg_str.startswith("recent-"):
        recent_paths = _recent_models(int(arg_str.split("-")[1]))
        print("loading from", recent_paths)
        return load_model(random.choice(recent_paths))
    assert False


strategy_from_string = RegisterByName(arg_readers={"model": get_model})


def one_hot(i, n, smooth=0.0):
    x = torch.ones(n, dtype=torch.float) * smooth / n
    x[i] += 1 - smooth
    return x


@strategy_from_string.register("random")
class RandomStrategy:
    async def __call__(self, g: Game):
        uniform = one_hot(0, len(g.moves), smooth=1)  # uniform distribution
        return uniform.log(), {
            "moves": dict(zip(g.moves, uniform.tolist())),
        }


@strategy_from_string.register("argmax")
class ArgmaxStrategy:
    def __init__(self, model, epsilon: float = 0.0):
        self.nn = model
        self.epsilon = epsilon

    async def __call__(self, g: Game):
        if torch.rand(1).item() < self.epsilon:
            distribution = one_hot(
                torch.randint(len(g.moves), (1,)).item(), len(g.moves)
            )
            return distribution.log(), {
                "moves": dict(zip(g.moves, distribution.tolist()))
            }
        else:
            policy = (await self.nn(g.display_with_moves())).policy[0].cpu()
            distribution = one_hot(torch.argmax(policy).item(), len(g.moves))
            return distribution.log(), {
                "moves": dict(zip(g.moves, torch.softmax(policy, dim=0).tolist()))
            }


def mean(xs):
    return sum(xs) / len(xs)


@strategy_from_string.register("longest_move")
class LongestMoveStrategy:
    async def __call__(self, g: Game):
        distribution = one_hot(
            max(range(len(g.moves)), key=lambda i: len(g.moves[i])), len(g.moves)
        )
        total = sum(len(move) for move in g.moves)
        return distribution, {"moves": {move: len(g.moves) / total for move in g.moves}}


@strategy_from_string.register("policy_sampling")
class PolicySamplingStrategy:
    def __init__(self, model, temperature: float = 1.0):
        self.nn = model
        self.temperature = temperature

    @torch.no_grad()
    async def __call__(self, g: Game):
        if len(g.moves) == 1:
            return torch.tensor([1.0]), {"moves": {g.moves[0]: 1.0}}

        policy = (await self.nn(g.display_with_moves())).policy[
            0
        ].cpu() / self.temperature
        # print(policy)
        return policy, {"moves": dict(zip(g.moves, policy.tolist()))}


@strategy_from_string.register("mcts")
class MCTS:
    def __init__(self, discount_factor: float, max_unroll: int, iterations: int):
        self.discount_factor = discount_factor
        self.max_unroll = max_unroll
        self.iterations = iterations

    async def __call__(self, g: Game):
        eval_fn = mcts.Simulate(self.max_unroll, self.discount_factor)
        searcher = mcts.MCTS(g.current_player(), self.discount_factor, eval_fn)
        visits = await searcher.search(g, self.iterations)
        tvisits = torch.tensor(visits, dtype=torch.float)
        tvisits /= tvisits.sum()
        return tvisits.log(), {"moves": dict(zip(g.moves, visits))}


@strategy_from_string.register("mcts_value")
class MCTSValue:
    def __init__(self, model, discount_factor: float, iterations: int):
        self.discount_factor = discount_factor
        self.iterations = iterations
        self.model = model

    async def __call__(self, g: Game):
        async def eval_fn(g):
            return (await self.model(g.display_with_moves())).value.mean.item()

        searcher = mcts.MCTS(g.current_player(), self.discount_factor, eval_fn)
        visits = await searcher.search(g, self.iterations)
        tvisits = torch.tensor(visits, dtype=torch.float)
        tvisits /= tvisits.sum()
        return tvisits.log(), {"moves": dict(zip(g.moves, visits))}
