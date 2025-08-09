import inspect
import os
import random

import torch

from boardrl.utils import BatchProcessor, Game, RegisterByName
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


async def _maybe_await(result):
    """Await ``result`` if it is awaitable and return its value."""
    if inspect.isawaitable(result):
        return await result
    return result


class ModelPool:
    """Utility to resolve model specifications to ``BatchProcessor`` instances."""

    def __init__(self, base_model: BatchProcessor, batch_size: int, timeout: float):
        self.base_model = base_model
        self.batch_size = batch_size
        self.timeout = timeout
        self.cache: dict[str, BatchProcessor] = {}

    def _load(self, path: str) -> BatchProcessor:
        model = load_model(path)
        model.eval()
        return BatchProcessor(self.batch_size, model, timeout=self.timeout)

    def _resolve_path(self, spec: str) -> str:
        if spec.startswith("recent-"):
            try:
                topk = int(spec.split("-", 1)[1])
            except ValueError as exc:  # pragma: no cover - defensive programming
                raise ValueError(f"invalid recent model spec: {spec}") from exc
            candidates = _recent_models(topk)
            if not candidates:
                raise ValueError("no recent model files found")
            return random.choice(candidates)
        return spec

    def __call__(self, spec: str | None):
        if spec in (None, "this"):
            if self.base_model is None:
                raise ValueError("model='this' requires a provided model")
            return self.base_model
        path = self._resolve_path(spec)
        if not os.path.exists(path):
            raise ValueError(f"model file '{path}' does not exist")
        if path not in self.cache:
            self.cache[path] = self._load(path)
        return self.cache[path]

def get_model(arg_str, default, provided_arg):
    """Resolve the model argument for strategy creation.

    ``arg_str`` is the model specification from the strategy string.  If a
    :class:`ModelPool` is supplied as ``provided_arg`` the resolution is
    delegated to it, allowing ``model=this``, paths, or ``recent-N`` specs to
    share caching and batching behaviour.  When no pool is supplied, fall back
    to loading a model directly from ``arg_str``.
    """

    if isinstance(provided_arg, ModelPool):
        return provided_arg(arg_str)

    # Fallback behaviour without a pool: either reuse the provided model or
    # load the requested checkpoint without batching.
    if arg_str in (None, "this"):
        if provided_arg is None:
            raise ValueError("model='this' requires a provided model")
        return provided_arg

    if arg_str.startswith("recent-"):
        try:
            topk = int(arg_str.split("-", 1)[1])
        except ValueError as exc:  # pragma: no cover - defensive programming
            raise ValueError(f"invalid recent model spec: {arg_str}") from exc
        candidates = _recent_models(topk)
        if not candidates:
            raise ValueError("no recent model files found")
        model_path = random.choice(candidates)
    else:
        model_path = arg_str
        if not os.path.exists(model_path):
            raise ValueError(f"model file '{model_path}' does not exist")

    model = load_model(model_path)
    model.eval()
    return model


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
            policy = (
                await _maybe_await(self.nn(g.display_with_moves()))
            ).policy[0].cpu()
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

        policy = (
            await _maybe_await(self.nn(g.display_with_moves()))
        ).policy[0].cpu()
        policy = policy / self.temperature
        return policy, {"moves": dict(zip(g.moves, policy.tolist()))}


@strategy_from_string.register("mcts")
class MCTS:
    def __init__(
        self, discount_factor: float, max_unroll: int, iterations: int, c: float = 0.0
    ):
        self.discount_factor = discount_factor
        self.max_unroll = max_unroll
        self.iterations = iterations
        self.c = c

    async def __call__(self, g: Game):
        eval_fn = mcts.Simulate(self.max_unroll, self.discount_factor)
        searcher = mcts.MCTS(
            g.current_player(),
            self.discount_factor,
            eval_fn,
            mcts.StochasticUCT(self.c),
        )
        root_node = await searcher.search(g, self.iterations)
        visits = [c.visits for c in root_node.children]
        tvisits = torch.tensor(visits, dtype=torch.float)
        tvisits /= tvisits.sum()
        return tvisits.log(), {"moves": dict(zip(g.moves, visits))}


@strategy_from_string.register("mcts_value")
class MCTSValue:
    def __init__(self, model, discount_factor: float, iterations: int, c: float = 0.0):
        self.discount_factor = discount_factor
        self.iterations = iterations
        self.model = model
        self.c = c

    async def __call__(self, g: Game):
        async def eval_fn(g):
            out = await _maybe_await(self.model(g.display_with_moves()))
            return out.value.mean.item()

        searcher = mcts.MCTS(
            g.current_player(),
            self.discount_factor,
            eval_fn,
            mcts.StochasticUCT(self.c),
        )
        root_node = await searcher.search(g, self.iterations)
        visits = [c.visits for c in root_node.children]
        tvisits = torch.tensor(visits, dtype=torch.float)
        tvisits /= tvisits.sum()
        return tvisits.log(), {"moves": dict(zip(g.moves, visits))}
