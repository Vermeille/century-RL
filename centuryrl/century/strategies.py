import torch
import inspect

from centuryrl.rl.model import load_model
import pyximport

pyximport.install()
from centuryrl.century.engine import Game, fast_sample, random_buy_fast

strategy_registry = {}


def register_strategy(name):
    def register(cls):
        # Extract the argument names, types, and defaults from the __init__ method
        if "__init__" in cls.__dict__:
            sig = inspect.signature(cls.__init__)
            params = sig.parameters
            arg_info = {
                name: (
                    param.annotation
                    if param.annotation != inspect.Parameter.empty
                    else lambda x: x,
                    param.default,
                )
                for name, param in params.items()
                if name != "self"
            }
        else:
            arg_info = {}

        strategy_registry[name] = (cls, arg_info)
        return cls

    return register


@register_strategy("random")
class RandomStrategy:
    def __call__(self, g: Game):
        uniform = torch.tensor([1 / len(g.moves)] * len(g.moves))
        return uniform.log(), {
            "moves": dict(zip(g.moves, uniform.tolist())),
        }


@register_strategy("random_buy")
class RandomBuyStrategy:
    def __call__(self, g: Game):
        moves = g.moves
        for mov in moves:
            if mov[0] == "V":
                one_hot = torch.zeros(len(g.moves), dtype=torch.float)
                one_hot[g.moves.index(mov)] = 1
                return one_hot.log(), {"moves": dict(zip(g.moves, one_hot.tolist()))}

        uniform = torch.tensor([1 / len(g.moves)] * len(g.moves))
        return uniform.log(), {"moves": dict(zip(g.moves, uniform.tolist()))}


@register_strategy("all_actions_then_random_buy")
class AllActionsThenRandomBuyStrategy:
    def __call__(self, g: Game):
        moves = g.moves
        for mov in moves:
            if mov.startswith("A0"):
                one_hot = torch.zeros(len(g.moves), dtype=torch.float)
                one_hot[g.moves.index(mov)] = 1
                return one_hot.log(), {"moves": dict(zip(g.moves, one_hot.tolist()))}

        for mov in moves:
            if mov[0] == "V":
                one_hot = torch.zeros(len(g.moves), dtype=torch.float)
                one_hot[g.moves.index(mov)] = 1
                return one_hot.log(), {"moves": dict(zip(g.moves, one_hot.tolist()))}

        uniform = torch.tensor([1 / len(g.moves)] * len(g.moves))
        return uniform.log(), {"moves": dict(zip(g.moves, uniform.tolist()))}


@register_strategy("no_actions_random_buy")
class NoActionsRandomBuyStrategy:
    def __call__(self, g: Game):
        moves = g.moves
        for mov in moves:
            if mov[0] == "V":
                one_hot = torch.zeros(len(g.moves), dtype=torch.float)
                one_hot[g.moves.index(mov)] = 1
                return one_hot.log(), {"moves": dict(zip(g.moves, one_hot.tolist()))}

        num_no_action = sum(1 for mov in g.moves if mov[0] != "A")
        dist = torch.tensor(
            [(1 / num_no_action) if move[0] != "A" else 0 for move in g.moves]
        )
        return dist.log(), {"moves": dict(zip(g.moves, dist.tolist()))}


@register_strategy("argmax")
class ArgmaxStrategy:
    def __init__(self, nn):
        nn.eval()
        self.nn = nn

    def __call__(self, g: Game):
        policy = self.nn([g.display_with_moves()]).policy[0]
        one_hot = torch.zeros(len(g.moves), dtype=torch.float)
        one_hot[torch.argmax(policy).item()] = 1
        return one_hot.log(), {
            "moves": dict(zip(g.moves, torch.softmax(policy, dim=0).tolist()))
        }


def mean(xs):
    return sum(xs) / len(xs)


@register_strategy("pick_best_mc_value")
class PickBestMCValueStrategy:
    def __init__(self, budget: int):
        self.budget = budget

    def __call__(self, g: Game):
        values = [[] for _ in g.moves]
        me = g.current_player()

        for _ in range(self.budget):
            for m_i, m in enumerate(g.moves):
                g2 = g.copy()
                g2.play_str(m)
                g2.simulate_to_end()
                values[m_i].append(g2.diff_points_for(me))
        means = [mean(vs) for vs in values]
        policy = torch.median(torch.tensor(values).float(), dim=1).values
        return policy, {
            "moves": dict(zip(g.moves, means)),
        }


import asyncio
from collections import deque
from typing import Any, List, Callable


class BatchProcessor:
    def __init__(
        self,
        batch_size: int,
        process_fn: Callable[[List[Any]], List[Any]],
        timeout: float = 1.0,
    ):
        self.batch_size = batch_size
        self.process_fn = process_fn
        self.queue = deque()
        self.lock = asyncio.Lock()
        self.timeout = timeout
        self.last_batch_time = asyncio.get_event_loop().time()

    async def process_batch(self):
        async with self.lock:
            if len(self.queue) == 0:
                return

            # Extract the current batch
            batch = [
                self.queue.popleft()
                for _ in range(min(self.batch_size, len(self.queue)))
            ]
            inputs = [task["input"] for task in batch]

            # Process the batch
            results = self.process_fn(inputs)

            # Update last batch processed time
            self.last_batch_time = asyncio.get_event_loop().time()

            # Return the results to the respective tasks
            for task, result in zip(batch, results.unbatched()):
                task["future"].set_result(result)

    async def wait_data(self):
        if len(self.queue) >= self.batch_size or (
            asyncio.get_event_loop().time() - self.last_batch_time >= self.timeout
        ):
            await self.process_batch()

    async def send(self, data: Any):
        # Create a future to hold the result
        future = asyncio.Future()
        task = {"input": data, "future": future}

        # Add the task to the queue
        self.queue.append(task)

        while not future.done():
            # Check if the batch is ready to process
            await self.wait_data()

            # Sleep briefly to prevent busy-waiting
            await asyncio.sleep(0.01)

        # Wait for the result
        return await future


@register_strategy("pick_best_value")
class PickBestValueStrategy:
    def __init__(self, budget: int, model):
        self.budget = budget
        self.model = model
        model.eval()

    @torch.no_grad()
    async def async_call(self, g: Game):
        values = [[] for _ in g.moves]
        me = g.current_player()
        temp = 0.01

        processor = BatchProcessor(batch_size=64, process_fn=self.model, timeout=0.1)

        async def try_move(m_i):
            m = g.moves[m_i]
            g2 = g.copy()
            g2.play_str(m)
            for p in range(g2.num_players - 1):
                if g2.ended():
                    break
                board = g2.display_with_moves()
                policy = (await processor.send(board)).policy[0]
                m_j = fast_sample(torch.softmax(policy / temp, dim=0))
                g2.play_idx(m_j)
            if g2.ended():
                values[m_i].append(g2.diff_points_for(me))
            else:
                assert g2.current_player() == me
                values[m_i].append(
                    g2.diff_points_for(me)
                    + 0.98
                    * (await processor.send(g2.display_with_moves())).value[0].item()
                )

        tasks = [
            asyncio.create_task(try_move(m_i))
            for m_i in range(len(g.moves))
            for _ in range(self.budget)
        ]
        await asyncio.gather(*tasks)
        means = [mean(vs) for vs in values]
        policy = torch.median(torch.tensor(values).float(), dim=1).values / temp
        sm = torch.softmax(policy, dim=0)
        sm = 0.95 * sm + 0.05 / len(g.moves)
        return sm.log(), {
            "moves": dict(zip(g.moves, means)),
        }

    def __call__(self, g: Game):
        return asyncio.run(self.async_call(g))


@register_strategy("longest_move")
class LongestMoveStrategy:
    def __call__(self, g: Game):
        one_hot = torch.zeros(len(g.moves), dtype=torch.float)
        one_hot[max(range(len(g.moves)), key=lambda i: len(g.moves[i]))] = 1
        total = sum(len(move) for move in g.moves)
        return one_hot, {"moves": {move: len(g.moves) / total for move in g.moves}}


@register_strategy("policy_sampling")
class PolicySamplingStrategy:
    def __init__(self, model, temperature: float = 1.0, epsilon: float = 0):
        self.nn = model
        model.eval()
        self.temperature = temperature
        self.epsilon = epsilon

    @torch.no_grad()
    def __call__(self, g: Game):
        if len(g.moves) == 1:
            return torch.tensor([1.0]), {"moves": {g.moves[0]: 1.0}}

        policy = self.nn([g.display_with_moves()]).policy[0] / self.temperature
        policy = torch.softmax(policy, dim=0)
        policy = (1 - self.epsilon) * policy + self.epsilon / len(g.moves)
        return policy.log(), {"moves": dict(zip(g.moves, policy.tolist()))}


def strategy_from_string(strategy_string, model=None):
    strategy_name, *arg_list = strategy_string.split(",")
    args = {arg.split("=")[0]: arg.split("=")[1] for arg in arg_list}

    if strategy_name not in strategy_registry:
        raise ValueError(f"Unknown strategy: {strategy_string}")

    strategy_class, arg_info = strategy_registry[strategy_name]
    init_args = {}

    for arg_name, (arg_type, default) in arg_info.items():
        if arg_name == "model":
            init_args[arg_name] = model or load_model(args.get(arg_name, "this"))
        elif arg_name in args:
            init_args[arg_name] = arg_type(args[arg_name])
        else:
            init_args[arg_name] = default

    return strategy_class(**init_args)
