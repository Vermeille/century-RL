import torch
from collections import deque
from typing import Any, List, Callable
import asyncio
import inspect
import pyximport

pyximport.install()
from boardrl.cyutils import fast_sample


class BatchProcessor:
    def __init__(
        self,
        batch_size: int,
        process_fn: Callable[[List[Any]], Any],
        timeout: float = 1.0,
    ):
        self.batch_size = batch_size
        self.process_fn = process_fn
        self.queue = deque()
        self.timeout = timeout
        self.last_batch_time = asyncio.get_event_loop().time()

    def process_batch(self):
        if len(self.queue) == 0:
            return

        # Extract the current batch
        batch = [
            self.queue.popleft() for _ in range(min(self.batch_size, len(self.queue)))
        ]
        inputs = [task["input"] for task in batch]
        print(f"Processing batch of size {len(inputs)}, timeouts: {self.timeout}")

        # Process the batch
        results = self.process_fn(inputs)

        # Update last batch processed time
        self.last_batch_time = asyncio.get_event_loop().time()

        # Return the results to the respective tasks
        for task, result in zip(batch, results.unbatched()):
            task["future"].set_result(result)

    def wait_data(self):
        queue_full = len(self.queue) >= self.batch_size
        has_timeout = (
            asyncio.get_event_loop().time() - self.last_batch_time >= self.timeout
        )
        if queue_full or has_timeout:
            self.process_batch()

    async def __call__(self, data: Any):
        # Create a future to hold the result
        future = asyncio.Future()
        task = {"input": data, "future": future}

        # Add the task to the queue
        self.queue.append(task)

        self.wait_data()
        while not future.done():
            # Sleep briefly to prevent busy-waiting
            await asyncio.sleep(0.0001)

            # Check if the batch is ready to process
            self.wait_data()

        # Wait for the result
        return await future


def run_tasks(tasks):
    async def do():
        return await asyncio.gather(*[asyncio.create_task(t) for t in tasks])

    ret = asyncio.run(do())
    return ret


class CachedBatchProcessor(BatchProcessor):
    def __init__(
        self,
        batch_size: int,
        process_fn: Callable[[List[Any]], Any],
        timeout: float = 1.0,
        cache_size: int = 100,
    ):
        super().__init__(batch_size, process_fn, timeout)
        self.cache = {}
        self.cache_size = cache_size

    async def send(self, data: Any):
        if data in self.cache:
            return self.cache[data]
        result = await super().send(data)
        if len(self.cache) >= self.cache_size:
            self.cache.popitem()
        self.cache[data] = result
        return result


class RegisterByName:
    def __init__(self, arg_readers=None):
        self.registry = {}
        self.arg_readers = arg_readers or {}

    def copy(self):
        new_register = RegisterByName()
        new_register.registry = self.registry.copy()
        new_register.arg_readers = self.arg_readers.copy()
        return new_register

    def register(self, name):
        def foo(cls):
            # Extract the argument names, types, and defaults from the __init__ method
            if "__init__" in cls.__dict__:
                sig = inspect.signature(cls.__init__)
                params = sig.parameters
                arg_info = {
                    name: (
                        param.annotation
                        if param.annotation != inspect.Parameter.empty
                        else lambda x: x,
                        param.default
                        if param.default != inspect.Parameter.empty
                        else None,
                    )
                    for name, param in params.items()
                    if name != "self"
                }
            else:
                arg_info = {}

            self.registry[name] = (cls, arg_info)
            return cls

        return foo

    def update(self, other: "RegisterByName"):
        self.registry.update(other.registry)
        self.arg_readers.update(other.arg_readers)
        return self

    def __call__(self, descr_string, **provided_args):
        name, *arg_list = descr_string.split(",")
        args = {arg.split("=")[0]: arg.split("=")[1] for arg in arg_list}

        if name not in self.registry:
            raise ValueError(f"Unknown class: {descr_string}")

        klass, arg_info = self.registry[name]
        init_args = {}

        for arg_name in args.keys():
            assert arg_name in arg_info, f"Unknown argument {arg_name} for {name}"

        for arg_name, (arg_type, default) in arg_info.items():
            if arg_name in self.arg_readers:
                init_args[arg_name] = self.arg_readers[arg_name](
                    args.get(arg_name, None), default, provided_args.get(arg_name, None)
                )
            elif arg_name in provided_args:
                init_args[arg_name] = provided_args[arg_name]
            elif arg_name in args:
                if arg_type is bool:
                    assert args[arg_name] in ["True", "False"]
                    init_args[arg_name] = args[arg_name] == "True"
                else:
                    init_args[arg_name] = arg_type(args[arg_name])
            else:
                init_args[arg_name] = default

        return klass(**init_args)

    def display(self):
        for fun, args in self.registry.items():
            fun_display = fun
            for arg, (arg_type, default) in args[1].items():
                if default == inspect.Parameter.empty:
                    default = "?"
                fun_display += f",{arg}={default}"
            print(fun_display)


def easydict_to_dict(d):
    if isinstance(d, dict):
        return {k: easydict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [easydict_to_dict(v) for v in d]
    else:
        return d


def entropy(logits, dim):
    log_probs = torch.log_softmax(logits, dim=dim)
    return -torch.sum(torch.exp(log_probs) * log_probs, dim=dim).mean()


class Game:
    moves: list[str]
    num_players: int

    def current_player(self) -> int: ...

    def display_with_moves(self) -> str: ...

    def copy(self) -> "Game": ...

    def play_str(self, move: str): ...

    def play_idx(self, move: int): ...

    def simulate_to_end(self): ...

    def diff_points_for(self, player: int) -> float: ...

    def ended(self) -> bool: ...

    ...
