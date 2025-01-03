from collections import deque
from typing import Any, List, Callable
import asyncio
import inspect


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
        self.last_batch_time = None

    async def process_batch(self):
        if len(self.queue) == 0:
            return

        # Extract the current batch
        batch = [
            self.queue.popleft() for _ in range(min(self.batch_size, len(self.queue)))
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

        await self.wait_data()
        while not future.done():
            # Sleep briefly to prevent busy-waiting
            await asyncio.sleep(0.01)

            # Check if the batch is ready to process
            await self.wait_data()

        # Wait for the result
        return await future

    def run_tasks(self, tasks):
        async def do():
            self.last_batch_time = asyncio.get_event_loop().time()
            return await asyncio.gather(*[asyncio.create_task(t) for t in tasks])

        ret = asyncio.run(do())
        return ret


class RegisterByName:
    def __init__(self, arg_readers=None):
        self.registry = {}
        self.arg_readers = arg_readers or {}

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
                        param.default,
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

    def __call__(self, descr_string, **provided_args):
        name, *arg_list = descr_string.split(",")
        args = {arg.split("=")[0]: arg.split("=")[1] for arg in arg_list}

        if name not in self.registry:
            raise ValueError(f"Unknown strategy: {descr_string}")

        strategy_class, arg_info = self.registry[name]
        init_args = {}

        for arg_name, (arg_type, default) in arg_info.items():
            if arg_name in provided_args:
                init_args[arg_name] = provided_args[arg_name]
            elif arg_name in self.arg_readers:
                init_args[arg_name] = self.arg_readers[arg_name](
                    args.get(arg_name, default)
                )
            elif arg_name in args:
                init_args[arg_name] = arg_type(args[arg_name])
            else:
                init_args[arg_name] = default

        return strategy_class(**init_args)


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
