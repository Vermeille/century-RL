from collections import deque
from typing import Any, List, Callable
import asyncio


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

    def process_batch(self):
        if len(self.queue) == 0:
            return

        # Extract the current batch
        batch = [
            self.queue.popleft() for _ in range(min(self.batch_size, len(self.queue)))
        ]
        inputs = [task["input"] for task in batch]
        # print(f"Processing batch of size {len(inputs)}, timeouts: {self.timeout}")

        # Process the batch
        results = self.process_fn(inputs)

        # Update last batch processed time
        self.last_batch_time = asyncio.get_event_loop().time()

        # Return the results to the respective tasks
        for task, result in zip(batch, results.unbatched()):
            task["future"].set_result(result)

    def wait_data(self):
        queue_full = len(self.queue) >= self.batch_size
        if self.last_batch_time is None:
            self.last_batch_time = asyncio.get_event_loop().time()
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

    async def __call__(self, data: Any):
        if data in self.cache:
            return self.cache[data]
        result = await super().__call__(data)
        if len(self.cache) >= self.cache_size:
            self.cache.popitem()
        self.cache[data] = result
        return result
