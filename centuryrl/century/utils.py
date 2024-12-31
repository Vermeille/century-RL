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
