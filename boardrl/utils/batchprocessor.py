from typing import Any, List, Callable, Awaitable
import asyncio
import inspect


class BatchProcessor:
    """Async batcher for request/response style compute.

    Collects inputs and processes them in batches using ``process_fn``. A batch
    is dispatched when either ``batch_size`` is reached or ``timeout`` seconds
    have elapsed since the first item of the pending batch was queued.

    ``process_fn`` is called with ``List[Any]`` and must return an object that
    provides ``.unbatched()`` to iterate per-input results. ``process_fn`` can
    be sync or async.
    """

    def __init__(
        self,
        batch_size: int,
        process_fn: Callable[[List[Any]], Any | Awaitable[Any]],
        timeout: float = 1.0,
        model_name: str = "undefined",
    ):
        self.batch_size = max(1, int(batch_size))
        self.process_fn = process_fn
        self.timeout = float(timeout)
        self.model_name = model_name

        self._queue: asyncio.Queue[tuple[Any, asyncio.Future]] | None = None
        self._runner: asyncio.Task | None = None
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    def _ensure_started(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._owner_loop is loop and self._runner is not None and not self._runner.done():
            return
        self._queue = asyncio.Queue()
        self._owner_loop = loop
        self._runner = asyncio.create_task(self._loop(self._queue))

        def _cleanup(_):
            self._queue = None
            self._runner = None
            self._owner_loop = None

        self._runner.add_done_callback(_cleanup)

    async def _maybe_await(self, x):
        return await x if inspect.isawaitable(x) else x

    async def _loop(self, queue: asyncio.Queue[tuple[Any, asyncio.Future]]) -> None:
        loop = asyncio.get_running_loop()
        pending: list[tuple[Any, asyncio.Future]] = []
        deadline: float | None = None

        async def flush():
            nonlocal pending, deadline
            if not pending:
                return
            inputs = [i for (i, _) in pending]
            futures = [f for (_, f) in pending]
            try:
                results = await self._maybe_await(self.process_fn(inputs))
                for fut, res in zip(futures, results.unbatched()):
                    if not fut.cancelled():
                        fut.set_result(res)
            except Exception as e:
                for fut in futures:
                    if not fut.cancelled():
                        fut.set_exception(e)
            finally:
                pending = []
                deadline = None

        while not self._closed:
            try:
                if not pending:
                    item = await queue.get()
                    pending.append(item)
                    deadline = loop.time() + self.timeout
                else:
                    if len(pending) >= self.batch_size:
                        await flush()
                        continue
                    assert deadline is not None
                    timeout_left = max(0.0, deadline - loop.time())
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=timeout_left)
                        pending.append(item)
                    except asyncio.TimeoutError:
                        await flush()
            except asyncio.CancelledError:
                break
        if pending:
            await flush()

    async def __call__(self, data: Any):
        if self._closed:
            raise RuntimeError("BatchProcessor is closed")
        loop = asyncio.get_running_loop()
        self._ensure_started(loop)
        assert self._queue is not None
        fut: asyncio.Future = loop.create_future()
        await self._queue.put((data, fut))
        return await fut

    async def aclose(self) -> None:
        """Gracefully stop background worker and flush remaining items."""
        self._closed = True
        if self._runner is not None and self._owner_loop is asyncio.get_running_loop():
            self._runner.cancel()
            try:
                await self._runner
            except asyncio.CancelledError:
                pass


def run_tasks(tasks):
    async def do():
        return await asyncio.gather(*[asyncio.create_task(t) for t in tasks])

    return asyncio.run(do())
