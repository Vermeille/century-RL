import asyncio
import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

if "pyximport" not in sys.modules:
    import types

    pyx = types.ModuleType("pyximport")
    pyx.install = lambda *a, **k: None
    sys.modules["pyximport"] = pyx
if "boardrl.cyutils" not in sys.modules:
    import types

    cyutils = types.ModuleType("boardrl.cyutils")
    cyutils.fast_sample = lambda x: 0
    sys.modules["boardrl.cyutils"] = cyutils

from boardrl.utils import BatchProcessor, RegisterByName
import torch
from boardrl.rl.utils import pearson_corr


class DummyOut:
    def __init__(self, outputs):
        self._outs = outputs

    def unbatched(self):
        return self._outs


@pytest.mark.timeout(10)
def test_batch_processor_collects_batch():
    async def run_bp():
        def process_fn(batch):
            return DummyOut([b * 2 for b in batch])

        bp = BatchProcessor(batch_size=3, process_fn=process_fn, timeout=100000)
        tasks = [bp(i) for i in range(3)]
        results = await asyncio.gather(*tasks)
        return results

    results = asyncio.run(run_bp())
    assert results == [0, 2, 4]


def test_batch_processor_timeouts():
    async def run_bp():
        def process_fn(batch):
            return DummyOut([b * 2 for b in batch])

        bp = BatchProcessor(batch_size=5, process_fn=process_fn, timeout=1)
        tasks = [bp(i) for i in range(3)]
        results = await asyncio.gather(*tasks)
        return results

    results = asyncio.run(run_bp())
    assert results == [0, 2, 4]


def test_register_by_name_basic():
    registry = RegisterByName()

    @registry.register("foo")
    class Foo:
        def __init__(self, x: int, y: str = "bar"):
            self.x = x
            self.y = y

    inst1 = registry("foo,x=1,y=baz")
    assert isinstance(inst1, Foo)
    assert inst1.x == 1 and inst1.y == "baz"

    inst2 = registry("foo,x=5")
    assert inst2.x == 5 and inst2.y == "bar"


def test_pearson_corr_identity():
    x = torch.randn(10)
    corr = pearson_corr(x, x)
    assert torch.isclose(corr, torch.tensor(1.0), atol=1e-5)
