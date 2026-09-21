import asyncio
import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

if "pyximport" not in sys.modules:
    import types

    pyx = types.ModuleType("pyximport")
    setattr(pyx, "install", lambda *a, **k: None)
    sys.modules["pyximport"] = pyx
if "boardrl.cyutils" not in sys.modules:
    import types

    cyutils = types.ModuleType("boardrl.cyutils")
    setattr(cyutils, "fast_sample", lambda x: 0)
    sys.modules["boardrl.cyutils"] = cyutils

from boardrl.utils import BatchProcessor, RegisterByName, chunk, parse_spec
import torch
from boardrl.rl.utils import pearson_corr, explained_variance


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


def test_parse_spec_string():
    assert parse_spec("foo, x=1, path=a=b") == (
        "foo",
        {"x": "1", "path": "a=b"},
    )


def test_parse_spec_mapping_does_not_mutate_input():
    spec = {"name": "foo", "x": 1}

    assert parse_spec(spec) == ("foo", {"x": 1})
    assert spec == {"name": "foo", "x": 1}


@pytest.mark.parametrize(
    "spec,match",
    [
        ("", "name"),
        (",x=1", "name"),
        ("foo,x", "Invalid spec argument"),
        ("foo,=1", "Invalid spec argument"),
        ("foo,x=1,x=2", "Duplicate spec argument"),
        ({"x": 1}, "requires a 'name'"),
    ],
)
def test_parse_spec_rejects_malformed_specs(spec, match):
    with pytest.raises(ValueError, match=match):
        parse_spec(spec)


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


def test_register_by_name_required_and_unknown_arguments_are_values_errors():
    registry = RegisterByName()

    @registry.register("foo")
    class Foo:
        def __init__(self, x: int):
            self.x = x

    with pytest.raises(ValueError, match="Missing required argument x"):
        registry("foo")
    with pytest.raises(ValueError, match="Unknown argument.*y"):
        registry("foo,x=1,y=2")


def test_register_by_name_explicit_injection_overrides_spec_value():
    registry = RegisterByName()

    @registry.register("foo")
    class Foo:
        def __init__(self, x):
            self.x = x

    injected = object()
    assert registry("foo,x=from-spec", x=injected).x is injected


def test_register_by_name_rejects_invalid_bool():
    registry = RegisterByName()

    @registry.register("foo")
    class Foo:
        def __init__(self, enabled: bool = False):
            self.enabled = enabled

    assert registry("foo,enabled=True").enabled is True
    with pytest.raises(ValueError, match="Invalid value for enabled"):
        registry("foo,enabled=yes")


@pytest.mark.parametrize(
    "n,size,expected",
    [
        (0, 4, []),
        (1, 4, []),
        (3, 4, []),
        (4, 4, [[0, 1, 2, 3]]),
        (5, 4, [[0, 1, 2, 3]]),
        (8, 4, [[0, 1, 2, 3], [4, 5, 6, 7]]),
        (9, 4, [[0, 1, 2, 3], [4, 5, 6, 7]]),
    ],
)
def test_chunk_skip_last_boundaries(n, size, expected):
    assert list(chunk(list(range(n)), size, skip_last=True)) == expected


def test_chunk_keeps_incomplete_tail_by_default():
    assert list(chunk(list(range(5)), 4)) == [[0, 1, 2, 3], [4]]


def test_chunk_rejects_non_positive_size():
    with pytest.raises(ValueError, match="positive"):
        list(chunk([1], 0))


def test_pearson_corr_identity():
    x = torch.randn(10)
    corr = pearson_corr(x, x)
    assert torch.isclose(corr, torch.tensor(1.0), atol=1e-5)


def test_explained_variance_identity():
    y = torch.randn(10)
    ev = explained_variance(y, y)
    assert torch.isclose(ev, torch.tensor(1.0), atol=1e-5)


def test_explained_variance_basic():
    target = torch.tensor([1.0, 2.0, 3.0])
    pred = torch.tensor([1.0, 2.5, 3.5])
    expected = 1 - (target - pred).var(unbiased=False) / target.var(unbiased=False)
    ev = explained_variance(pred, target)
    assert torch.isclose(ev, expected, atol=1e-5)


def test_explained_variance_zero_var_target():
    target = torch.ones(5)
    pred = torch.zeros(5)
    ev = explained_variance(pred, target)
    assert ev == torch.tensor(0.0)
