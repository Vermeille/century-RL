import pytest

from boardrl.utils import BatchProcessor, ModelPool


class _DummyResult:
    def unbatched(self):
        return []


def _bp() -> BatchProcessor:
    return BatchProcessor(1, lambda _: _DummyResult(), timeout=0.0)


class _DummyModel:
    def __init__(self, path: str):
        self.path = path

    def eval(self):
        pass


def test_modelpool_uses_lru(tmp_path, monkeypatch):
    loaded: list[str] = []

    def fake_load(path: str):
        loaded.append(path)
        return _DummyModel(path)

    monkeypatch.setattr("boardrl.utils.modelpool.load_model", fake_load)

    base = _bp()
    ref = _bp()
    pool = ModelPool(base, 1, 0.0, reference_model=ref, max_cache_size=3)

    paths = [tmp_path / f"model_{idx}.pth" for idx in range(3)]
    for path in paths:
        path.touch()

    # Fill cache with two dynamic models; the oldest should be evicted when a
    # third is requested because "this" and "reference" remain pinned.
    first = str(paths[0])
    second = str(paths[1])
    third = str(paths[2])

    pool(first)
    pool(second)

    assert "this" in pool.cache
    assert "reference" in pool.cache
    assert first not in pool.cache

    pool(first)

    # Loading "first" again reloads the model because it was evicted and
    # removes "second" as the least recently used non-pinned entry.
    assert loaded.count(first) == 2
    assert loaded.count(second) == 1
    assert second not in pool.cache

    pool(third)

    # Pinned entries remain regardless of additional evictions.
    assert "this" in pool.cache
    assert "reference" in pool.cache


def test_modelpool_cache_size_too_small():
    base = _bp()
    ref = _bp()

    with pytest.raises(ValueError):
        ModelPool(base, 1, 0.0, reference_model=ref, max_cache_size=1)
