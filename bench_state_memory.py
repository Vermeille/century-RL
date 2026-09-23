import ast
import dataclasses
import functools
import gc
import inspect
import os
import random
import sys
import textwrap
import time
import tracemalloc
from collections import defaultdict, deque

from boardrl.games import games_library


SEED = 12345
PREFIX_STEPS = int(os.environ.get("STATE_BENCH_PREFIX_STEPS", "20"))
COPIES = int(os.environ.get("STATE_BENCH_COPIES", "3000"))


def _walk_objects(root):
    """Walk the instance-owned Python object graph without descending into types/modules."""
    seen = set()
    stack = [root]
    atomic = (str, bytes, bytearray, int, float, complex, bool, type(None))

    while stack:
        obj = stack.pop()
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        yield obj

        if isinstance(obj, atomic):
            continue
        if isinstance(obj, dict):
            stack.extend(obj.keys())
            stack.extend(obj.values())
            continue
        if isinstance(obj, (list, tuple, set, frozenset, deque)):
            stack.extend(obj)
            continue
        if inspect.ismodule(obj) or inspect.isclass(obj) or inspect.isroutine(obj):
            continue
        d = getattr(obj, "__dict__", None)
        if d is not None:
            stack.extend(d.values())
        for cls in type(obj).__mro__:
            for slot in getattr(cls, "__slots__", ()):
                if isinstance(slot, str) and slot not in ("__dict__", "__weakref__"):
                    try:
                        stack.append(getattr(obj, slot))
                    except AttributeError:
                        pass


def _local_state_classes(root, module_name):
    attrs = defaultdict(set)
    examples = {}
    for obj in _walk_objects(root):
        cls = type(obj)
        if cls.__module__ != module_name or not hasattr(obj, "__dict__"):
            continue
        examples.setdefault(cls, obj)
        attrs[cls].update(obj.__dict__.keys())
    return attrs, examples


def _slotify_source(cls, attrs):
    source = textwrap.dedent(inspect.getsource(cls))
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef))

    if dataclasses.is_dataclass(cls):
        new_decorators = []
        found = False
        for dec in node.decorator_list:
            is_dataclass = isinstance(dec, ast.Name) and dec.id == "dataclass"
            is_dataclass_call = (
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Name)
                and dec.func.id == "dataclass"
            )
            if is_dataclass:
                dec = ast.Call(
                    func=ast.Name(id="dataclass", ctx=ast.Load()),
                    args=[],
                    keywords=[ast.keyword(arg="slots", value=ast.Constant(True))],
                )
                found = True
            elif is_dataclass_call:
                if not any(kw.arg == "slots" for kw in dec.keywords):
                    dec.keywords.append(ast.keyword(arg="slots", value=ast.Constant(True)))
                found = True
            new_decorators.append(dec)
        node.decorator_list = new_decorators
        if not found:
            raise RuntimeError(f"Could not find dataclass decorator for {cls.__name__}")
    else:
        slots = ast.Assign(
            targets=[ast.Name(id="__slots__", ctx=ast.Store())],
            value=ast.Tuple(
                elts=[ast.Constant(name) for name in sorted(attrs)],
                ctx=ast.Load(),
            ),
        )
        node.body.insert(0, slots)

    ast.fix_missing_locations(tree)
    return compile(tree, inspect.getsourcefile(cls) or "<slotified>", "exec")


def make_slotted_root(root_cls, representative):
    """Rebuild game-local state classes with slots, without changing repository code."""
    module = sys.modules[root_cls.__module__]
    attrs_by_cls, _ = _local_state_classes(representative, root_cls.__module__)
    attrs_by_cls.setdefault(root_cls, set(representative.__dict__.keys()))

    namespace = dict(vars(module))
    classes = sorted(
        attrs_by_cls,
        key=lambda cls: inspect.getsourcelines(cls)[1],
    )
    for cls in classes:
        code = _slotify_source(cls, attrs_by_cls[cls])
        exec(code, namespace, namespace)

    return namespace[root_cls.__name__], [cls.__name__ for cls in classes]


def _maker_info(desc):
    maker = desc.make_game
    if isinstance(maker, functools.partial):
        return maker.func, dict(maker.keywords or {})
    return maker, {}


def _advance(game, steps, rng):
    indices = []
    for _ in range(steps):
        if game.ended() or not game.moves:
            break
        idx = rng.randrange(len(game.moves))
        indices.append(idx)
        game.play_idx(idx)
    return indices


def _replay(game, indices):
    for idx in indices:
        if game.ended() or not game.moves:
            break
        if idx >= len(game.moves):
            raise RuntimeError(f"replay action {idx} out of {len(game.moves)}")
        game.play_idx(idx)


def _retained_bytes_per_copy(game, n):
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    copies = [game.copy() for _ in range(n)]
    current, peak = tracemalloc.get_traced_memory()
    retained = max(current - before, 0)
    peak_delta = max(peak - before, 0)
    result = retained / n, peak_delta / n
    del copies
    tracemalloc.stop()
    gc.collect()
    return result


def _copy_us(game, n):
    for _ in range(min(100, n)):
        game.copy()
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter()
        copies = [game.copy() for _ in range(n)]
        elapsed = time.perf_counter() - start
    finally:
        if enabled:
            gc.enable()
    del copies
    gc.collect()
    return elapsed * 1e6 / n


def _root_overhead(game):
    root = sys.getsizeof(game)
    d = getattr(game, "__dict__", None)
    return root, 0 if d is None else sys.getsizeof(d), len(d or {})


def _fmt_bytes(value):
    if value < 1024:
        return f"{value:.0f} B"
    return f"{value / 1024:.2f} KiB"


def _bench_variant(game, n):
    retained, peak = _retained_bytes_per_copy(game, n)
    copy_us = _copy_us(game, n)
    root, dct, attrs = _root_overhead(game)
    return {
        "retained": retained,
        "peak": peak,
        "copy_us": copy_us,
        "root": root,
        "dict": dct,
        "attrs": attrs,
    }


def main():
    print(f"copies={COPIES} prefix_steps={PREFIX_STEPS} python={sys.version.split()[0]}")
    print(
        "game\tvariant\tretained/copy\t100k retained\tcopy us\troot\tdict\tattrs\tslotted classes"
    )

    for name in sorted(games_library.registry):
        try:
            random.seed(SEED)
            desc = games_library(name)
            root_cls, kwargs = _maker_info(desc)
            original = root_cls(**kwargs)
            indices = _advance(original, PREFIX_STEPS, random.Random(SEED + 1))
            current = _bench_variant(original, COPIES)
            print(
                f"{name}\tcurrent\t{_fmt_bytes(current['retained'])}\t"
                f"{current['retained'] * 100000 / (1024**2):.1f} MiB\t"
                f"{current['copy_us']:.2f}\t{current['root']}\t{current['dict']}\t"
                f"{current['attrs']}\t-"
            )

            try:
                if not inspect.isclass(root_cls) or not hasattr(original, "__dict__"):
                    raise RuntimeError("root class has no Python __dict__ (already compact or extension type)")
                slotted_cls, slotted_classes = make_slotted_root(root_cls, original)
                random.seed(SEED)
                slotted = slotted_cls(**kwargs)
                _replay(slotted, indices)
                compact = _bench_variant(slotted, COPIES)
                delta_mem = 100 * (compact["retained"] / current["retained"] - 1)
                delta_copy = 100 * (compact["copy_us"] / current["copy_us"] - 1)
                print(
                    f"{name}\tslotted\t{_fmt_bytes(compact['retained'])}\t"
                    f"{compact['retained'] * 100000 / (1024**2):.1f} MiB\t"
                    f"{compact['copy_us']:.2f}\t{compact['root']}\t{compact['dict']}\t"
                    f"{compact['attrs']}\t{','.join(slotted_classes)}"
                )
                print(f"{name}\tdelta\t{delta_mem:+.1f}% memory\t-\t{delta_copy:+.1f}% copy\t-\t-\t-\t-")
            except Exception as exc:
                print(f"{name}\tslotted-unavailable\t{type(exc).__name__}: {exc}")
        except Exception as exc:
            print(f"{name}\tERROR\t{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
