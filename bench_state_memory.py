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
    for obj in _walk_objects(root):
        cls = type(obj)
        if cls.__module__ != module_name or not hasattr(obj, "__dict__"):
            continue
        attrs[cls].update(obj.__dict__.keys())
    return attrs


class _CopyOptimizer(ast.NodeTransformer):
    """Model the boring source changes we would actually keep in the games.

    - copies share `moves`: the repository never mutates a moves list in place,
      it replaces it on transitions;
    - constructor-based copy methods allocate with object.__new__ and then copy
      fields instead of initializing a fresh random game just to overwrite it.
    """

    def __init__(self, cls_name, attrs):
        self.cls_name = cls_name
        self.attrs = sorted(attrs)
        self.in_copy = False

    def visit_FunctionDef(self, node):
        if node.name != "copy":
            return self.generic_visit(node)

        previous = self.in_copy
        self.in_copy = True
        node = self.generic_visit(node)
        self.in_copy = previous

        new_body = []
        bypassed_constructor = False
        for stmt in node.body:
            if (
                not bypassed_constructor
                and isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id == self.cls_name
            ):
                target_name = stmt.targets[0].id
                stmt.value = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="object", ctx=ast.Load()),
                        attr="__new__",
                        ctx=ast.Load(),
                    ),
                    args=[ast.Name(id=self.cls_name, ctx=ast.Load())],
                    keywords=[],
                )
                new_body.append(stmt)
                # Preserve fields that the old constructor supplied but the copy
                # method did not explicitly overwrite (usually static config).
                for attr in self.attrs:
                    new_body.append(
                        ast.Assign(
                            targets=[
                                ast.Attribute(
                                    value=ast.Name(id=target_name, ctx=ast.Load()),
                                    attr=attr,
                                    ctx=ast.Store(),
                                )
                            ],
                            value=ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr=attr,
                                ctx=ast.Load(),
                            ),
                        )
                    )
                bypassed_constructor = True
            else:
                new_body.append(stmt)
        node.body = new_body
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if not self.in_copy:
            return node
        if any(
            isinstance(target, ast.Attribute)
            and target.attr == "moves"
            and isinstance(target.value, ast.Name)
            and target.value.id != "self"
            for target in node.targets
        ):
            node.value = ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load()),
                attr="moves",
                ctx=ast.Load(),
            )
        return node


class _ConnectFourFlattener(ast.NodeTransformer):
    """Use one 42-cell list instead of 7 separately allocated column lists."""

    @staticmethod
    def _self_board(node):
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr == "board"
        )

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Attribute):
            return node
        target = node.targets[0]
        if self._self_board(target):
            node.value = ast.BinOp(
                left=ast.List(elts=[], ctx=ast.Load()),
                op=ast.Mult(),
                right=ast.BinOp(
                    left=ast.Attribute(ast.Name("self", ast.Load()), "width", ast.Load()),
                    op=ast.Mult(),
                    right=ast.Attribute(ast.Name("self", ast.Load()), "height", ast.Load()),
                ),
            )
            # [None] rather than []
            node.value.left.elts.append(ast.Constant(None))
        elif (
            target.attr == "board"
            and isinstance(target.value, ast.Name)
            and target.value.id != "self"
        ):
            node.value = ast.Subscript(
                value=ast.Attribute(ast.Name("self", ast.Load()), "board", ast.Load()),
                slice=ast.Slice(lower=None, upper=None, step=None),
                ctx=ast.Load(),
            )
        return node

    def visit_Subscript(self, node):
        node = self.generic_visit(node)
        if not isinstance(node.value, ast.Subscript):
            return node
        inner = node.value
        if not self._self_board(inner.value):
            return node
        index = ast.BinOp(
            left=ast.BinOp(
                left=inner.slice,
                op=ast.Mult(),
                right=ast.Attribute(ast.Name("self", ast.Load()), "height", ast.Load()),
            ),
            op=ast.Add(),
            right=node.slice,
        )
        return ast.copy_location(
            ast.Subscript(
                value=ast.Attribute(ast.Name("self", ast.Load()), "board", ast.Load()),
                slice=index,
                ctx=node.ctx,
            ),
            node,
        )


class _HanabiKnowledgeTuples(ast.NodeTransformer):
    """Possible colors/ranks are tiny immutable domains; tuples beat per-card sets."""

    def visit_ClassDef(self, node):
        if node.name != "CardKnowledge":
            return self.generic_visit(node)
        for fn in node.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if fn.name == "unknown":
                fn.body = [
                    ast.Return(
                        ast.Call(
                            func=ast.Name("cls", ast.Load()),
                            args=[
                                ast.Call(ast.Name("tuple", ast.Load()), [ast.Name("colors", ast.Load())], []),
                                ast.Call(ast.Name("tuple", ast.Load()), [ast.Name("ranks", ast.Load())], []),
                            ],
                            keywords=[],
                        )
                    )
                ]
            elif fn.name == "copy":
                fn.body = [
                    ast.Return(
                        ast.Call(
                            func=ast.Name("CardKnowledge", ast.Load()),
                            args=[
                                ast.Attribute(ast.Name("self", ast.Load()), "colors", ast.Load()),
                                ast.Attribute(ast.Name("self", ast.Load()), "ranks", ast.Load()),
                                ast.Attribute(ast.Name("self", ast.Load()), "hinted_color", ast.Load()),
                                ast.Attribute(ast.Name("self", ast.Load()), "hinted_rank", ast.Load()),
                            ],
                            keywords=[],
                        )
                    )
                ]
            elif fn.name == "reveal_color":
                fn.body = ast.parse(
                    """
if true_color == hinted_color:
    self.colors = (hinted_color,)
    self.hinted_color = hinted_color
else:
    self.colors = tuple(color for color in self.colors if color != hinted_color)
"""
                ).body
            elif fn.name == "reveal_rank":
                fn.body = ast.parse(
                    """
if true_rank == hinted_rank:
    self.ranks = (hinted_rank,)
    self.hinted_rank = hinted_rank
else:
    self.ranks = tuple(rank for rank in self.ranks if rank != hinted_rank)
"""
                ).body
        return self.generic_visit(node)


def _slotify_source(cls, attrs, *, optimize_copy=False, compact_special=False):
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
        node.body.insert(
            0,
            ast.Assign(
                targets=[ast.Name(id="__slots__", ctx=ast.Store())],
                value=ast.Tuple(
                    elts=[ast.Constant(name) for name in sorted(attrs)],
                    ctx=ast.Load(),
                ),
            ),
        )

    if compact_special and cls.__name__ == "CardKnowledge":
        tree = _HanabiKnowledgeTuples().visit(tree)
    if compact_special and cls.__name__ == "ConnectFour":
        tree = _ConnectFourFlattener().visit(tree)
    if optimize_copy:
        tree = _CopyOptimizer(cls.__name__, attrs).visit(tree)

    ast.fix_missing_locations(tree)
    return compile(tree, inspect.getsourcefile(cls) or "<slotified>", "exec")


def make_compact_root(root_cls, representative):
    """Rebuild game state classes with readable candidate optimizations."""
    module = sys.modules[root_cls.__module__]
    attrs_by_cls = _local_state_classes(representative, root_cls.__module__)
    attrs_by_cls.setdefault(root_cls, set(representative.__dict__.keys()))

    # Rebuilding nested classes only helps when copies actually allocate lots of
    # them. It also avoids needlessly disturbing module-global helper functions.
    keep_nested = root_cls.__module__ in {
        "boardrl.games.hanabi.game",
        "boardrl.games.splendor.game",
    }
    classes = [root_cls]
    if keep_nested:
        classes = [cls for cls, attrs in attrs_by_cls.items() if attrs or cls is root_cls]
        classes.sort(key=lambda cls: inspect.getsourcelines(cls)[1])

    namespace = dict(vars(module))
    for cls in classes:
        code = _slotify_source(
            cls,
            attrs_by_cls[cls],
            optimize_copy=cls is root_cls,
            compact_special=True,
        )
        exec(code, namespace, namespace)

    return namespace[root_cls.__name__], [cls.__name__ for cls in classes]


def _maker_info(desc):
    maker = desc.make_game
    if isinstance(maker, functools.partial):
        return maker.func, dict(maker.keywords or {})
    return maker, {}


def _advance(game, steps, rng):
    actions = []
    for _ in range(steps):
        if game.ended() or not game.moves:
            break
        move = game.moves[rng.randrange(len(game.moves))]
        actions.append(move)
        game.play_str(move)
    return actions


def _replay(game, actions):
    for move in actions:
        if game.ended() or not game.moves:
            break
        if move not in game.moves:
            raise RuntimeError(f"replay move {move!r} is not legal")
        game.play_str(move)


def _copy_game(game):
    try:
        params = inspect.signature(game.copy).parameters
    except (TypeError, ValueError):
        params = {}
    if "randomize" in params:
        return game.copy(randomize=False)
    return game.copy()


def _retained_bytes_per_copy(game, n):
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    copies = [_copy_game(game) for _ in range(n)]
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
        _copy_game(game)
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter()
        copies = [_copy_game(game) for _ in range(n)]
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
        "game\tvariant\tretained/copy\t100k retained\tcopy us\troot\tdict\tattrs\tclasses"
    )

    for name in sorted(games_library.registry):
        try:
            random.seed(SEED)
            desc = games_library(name)
            root_cls, kwargs = _maker_info(desc)
            original = root_cls(**kwargs)
            actions = _advance(original, PREFIX_STEPS, random.Random(SEED + 1))
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
                compact_cls, compact_classes = make_compact_root(root_cls, original)
                random.seed(SEED)
                compact_game = compact_cls(**kwargs)
                _replay(compact_game, actions)
                compact = _bench_variant(compact_game, COPIES)
                delta_mem = 100 * (compact["retained"] / current["retained"] - 1)
                delta_copy = 100 * (compact["copy_us"] / current["copy_us"] - 1)
                print(
                    f"{name}\tcompact\t{_fmt_bytes(compact['retained'])}\t"
                    f"{compact['retained'] * 100000 / (1024**2):.1f} MiB\t"
                    f"{compact['copy_us']:.2f}\t{compact['root']}\t{compact['dict']}\t"
                    f"{compact['attrs']}\t{','.join(compact_classes)}"
                )
                print(f"{name}\tdelta\t{delta_mem:+.1f}% memory\t-\t{delta_copy:+.1f}% copy\t-\t-\t-\t-")
            except Exception as exc:
                print(f"{name}\tcompact-unavailable\t{type(exc).__name__}: {exc}")
        except Exception as exc:
            print(f"{name}\tERROR\t{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
