import gc
import inspect
import random
import time

from boardrl.games import games_library
from bench_state_memory import SEED, _advance, _maker_info, _replay, make_compact_root


COPIES = 20_000
PREFIX_STEPS = 20


def make_copier(game):
    """Resolve optional randomization once, outside the timed loop."""
    try:
        has_randomize = "randomize" in inspect.signature(game.copy).parameters
    except (TypeError, ValueError):
        has_randomize = False
    if has_randomize:
        return lambda: game.copy(randomize=False)
    return game.copy


def copy_us(game):
    copier = make_copier(game)
    for _ in range(200):
        copier()
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter()
        copies = [copier() for _ in range(COPIES)]
        elapsed = time.perf_counter() - start
    finally:
        if enabled:
            gc.enable()
    del copies
    gc.collect()
    return elapsed * 1e6 / COPIES


def main():
    print(f"copies={COPIES} prefix_steps={PREFIX_STEPS}")
    print("game\tcurrent_us\tcompact_us\tdelta")
    for name in sorted(games_library.registry):
        random.seed(SEED)
        desc = games_library(name)
        root_cls, kwargs = _maker_info(desc)
        game = root_cls(**kwargs)
        actions = _advance(game, PREFIX_STEPS, random.Random(SEED + 1))
        current = copy_us(game)

        if not inspect.isclass(root_cls) or not hasattr(game, "__dict__"):
            print(f"{name}\t{current:.3f}\t-\t-")
            continue
        try:
            compact_cls, _ = make_compact_root(root_cls, game)
            random.seed(SEED)
            compact_game = compact_cls(**kwargs)
            _replay(compact_game, actions)
            compact = copy_us(compact_game)
            delta = 100 * (compact / current - 1)
            print(f"{name}\t{current:.3f}\t{compact:.3f}\t{delta:+.1f}%")
        except Exception as exc:
            print(f"{name}\t{current:.3f}\tERROR {type(exc).__name__}: {exc}\t-")


if __name__ == "__main__":
    main()
