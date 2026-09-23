import functools
import gc
import inspect
import os
import random
import tracemalloc

from boardrl.games import games_library
from bench_state_memory import make_compact_root


SEED = 24680
STATES = int(os.environ.get("STATE_STORAGE_STATES", "5000"))
GAME_FILTER = os.environ.get("GAME_FILTER")


def maker_info(desc):
    maker = desc.make_game
    if isinstance(maker, functools.partial):
        return maker.func, dict(maker.keywords or {})
    return maker, {}


def has_randomize_copy(cls):
    try:
        return "randomize" in inspect.signature(cls.copy).parameters
    except (TypeError, ValueError):
        return False


def snapshot(game, has_randomize):
    return game.copy(randomize=False) if has_randomize else game.copy()


def retain_trajectory_states(cls, kwargs, n, seed):
    """Retain snapshots from distinct states, restarting episodes as needed.

    This models rollout/replay storage rather than N branches from one identical
    node. A copied state may share its current moves list with the live game, but
    once the live game advances that old list is retained only by the snapshot,
    which is the real memory behavior we care about.
    """
    rng = random.Random(seed + 1)
    random.seed(seed)
    has_randomize = has_randomize_copy(cls)
    game = cls(**kwargs)
    states = []

    while len(states) < n:
        if game.ended() or not game.moves:
            game = cls(**kwargs)
            continue
        states.append(snapshot(game, has_randomize))
        move = game.moves[rng.randrange(len(game.moves))]
        game.play_str(move)

    return states


def retained_bytes(cls, kwargs, n, seed):
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    states = retain_trajectory_states(cls, kwargs, n, seed)
    current, peak = tracemalloc.get_traced_memory()
    retained = max(0, current - before)
    result = retained / len(states), max(0, peak - before) / len(states)
    del states
    tracemalloc.stop()
    gc.collect()
    return result


def fmt_bytes(value):
    return f"{value:.0f} B" if value < 1024 else f"{value / 1024:.2f} KiB"


def main():
    print(f"states={STATES}")
    print("game\tvariant\tretained/state\t100k retained\tdelta")

    names = sorted(games_library.registry)
    if GAME_FILTER:
        names = [name for name in names if name == GAME_FILTER]

    for offset, name in enumerate(names):
        desc = games_library(name)
        cls, kwargs = maker_info(desc)
        seed = SEED + offset * 1000

        current, _ = retained_bytes(cls, kwargs, STATES, seed)
        print(
            f"{name}\tcurrent\t{fmt_bytes(current)}\t"
            f"{current * 100_000 / (1024**2):.1f} MiB\t-"
        )

        # Cython extension states are already slotted/compact at the object
        # level. Game-specific source experiments are benchmarked separately.
        representative = cls(**kwargs)
        if not inspect.isclass(cls) or not hasattr(representative, "__dict__"):
            continue

        try:
            compact_cls, _ = make_compact_root(cls, representative)
            compact, _ = retained_bytes(compact_cls, kwargs, STATES, seed)
            delta = 100 * (compact / current - 1)
            print(
                f"{name}\tcompact\t{fmt_bytes(compact)}\t"
                f"{compact * 100_000 / (1024**2):.1f} MiB\t{delta:+.1f}%"
            )
        except Exception as exc:
            print(f"{name}\tcompact-error\t{type(exc).__name__}: {exc}\t-\t-")


if __name__ == "__main__":
    main()
