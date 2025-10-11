import argparse
import os
import random
from itertools import combinations
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from natsort import natsorted

from boardrl.games import games_library
from boardrl.rl.eval.matchmaker import MatchMaker
from boardrl.rl.eval.selfplay import pit
from boardrl.utils import ModelPool


def discover_strategies(
    game_desc,
    model_dir: str | None = None,
    *,
    include_zero_arg: bool = True,
    include_greedy: bool = False,
):
    """Return a list of strategy descriptor strings for the chosen game.

    - Includes all zero-argument strategies registered for the game (e.g. "random",
      game-specific baselines like "random_buy").
    - Discovers model-based strategies from any .pth files and exposes them as
      policy sampling strategies using the updated descriptor format
      (e.g. "policy_sampling,model=/path/model.pth").
    """

    # Zero-arg strategies registered for this game
    zero_arg: list[str] = []
    if include_zero_arg:
        zero_arg = [
            name
            for name, (_, arg_info) in game_desc.strategy_from_string.registry.items()
            if len(arg_info) == 0
        ]

    # Model-based strategies discovered from files
    model_strats = []
    search_root = model_dir or "."
    for root, dirs, files in os.walk(search_root):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if f.endswith(".pth"):
                model_path = os.path.join(root, f)
                model_strats.append(f"policy_sampling,model={model_path}")
                if include_greedy:
                    model_strats.append(f"argmax,model={model_path}")

    return natsorted(zero_arg + model_strats)


def anchor_elo(maker: MatchMaker, name: str, rating: float) -> None:
    """Shift all ratings so that ``name`` has exactly ``rating``.

    Elo expected scores depend only on differences, so adding a constant offset
    to all ratings preserves pairwise expectations while providing a stable
    scale anchored to a baseline strategy like ``random``.
    """
    # Ensure the anchor exists in the table
    maker.ensure_strategy_registered(name)
    current = maker._elo[name]
    offset = current - rating
    if offset == 0:
        return
    for k in list(maker._elo.keys()):
        maker._elo[k] -= offset


def P_win(r1: torch.Tensor, r2: torch.Tensor, scale: float):
    return 1.0 / (1.0 + 10.0 ** ((r2 - r1) / scale))


def bce_fit_elo(
    players: list[str],
    matches: list[dict],
    *,
    scale: float,
    anchor_name: str,
    anchor_rating: float,
    epochs: int = 60,
    lr: float = 2.0,
):
    # Map players to indices
    p_to_i = {p: i for i, p in enumerate(players)}
    i1 = torch.tensor([p_to_i[m["player1"]] for m in matches])
    i2 = torch.tensor([p_to_i[m["player2"]] for m in matches])
    results = torch.tensor(
        [float(m["points"] >= 0) for m in matches], dtype=torch.float32
    )

    # Ratings to learn
    R = nn.Embedding(len(players), 1)
    with torch.no_grad():
        R.weight.data.fill_(anchor_rating)

    opt = SGD([R.weight], lr=lr, momentum=0.8)
    for _ in range(epochs):
        opt.zero_grad()
        r1 = R(i1).squeeze(-1)
        r2 = R(i2).squeeze(-1)
        p_hat = P_win(r1, r2, scale)
        loss = F.binary_cross_entropy(p_hat, results, reduction="sum")
        loss.backward()
        opt.step()
        # Recenter to fixed anchor each step
        if anchor_name in p_to_i:
            with torch.no_grad():
                offset = R.weight[p_to_i[anchor_name], 0] - anchor_rating
                R.weight[:, 0] -= offset

    with torch.no_grad():
        ratings = R.weight[:, 0].cpu().numpy().tolist()
    return {p: ratings[i] for p, i in p_to_i.items()}


def run_opt_method(
    game_desc,
    strategies: list[str],
    *,
    pool: ModelPool,
    games_per_pair: int,
    max_len: int,
    rotate: bool,
    scale: float,
    anchor_name: str,
    anchor_rating: float,
    epochs: int,
    lr: float,
    iterations: int,
    convergence_tol: float,
    convergence_patience: int,
):
    # For now, assume 2-player games for pairwise outcomes
    n_players = game_desc.make_game().num_players
    if n_players != 2:
        raise SystemExit("Cross-entropy Elo currently supports 2-player games only.")

    matches: list[dict] = []
    if len(strategies) < 2:
        raise SystemExit("Need at least two strategies to compare.")

    # Initialize current ratings and pair counts
    ratings: dict[str, float] = {s: anchor_rating for s in strategies}
    pair_counts: dict[tuple[str, str], int] = {}
    order = strategies[:]
    cursor = 0
    prev: dict[str, float] | None = None
    stable = 0
    step = 0

    def pair_key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def choose_closest(a: str) -> str:
        # Pick opponent with closest current Elo; tie-break on fewest prior matches, then name
        opps = [s for s in strategies if s != a]
        best = min(
            opps,
            key=lambda o: (
                abs(ratings[a] - ratings[o]),
                pair_counts.get(pair_key(a, o), 0),
                o,
            ),
        )
        return best

    target_steps = None if iterations == -1 else max(iterations, len(strategies))

    while True:
        if target_steps is not None and step >= target_steps:
            return ratings

        a = order[cursor]
        cursor = (cursor + 1) % len(order)
        b = choose_closest(a)

        # Play the scheduled pair
        strats = [
            game_desc.strategy_from_string(a, model=pool),
            game_desc.strategy_from_string(b, model=pool),
        ]
        res = pit(game_desc.make_game, strats, games_per_pair, max_len, rotate=rotate)
        for pts in res.my_points(0):
            matches.append({"player1": a, "player2": b, "points": pts})
        pk = pair_key(a, b)
        pair_counts[pk] = pair_counts.get(pk, 0) + 1
        step += 1

        # Fit ratings on accumulated matches; pass full candidate list so unseen remain anchored
        ratings = bce_fit_elo(
            strategies,
            matches,
            scale=scale,
            anchor_name=anchor_name,
            anchor_rating=anchor_rating,
            epochs=epochs,
            lr=lr,
        )

        # Convergence check (only meaningful if iterations == -1)
        if iterations == -1 and prev is not None:
            keys = set(prev) | set(ratings)
            delta = max(
                abs(ratings.get(k, anchor_rating) - prev.get(k, anchor_rating))
                for k in keys
            )
            if delta < convergence_tol:
                stable += 1
            else:
                stable = 0
            if stable >= convergence_patience:
                top = "\n".join(
                    f"{name}:{ratings[name]:.1f}"
                    for name in sorted(ratings, key=ratings.get, reverse=True)
                )
                print(f"[opt converged @ step {step}] {a} vs {b}  ->  {top}")
                return ratings
        prev = ratings

        top = "\n".join(
            f"{name}:{ratings[name]:.1f}"
            for name in sorted(ratings, key=ratings.get, reverse=True)
        )
        tag = f"{step}/{target_steps}" if target_steps is not None else f"step {step}"
        print(f"[opt {tag}] {a} vs {b}  ->  {top}")


def main():
    parser = argparse.ArgumentParser(description="Run Elo by playing games.")
    parser.add_argument("--game", type=str, default="century", help="Game name")
    parser.add_argument(
        "--method",
        choices=["opt", "online"],
        default="opt",
        help="Elo estimation method: 'opt' (cross-entropy fit) or 'online' (incremental updates)",
    )
    # Online method controls
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Match iterations (online: rounds; opt: sampled pairs, -1 for convergence)",
    )
    parser.add_argument(
        "--num_games", type=int, default=32, help="[online] Games per iteration"
    )
    parser.add_argument("--max_len", type=int, default=200, help="Max length per game")
    parser.add_argument(
        "--elo_k", type=float, default=32.0, help="[online] Elo K-factor"
    )
    parser.add_argument(
        "--discount_factor", type=float, default=0.99, help="Discount for returns"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="ModelPool batch size"
    )
    parser.add_argument(
        "--timeout", type=float, default=0.01, help="ModelPool batch timeout"
    )
    parser.add_argument(
        "--rotate", action="store_true", help="Rotate player order across games"
    )
    parser.add_argument(
        "--list", action="store_true", help="List discovered strategies and exit"
    )
    parser.add_argument(
        "--anchor_name",
        type=str,
        default="random",
        help="Strategy name to anchor Elo (default: random)",
    )
    parser.add_argument(
        "--anchor_rating",
        type=float,
        default=1500.0,
        help="Fixed Elo rating for the anchor strategy",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Restrict model discovery to this directory",
    )
    # Cross-entropy method controls
    parser.add_argument(
        "--games_per_pair",
        type=int,
        default=32,
        help="[opt] Games to play per pair of strategies",
    )
    parser.add_argument(
        "--elo_scale",
        type=float,
        default=400.0,
        help="[opt] Elo scale (400 for standard Elo)",
    )
    parser.add_argument(
        "--epochs", type=int, default=80, help="[opt] Optimization epochs for BCE fit"
    )
    parser.add_argument(
        "--lr", type=float, default=32.0, help="[opt] SGD learning rate for BCE fit"
    )
    parser.add_argument(
        "--convergence_tol",
        type=float,
        default=0.5,
        help="[opt] Elo convergence tolerance in rating points when --iterations=-1",
    )
    parser.add_argument(
        "--convergence_patience",
        type=int,
        default=5,
        help="[opt] Consecutive steps below tolerance required to stop when --iterations=-1",
    )
    args = parser.parse_args()

    game_desc = games_library(args.game)
    strategies = discover_strategies(game_desc, model_dir=args.model_dir)
    if args.list:
        print("Discovered strategies:")
        for s in strategies:
            print(" -", s)
        return

    if not strategies:
        raise SystemExit("No strategies found. Use --list to debug strategy discovery.")

    # Initialize ModelPool for loading models referenced by strategies
    pool = ModelPool(None, args.batch_size, args.timeout)

    if args.method == "online":
        maker = MatchMaker(
            game_desc,
            model_pool=pool,
            discount_factor=args.discount_factor,
            elo_k=args.elo_k,
        )

        # Determine number of seats/players
        n_players = game_desc.make_game().num_players

        print(f"Game='{args.game}', players={n_players}")
        print(f"Playing {args.iterations} iterations x {args.num_games} games each")
        print(f"Candidates: {len(strategies)} strategies (use --list to view)")

        for i in range(1, args.iterations + 1):
            # Sample strategies for this iteration
            chosen = (
                random.sample(strategies, k=n_players)
                if len(strategies) >= n_players
                else random.choices(strategies, k=n_players)
            )

            maker.run_self_play(
                chosen,
                num_games=args.num_games,
                max_len=args.max_len,
                rotate=args.rotate,
                desc=f"iter {i}/{args.iterations}",
            )
            # Recenter Elo so that the chosen anchor remains at the fixed rating
            anchor_elo(maker, args.anchor_name, args.anchor_rating)

            # Print a quick Elo snapshot each iteration
            ranking = sorted(maker.elo.items(), key=lambda kv: kv[1], reverse=True)
            top = "\n".join([f"{name}:{elo:.1f}" for name, elo in ranking])
            print(f"[iter {i}] {top}")

        # Final standings
        final = sorted(maker.elo.items(), key=lambda kv: kv[1], reverse=True)
        print("\nFinal Elo ratings:")
        for name, rating in final:
            print(f"{name}: {rating:.2f}")

    else:  # args.method == 'opt'
        print(f"Game='{args.game}' — cross-entropy Elo fit (sampled pairs)")
        if args.iterations == -1:
            print(
                f"Evaluating {len(strategies)} strategies; until convergence, games_per_pair={args.games_per_pair}"
            )
        else:
            print(
                f"Evaluating {len(strategies)} strategies; iterations={args.iterations}, games_per_pair={args.games_per_pair}"
            )
        ratings = run_opt_method(
            game_desc,
            strategies,
            pool=pool,
            games_per_pair=args.games_per_pair,
            max_len=args.max_len,
            rotate=args.rotate,
            scale=args.elo_scale,
            anchor_name=args.anchor_name,
            anchor_rating=args.anchor_rating,
            epochs=args.epochs,
            lr=args.lr,
            iterations=args.iterations,
            convergence_tol=args.convergence_tol,
            convergence_patience=args.convergence_patience,
        )
        final = sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)
        print("\nFitted Elo ratings:")
        for name, rating in final:
            print(f"{name}: {rating:.2f}")


if __name__ == "__main__":
    main()
