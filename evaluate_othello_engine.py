"""Evaluate a Century Othello checkpoint against a GTP engine."""

from __future__ import annotations

import argparse
import json
import shlex

from boardrl.benchmarks.gtp import GTPEngine
from boardrl.benchmarks.othello import OthelloEngineBenchmark
from boardrl.rl.model import load_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument(
        "--engine",
        required=True,
        help="GTP engine command, including its GTP/level flags",
    )
    parser.add_argument("--model", default=None, help="checkpoint model name")
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--opening-plies", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = load_model(args.checkpoint, name=args.model, device=args.device)
    with GTPEngine(shlex.split(args.engine)) as engine:
        result = OthelloEngineBenchmark(
            model,
            engine,
            temperature=args.temperature,
            opening_plies=args.opening_plies,
            seed=args.seed,
        ).run(pairs=args.pairs)
    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "model": args.model,
                "engine": args.engine,
                "pairs": args.pairs,
                "opening_plies": args.opening_plies,
                "seed": args.seed,
                "temperature": args.temperature,
                "games": result.games,
                "wins": result.wins,
                "draws": result.draws,
                "losses": result.losses,
                "score": result.score,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
