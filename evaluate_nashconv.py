"""Evaluate a BoardRL checkpoint with OpenSpiel NashConv."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from boardrl.checkpoints import Checkpoints
from boardrl.games import games_library
from boardrl.games.openspiel.evaluation import nash_conv
from boardrl.rl.model import load_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("checkpoint", nargs="?", type=Path)
    source.add_argument(
        "--training-run",
        type=Path,
        help=(
            "evaluate the final agent checkpoint using saved run settings; "
            "skip games without a registered exact evaluator"
        ),
    )
    parser.add_argument("--model", default=None, help="checkpoint model name")
    parser.add_argument("--game", default="leduc", help="registered BoardRL game")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, help="also save the JSON result here")
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.training_run is not None:
        from boardrl.arena import read_run_arguments

        arguments = read_run_arguments(args.training_run / "run.txt")
        parser.set_defaults(
            game=arguments["game"],
            model="agent",
            temperature=arguments["eval_temperature"],
            batch_size=arguments["inference_batch_size"],
            device=arguments["device"],
            output=args.training_run / "exploitability.json",
        )
        # Explicit evaluation options still override the recorded run settings.
        args = parser.parse_args(argv)
        if (
            args.game.split(",", 1)[0] in games_library.registry
            and games_library(args.game).exact_evaluation is None
        ):
            print(
                f"Skipping final exploitability: {args.game} has no registered exact evaluator.",
                file=sys.stderr,
            )
            return
        args.checkpoint = Checkpoints(args.training_run).latest
        if args.checkpoint is None:
            parser.error(f"no step checkpoint found in {args.training_run}")

    descriptor = games_library(args.game)
    if descriptor.exact_evaluation is None:
        parser.error(f"{args.game} has no registered exact evaluator")
    game = resolve_game(args.game)
    model = load_model(args.checkpoint, name=args.model, device=args.device)
    result = nash_conv(
        model,
        args.game,
        temperature=args.temperature,
        spiel_game=game,
        batch_size=args.batch_size,
        backend=descriptor.exact_evaluation(game),
    )
    report = json.dumps(
        {
            "checkpoint": str(args.checkpoint),
            "model": args.model,
            "game": args.game,
            "native_game": str(game),
            "temperature": args.temperature,
            "nash_conv": result.nash_conv,
            "exploitability": result.exploitability,
            "player_improvements": result.player_improvements,
            "evaluated_states": result.evaluated_states,
        },
        indent=2,
    )
    if args.output is not None:
        args.output.write_text(report + "\n")
    print(report)


def resolve_game(spec):
    return games_library(spec).make_game(resolve_chance=False)._game


if __name__ == "__main__":
    main()
