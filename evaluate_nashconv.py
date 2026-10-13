"""Evaluate a Century checkpoint with OpenSpiel NashConv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from boardrl.checkpoints import Checkpoints
from boardrl.games.openspiel.evaluation import nash_conv
from boardrl.games import games_library
from boardrl.games.openspiel.game import _require_pyspiel
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
    parser.add_argument(
        "--game", default="leduc", help="BoardRL preset or native OpenSpiel spec"
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--backend", choices=("auto", "openspiel"), default="auto")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, help="also save the JSON result here")
    parser.add_argument(
        "--python-br",
        action="store_true",
        help="use OpenSpiel's Python best response instead of the faster C++ implementation",
    )
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
            and games_library(args.game).model_evaluation is None
        ):
            print(
                f"Skipping final exploitability: {args.game} has no registered exact evaluator.",
                file=sys.stderr,
            )
            return
        args.checkpoint = Checkpoints(args.training_run).latest
        if args.checkpoint is None:
            parser.error(f"no step checkpoint found in {args.training_run}")

    model = load_model(args.checkpoint, name=args.model, device=args.device)
    game = resolve_game(args.game)
    result = nash_conv(
        model,
        args.game,
        temperature=args.temperature,
        use_cpp_br=not args.python_br,
        spiel_game=game,
        batch_size=args.batch_size,
        backend="openspiel" if args.python_br else args.backend,
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
    if spec.split(",", 1)[0] in games_library.registry:
        return games_library(spec).make_game()._game
    return _require_pyspiel().load_game(spec)


if __name__ == "__main__":
    main()
