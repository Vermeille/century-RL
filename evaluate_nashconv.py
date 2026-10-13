"""Evaluate a Century checkpoint with OpenSpiel NashConv."""

from __future__ import annotations

import argparse
import json

from boardrl.games.openspiel.evaluation import nash_conv
from boardrl.games import games_library
from boardrl.games.openspiel.game import _require_pyspiel
from boardrl.rl.model import load_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--model", default=None, help="checkpoint model name")
    parser.add_argument(
        "--game", default="leduc", help="BoardRL preset or native OpenSpiel spec"
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--backend", choices=("auto", "openspiel"), default="auto")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--python-br",
        action="store_true",
        help="use OpenSpiel's Python best response instead of the faster C++ implementation",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
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
    )


def resolve_game(spec):
    if spec.split(",", 1)[0] in games_library.registry:
        return games_library(spec).make_game()._game
    return _require_pyspiel().load_game(spec)


if __name__ == "__main__":
    main()
