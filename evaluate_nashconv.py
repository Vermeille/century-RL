"""Evaluate a Century checkpoint with OpenSpiel NashConv."""

from __future__ import annotations

import argparse
import json

from boardrl.benchmarks.openspiel import nash_conv
from boardrl.rl.model import load_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--model", default=None, help="checkpoint model name")
    parser.add_argument("--game", default="leduc_poker")
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
    result = nash_conv(
        model,
        args.game,
        temperature=args.temperature,
        use_cpp_br=not args.python_br,
    )
    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "model": args.model,
                "game": args.game,
                "temperature": args.temperature,
                "nash_conv": result.nash_conv,
                "exploitability": result.exploitability,
                "player_improvements": result.player_improvements,
                "evaluated_states": result.evaluated_states,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
