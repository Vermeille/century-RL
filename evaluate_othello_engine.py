"""Evaluate a BoardRL Othello checkpoint against a GTP engine."""

from __future__ import annotations

import argparse
import json
import random
import shlex

from boardrl import Evaluator, Inference
from boardrl.games.openspiel.othello.game import Othello
from boardrl.games.openspiel.othello.gtp import GTPEngine
from boardrl.games.openspiel.othello.strategies import GTPStrategy
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


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.pairs <= 0 or args.opening_plies < 0:
        parser.error("pairs must be positive and opening-plies non-negative")
    model = load_model(args.checkpoint, name=args.model, device=args.device)
    inference = Inference(model, batch_size=2)
    rng = random.Random(args.seed)
    payoffs = []
    with GTPEngine(shlex.split(args.engine)) as engine, inference.evaluating():
        player = inference.policy(temperature=args.temperature)
        opponent = GTPStrategy(engine)
        for _ in range(args.pairs):
            opening = Othello()
            for _ in range(args.opening_plies):
                if opening.ended():
                    break
                opening.play_idx(rng.randrange(len(opening.moves)))
            evaluator = Evaluator(
                lambda num_players, opening=opening: opening.copy(), progress=False
            )
            result = evaluator.compare(
                [player, opponent],
                names=["model", "engine"],
                games=2,
                max_steps=opening._game.max_game_length(),
                rotate=True,
            )
            payoffs.extend(result.rollouts.my_points(0))
    wins = sum(payoff > 0 for payoff in payoffs)
    draws = payoffs.count(0)
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
                "games": len(payoffs),
                "wins": wins,
                "draws": draws,
                "losses": len(payoffs) - wins - draws,
                "score": (wins + 0.5 * draws) / len(payoffs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
