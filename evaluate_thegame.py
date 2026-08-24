"""Evaluate The Game policies under one reproducible protocol."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import torch

from boardrl.evaluation import Evaluator
from boardrl.games import games_library
from boardrl.games.strategies import Gumbel
from boardrl.games.thegame.metrics import Metrics
from boardrl.games.thegame.strategies import LowestCostStrategy
from boardrl.rl.model import load_model
from boardrl.rollouts import Inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="*")
    parser.add_argument("--game", default="thegame,mode=strict")
    parser.add_argument("--games", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        help="evaluate every checkpoint at each temperature",
    )
    parser.add_argument("--gumbel-evals", type=int)
    parser.add_argument(
        "--gumbel-q-scales",
        type=float,
        nargs="+",
        default=[1.0],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress", action="store_true")
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    from boardrl.cyutils import init_seed

    init_seed(seed)


def score_summary(values) -> dict[str, float | int | list[float]]:
    values = list(values)
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    half_width = 1.96 * stdev / math.sqrt(len(values))
    return {
        "games": len(values),
        "mean": mean,
        "stdev": stdev,
        "ci95": [mean - half_width, mean + half_width],
        "min": min(values),
        "max": max(values),
    }


def evaluate(
    name,
    player,
    *,
    game,
    games: int,
    seed: int,
    progress: bool,
) -> dict:
    seed_everything(seed)
    evaluation = Evaluator(
        game.make_game, progress=progress, coop=game.coop
    ).compare(
        [player, player],
        names=[name, name],
        games=games,
        max_steps=800,
        rotate=False,
    )
    game_metrics = Metrics(evaluation.rollouts).metrics()
    ten_rule_moves = game_metrics["ten_rule_moves"].values
    return {
        "name": name,
        "points": score_summary(evaluation.rollouts.my_points(0)),
        "ratio_lowest_cost": game_metrics["ratio_lowest_cost"],
        "avg_cost": game_metrics["avg_cost"],
        "ten_rule_moves": score_summary(ten_rule_moves),
    }


def main() -> None:
    args = build_parser().parse_args()
    game = games_library(args.game)
    results = [
        evaluate(
            "lowest_cost",
            LowestCostStrategy(),
            game=game,
            games=args.games,
            seed=args.seed,
            progress=args.progress,
        )
    ]
    temperatures = args.temperatures or [args.temperature]
    for path in args.checkpoints:
        model = load_model(path)
        inference = Inference(model, batch_size=args.batch_size, name=str(path))
        for temperature in temperatures:
            results.append(
                evaluate(
                    f"{path} @ temperature={temperature}",
                    inference.policy(temperature=temperature),
                    game=game,
                    games=args.games,
                    seed=args.seed,
                    progress=args.progress,
                )
            )
        if args.gumbel_evals:
            for q_scale in args.gumbel_q_scales:
                results.append(
                    evaluate(
                        (
                            f"{path} @ gumbel-evals={args.gumbel_evals},"
                            f"q-scale={q_scale}"
                        ),
                        Gumbel(
                            inference.processor,
                            discount_factor=1.0,
                            num_evals=args.gumbel_evals,
                            q_scale=q_scale,
                        ),
                        game=game,
                        games=args.games,
                        seed=args.seed,
                        progress=args.progress,
                    )
                )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
