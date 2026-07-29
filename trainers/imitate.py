"""Supervised strategy-fit diagnostic for model architectures."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from boardrl import (
    Checkpoints,
    Console,
    Visdom,
    Evaluator,
    Inference,
    MetricLogger,
    Range,
    RolloutRunner,
    RunInfo,
)
from boardrl.games import games_library
from boardrl.metrics import rollout_metrics
from boardrl.models import architectures, make
from boardrl.rl.model.loss import (
    CELoss,
)
from boardrl.training import (
    ComputeReturns,
    Learner,
    LinearWarmupDecay,
    Pipeline,
    PolicyMetrics,
    ToSamples,
)
from boardrl.utils.visualizer import OfflineVisualizer, VisdomVisualizer


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--game",
        "-g",
        default="thegame",
        help=f"game specification; available games: {', '.join(games_library.registry)}",
    )
    parser.add_argument(
        "--architecture",
        "-a",
        choices=sorted(architectures),
        default="cnn",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--strategy")
    parser.add_argument("--rollout-games", type=int, default=256)
    parser.add_argument("--evaluation-games", type=int, default=256)
    parser.add_argument("--evaluation-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-beta1", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-clip", type=float)
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--perplexity-start", type=float, default=0.8)
    parser.add_argument("--perplexity-end", type=float, default=0.05)
    parser.add_argument("--entropy-strength", type=float, default=0.1)
    parser.add_argument("--value-strength", type=float, default=1.0)
    parser.add_argument("--kl-target", type=float, default=0.003)
    parser.add_argument("--kl-strength", type=float, default=1.0)
    parser.add_argument("--eval-temperature", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--min-lr-scale", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    starting_point = parser.add_mutually_exclusive_group()
    starting_point.add_argument("--resume", type=Path)
    starting_point.add_argument(
        "--initialize-from",
        type=Path,
        help="load model weights but start a fresh optimizer and step count",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--tag", default="coop")
    parser.add_argument("--visdom-url")
    parser.add_argument("--visdom-port")
    return parser


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    from boardrl.cyutils import init_seed

    init_seed(seed)


class AccuracyMetrics:
    def __call__(self, policy, value, batch):
        correct = sum(
            logits.argmax().item() == action.item()
            for logits, action in zip(policy, batch.action_idx)
        )
        return {"accuracy": correct / len(policy)}


def make_learner(model, args):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, 0.999),
        weight_decay=args.weight_decay,
    )
    learner = Learner(
        model,
        optimizer,
        [CELoss()],
        batch_size=args.batch_size,
        device=args.device,
        epochs=args.epochs,
        gradient_clip=args.gradient_clip,
        # Shuffling destroys lowest_cost's deterministic first-action tie break,
        # making exact supervised labels impossible to recover.
        augmentations=(),
        batch_metrics=[PolicyMetrics(), AccuracyMetrics()],
        normalize_lr=True,
    )
    return learner, optimizer


def run(args):
    seed_everything(args.seed)
    game = games_library(args.game)
    model = make(args.architecture).to(args.device)
    if args.initialize_from:
        Checkpoints(args.initialize_from.parent).load(
            args.initialize_from,
            models={"current": model},
            map_location=args.device,
        )
    learner, optimizer = make_learner(model, args)
    schedule = LinearWarmupDecay(
        optimizer,
        steps=args.steps,
        warmup=args.warmup,
        min_scale=args.min_lr_scale,
    )
    checkpoint_dir = (
        args.checkpoint_root / "imitate" / args.game / args.architecture / args.tag
    )
    checkpoints = Checkpoints(checkpoint_dir, prefix="step")

    visualizer = (
        VisdomVisualizer(args.tag, args.visdom_url, args.visdom_port)
        if args.visdom_url
        else OfflineVisualizer()
    )
    visdom = Visdom(visualizer)
    run_info = RunInfo.capture(args, __file__)
    run_info.publish(visdom)
    run_info.save(checkpoint_dir)

    start = 0
    if args.resume:
        state = checkpoints.load(
            args.resume,
            models={"current": model},
            optimizers={"current": optimizer},
            map_location=args.device,
        )
        start = state["step"]

    inference = Inference(model, batch_size=args.batch_size)
    rollouts = RolloutRunner(game.make_game, progress=not args.no_progress)
    evaluator = Evaluator(game.make_game, progress=not args.no_progress)
    metrics = MetricLogger(Console(), visdom)
    prepare = Pipeline(
        ComputeReturns(args.discount, reward_scale=game.reward_rescale),
        ToSamples(),
    )

    for step in range(start, args.steps):
        schedule.step(step)

        if step % args.evaluation_every == 0:
            with inference.evaluating():
                player = inference.policy(temperature=args.eval_temperature)
                evaluation = evaluator.compare(
                    [player, player],
                    names=["current", "current"],
                    games=args.evaluation_games,
                    max_steps=800,
                )
            metrics.log(
                step,
                evaluation={
                    "win_rate": evaluation.win_rate(),
                    "points": Range(evaluation.rollouts.my_points(0)),
                },
            )

        with inference.evaluating():
            player = inference.policy()
            games = rollouts.play(
                [
                    game.strategy_from_string(args.strategy),
                    game.strategy_from_string(args.strategy),
                ],
                games=args.rollout_games,
                max_steps=5_000,
                rotate=True,
            )

        result = learner.train(prepare(games), progress=step / args.steps)
        completed = step + 1

        if completed % 1 == 0:
            metrics.log(
                completed,
                rollout=rollout_metrics(games),
                train=result.metrics,
            )
            metrics.game(completed, game.make_metrics(games))

        if completed % args.save_every == 0:
            save(checkpoints, completed, model, optimizer, args)

    return save(checkpoints, args.steps, model, optimizer, args)


def save(checkpoints, step, model, optimizer, args):
    return checkpoints.save(
        step,
        {"current": model},
        optimizers={"current": optimizer},
        metadata={
            "trainer": "imitate",
            "game": args.game,
            "architecture": args.architecture,
        },
    )


def main():
    torch.set_float32_matmul_precision("medium")
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
