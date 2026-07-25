"""Cooperative PPO training from scratch."""

from __future__ import annotations

import argparse
import copy
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
from boardrl.models import architectures, copy_weights, make
from boardrl.rl.model.loss import (
    AdaptiveKLPenalty,
    BootstrapValueMSELoss,
    PolicyGradientLoss,
    ScheduledPerplexity,
)
from boardrl.training import (
    ComputeReturns,
    Learner,
    LinearWarmupDecay,
    Pipeline,
    PolicyMetrics,
    ReferenceTargets,
    ToSamples,
    ValueMetrics,
)
from boardrl.utils.visualizer import OfflineVisualizer, VisdomVisualizer


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


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
    parser.add_argument(
        "--schedule-steps",
        type=positive_int,
        help="anneal learning rate and exploration over this many steps, then hold",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=positive_int,
        default=512,
        help="maximum batch for live rollout and evaluation inference",
    )
    parser.add_argument(
        "--learner-batch-size",
        type=positive_int,
        default=512,
        help="batch size for reference targets and PPO updates",
    )
    parser.add_argument(
        "--patch-size",
        type=positive_int,
        help="override shared-patch compression width",
    )
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
    parser.add_argument("--trace-decay", type=float, default=1.0)
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


def make_learner(model, game, args):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, 0.999),
        weight_decay=args.weight_decay,
    )
    learner = Learner(
        model,
        optimizer,
        [
            PolicyGradientLoss(
                weight="normalized_gae",
                drift="ppo",
                imp_ratio_clip=0.2,
            ),
            ScheduledPerplexity(
                start=args.perplexity_start,
                end=args.perplexity_end,
                init_strength=args.entropy_strength,
                baseline_ratio=0.2,
                adaptation_rate=0.005,
                ppl_beta=0.99,
                deadband=0.02,
            ),
            AdaptiveKLPenalty(
                target=args.kl_target,
                init_strength=args.kl_strength,
                deadband=0.001,
            ),
            BootstrapValueMSELoss(strength=args.value_strength),
        ],
        batch_size=args.learner_batch_size,
        device=args.device,
        epochs=args.epochs,
        gradient_clip=args.gradient_clip,
        augmentations=game.augmentations,
        batch_metrics=[PolicyMetrics(), ValueMetrics()],
        normalize_lr=True,
    )
    return learner, optimizer


def run(args):
    seed_everything(args.seed)
    game = games_library(args.game)
    model_overrides = (
        {"backbone_kwargs": {"patch_size": args.patch_size}}
        if args.patch_size is not None
        else {}
    )
    model = make(args.architecture, **model_overrides).to(args.device)
    if args.initialize_from:
        Checkpoints(args.initialize_from.parent).load(
            args.initialize_from,
            models={"current": model},
            map_location=args.device,
        )
    reference = copy.deepcopy(model).eval()
    learner, optimizer = make_learner(model, game, args)
    schedule_steps = args.schedule_steps or args.steps
    schedule = LinearWarmupDecay(
        optimizer,
        steps=schedule_steps,
        warmup=args.warmup,
        min_scale=args.min_lr_scale,
    )
    checkpoint_dir = (
        args.checkpoint_root / "coop" / args.game / args.architecture / args.tag
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
        copy_weights(reference, model)

    inference = Inference(model, batch_size=args.inference_batch_size)
    rollouts = RolloutRunner(game.make_game, progress=not args.no_progress)
    evaluator = Evaluator(game.make_game, progress=not args.no_progress)
    metrics = MetricLogger(Console(), visdom)
    prepare = Pipeline(
        ComputeReturns(args.discount, reward_scale=game.reward_rescale),
        ToSamples(),
        ReferenceTargets(
            reference,
            batch_size=args.learner_batch_size,
            discount=args.discount,
            trace_decay=args.trace_decay,
            reuse_rollout_predictions=True,
        ),
    )

    for step in range(start, args.steps):
        schedule.step(step)
        schedule_progress = min(step / schedule_steps, 1.0)

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
                [player, player],
                games=args.rollout_games,
                max_steps=5_000,
                rotate=True,
            )

        copy_weights(reference, model)
        result = learner.train(prepare(games), progress=schedule_progress)
        completed = step + 1

        if completed % 5 == 0:
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
            "trainer": "coop",
            "game": args.game,
            "architecture": args.architecture,
        },
    )


def main():
    torch.set_float32_matmul_precision("medium")
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
