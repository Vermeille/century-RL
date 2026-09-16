"""Cooperative PPO training from scratch."""

from __future__ import annotations

import argparse
import copy
import os
import random
from pathlib import Path

import torch

from boardrl import (
    Checkpoints,
    Console,
    Evaluator,
    Inference,
    MetricLogger,
    make_trackio,
    Range,
    RolloutRunner,
    RunInfo,
)
from boardrl.games import games_library
from boardrl.metrics import rollout_metrics
from boardrl.models import architectures, copy_weights, make_for_game
from boardrl.rl.model.loss import (
    AdaptiveKLPenalty,
    BootstrapValueLogProbLoss,
    EntropyBonus,
    LinearEntropyBonus,
    LinearReverseEntropyBonus,
    LinearSymmetricUniformKLPenalty,
    PolicyGradientLoss,
    ReverseEntropyBonus,
    ScheduledPerplexity,
    SymmetricUniformKLPenalty,
)
from boardrl.training import (
    ComputeReturns,
    Learner,
    Pipeline,
    PolicyMetrics,
    ReferenceTargets,
    ToSamples,
    ValueMetrics,
    SCHEDULE_SHAPES,
    Scheduler,
)


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def probability_float(value):
    value = float(value)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("must be a probability")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def percentage(value):
    value = nonnegative_int(value)
    if value > 100:
        raise argparse.ArgumentTypeError("must be between 0 and 100")
    return value


def positive_float(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def optional_positive_float(value):
    if value.lower() == "none":
        return None
    return positive_float(value)


exploration_regularizers = {
    "entropy": EntropyBonus,
    "reverse-kl": ReverseEntropyBonus,
    "symmetric-kl": SymmetricUniformKLPenalty,
}

linear_exploration_regularizers = {
    "entropy": LinearEntropyBonus,
    "reverse-kl": LinearReverseEntropyBonus,
    "symmetric-kl": LinearSymmetricUniformKLPenalty,
}

NUCLEUS_THRESHOLD = 0.95


class RandomOpeningGameFactory:
    """Create games after a probabilistic prefix of random legal moves."""

    def __init__(self, make_game, move_prob):
        self.make_game = make_game
        self.move_prob = move_prob

    def __call__(self, **kwargs):
        game = self.make_game(**kwargs)
        while self.move_prob and random.random() < self.move_prob:
            game.play_idx(random.randrange(len(game.moves)))
            if game.ended():
                game = self.make_game(**kwargs)
        return game


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--game",
        "-g",
        default="thegame,mode=omni",
        help=f"game specification; available games: {', '.join(games_library.registry)}",
    )
    parser.add_argument(
        "--architecture",
        "-a",
        choices=sorted(architectures),
        default="patchformer-medium-p8",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--steps", type=int, default=2_400)
    parser.add_argument(
        "--schedule-steps",
        type=positive_int,
        help=(
            "anneal exploration over this many steps, then hold; defaults to the "
            "window defined by --schedule-start-percent and --schedule-end-percent"
        ),
    )
    parser.add_argument(
        "--schedule-start",
        type=nonnegative_int,
        help=(
            "global step at which the exploration schedule starts; defaults to "
            "--schedule-start-percent of --steps"
        ),
    )
    parser.add_argument(
        "--schedule-start-percent",
        type=percentage,
        default=1,
        help="default exploration schedule start as a percentage of --steps",
    )
    parser.add_argument(
        "--schedule-end-percent",
        type=percentage,
        default=99,
        help="default exploration schedule end as a percentage of --steps",
    )
    parser.add_argument(
        "--lr-schedule-steps",
        type=positive_int,
        help=(
            "anneal learning rate over this many steps independently of exploration; "
            "defaults to the window from --lr-schedule-start-percent through --steps"
        ),
    )
    parser.add_argument(
        "--lr-schedule-start",
        type=nonnegative_int,
        help=(
            "learner update iteration at which learning-rate decay starts; defaults "
            "to --lr-schedule-start-percent of --steps"
        ),
    )
    parser.add_argument(
        "--lr-schedule-start-percent",
        type=percentage,
        default=50,
        help="default learning-rate decay start as a percentage of --steps",
    )
    parser.add_argument(
        "--lr-schedule-shape",
        choices=sorted(SCHEDULE_SHAPES),
        default="cosine",
        help="shape of the learning-rate decay",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=positive_int,
        default=1024,
        help="maximum batch for live rollout and evaluation inference",
    )
    parser.add_argument(
        "--learner-batch-size",
        type=positive_int,
        default=384,
        help="batch size for reference targets and PPO updates",
    )
    parser.add_argument("--rollout-games", type=positive_int, default=128)
    parser.add_argument(
        "--random-move-prob",
        type=probability_float,
        default=0,
        help="init state random move probability. Play while satisfied.",
    )
    parser.add_argument(
        "--value-clip-epsilon",
        type=optional_positive_float,
        default=3.0,
        help=(
            "clip TD(lambda) targets to this many rollout value standard deviations; "
            "use 'none' to disable"
        ),
    )
    parser.add_argument(
        "--rollout-max-steps",
        type=positive_int,
        default=5_000,
        help="maximum number of environment steps per rollout game",
    )
    parser.add_argument("--evaluation-games", type=positive_int, default=512)
    parser.add_argument("--evaluation-every", type=positive_int, default=50)
    parser.add_argument("--save-every", type=positive_int, default=25)
    parser.add_argument(
        "--keep-checkpoints",
        type=positive_int,
        default=10,
        help="retain only this many latest periodic checkpoints",
    )
    parser.add_argument(
        "--save-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="retain the checkpoint with the highest evaluation points",
    )
    parser.add_argument("--learning-rate", type=positive_float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=positive_float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.1,
        help="trace decay for generalized advantage estimation",
    )
    parser.add_argument(
        "--value-lambda",
        type=float,
        default=0.9,
        help="trace decay for critic targets (1 uses pure Monte Carlo returns)",
    )
    parser.add_argument("--perplexity-start", type=float, default=2.5)
    parser.add_argument("--perplexity-end", type=float, default=1.5)
    parser.add_argument(
        "--perplexity-curve",
        type=positive_float,
        default=1.0,
        help="power applied to thermostat target progress (<1 anneals earlier)",
    )
    parser.add_argument(
        "--opponent-eval-strategy",
        help="For non coop game, what evaluation strategy is used as opponent",
        default="random",
    )
    parser.add_argument(
        "--perplexity-schedule-shape",
        choices=sorted(SCHEDULE_SHAPES),
        default="cosine",
        help="shape used to interpolate the thermostat perplexity target",
    )
    parser.add_argument(
        "--perplexity-adaptation-rate",
        type=positive_float,
        default=0.004,
        help="thermostat response rate for perplexity error",
    )
    parser.add_argument("--entropy-strength", type=float, default=0.1)
    parser.add_argument(
        "--exploration-regularizer",
        choices=sorted(exploration_regularizers),
        default="entropy",
        help="regularizer controlled by the exploration controller",
    )
    parser.add_argument(
        "--exploration-controller",
        choices=("thermostat", "linear"),
        default="thermostat",
        help="use adaptive perplexity control or linear strength decay",
    )
    parser.add_argument(
        "--entropy-baseline-ratio",
        type=float,
        default=0.05,
        help="fraction of initial entropy strength retained after annealing",
    )
    parser.add_argument("--value-strength", type=float, default=1.0)
    parser.add_argument("--kl-target", type=float, default=0.05)
    parser.add_argument("--kl-strength", type=float, default=0.05)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--eval-temperature", type=float, default=0.02)
    parser.add_argument("--warmup", type=nonnegative_int, default=20)
    parser.add_argument("--min-lr-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("checkpoints/schedule-search")
    )
    starting_point = parser.add_mutually_exclusive_group()
    starting_point.add_argument("--resume", type=Path)
    starting_point.add_argument(
        "--initialize-from",
        type=Path,
        help="load model weights but start a fresh optimizer and step count",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--tag", default="coop")
    parser.add_argument("--trackio", action="store_true")
    parser.add_argument("--trackio-url", default=os.environ.get("TRACKIO_URL"))
    return parser


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    from boardrl.cyutils import init_seed

    init_seed(seed)


def resolve_schedule_steps(args):
    schedule_start = resolve_schedule_start(args)
    exploration_end = args.steps * args.schedule_end_percent // 100
    exploration_steps = args.schedule_steps or max(exploration_end - schedule_start, 1)
    lr_schedule_start = resolve_lr_schedule_start(args)
    if args.lr_schedule_steps is not None:
        lr_steps = args.lr_schedule_steps
    elif args.schedule_steps is not None:
        lr_steps = exploration_steps
    else:
        lr_steps = max(args.steps - lr_schedule_start - 1, 1)
    return lr_steps, exploration_steps


def resolve_schedule_start(args):
    if args.schedule_start is not None:
        return args.schedule_start
    return args.steps * args.schedule_start_percent // 100


def resolve_lr_schedule_start(args):
    if args.lr_schedule_start is not None:
        return args.lr_schedule_start
    return args.steps * args.lr_schedule_start_percent // 100


def training_progress(step, args):
    return min(max(step / max(args.steps - 1, 1), 0.0), 1.0)


def make_exploration_schedule(args):
    _, schedule_steps = resolve_schedule_steps(args)
    start = resolve_schedule_start(args)
    if args.exploration_controller == "thermostat":
        shape = args.perplexity_schedule_shape
        curve = args.perplexity_curve
    else:
        shape = "linear"
        curve = 1.0
    return Scheduler.from_steps(
        total_steps=args.steps,
        start_step=start,
        end_step=start + schedule_steps,
        shape=shape,
        curve=curve,
    )


def make_learner(model, reference, game, args):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    lr_schedule_steps, _ = resolve_schedule_steps(args)
    lr_start = resolve_lr_schedule_start(args)
    lr_schedule = Scheduler.from_steps(
        total_steps=args.steps,
        start_step=lr_start,
        end_step=lr_start + lr_schedule_steps,
        warmup_steps=args.warmup,
        shape=args.lr_schedule_shape,
        start_value=1.0,
        end_value=args.min_lr_scale,
    )
    exploration_schedule = make_exploration_schedule(args)
    if args.exploration_controller == "thermostat":
        exploration_loss = ScheduledPerplexity(
            start=args.perplexity_start,
            end=args.perplexity_end,
            init_strength=args.entropy_strength,
            baseline_ratio=args.entropy_baseline_ratio,
            adaptation_rate=args.perplexity_adaptation_rate,
            ppl_beta=0.98,
            deadband=0.02,
            regularizer_factory=exploration_regularizers[args.exploration_regularizer],
            schedule=exploration_schedule,
        )
    else:
        exploration_loss = linear_exploration_regularizers[
            args.exploration_regularizer
        ](
            start=args.entropy_strength,
            end=args.entropy_strength * args.entropy_baseline_ratio,
            schedule=exploration_schedule,
        )

    losses = [
        PolicyGradientLoss(
            weight="normalized_gae",
            drift="ppo",
            imp_ratio_clip=args.ppo_clip,
        ),
        exploration_loss,
    ]
    losses.extend(
        [
            AdaptiveKLPenalty(
                target=args.kl_target,
                init_strength=args.kl_strength,
                adaptation_rate=args.kl_strength,
                deadband=0.001,
            ),
            BootstrapValueLogProbLoss(
                strength=args.value_strength,
                epsilon=args.value_clip_epsilon,
            ),
        ]
    )
    learner = Learner(
        model,
        optimizer,
        losses,
        batch_size=args.learner_batch_size,
        device=args.device,
        epochs=args.epochs,
        gradient_clip=args.gradient_clip,
        augmentations=game.augmentations,
        batch_metrics=[
            PolicyMetrics(nucleus_threshold=NUCLEUS_THRESHOLD),
            ValueMetrics(),
        ],
        normalize_lr=False,
        lr_schedule=lr_schedule,
        offload_modules=(reference,),
    )
    return learner, optimizer


def apply_optimizer_hyperparameters(optimizer, args):
    """Keep restored Adam state while making the CLI recipe authoritative."""

    for group in optimizer.param_groups:
        group.update(
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )


def load_resumed_learner_state(learner, state):
    learner.load_state_dict(state)


def run(args):
    trackio_sink = make_trackio(
        project=args.game if args.trackio else None,
        name=args.tag,
        server_url=args.trackio_url,
        config=vars(args),
    )
    try:
        return _run(args, trackio_sink)
    finally:
        if trackio_sink is not None:
            trackio_sink.finish()


def _run(args, trackio_sink):
    seed_everything(args.seed)
    game = games_library(args.game)
    model = make_for_game(args.architecture, game).to(args.device)
    if args.initialize_from:
        Checkpoints(args.initialize_from.parent).load(
            args.initialize_from,
            models={"current": model},
            map_location=args.device,
        )
    reference = copy.deepcopy(model).eval()
    learner, optimizer = make_learner(model, reference, game, args)
    checkpoint_dir = (
        args.checkpoint_root / "coop" / args.game / args.architecture / args.tag
    )
    checkpoints = Checkpoints(
        checkpoint_dir,
        prefix="step",
        keep=args.keep_checkpoints,
    )
    best_checkpoints = Checkpoints(checkpoint_dir, prefix="best", keep=1)
    best_score = float("-inf")

    repo_root = Path(__file__).resolve().parents[1]
    run_info = RunInfo.capture(
        args,
        __file__,
        additional_sources=(
            repo_root / "boardrl/models.py",
            repo_root / "boardrl/rl/model/cnn.py",
            repo_root / "boardrl/rl/model/loss.py",
            repo_root / "boardrl/rl/model/model.py",
            repo_root / "boardrl/rl/model/transformer.py",
            repo_root / "boardrl/rl/eval/selfplay.py",
            repo_root / "boardrl/games/strategies.py",
            repo_root / "boardrl/games/thegame/game.py",
            repo_root / "boardrl/training/learner.py",
            repo_root / "boardrl/schedules.py",
            repo_root / "boardrl/training/postprocess.py",
            repo_root / "boardrl/training/returns.py",
        ),
    )
    run_info.save(checkpoint_dir)

    start = 0
    if args.resume:
        state = checkpoints.load(
            args.resume,
            models={"current": model},
            optimizers={"current": optimizer},
            map_location=args.device,
        )
        load_resumed_learner_state(
            learner,
            state["states"]["learner"],
        )
        apply_optimizer_hyperparameters(optimizer, args)
        start = state["step"]
        copy_weights(reference, model)
        if args.save_best and best_checkpoints.latest:
            best_state = best_checkpoints.load(map_location="cpu")
            best_score = best_state["metadata"]["evaluation_points"]

    inference = Inference(model, batch_size=args.inference_batch_size)
    training_games = RandomOpeningGameFactory(
        game.make_game,
        args.random_move_prob,
    )
    rollouts = RolloutRunner(
        training_games, progress=not args.no_progress, coop=game.coop
    )
    evaluator = Evaluator(game.make_game, progress=not args.no_progress, coop=game.coop)
    sinks = [Console()]
    if trackio_sink is not None:
        sinks.append(trackio_sink)
    metrics = MetricLogger(*sinks)
    prepare = Pipeline(
        ComputeReturns(args.discount, reward_scale=game.reward_rescale),
        ToSamples(),
        ReferenceTargets(
            reference,
            batch_size=args.inference_batch_size,
            discount=args.discount,
            gae_lambda=args.gae_lambda,
            value_lambda=args.value_lambda,
            reuse_rollout_predictions=False,
        ),
    )

    def evaluate(step):
        nonlocal best_score

        with inference.evaluating():
            player = inference.policy(temperature=args.eval_temperature)
            evaluation = evaluator.compare(
                [player, player]
                if game.coop
                else [player, game.strategy_from_string(args.opponent_eval_strategy)],
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
        score = evaluation.avg_points()
        if args.save_best and score > best_score:
            best_score = score
            save(
                best_checkpoints,
                step,
                model,
                optimizer,
                learner,
                args,
                evaluation_points=score,
            )

    with learner:
        # Always establish and log a baseline before the first training step,
        # including zero-step runs and resumed runs.
        evaluate(start)
        learner.safe_point()

        for step in range(start, args.steps):
            learner.safe_point()
            schedule_progress = training_progress(step, args)

            if step != start and step % args.evaluation_every == 0:
                evaluate(step)
                learner.safe_point()

            with inference.evaluating():
                player = inference.policy()
                games = rollouts.play(
                    [player, player],
                    games=args.rollout_games,
                    max_steps=args.rollout_max_steps,
                    rotate=True,
                )
            learner.safe_point()

            copy_weights(reference, model)
            result = learner.train(prepare(games), progress=schedule_progress)
            learner.safe_point()
            completed = step + 1

            if completed % 5 == 0:
                metrics.log(
                    completed,
                    rollout=rollout_metrics(games, coop=game.coop),
                    train=result.metrics,
                )
                metrics.game(completed, game.make_metrics(games))

            if completed % args.save_every == 0:
                save(checkpoints, completed, model, optimizer, learner, args)

        learner.safe_point()
        # Persist the terminal state even when args.steps is not a save interval.
        final_path = save(checkpoints, args.steps, model, optimizer, learner, args)
        if args.save_best and args.steps != start:
            evaluate(args.steps)
            learner.safe_point()
    return final_path


def save(
    checkpoints,
    step,
    model,
    optimizer,
    learner,
    args,
    **metadata,
):
    return checkpoints.save(
        step,
        {"current": model},
        optimizers={"current": optimizer},
        states={"learner": learner},
        metadata={
            "trainer": "coop",
            "game": args.game,
            "architecture": args.architecture,
            **metadata,
        },
    )


def main():
    torch.set_float32_matmul_precision("medium")
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
