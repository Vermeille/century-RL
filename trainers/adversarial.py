"""Adversarial Neural Fictitious Self-Play (NFSP) training."""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import torch

if __package__:
    from . import coop
else:
    import coop

from boardrl import (
    Checkpoints,
    Console,
    Evaluator,
    Inference,
    MetricLogger,
    Range,
    RolloutRunner,
    RunInfo,
    make_trackio,
)
from boardrl.games import games_library
from boardrl.metrics import rollout_metrics
from boardrl.models import copy_weights, make_for_game
from boardrl.rl.model.loss import ImitationCELoss
from boardrl.training import (
    ComputeReturns,
    Learner,
    LR_SCHEDULES,
    Pipeline,
    PolicyMetrics,
    ReferenceTargets,
    ToSamples,
)
from boardrl.training.cuda_pause import CudaOffloadPause
from boardrl.training.sample import TrainingSample


NUCLEUS_THRESHOLD = coop.NUCLEUS_THRESHOLD


def probability(value):
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


class Reservoir:
    """Uniform reservoir of best-response state/policy pairs."""

    def __init__(self, capacity: int, *, seed: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.samples: list[tuple[str, torch.Tensor]] = []
        self.seen = 0
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.samples)

    def add(self, samples):
        for sample in samples:
            item = (
                sample.state,
                sample.action_distribution.detach().cpu().clone(),
            )
            self.seen += 1
            if len(self.samples) < self.capacity:
                self.samples.append(item)
                continue

            index = self.rng.randrange(self.seen)
            if index < self.capacity:
                self.samples[index] = item

    def sample(self, count: int) -> list[TrainingSample]:
        count = min(count, len(self.samples))
        if count == 0:
            return []
        return [
            TrainingSample(state=state, action_distribution=action_distribution)
            for state, action_distribution in self.rng.sample(self.samples, count)
        ]

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "samples": self.samples,
            "seen": self.seen,
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state):
        saved_capacity = state["capacity"]
        if saved_capacity != self.capacity:
            raise ValueError(
                "checkpoint reservoir capacity does not match "
                f"--reservoir-capacity ({saved_capacity} != {self.capacity})"
            )
        self.samples = list(state["samples"])
        self.seen = state["seen"]
        self.rng.setstate(state["rng_state"])


def build_parser():
    parser = coop.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        game="connectfour",
        tag="nfsp",
        opponent_eval_strategy="tactical_random",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-5,
    )
    parser.add_argument(
        "--anticipatory",
        type=probability,
        default=0.1,
        help=(
            "probability that the opponent uses the current best response "
            "instead of the average policy"
        ),
    )
    parser.add_argument(
        "--fixed-training-opponent-fraction",
        type=probability,
        default=0.0,
        help=(
            "fraction of rollout games against the fixed evaluation opponent; "
            "the default keeps pure NFSP self-play"
        ),
    )
    parser.add_argument(
        "--reservoir-capacity",
        type=coop.positive_int,
        default=100_000,
        help="maximum number of best-response state/policy pairs retained",
    )
    parser.add_argument(
        "--average-samples-per-update",
        type=coop.positive_int,
        default=4096,
        help="uniform reservoir samples used for each average-policy update",
    )
    parser.add_argument(
        "--reservoir-insert-ratio",
        type=coop.positive_float,
        default=4.0,
        help=(
            "random BR transitions inserted per iteration as a multiple of "
            "--average-samples-per-update; oversamples with replacement when "
            "the rollout contains fewer transitions"
        ),
    )
    parser.add_argument(
        "--average-batch-size",
        type=coop.positive_int,
        help="average-policy learner batch size; defaults to --learner-batch-size",
    )
    parser.add_argument(
        "--average-learning-rate",
        type=coop.positive_float,
        help="average-policy AdamW learning rate; defaults to --learning-rate",
    )
    parser.add_argument(
        "--average-epochs",
        type=coop.positive_int,
        default=1,
        help="supervised epochs over each sampled reservoir batch",
    )
    return parser


def make_average_learner(model, game, args):
    learning_rate = args.average_learning_rate or args.learning_rate
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    lr_schedule_steps, _ = coop.resolve_schedule_steps(args)
    lr_schedule = LR_SCHEDULES[args.lr_schedule_shape](
        optimizer,
        steps=lr_schedule_steps + args.warmup,
        warmup=args.warmup,
        min_scale=args.min_lr_scale,
        start=coop.resolve_lr_schedule_start(args),
    )
    learner = Learner(
        model,
        optimizer,
        [ImitationCELoss()],
        batch_size=args.average_batch_size or args.learner_batch_size,
        device=args.device,
        epochs=args.average_epochs,
        gradient_clip=args.gradient_clip,
        augmentations=game.augmentations,
        batch_metrics=[PolicyMetrics(nucleus_threshold=NUCLEUS_THRESHOLD)],
        normalize_lr=False,
        lr_schedule=lr_schedule,
    )
    return learner, optimizer


def apply_average_optimizer_hyperparameters(optimizer, args):
    learning_rate = args.average_learning_rate or args.learning_rate
    for group in optimizer.param_groups:
        group.update(
            lr=learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )


def initialize_models(path, best_response, average, device):
    payload = Checkpoints(path.parent).load(path, map_location=device)
    models = payload["models"]
    if "current" in models:
        state = models["current"]
    elif "average" in models:
        state = models["average"]
    elif "best_response" in models:
        state = models["best_response"]
    else:
        raise KeyError(
            "checkpoint contains none of the supported model names: "
            "'current', 'average', 'best_response'"
        )
    best_response.load_state_dict(state)
    average.load_state_dict(state)


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
    coop.seed_everything(args.seed)
    game = games_library(args.game)
    if game.coop:
        raise ValueError("adversarial.py requires a non-cooperative game")

    best_response = make_for_game(args.architecture, game).to(args.device)
    average = copy.deepcopy(best_response)
    if args.initialize_from:
        initialize_models(
            args.initialize_from,
            best_response,
            average,
            args.device,
        )
    reference = copy.deepcopy(best_response).eval()

    best_response_learner, best_response_optimizer = coop.make_learner(
        best_response,
        reference,
        game,
        args,
    )
    average_learner, average_optimizer = make_average_learner(average, game, args)
    reservoir = Reservoir(args.reservoir_capacity, seed=args.seed + 1)
    pause = CudaOffloadPause(
        (best_response, reference, average),
        best_response_optimizer,
        device=args.device,
        extra_optimizers=(average_optimizer,),
    )

    _, exploration_schedule_steps = coop.resolve_schedule_steps(args)
    schedule_start = coop.resolve_schedule_start(args)

    checkpoint_dir = (
        args.checkpoint_root / "adversarial" / args.game / args.architecture / args.tag
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
            repo_root / "trainers/coop.py",
            repo_root / "boardrl/models.py",
            repo_root / "boardrl/rollouts.py",
            repo_root / "boardrl/rl/model/loss.py",
            repo_root / "boardrl/training/learner.py",
            repo_root / "boardrl/training/postprocess.py",
            repo_root / "boardrl/training/returns.py",
        ),
    )
    run_info.save(checkpoint_dir)

    start = 0
    if args.resume:
        state = checkpoints.load(
            args.resume,
            models={
                "best_response": best_response,
                "average": average,
            },
            optimizers={
                "best_response": best_response_optimizer,
                "average": average_optimizer,
            },
            map_location=args.device,
        )
        coop.load_resumed_learner_state(
            best_response_learner,
            state["states"]["best_response_learner"],
            iteration=state["step"],
        )
        average_learner.load_state_dict(state["states"]["average_learner"])
        if "iteration" not in state["states"]["average_learner"]:
            average_learner.iteration = state["step"]
        reservoir.load_state_dict(state["states"]["reservoir"])
        coop.apply_optimizer_hyperparameters(best_response_optimizer, args)
        apply_average_optimizer_hyperparameters(average_optimizer, args)
        start = state["step"]
        copy_weights(reference, best_response)
        if args.save_best and best_checkpoints.latest:
            best_state = best_checkpoints.load(map_location="cpu")
            best_score = best_state["metadata"]["evaluation_points"]

    best_response_inference = Inference(
        best_response,
        batch_size=args.inference_batch_size,
        name="best-response",
    )
    average_inference = Inference(
        average,
        batch_size=args.inference_batch_size,
        name="average",
    )
    training_games = coop.RandomOpeningGameFactory(
        game.make_game,
        args.random_move_prob,
    )
    rollouts = RolloutRunner(
        training_games,
        progress=not args.no_progress,
        coop=False,
    )
    evaluator = Evaluator(
        game.make_game,
        progress=not args.no_progress,
        coop=False,
    )

    sinks = [Console()]
    if trackio_sink is not None:
        sinks.append(trackio_sink)
    metrics = MetricLogger(*sinks)

    compute_rollout_returns = ComputeReturns(
        args.discount,
        reward_scale=game.reward_rescale,
    )
    reinforcement_targets = Pipeline(
        ToSamples(),
        ReferenceTargets(
            reference,
            batch_size=args.learner_batch_size,
            discount=args.discount,
            gae_lambda=args.gae_lambda,
            value_lambda=args.value_lambda,
            reuse_rollout_predictions=False,
        ),
    )

    def evaluate(step):
        nonlocal best_score

        with best_response_inference.evaluating(), average_inference.evaluating():
            average_player = average_inference.policy(temperature=args.eval_temperature)
            best_response_player = best_response_inference.policy(
                temperature=args.eval_temperature
            )
            fixed_opponent = game.strategy_from_string(args.opponent_eval_strategy)

            average_evaluation = evaluator.compare(
                [average_player, fixed_opponent],
                names=["average", "fixed"],
                games=args.evaluation_games,
                max_steps=args.rollout_max_steps,
            )
            best_response_fixed_evaluation = evaluator.compare(
                [best_response_player, fixed_opponent],
                names=["best_response", "fixed"],
                games=args.evaluation_games,
                max_steps=args.rollout_max_steps,
            )

        metrics.log(
            step,
            evaluation={
                "average_vs_fixed": {
                    "win_rate": average_evaluation.win_rate(),
                    "points": Range(average_evaluation.rollouts.my_points(0)),
                },
                "best_response_vs_fixed": {
                    "win_rate": best_response_fixed_evaluation.win_rate(),
                    "points": Range(
                        best_response_fixed_evaluation.rollouts.my_points(0)
                    ),
                },
            },
            nfsp={
                "reservoir_size": len(reservoir),
                "reservoir_seen": reservoir.seen,
            },
        )

        score = best_response_fixed_evaluation.avg_points()
        if args.save_best and score > best_score:
            best_score = score
            save(
                best_checkpoints,
                step,
                best_response,
                average,
                best_response_optimizer,
                average_optimizer,
                best_response_learner,
                average_learner,
                reservoir,
                args,
                evaluation_points=score,
            )

    with pause:
        evaluate(start)
        pause.service()

        for step in range(start, args.steps):
            pause.service()
            schedule_progress = coop.exploration_schedule_progress(
                step, args, schedule_start, exploration_schedule_steps
            )

            with best_response_inference.evaluating(), average_inference.evaluating():
                best_response_player = best_response_inference.policy()
                average_player = average_inference.policy()
                fixed_player = (
                    game.strategy_from_string(args.opponent_eval_strategy)
                    if args.fixed_training_opponent_fraction
                    else None
                )
                # Keep the anticipatory mixture at the batch level. This avoids
                # per-game opponent branching and lets the rollout engine process
                # each homogeneous opponent batch efficiently.
                fixed_opponent_games = round(
                    args.rollout_games * args.fixed_training_opponent_fraction
                )
                best_response_opponent_games = round(
                    (args.rollout_games - fixed_opponent_games) * args.anticipatory
                )
                average_opponent_games = (
                    args.rollout_games
                    - fixed_opponent_games
                    - best_response_opponent_games
                )
                average_opponent_rollouts = rollouts.play(
                    [best_response_player, average_player],
                    games=average_opponent_games,
                    max_steps=args.rollout_max_steps,
                    rotate=True,
                    description="rollouts vs average",
                )
                best_response_opponent_rollouts = rollouts.play(
                    [best_response_player, best_response_player],
                    games=best_response_opponent_games,
                    max_steps=args.rollout_max_steps,
                    rotate=True,
                    description="rollouts vs best response",
                )
                games = type(average_opponent_rollouts)(
                    list(average_opponent_rollouts)
                    + list(best_response_opponent_rollouts)
                )
                best_response_games = type(average_opponent_rollouts)(
                    list(average_opponent_rollouts.only_strategy([0]))
                    + list(best_response_opponent_rollouts.only_strategy([0]))
                )

            pause.service()
            compute_rollout_returns(games)
            copy_weights(reference, best_response)
            reinforcement_samples = reinforcement_targets(best_response_games)
            reservoir_inserted = reservoir.add(
                reinforcement_targets(
                    type(average_opponent_rollouts)(
                        average_opponent_rollouts.only_strategy([0])
                    )
                )
            )
            best_response_result = best_response_learner.train(
                reinforcement_samples,
                progress=schedule_progress,
            )
            pause.service()
            average_result = average_learner.train(
                reservoir.sample(args.average_samples_per_update),
                progress=schedule_progress,
            )
            pause.service()
            completed = step + 1

            if completed % 5 == 0:
                rollout_log = rollout_metrics(games, coop=False)
                if average_opponent_games:
                    rollout_log["best_response_vs_average"] = {
                        "win_rate": average_opponent_rollouts.win_rate(0),
                    }
                metrics.log(
                    completed,
                    rollout=rollout_log,
                    nfsp={
                        "reservoir_size": len(reservoir),
                        "reservoir_seen": reservoir.seen,
                        "reservoir_inserted": reservoir_inserted,
                        "opponent_best_response_fraction": (
                            best_response_opponent_games / args.rollout_games
                        ),
                        "fixed_training_opponent_fraction": (
                            fixed_opponent_games / args.rollout_games
                        ),
                    },
                    train={
                        "best_response": {
                            **best_response_result.metrics,
                            "batches": best_response_result.batches,
                            "samples": best_response_result.samples,
                        },
                        "average": {
                            **average_result.metrics,
                            "batches": average_result.batches,
                            "samples": average_result.samples,
                        },
                    },
                )
                metrics.game(completed, game.make_metrics(games))

            # Evaluate post-update so evaluation step N matches step-N.pth.
            if completed % args.evaluation_every == 0:
                evaluate(completed)
                pause.service()

            if completed % args.save_every == 0:
                save(
                    checkpoints,
                    completed,
                    best_response,
                    average,
                    best_response_optimizer,
                    average_optimizer,
                    best_response_learner,
                    average_learner,
                    reservoir,
                    args,
                )
                pause.service()

        pause.service()
        final_path = save(
            checkpoints,
            args.steps,
            best_response,
            average,
            best_response_optimizer,
            average_optimizer,
            best_response_learner,
            average_learner,
            reservoir,
            args,
        )
        pause.service()
        if (
            args.save_best
            and args.steps != start
            and args.steps % args.evaluation_every != 0
        ):
            evaluate(args.steps)
            pause.service()
    return final_path


def save(
    checkpoints,
    step,
    best_response,
    average,
    best_response_optimizer,
    average_optimizer,
    best_response_learner,
    average_learner,
    reservoir,
    args,
    **metadata,
):
    return checkpoints.save(
        step,
        {
            "best_response": best_response,
            "average": average,
        },
        optimizers={
            "best_response": best_response_optimizer,
            "average": average_optimizer,
        },
        states={
            "best_response_learner": best_response_learner,
            "average_learner": average_learner,
            "reservoir": reservoir,
        },
        metadata={
            "trainer": "adversarial",
            "algorithm": "nfsp",
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
