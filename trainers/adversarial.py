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
from boardrl.models import copy_weights, make
from boardrl.rl.model.loss import CELoss
from boardrl.training import (
    ComputeReturns,
    Learner,
    Pipeline,
    PolicyMetrics,
    ReferenceTargets,
    ToSamples,
)
from boardrl.training.sample import TrainingSample


NUCLEUS_THRESHOLD = coop.NUCLEUS_THRESHOLD


def probability(value):
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


class Reservoir:
    """Uniform reservoir of best-response state/action pairs."""

    def __init__(self, capacity: int, *, seed: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.samples: list[tuple[str, int]] = []
        self.seen = 0
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.samples)

    def add(self, samples):
        for sample in samples:
            item = (sample.state, int(sample.action_idx))
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
            TrainingSample(state=state, action_idx=action_idx)
            for state, action_idx in self.rng.sample(self.samples, count)
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
    parser.set_defaults(game="connectfour", tag="nfsp")
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
        "--reservoir-capacity",
        type=coop.positive_int,
        default=100_000,
        help="maximum number of best-response state/action pairs retained",
    )
    parser.add_argument(
        "--average-samples-per-update",
        type=coop.positive_int,
        default=4096,
        help="uniform reservoir samples used for each average-policy update",
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
    learner = Learner(
        model,
        optimizer,
        [CELoss()],
        batch_size=args.average_batch_size or args.learner_batch_size,
        device=args.device,
        epochs=args.average_epochs,
        gradient_clip=args.gradient_clip,
        augmentations=game.augmentations,
        batch_metrics=[PolicyMetrics(nucleus_threshold=NUCLEUS_THRESHOLD)],
        normalize_lr=False,
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

    best_response = make(args.architecture).to(args.device)
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

    lr_schedule_steps, exploration_schedule_steps = coop.resolve_schedule_steps(args)
    best_response_schedule = coop.lr_schedulers[args.lr_schedule_shape](
        best_response_optimizer,
        steps=lr_schedule_steps + args.warmup,
        warmup=args.warmup,
        min_scale=args.min_lr_scale,
    )
    average_schedule = coop.lr_schedulers[args.lr_schedule_shape](
        average_optimizer,
        steps=lr_schedule_steps + args.warmup,
        warmup=args.warmup,
        min_scale=args.min_lr_scale,
    )

    checkpoint_dir = (
        args.checkpoint_root
        / "adversarial"
        / args.game
        / args.architecture
        / args.tag
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
            reset_exploration_state=args.resume_reset_exploration_state,
        )
        average_learner.load_state_dict(state["states"]["average_learner"])
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
    rollouts = RolloutRunner(
        game.make_game,
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

    reinforcement_targets = Pipeline(
        ComputeReturns(args.discount, reward_scale=game.reward_rescale),
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
    imitation_targets = ToSamples()

    def evaluate(step):
        nonlocal best_score

        with best_response_inference.evaluating(), average_inference.evaluating():
            average_player = average_inference.policy(
                temperature=args.eval_temperature
            )
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
            exploitability_probe = evaluator.compare(
                [best_response_player, average_player],
                names=["best_response", "average"],
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
                "best_response_vs_average": {
                    "win_rate": exploitability_probe.win_rate(),
                    "points": Range(exploitability_probe.rollouts.my_points(0)),
                },
            },
            nfsp={
                "reservoir_size": len(reservoir),
                "reservoir_seen": reservoir.seen,
            },
        )

        score = average_evaluation.avg_points()
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

    evaluate(start)

    for step in range(start, args.steps):
        schedule_position = coop.optimizer_schedule_position(step, args)
        if schedule_position is not None:
            best_response_schedule.step(schedule_position)
            average_schedule.step(schedule_position)

        schedule_progress = min(
            max((step - args.schedule_start) / exploration_schedule_steps, 0.0),
            1.0,
        )
        if args.exploration_controller == "thermostat":
            schedule_progress **= args.perplexity_curve
            schedule_progress = coop.SCHEDULE_SHAPES[
                args.perplexity_schedule_shape
            ](schedule_progress)

        if step != start and step % args.evaluation_every == 0:
            evaluate(step)

        with best_response_inference.evaluating(), average_inference.evaluating():
            best_response_player = best_response_inference.policy()
            average_player = average_inference.policy()
            opponent_uses_best_response = [
                random.random() < args.anticipatory
                for _ in range(args.rollout_games)
            ]

            # Strategy 0 is always the BR so every game yields on-policy PPO
            # data. The opponent is the NFSP anticipatory mixture; rotation
            # exposes the shared BR to both seats without collecting
            # average-vs-average games that contain no BR training samples.
            def lineup(game_index):
                opponent = (
                    best_response_player
                    if opponent_uses_best_response[game_index]
                    else average_player
                )
                return [best_response_player, opponent]

            games = rollouts.play(
                lineup,
                games=args.rollout_games,
                max_steps=args.rollout_max_steps,
                rotate=True,
            )

        best_response_games = games.only_strategy([0])
        reservoir.add(imitation_targets(best_response_games))

        copy_weights(reference, best_response)
        best_response_result = best_response_learner.train(
            reinforcement_targets(best_response_games),
            progress=schedule_progress,
        )
        average_result = average_learner.train(
            reservoir.sample(args.average_samples_per_update),
            progress=schedule_progress,
        )
        completed = step + 1

        if completed % 5 == 0:
            metrics.log(
                completed,
                rollout=rollout_metrics(games, coop=False),
                nfsp={
                    "reservoir_size": len(reservoir),
                    "reservoir_seen": reservoir.seen,
                    "opponent_best_response_fraction": (
                        sum(opponent_uses_best_response)
                        / len(opponent_uses_best_response)
                    ),
                },
                train={
                    "best_response": best_response_result.metrics,
                    "average": average_result.metrics,
                },
            )
            metrics.game(completed, game.make_metrics(games))

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
    if args.save_best and args.steps != start:
        evaluate(args.steps)
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
