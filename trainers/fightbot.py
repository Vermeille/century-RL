"""PPO training for one agent against a fixed strategy."""

from __future__ import annotations

import copy
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
from boardrl.training import (
    ComputeReturns,
    Learner,
    Pipeline,
    ReferenceTargets,
    Select,
    ToSamples,
)


def build_parser():
    parser = coop.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        game="connectfour",
        tag="fightbot",
        opponent_bot="tactical_random",
        opponent_eval_strategy="tactical_random",
        value_lambda=1.0,
        value_clip_epsilon=None,
    )
    parser.add_argument(
        "--opponent-bot",
        "--opponent-strategy",
        dest="opponent_bot",
        help="fixed strategy used by the opponent during training and evaluation",
    )
    return parser


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
        raise ValueError("fightbot.py requires a non-cooperative game")

    model = make_for_game(args.architecture, game).to(args.device)
    if args.initialize_from:
        Checkpoints(args.initialize_from.parent).load(
            args.initialize_from,
            models={"current": model},
            map_location=args.device,
        )
    reference = copy.deepcopy(model).eval()
    learner, optimizer = coop.make_learner(
        model, game, args, offload_modules=(reference,)
    )

    checkpoint_dir = (
        args.checkpoint_root / "fightbot" / args.game / args.architecture / args.tag
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
            repo_root / "boardrl/rl/eval/selfplay.py",
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
        coop.load_resumed_learner_state(
            learner,
            state["states"]["learner"],
        )
        coop.apply_optimizer_hyperparameters(optimizer, args)
        start = state["step"]
        copy_weights(reference, model)
        if args.save_best and best_checkpoints.latest:
            best_state = best_checkpoints.load(map_location="cpu")
            best_score = best_state["metadata"]["evaluation_points"]

    inference = Inference(model, batch_size=args.inference_batch_size)
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
    training_opponent = game.strategy_from_string(args.opponent_bot)
    evaluation_opponent = game.strategy_from_string(args.opponent_eval_strategy)
    sinks = [Console()]
    if trackio_sink is not None:
        sinks.append(trackio_sink)
    metrics = MetricLogger(*sinks)

    # Compute returns while both player traces are still present. Select the
    # model's strategy identity before flattening so the fixed bot can never
    # contribute a training sample, even when seats are rotated.
    prepare = Pipeline(
        ComputeReturns(args.discount, reward_scale=game.reward_rescale),
        Select(strategies=[0]),
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

        with inference.evaluating():
            player = inference.policy(temperature=args.eval_temperature)
            evaluation = evaluator.compare(
                [player, evaluation_opponent],
                names=["current", args.opponent_eval_strategy],
                games=args.evaluation_games,
                max_steps=args.rollout_max_steps,
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
        # Establish a baseline even for zero-step runs and resumed runs.
        evaluate(start)
        learner.safe_point()

        for step in range(start, args.steps):
            learner.safe_point()
            schedule_progress = coop.training_progress(step, args)

            if step != start and step % args.evaluation_every == 0:
                evaluate(step)
                learner.safe_point()

            with inference.evaluating():
                player = inference.policy()
                games = rollouts.play(
                    [player, training_opponent],
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
                    rollout=rollout_metrics(games, coop=False),
                    train=result.metrics,
                )
                metrics.game(completed, game.make_metrics(games))

            if completed % args.save_every == 0:
                save(checkpoints, completed, model, optimizer, learner, args)

        learner.safe_point()
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
            "trainer": "fightbot",
            "game": args.game,
            "architecture": args.architecture,
            "opponent_bot": args.opponent_bot,
            **metadata,
        },
    )


def main():
    torch.set_float32_matmul_precision("medium")
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
