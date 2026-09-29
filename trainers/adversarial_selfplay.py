"""Shared one-policy adversarial self-play training loop."""

from __future__ import annotations

import os
from pathlib import Path
import uuid

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
    RolloutRunner,
    RunInfo,
    make_trackio,
    seed_everything,
    trackio_run,
)
from boardrl.games import games_library
from boardrl.metrics import rollout_metrics
from boardrl.models import make_for_game
from boardrl.training import Pipeline, ReferenceTargets, Scheduler, ToSamples


def build_parser(*, description: str, tag: str):
    parser = coop.build_parser()
    parser.description = description
    parser.set_defaults(
        game="connectfour",
        tag=tag,
        opponent_eval_strategy="tactical_random",
    )
    return parser


def make_optimizer(model, args):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    lr_schedule_steps, _ = coop.resolve_schedule_steps(args)
    lr_start = coop.resolve_lr_schedule_start(args)
    lr_schedule = Scheduler.from_steps(
        total_steps=args.steps,
        start_step=lr_start,
        end_step=lr_start + lr_schedule_steps,
        warmup_steps=args.warmup,
        shape=args.lr_schedule_shape,
        start_value=1.0,
        end_value=args.min_lr_scale,
    )
    return optimizer, lr_schedule


def make_prepare(model, args):
    """Build PPO targets from both seats of the shared self-play policy."""

    return Pipeline(
        ToSamples(),
        ReferenceTargets(
            model,
            batch_size=args.inference_batch_size,
            discount=args.discount,
            gae_lambda=args.gae_lambda,
            value_lambda=args.value_lambda,
            reuse_rollout_predictions=True,
        ),
    )


def initialize_model(path: Path, model, device):
    payload = Checkpoints(path.parent).load(path, map_location=device)
    models = payload["models"]
    for name in ("agent", "current", "best_response", "average", "environment"):
        if name in models:
            model.load_state_dict(models[name])
            return
    raise KeyError("checkpoint contains no supported policy model")


def checkpoint_directory(args, trainer_name: str):
    return (
        args.checkpoint_root
        / trainer_name
        / args.game
        / args.architecture
        / args.tag
    )


def run(args, *, trainer_name, algorithm, make_learner, script_path):
    if args.trackio:
        args.trackio_run_name = f"{args.tag}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    with trackio_run(
        project=args.game if args.trackio else None,
        name=getattr(args, "trackio_run_name", args.tag),
        server_url=args.trackio_url,
        config=vars(args),
        factory=make_trackio,
    ) as trackio_sink:
        return _run(
            args,
            trackio_sink,
            trainer_name=trainer_name,
            algorithm=algorithm,
            make_learner=make_learner,
            script_path=script_path,
        )


def _run(
    args,
    trackio_sink,
    *,
    trainer_name,
    algorithm,
    make_learner,
    script_path,
):
    seed_everything(args.seed)
    game = games_library(args.game)
    if game.coop:
        raise ValueError(f"{trainer_name} requires a non-cooperative game")

    agent = make_for_game(args.architecture, game).to(args.device)
    if args.initialize_from:
        initialize_model(args.initialize_from, agent, args.device)

    learner, optimizer = make_learner(agent, game, args)

    checkpoint_dir = checkpoint_directory(args, trainer_name)
    checkpoints = Checkpoints(
        checkpoint_dir,
        prefix="step",
        keep=args.keep_checkpoints,
    )
    best_checkpoints = Checkpoints(checkpoint_dir, prefix="best", keep=1)
    best_score = float("-inf")

    RunInfo.capture(args, script_path).save(checkpoint_dir)

    start = 0
    if args.resume:
        state = checkpoints.load(
            args.resume,
            models={"agent": agent},
            optimizers={"agent": optimizer},
            map_location=args.device,
        )
        coop.load_resumed_learner_state(
            learner,
            state["states"]["agent_learner"],
        )
        coop.apply_optimizer_hyperparameters(optimizer, args)
        start = state["step"]
        if args.save_best and best_checkpoints.latest:
            best_state = best_checkpoints.load(map_location="cpu")
            best_score = best_state["metadata"]["evaluation_points"]

    inference = Inference(
        agent,
        batch_size=args.inference_batch_size,
        name="agent",
    )
    training_games = coop.RandomOpeningGameFactory(
        game.make_game,
        args.random_move_prob,
    )
    rollouts = RolloutRunner(
        training_games,
        progress=not args.no_progress,
        outcome=game.outcome,
    )
    evaluator = Evaluator(
        game.make_game,
        progress=not args.no_progress,
        outcome=game.outcome,
    )
    evaluation_opponent = game.strategy_from_string(args.opponent_eval_strategy)

    sinks = [Console()]
    if trackio_sink is not None:
        sinks.append(trackio_sink)
    metrics = MetricLogger(*sinks)
    compute_returns = game.rewards.make_returns(args.discount)
    prepare = make_prepare(agent, args)

    def evaluate(step):
        nonlocal best_score

        with inference.evaluating():
            agent_player = inference.policy(temperature=args.eval_temperature)
            evaluation = evaluator.compare(
                [agent_player, evaluation_opponent],
                names=["agent", args.opponent_eval_strategy],
                games=args.evaluation_games,
                max_steps=args.rollout_max_steps,
                rotate=True,
            )

        metrics.log(
            step,
            evaluation={
                "agent": {
                    "win_rate": evaluation.win_rate(),
                    **game.scores.evaluation_metrics(evaluation),
                },
            },
        )
        score = game.rewards.evaluation_score(evaluation)
        if args.save_best and score > best_score:
            best_score = score
            save(
                best_checkpoints,
                step,
                agent,
                optimizer,
                learner,
                args,
                trainer_name=trainer_name,
                algorithm=algorithm,
                evaluation_points=score,
            )

    with learner:
        evaluate(start)
        learner.safe_point()

        completed = start
        for step in range(start, args.steps):
            learner.safe_point()
            with inference.evaluating():
                games = rollouts.play(
                    [inference.policy(), inference.policy()],
                    games=args.rollout_games,
                    max_steps=args.rollout_max_steps,
                    rotate=True,
                )
            learner.safe_point()

            compute_returns(games)
            with inference.evaluating():
                samples = prepare(games)
            result = learner.train(
                samples,
                progress=coop.training_progress(step, args),
            )
            learner.safe_point()
            completed = step + 1

            if step % 5 == 0:
                metrics.log(
                    step,
                    rollout=rollout_metrics(
                        games, outcome=game.outcome, scores=game.scores
                    ),
                    train={"agent": result.metrics},
                )
                metrics.game(step, game.make_metrics(games))

            if completed % args.evaluation_every == 0:
                evaluate(completed)
                learner.safe_point()

            if completed % args.save_every == 0:
                save(
                    checkpoints,
                    completed,
                    agent,
                    optimizer,
                    learner,
                    args,
                    trainer_name=trainer_name,
                    algorithm=algorithm,
                )
                learner.safe_point()

        learner.safe_point()
        final_path = save(
            checkpoints,
            completed,
            agent,
            optimizer,
            learner,
            args,
            trainer_name=trainer_name,
            algorithm=algorithm,
        )
        learner.safe_point()
        if (
            args.save_best
            and completed != start
            and completed % args.evaluation_every != 0
        ):
            evaluate(completed)
            learner.safe_point()
    return final_path


def save(
    checkpoints,
    step,
    agent,
    optimizer,
    learner,
    args,
    *,
    trainer_name,
    algorithm,
    **metadata,
):
    return checkpoints.save(
        step,
        {"agent": agent},
        optimizers={"agent": optimizer},
        states={"agent_learner": learner},
        metadata={
            "trainer": trainer_name,
            "algorithm": algorithm,
            "game": args.game,
            "architecture": args.architecture,
            **metadata,
        },
    )
