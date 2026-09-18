"""Two-policy adversarial PPO balanced by batch advantage shaping."""

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
from boardrl.models import make_for_game
from boardrl.training import (
    ComputeReturns,
    Pipeline,
    ReferenceTargets,
    Select,
    ToSamples,
)
from boardrl.training.cuda_pause import CudaOffloadPause


def build_parser():
    parser = coop.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        game="connectfour",
        tag="adversarial-advshape",
        opponent_eval_strategy="tactical_random",
        perplexity_schedule_shape="cosine",
    )
    parser.add_argument(
        "--environment-perplexity-start",
        type=coop.positive_float,
        default=3.0,
        help="initial perplexity target for the environment policy",
    )
    parser.add_argument(
        "--environment-perplexity-end",
        type=coop.positive_float,
        default=1.0,
        help="final perplexity target for the environment policy",
    )
    parser.add_argument(
        "--environment-entropy-strength",
        type=coop.positive_float,
        default=1.0,
        help="start value for entropy regularizer strength",
    )
    parser.add_argument(
        "--agent-threshold",
        type=coop.probability_float,
        default=0.98,
        help="batch win rate at which the agent has zero policy advantage",
    )
    parser.add_argument(
        "--environment-threshold",
        type=coop.probability_float,
        default=0.7,
        help="batch win rate at which the environment has zero policy advantage",
    )
    return parser


def make_strategy_learner(
    model,
    game,
    args,
    *,
    perplexity_start=None,
    perplexity_end=None,
    entropy_strength=None,
):
    learner_args = copy.copy(args)
    if perplexity_start is not None:
        learner_args.perplexity_start = perplexity_start
    if perplexity_end is not None:
        learner_args.perplexity_end = perplexity_end
    if entropy_strength is not None:
        learner_args.entropy_strength = entropy_strength
    return coop.make_learner(model, game, learner_args)


class ScaleAdvantages:
    """Multiply raw and normalized GAE by a batch-dependent scalar."""

    def __init__(self):
        self.factor = 1.0

    def __call__(self, samples):
        for sample in samples:
            sample.gae *= self.factor
            sample.normalized_gae *= self.factor
        return samples


class AdvantageShapingPipeline(Pipeline):
    """Shape one strategy's advantages from its current batch win rate."""

    def __init__(self, *steps, strategy_id, threshold, scale):
        super().__init__(*steps, scale)
        self.strategy_id = strategy_id
        self.threshold = threshold
        self.scale = scale

    def __call__(self, games):
        win_rate = games.by_strategy.group(self.strategy_id).win_rate()
        self.scale.factor = self.threshold - win_rate
        return super().__call__(games)


def make_prepare(model, args, strategy_id, threshold):
    """Build PPO targets for one stable rollout strategy identity."""

    scale = ScaleAdvantages()
    return AdvantageShapingPipeline(
        Select(strategies=[strategy_id]),
        ToSamples(),
        ReferenceTargets(
            model,
            batch_size=args.inference_batch_size,
            discount=args.discount,
            gae_lambda=args.gae_lambda,
            value_lambda=args.value_lambda,
            reuse_rollout_predictions=False,
        ),
        strategy_id=strategy_id,
        threshold=threshold,
        scale=scale,
    )


def initialize_models(path, agent, environment, device):
    payload = Checkpoints(path.parent).load(path, map_location=device)
    models = payload["models"]
    if "agent" in models and "environment" in models:
        agent.load_state_dict(models["agent"])
        environment.load_state_dict(models["environment"])
        return

    for name in ("current", "best_response", "average"):
        if name in models:
            agent.load_state_dict(models[name])
            environment.load_state_dict(models[name])
            return
    raise KeyError(
        "checkpoint contains neither adversarial policies nor a supported "
        "single-policy model"
    )


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
        raise ValueError("adversarial-advshape.py requires a non-cooperative game")

    agent = make_for_game(args.architecture, game).to(args.device)
    environment = make_for_game(args.architecture, game).to(args.device)
    if args.initialize_from:
        initialize_models(
            args.initialize_from,
            agent,
            environment,
            args.device,
        )

    agent_learner, agent_optimizer = make_strategy_learner(
        agent,
        game,
        args,
    )
    environment_learner, environment_optimizer = make_strategy_learner(
        environment,
        game,
        args,
        perplexity_start=args.environment_perplexity_start,
        perplexity_end=args.environment_perplexity_end,
        entropy_strength=args.environment_entropy_strength,
    )

    pause = CudaOffloadPause(
        (agent, environment),
        agent_optimizer,
        device=args.device,
        extra_optimizers=(environment_optimizer,),
    )

    checkpoint_dir = (
        args.checkpoint_root
        / "adversarial-advshape"
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
    RunInfo.capture(
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
    ).save(checkpoint_dir)

    start = 0
    if args.resume:
        state = checkpoints.load(
            args.resume,
            models={"agent": agent, "environment": environment},
            optimizers={
                "agent": agent_optimizer,
                "environment": environment_optimizer,
            },
            map_location=args.device,
        )
        coop.load_resumed_learner_state(
            agent_learner,
            state["states"]["agent_learner"],
        )
        coop.load_resumed_learner_state(
            environment_learner,
            state["states"]["environment_learner"],
        )
        coop.apply_optimizer_hyperparameters(agent_optimizer, args)
        coop.apply_optimizer_hyperparameters(environment_optimizer, args)
        start = state["step"]
        if args.save_best and best_checkpoints.latest:
            best_state = best_checkpoints.load(map_location="cpu")
            best_score = best_state["metadata"]["evaluation_points"]

    agent_inference = Inference(
        agent,
        batch_size=args.inference_batch_size,
        name="agent",
    )
    environment_inference = Inference(
        environment,
        batch_size=args.inference_batch_size,
        name="environment",
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
    evaluation_opponent = game.strategy_from_string(args.opponent_eval_strategy)

    sinks = [Console()]
    if trackio_sink is not None:
        sinks.append(trackio_sink)
    metrics = MetricLogger(*sinks)

    compute_returns = ComputeReturns(
        args.discount,
        reward_scale=game.reward_rescale,
    )

    # Rollout strategy IDs are list positions: agent=0, environment=1. They
    # remain stable when physical seats rotate.
    prepare_agent = make_prepare(agent, args, 0, args.agent_threshold)
    prepare_environment = make_prepare(
        environment,
        args,
        1,
        args.environment_threshold,
    )

    def evaluate(step):
        nonlocal best_score

        with agent_inference.evaluating(), environment_inference.evaluating():
            agent_player = agent_inference.policy(temperature=args.eval_temperature)
            environment_player = environment_inference.policy(
                temperature=args.eval_temperature
            )
            agent_evaluation = evaluator.compare(
                [agent_player, evaluation_opponent],
                names=["agent", args.opponent_eval_strategy],
                games=args.evaluation_games,
                max_steps=args.rollout_max_steps,
                rotate=True,
            )
            environment_evaluation = evaluator.compare(
                [environment_player, evaluation_opponent],
                names=["environment", args.opponent_eval_strategy],
                games=args.evaluation_games,
                max_steps=args.rollout_max_steps,
                rotate=True,
            )

        metrics.log(
            step,
            evaluation={
                "agent": {
                    "win_rate": agent_evaluation.win_rate(),
                    "points": Range(agent_evaluation.rollouts.my_points(0)),
                },
                "environment": {
                    "win_rate": environment_evaluation.win_rate(),
                    "points": Range(environment_evaluation.rollouts.my_points(0)),
                },
            },
        )
        score = agent_evaluation.avg_points()
        if args.save_best and score > best_score:
            best_score = score
            save(
                best_checkpoints,
                step,
                agent,
                environment,
                agent_optimizer,
                environment_optimizer,
                agent_learner,
                environment_learner,
                args,
                evaluation_points=score,
            )

    with pause:
        evaluate(start)
        pause.service()

        completed = start
        for step in range(start, args.steps):
            pause.service()
            with agent_inference.evaluating(), environment_inference.evaluating():
                games = rollouts.play(
                    [agent_inference.policy(), environment_inference.policy()],
                    games=args.rollout_games,
                    max_steps=args.rollout_max_steps,
                    rotate=True,
                )
            pause.service()

            compute_returns(games)
            with agent_inference.evaluating(), environment_inference.evaluating():
                agent_samples = prepare_agent(games)
                environment_samples = prepare_environment(games)
            schedule_progress = coop.training_progress(step, args)
            agent_result = agent_learner.train(
                agent_samples,
                progress=schedule_progress,
            )
            pause.service()
            environment_result = environment_learner.train(
                environment_samples,
                progress=schedule_progress,
            )
            pause.service()
            completed = step + 1

            if completed % 5 == 0:
                metrics.log(
                    completed,
                    rollout=rollout_metrics(games, coop=False),
                    train={
                        "agent": agent_result.metrics,
                        "environment": environment_result.metrics,
                    },
                )
                metrics.game(completed, game.make_metrics(games))

            if completed % args.evaluation_every == 0:
                evaluate(completed)
                pause.service()

            if completed % args.save_every == 0:
                save(
                    checkpoints,
                    completed,
                    agent,
                    environment,
                    agent_optimizer,
                    environment_optimizer,
                    agent_learner,
                    environment_learner,
                    args,
                )
                pause.service()

        pause.service()
        final_path = save(
            checkpoints,
            completed,
            agent,
            environment,
            agent_optimizer,
            environment_optimizer,
            agent_learner,
            environment_learner,
            args,
        )
        pause.service()
        if (
            args.save_best
            and completed != start
            and completed % args.evaluation_every != 0
        ):
            evaluate(completed)
            pause.service()
    return final_path


def save(
    checkpoints,
    step,
    agent,
    environment,
    agent_optimizer,
    environment_optimizer,
    agent_learner,
    environment_learner,
    args,
    **metadata,
):
    return checkpoints.save(
        step,
        {"agent": agent, "environment": environment},
        optimizers={"agent": agent_optimizer, "environment": environment_optimizer},
        states={
            "agent_learner": agent_learner,
            "environment_learner": environment_learner,
        },
        metadata={
            "trainer": "adversarial-advshape",
            "algorithm": "two-policy-advantage-shaped-ppo",
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
