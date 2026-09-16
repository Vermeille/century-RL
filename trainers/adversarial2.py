"""Two-policy adversarial PPO training with asymmetric exploration targets."""

from __future__ import annotations

import copy
from dataclasses import dataclass
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
    Pipeline,
    ReferenceTargets,
    Select,
    ToSamples,
)
from boardrl.training.cuda_pause import CudaOffloadPause


@dataclass(frozen=True)
class ControlledUpdate:
    win_rate: float
    result: object | None
    paused: bool

    @property
    def updated(self):
        return self.result is not None

    @property
    def metrics(self):
        return self.result.metrics if self.result is not None else {}


class BatchWinRateController:
    """Pause dominant policies until their batch win rate falls sufficiently."""

    def __init__(self, stop_threshold, restart_threshold):
        if restart_threshold >= stop_threshold:
            raise ValueError(
                "restart win-rate threshold must be below stop threshold"
            )
        self.stop_threshold = stop_threshold
        self.restart_threshold = restart_threshold
        self.paused_strategy_ids = set()

    def state_dict(self):
        return {"paused_strategy_ids": sorted(self.paused_strategy_ids)}

    def load_state_dict(self, state):
        self.paused_strategy_ids = set(state["paused_strategy_ids"])

    def is_paused(self, strategy_id):
        return strategy_id in self.paused_strategy_ids

    def should_update(self, strategy_id, win_rate):
        if self.is_paused(strategy_id):
            if win_rate < self.restart_threshold:
                self.paused_strategy_ids.remove(strategy_id)
                return True
            return False

        if win_rate > self.stop_threshold:
            self.paused_strategy_ids.add(strategy_id)
            return False
        return True

    def update(
        self,
        games,
        strategy_id,
        *,
        model,
        reference,
        prepare,
        learner,
        progress,
    ):
        win_rate = games.by_strategy.group(strategy_id).win_rate()
        if not self.should_update(strategy_id, win_rate):
            return ControlledUpdate(win_rate, None, paused=True)

        copy_weights(reference, model)
        result = learner.train(prepare(games), progress=progress)
        return ControlledUpdate(win_rate, result, paused=False)


def build_parser():
    parser = coop.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        game="connectfour",
        tag="adversarial2",
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
        "--stop-win-rate-threshold",
        type=coop.probability_float,
        default=0.7,
        help=(
            "pause a policy's PPO updates when its current batch win rate "
            "exceeds this threshold"
        ),
    )
    parser.add_argument(
        "--restart-win-rate-threshold",
        type=coop.probability_float,
        default=0.5,
        help=(
            "restart a paused policy's PPO updates when its current batch "
            "win rate falls below this threshold"
        ),
    )
    return parser


def make_strategy_learner(
    model,
    reference,
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
    return coop.make_learner(model, reference, game, learner_args)


def make_prepare(reference, args, strategy_id):
    """Build PPO targets for one stable rollout strategy identity."""

    return Pipeline(
        Select(strategies=[strategy_id]),
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
        "checkpoint contains neither adversarial2 policies nor a supported "
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
        raise ValueError("adversarial2.py requires a non-cooperative game")

    agent = make_for_game(args.architecture, game).to(args.device)
    environment = make_for_game(args.architecture, game).to(args.device)
    if args.initialize_from:
        initialize_models(
            args.initialize_from,
            agent,
            environment,
            args.device,
        )

    agent_reference = copy.deepcopy(agent).eval()
    environment_reference = copy.deepcopy(environment).eval()
    agent_learner, agent_optimizer = make_strategy_learner(
        agent,
        agent_reference,
        game,
        args,
    )
    environment_learner, environment_optimizer = make_strategy_learner(
        environment,
        environment_reference,
        game,
        args,
        perplexity_start=args.environment_perplexity_start,
        perplexity_end=args.environment_perplexity_end,
        entropy_strength=args.environment_entropy_strength,
    )

    pause = CudaOffloadPause(
        (agent, agent_reference, environment, environment_reference),
        agent_optimizer,
        device=args.device,
        extra_optimizers=(environment_optimizer,),
    )
    lr_schedule_steps, exploration_schedule_steps = coop.resolve_schedule_steps(args)
    schedule_start = coop.resolve_schedule_start(args)
    agent_schedule = coop.lr_schedulers[args.lr_schedule_shape](
        agent_optimizer,
        steps=lr_schedule_steps + args.warmup,
        warmup=args.warmup,
        min_scale=args.min_lr_scale,
    )
    environment_schedule = coop.lr_schedulers[args.lr_schedule_shape](
        environment_optimizer,
        steps=lr_schedule_steps + args.warmup,
        warmup=args.warmup,
        min_scale=args.min_lr_scale,
    )
    update_controller = BatchWinRateController(
        args.stop_win_rate_threshold,
        args.restart_win_rate_threshold,
    )

    checkpoint_dir = (
        args.checkpoint_root / "adversarial2" / args.game / args.architecture / args.tag
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
        if "update_controller" in state["states"]:
            update_controller.load_state_dict(state["states"]["update_controller"])
        coop.apply_optimizer_hyperparameters(agent_optimizer, args)
        coop.apply_optimizer_hyperparameters(environment_optimizer, args)
        start = state["step"]
        copy_weights(agent_reference, agent)
        copy_weights(environment_reference, environment)
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
    prepare_agent = make_prepare(agent_reference, args, 0)
    prepare_environment = make_prepare(environment_reference, args, 1)

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
                update_controller,
                args,
                evaluation_points=score,
            )

    with pause:
        evaluate(start)
        pause.service()

        for step in range(start, args.steps):
            pause.service()
            schedule_position = coop.optimizer_schedule_position(step, args)
            if schedule_position is not None:
                agent_schedule.step(schedule_position)
                environment_schedule.step(schedule_position)
            schedule_progress = coop.exploration_schedule_progress(
                step,
                args,
                schedule_start,
                exploration_schedule_steps,
            )

            with agent_inference.evaluating(), environment_inference.evaluating():
                games = rollouts.play(
                    [agent_inference.policy(), environment_inference.policy()],
                    games=args.rollout_games,
                    max_steps=args.rollout_max_steps,
                    rotate=True,
                )
            pause.service()

            compute_returns(games)
            agent_update = update_controller.update(
                games,
                0,
                model=agent,
                reference=agent_reference,
                prepare=prepare_agent,
                learner=agent_learner,
                progress=schedule_progress,
            )
            pause.service()
            environment_update = update_controller.update(
                games,
                1,
                model=environment,
                reference=environment_reference,
                prepare=prepare_environment,
                learner=environment_learner,
                progress=schedule_progress,
            )
            pause.service()
            completed = step + 1

            if completed % 5 == 0:
                metrics.log(
                    completed,
                    rollout=rollout_metrics(games, coop=False),
                    train={
                        "agent": agent_update.metrics,
                        "environment": environment_update.metrics,
                    },
                    update_controller={
                        "stop_threshold": args.stop_win_rate_threshold,
                        "restart_threshold": args.restart_win_rate_threshold,
                        "agent": {
                            "batch_win_rate": agent_update.win_rate,
                            "updated": agent_update.updated,
                            "paused": agent_update.paused,
                        },
                        "environment": {
                            "batch_win_rate": environment_update.win_rate,
                            "updated": environment_update.updated,
                            "paused": environment_update.paused,
                        },
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
                    update_controller,
                    args,
                )
                pause.service()

        pause.service()
        final_path = save(
            checkpoints,
            args.steps,
            agent,
            environment,
            agent_optimizer,
            environment_optimizer,
            agent_learner,
            environment_learner,
            update_controller,
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
    agent,
    environment,
    agent_optimizer,
    environment_optimizer,
    agent_learner,
    environment_learner,
    update_controller,
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
            "update_controller": update_controller,
        },
        metadata={
            "trainer": "adversarial2",
            "algorithm": "two-policy-ppo",
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
