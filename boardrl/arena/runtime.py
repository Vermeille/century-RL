"""CPU arena process orchestration."""

from __future__ import annotations

import gc
import hashlib
import random
import re
import time
from contextlib import ExitStack
from pathlib import Path

import torch

from boardrl.evaluation import Evaluator
from boardrl.games import games_library
from boardrl.rl.model import load_model
from boardrl.rollouts import Inference

from .events import TrackioTelemetry
from .ratings import OPPONENT_ID, RANDOM_ID, RatingFit, fit_ratings, residuals
from .reporting import ArenaReporter
from .scheduler import MatchScheduler
from .snapshots import capture_agent_checkpoint, choose_retained_pool
from .store import ArenaStore


class RatingArena:
    """Watch checkpoints, schedule CPU matches, and maintain arena state."""

    def __init__(
        self,
        checkpoint_directory: Path,
        arguments: dict[str, object],
        *,
        run_id: str,
        telemetry: TrackioTelemetry | None = None,
        started_at: float = 0.0,
        pool_size: int = 32,
        games_per_batch: int = 32,
        calibration_games: int = 64,
        placement_batches: int = 3,
        plot_every_batches: int = 10,
        poll_seconds: float = 1.0,
    ):
        self.checkpoint_directory = Path(checkpoint_directory)
        self.arguments = arguments
        self.run_id = run_id
        self.telemetry = telemetry
        self.started_at = started_at
        self.pool_size = pool_size
        self.games_per_batch = games_per_batch
        self.calibration_games = calibration_games
        self.placement_batches = placement_batches
        self.plot_every_batches = plot_every_batches
        self.poll_seconds = poll_seconds
        self.stop_requested = False
        self.completed_batches = 0

        game_spec = str(arguments["game"])
        self.game = games_library(game_spec)
        if self.game.coop:
            raise ValueError("the rating arena requires a competitive game")
        probe = self.game.make_game(num_players=2)
        if probe.num_players != 2:
            raise ValueError("the rating arena requires exactly two players")

        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]", "-", run_id)
        self.arena_directory = self.checkpoint_directory / ".arena" / safe_run_id
        self.pool_directory = self.arena_directory / "pool"
        self.pool_directory.mkdir(parents=True, exist_ok=True)
        self.store = ArenaStore(self.arena_directory / "arena.sqlite")
        self.store.add_policy(RANDOM_ID, kind="strategy")
        opponent = str(arguments["opponent_eval_strategy"])
        self.reference_id = RANDOM_ID if opponent == "random" else OPPONENT_ID
        if self.reference_id == OPPONENT_ID:
            self.store.add_policy(OPPONENT_ID, kind="strategy")
        self.scheduler = MatchScheduler(
            self.store,
            reference_id=self.reference_id,
            placement_batches=placement_batches,
        )
        self.reporter = ArenaReporter(
            self.store,
            telemetry=telemetry,
            reference_id=self.reference_id,
            opponent_name=opponent,
            placement_batches=placement_batches,
        )

    def close(self):
        self.store.close()

    def request_stop(self, *_):
        self.stop_requested = True

    def capture_checkpoints(self) -> int:
        captured = 0
        candidates = []
        for source in self.checkpoint_directory.glob("step-*.pth"):
            match = re.fullmatch(r"step-(\d+)", source.stem)
            if match:
                candidates.append((int(match.group(1)), source))
        for step, source in sorted(candidates):
            try:
                modified_at = source.stat().st_mtime
            except FileNotFoundError:
                continue
            if self.started_at and modified_at < self.started_at:
                continue
            policy_id = f"checkpoint-{step}"
            try:
                self.store.policy(policy_id)
                continue
            except KeyError:
                pass
            destination = self.pool_directory / f"agent-step-{step}.pth"
            try:
                captured_id, captured_step = capture_agent_checkpoint(
                    source,
                    destination,
                    expected_game=str(self.arguments["game"]),
                )
            except (FileNotFoundError, EOFError, RuntimeError, KeyError, ValueError):
                continue
            self.store.add_policy(
                captured_id,
                kind="checkpoint",
                step=captured_step,
                path=destination,
            )
            print(f"arena: captured agent checkpoint at step {captured_step}", flush=True)
            captured += 1
        return captured

    def fit(self) -> RatingFit:
        policies = self.store.policies()
        fit = fit_ratings(
            [policy.id for policy in policies],
            self.store.matches(),
        )
        self.store.record_fit(fit)
        return fit

    def rebalance_pool(self, fit: RatingFit):
        checkpoint_policies = self.store.policies(checkpoints_only=True)
        retained = choose_retained_pool(
            checkpoint_policies,
            fit,
            residuals(self.store.matches(), fit),
            limit=self.pool_size,
        )
        self.store.set_pool(retained)

    def _strategy(self, policy_id: str, stack: ExitStack):
        if policy_id == RANDOM_ID:
            return self.game.strategy_from_string("random")
        if policy_id == OPPONENT_ID:
            return self.game.strategy_from_string(
                str(self.arguments["opponent_eval_strategy"])
            )
        policy = self.store.policy(policy_id)
        if policy.path is None:
            raise RuntimeError(f"policy {policy_id} is no longer playable")
        model = load_model(policy.path, name="agent", device="cpu")
        inference = Inference(
            model,
            batch_size=min(int(self.arguments.get("inference_batch_size", 256)), 256),
            name=policy_id,
        )
        stack.enter_context(inference.evaluating())
        return inference.policy(
            temperature=float(self.arguments.get("eval_temperature", 1.0))
        )

    def _seed(self, first: str, second: str, purpose: str, batch_number: int) -> int:
        digest = hashlib.sha256(
            f"{self.run_id}|{first}|{second}|{purpose}|{batch_number}".encode()
        ).digest()
        return int.from_bytes(digest[:4], "big")

    @staticmethod
    def _seed_everything(seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        try:
            from boardrl.cyutils import init_seed

            init_seed(seed)
        except ImportError:
            pass

    def play_batch(self, first: str, second: str, purpose: str):
        batch_number = self.store.get_int("next_batch")
        seed = self._seed(first, second, purpose, batch_number)
        batch_id = hashlib.sha256(
            f"{self.run_id}|{batch_number}|{first}|{second}|{purpose}".encode()
        ).hexdigest()
        self._seed_everything(seed)
        evaluator = Evaluator(
            self.game.make_game,
            progress=False,
            outcome=self.game.outcome,
        )
        with ExitStack() as stack:
            first_player = self._strategy(first, stack)
            second_player = self._strategy(second, stack)
            evaluation = evaluator.compare(
                [first_player, second_player],
                names=[first, second],
                games=self.games_per_batch,
                max_steps=int(self.arguments.get("rollout_max_steps", 5_000)),
                rotate=True,
            )
        score = evaluation.win_rate(0) * self.games_per_batch
        inserted = self.store.record_match(
            batch_id=batch_id,
            first=first,
            second=second,
            games=self.games_per_batch,
            score=score,
            seed=seed,
            purpose=purpose,
        )
        if inserted:
            self.store.set_int("next_batch", batch_number + 1)
            self.completed_batches += 1
            print(
                f"arena: {purpose} {first} vs {second}: "
                f"{score:g}/{self.games_per_batch}",
                flush=True,
            )
        gc.collect()

    def publish(self, fit: RatingFit):
        self.reporter.publish(fit)

    def run(self):
        opponent = str(self.arguments["opponent_eval_strategy"])
        if opponent != "random":
            while self.store.games_between(RANDOM_ID, OPPONENT_ID) < self.calibration_games:
                self.play_batch(OPPONENT_ID, RANDOM_ID, "calibration")
                captured = self.capture_checkpoints()
                if captured:
                    self.rebalance_pool(self.fit())

        fit = self.fit()
        if opponent != "random":
            self.reporter.publish_calibration(fit)
        while True:
            captured = self.capture_checkpoints()
            if captured:
                fit = self.fit()
                self.rebalance_pool(fit)

            placement = self.scheduler.placement(fit)
            if placement is not None:
                policy_id = placement[0]
                self.play_batch(*placement)
                fit = self.fit()
                self.rebalance_pool(fit)
                if (
                    self.store.policy(policy_id).placement_batches
                    >= self.placement_batches
                ):
                    self.publish(fit)
                continue

            if self.stop_requested:
                captured = self.capture_checkpoints()
                if captured:
                    fit = self.fit()
                    self.rebalance_pool(fit)
                placement = self.scheduler.placement(fit)
                if placement is not None:
                    continue
                self.publish(fit)
                return

            idle = self.scheduler.idle(fit)
            if idle is None:
                time.sleep(self.poll_seconds)
                continue
            self.play_batch(*idle)
            fit = self.fit()
            self.rebalance_pool(fit)
            if self.store.get_int("idle_batches") % self.plot_every_batches == 0:
                self.publish(fit)
