"""Composable rollout post-processing."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
import random
from typing import Any

import torch

from boardrl.rl.eval.selfplay import SelfPlayResults
from boardrl.training.returns import annotate_with_model, compute_returns
from boardrl.training.sample import TrainingSample
from boardrl.utils import chunk


Processor = Callable[[Any], Any]


class Pipeline:
    """Compose post-processing objects and ordinary callables."""

    def __init__(self, *steps: Processor):
        self.steps = steps

    def __call__(self, value):
        for step in self.steps:
            value = step(value)
        return value

    def then(self, *steps: Processor) -> "Pipeline":
        return Pipeline(*self.steps, *steps)


class ComputeReturns:
    def __init__(self, discount, *, entropy_bonus=None, reward_scale=None):
        self.discount = discount
        self.entropy_bonus = entropy_bonus
        self.reward_scale = reward_scale

    def __call__(self, games: SelfPlayResults) -> SelfPlayResults:
        compute_returns(
            games,
            self.discount,
            entropy_reward_scale=self.entropy_bonus,
            reward_rescale=self.reward_scale,
        )
        return games


class Select:
    """Keep selected seats or strategy identities from every game."""

    def __init__(self, *, seats=None, strategies=None):
        if seats is not None and strategies is not None:
            raise ValueError("select seats or strategies, not both")
        self.seats = list(seats) if seats is not None else None
        self.strategies = list(strategies) if strategies is not None else None

    def __call__(self, games: SelfPlayResults) -> SelfPlayResults:
        if self.seats is not None:
            return games.only_player(self.seats)
        if self.strategies is not None:
            return games.only_strategy(self.strategies)
        return games


class ToSamples:
    """Flatten player trajectories into linked training samples."""

    def __call__(self, games: SelfPlayResults) -> list[TrainingSample]:
        samples = []
        for game in games:
            for trace in game:
                if len(trace) < 2:
                    continue
                previous = None
                for record in trace[:-1]:
                    sample = record.training_sample()
                    if previous is not None:
                        previous.next = sample
                    samples.append(sample)
                    previous = sample
                if previous is not None:
                    previous.next = trace[-1]
        return samples


class DropFields:
    """Remove rollout-only annotations that a learner does not consume."""

    def __init__(self, *fields: str):
        self.fields = fields

    def __call__(self, samples: list[TrainingSample]) -> list[TrainingSample]:
        for sample in samples:
            for field in self.fields:
                sample.__dict__.pop(field, None)
        return samples


class ReferenceTargets:
    """Add values, advantages, GAE and TD(lambda) targets from a model."""

    def __init__(
        self,
        model,
        *,
        batch_size: int,
        discount: float,
        trace_decay: float,
        reuse_rollout_predictions: bool = False,
    ):
        self.model = model
        self.batch_size = batch_size
        self.discount = discount
        self.trace_decay = trace_decay
        self.reuse_rollout_predictions = reuse_rollout_predictions

    def __call__(self, samples: list[TrainingSample]) -> list[TrainingSample]:
        if samples:
            annotate_with_model(
                self.model,
                samples,
                self.batch_size,
                self.discount,
                self.trace_decay,
                use_cached_rollout=self.reuse_rollout_predictions,
            )
        return samples


class ReplayBuffer:
    """Retain transitions and return a uniformly sampled training batch."""

    def __init__(self, capacity: int, samples_per_update: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if samples_per_update <= 0:
            raise ValueError("samples_per_update must be positive")
        self.samples = deque(maxlen=capacity)
        self.samples_per_update = samples_per_update

    def __call__(self, samples: list[TrainingSample]) -> list[TrainingSample]:
        self.samples.extend(samples)
        count = min(len(self.samples), self.samples_per_update)
        return random.sample(list(self.samples), count)


class DoubleQTargets:
    """Compute Double-DQN bootstrap values for sampled transitions."""

    def __init__(self, online_model, target_model, *, batch_size: int):
        self.online_model = online_model
        self.target_model = target_model
        self.batch_size = batch_size

    def __call__(self, samples: list[TrainingSample]) -> list[TrainingSample]:
        pending = [sample for sample in samples if not sample.next.terminal]
        for sample in samples:
            if sample.next.terminal:
                sample.next_reference_max_q = 0.0

        if not pending:
            return samples

        states = [sample.next.state for sample in pending]
        online_training = self.online_model.training
        self.online_model.eval()
        self.target_model.eval()
        try:
            with torch.no_grad():
                online = self._evaluate(self.online_model, states)
                target = self._evaluate(self.target_model, states)
        finally:
            self.online_model.train(online_training)

        for sample, online_pred, target_pred in zip(pending, online, target):
            action = online_pred.q_value()[0].argmax()
            sample.next_reference_max_q = target_pred.q_value()[0][action].item()
        return samples

    def _evaluate(self, model, states):
        predictions = []
        for batch in chunk(states, self.batch_size):
            predictions.extend(model(batch).unbatched())
        return predictions


def samples_from(games: SelfPlayResults) -> list[TrainingSample]:
    return ToSamples()(games)
