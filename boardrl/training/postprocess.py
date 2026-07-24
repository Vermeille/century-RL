"""Composable rollout post-processing."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from boardrl.rl.eval.selfplay import SelfPlayResults
from boardrl.training.returns import annotate_with_model, compute_returns
from boardrl.training.sample import TrainingSample


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


def samples_from(games: SelfPlayResults) -> list[TrainingSample]:
    return ToSamples()(games)
