"""Reusable optimization mechanics with no experiment loop."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import random
from typing import ClassVar

import torch

from boardrl.rl.utils import explained_variance, pearson_corr
from boardrl.training.sample import TrainingSample
from boardrl.utils import chunk


def gradient_norm(parameters) -> torch.Tensor:
    norms = [p.grad.detach().norm() for p in parameters if p.grad is not None]
    if not norms:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(norms))


def _float(value) -> float:
    return value.detach().item() if torch.is_tensor(value) else float(value)


@dataclass(frozen=True)
class TrainResult:
    metrics: dict[str, float]
    samples: int
    batches: int


class PolicyMetrics:
    """Prediction metrics that work for RL and imitation datasets."""

    def __call__(self, policy, value, batch):
        perplexity = sum(
            torch.exp(
                torch.sum(-torch.softmax(logits, 0) * torch.log_softmax(logits, 0))
            ).item()
            for logits in policy
        ) / len(policy)
        return {"perplexity": perplexity}


class ValueMetrics:
    """Value-prediction metrics for samples that contain ``returns``."""

    def __call__(self, policy, value, batch):
        return {
            "pearson": pearson_corr(value.mean, batch.returns),
            "explained_variance": explained_variance(value.mean, batch.returns),
            "mae": torch.nn.functional.l1_loss(value.mean, batch.returns),
        }


class Averages:
    def __init__(self):
        self.totals = {}
        self.counts = {}

    def add(self, values):
        for name, value in values.items():
            self.totals[name] = self.totals.get(name, 0.0) + _float(value)
            self.counts[name] = self.counts.get(name, 0) + 1

    def result(self):
        return {name: total / self.counts[name] for name, total in self.totals.items()}


class Updates:
    modes: ClassVar[dict[bool, type["Updates"]]] = {}

    def __init_subclass__(cls, *, reusable, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.modes[reusable] = cls

    def __init__(self, learner, samples):
        self.learner = learner
        self.samples = samples

    @classmethod
    def for_losses(cls, learner, samples):
        reusable = all(loss.supports_off_policy for loss in learner.losses)
        return cls.modes[reusable](learner, samples)

    def start(self):
        pass

    def finish(self):
        return {}


class BatchUpdates(Updates, reusable=True):
    @property
    def epochs(self):
        return self.learner.epochs

    def begin_batch(self):
        self.learner.optimizer.zero_grad(set_to_none=True)

    def objective(self, objective, batch):
        return objective

    def end_batch(self):
        return {"gradient_norm": self.learner._finish_batch()}


class RolloutUpdate(Updates, reusable=False):
    epochs = 1

    def start(self):
        self.learner.optimizer.zero_grad(set_to_none=True)

    def begin_batch(self):
        pass

    def objective(self, objective, batch):
        return objective * (len(batch) / len(self.samples))

    def end_batch(self):
        return {}

    def finish(self):
        return {"gradient_norm": self.learner._finish_batch()}


class Learner:
    """Train one model on an already prepared list of samples.

    The surrounding algorithm remains plain Python: it decides where samples
    came from, which model supplies targets, when to train, and when to update
    any opponent.
    """

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        losses: Sequence[Callable],
        *,
        batch_size: int,
        device: str | torch.device,
        epochs: int = 1,
        gradient_clip: float | None = None,
        augmentations: Sequence[Callable] = (),
        batch_metrics: Sequence[Callable] = (),
        normalize_lr: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.losses = list(losses)
        self.batch_size = batch_size
        self.device = device
        self.epochs = epochs
        self.gradient_clip = gradient_clip
        self.augmentations = tuple(augmentations)
        self.batch_metrics = tuple(batch_metrics)
        self.normalize_lr = normalize_lr
        self.base_batches = None

    def train(self, samples: Sequence[TrainingSample], *, progress=0.0) -> TrainResult:
        if not samples:
            return TrainResult({}, 0, 0)

        self.model.train()
        updates = Updates.for_losses(self, samples)
        metrics = Averages()
        batches = 0
        seen = 0

        updates.start()

        for _ in range(updates.epochs):
            epoch_samples = list(samples)
            for augmentation in self.augmentations:
                epoch_samples = augmentation(epoch_samples)
            random.shuffle(epoch_samples)

            num_batches = math.ceil(len(epoch_samples) / self.batch_size)
            for raw_batch in chunk(epoch_samples, self.batch_size):
                if self.base_batches is None or not self.normalize_lr:
                    self.base_batches = num_batches
                updates.begin_batch()
                batch = TrainingSample.collate(raw_batch).to(
                    self.device, non_blocking=True
                )
                policy, value = self.model(batch.state)
                training_state = {"progress": float(progress)}
                results = [
                    (type(loss).__name__, loss(policy, value, batch, training_state))
                    for loss in self.losses
                ]
                objective = sum(result.objective for _, result in results)
                objective *= self.base_batches / num_batches
                updates.objective(objective, raw_batch).backward()

                with torch.no_grad():
                    for name, result in results:
                        values = {name: result.objective}
                        for metric_name, metric in result.metrics.items():
                            values[f"{name}.{metric_name}"] = metric
                        metrics.add(values)
                    for metric_group in self.batch_metrics:
                        metrics.add(metric_group(policy, value, batch))
                    metrics.add(updates.end_batch())
                batches += 1
                seen += len(raw_batch)

        metrics.add(updates.finish())
        return TrainResult(metrics.result(), seen, batches)

    def _finish_batch(self):
        if self.gradient_clip is None:
            norm = gradient_norm(self.model.parameters())
        else:
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip
            )
        self.optimizer.step()
        return norm


class LinearWarmupDecay:
    """Small explicit scheduler used by the former config-driven trainer."""

    def __init__(
        self,
        optimizer,
        *,
        steps: int,
        warmup: int | None = None,
        min_scale: float = 0.0,
    ):
        self.optimizer = optimizer
        self.steps = steps
        self.warmup = min(100, steps * 0.05) if warmup is None else warmup
        self.min_scale = min_scale
        self.initial_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, step: int) -> float:
        if self.warmup > 0 and step < self.warmup:
            scale = step / self.warmup
        else:
            decay_steps = max(self.steps - self.warmup, 1)
            progress = (step - self.warmup) / decay_steps
            scale = self.min_scale + (1 - self.min_scale) * (1 - progress)
        scale = max(scale, self.min_scale if step <= self.steps else 0.0)
        for initial_lr, group in zip(self.initial_lrs, self.optimizer.param_groups):
            group["lr"] = initial_lr * scale
        return self.optimizer.param_groups[0]["lr"]
