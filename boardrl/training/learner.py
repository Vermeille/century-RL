"""Reusable optimization mechanics with no experiment loop."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import random
from typing import ClassVar

import torch

from boardrl.rl.model.loss import Loss
from boardrl.rl.utils import explained_variance, pearson_corr
from boardrl.training.cuda_pause import CudaOffloadPause
from boardrl.training.sample import TrainingSample
from boardrl.utils import chunk


def gradient_norm(parameters) -> torch.Tensor:
    norms = [p.grad.detach().norm() for p in parameters if p.grad is not None]
    if not norms:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(norms))


def _float(value) -> float:
    return value.detach().item() if torch.is_tensor(value) else float(value)


def normalized_nucleus_size(
    probs: torch.Tensor,
    threshold: float = 0.95,
) -> torch.Tensor:
    """
    Return the normalized number of actions needed to reach ``threshold``.

    ``probs`` may have any leading dimensions and must contain normalized
    probabilities along its last dimension. A deterministic distribution is
    mapped to 0, while a distribution requiring the largest possible nucleus
    for the threshold is mapped to 1.
    """
    n = probs.shape[-1]

    sorted_probs = probs.sort(dim=-1, descending=True).values
    cumulative = sorted_probs.cumsum(dim=-1)

    # Nombre minimal d'actions pour dépasser le seuil.
    k = (cumulative < threshold).sum(dim=-1) + 1

    max_k = min(n, math.ceil(threshold * n))

    if max_k <= 1:
        return torch.zeros_like(k, dtype=probs.dtype)

    return (k.to(probs.dtype) - 1) / (max_k - 1)


@dataclass(frozen=True)
class TrainResult:
    metrics: dict[str, float]
    samples: int
    batches: int


class PolicyMetrics:
    """Prediction metrics that work for RL and imitation datasets."""

    def __init__(self, nucleus_threshold: float = 0.95):
        self.nucleus_threshold = nucleus_threshold

    def __call__(self, policy, value, batch):
        probabilities = [torch.softmax(logits, dim=0) for logits in policy]
        entropies = torch.stack(
            [
                torch.sum(
                    -probs * torch.log_softmax(logits, dim=0)
                )
                for logits, probs in zip(policy, probabilities)
            ]
        )
        nucleus_sizes = torch.stack(
            [
                normalized_nucleus_size(probs, self.nucleus_threshold)
                for probs in probabilities
            ]
        )
        # Keep the reduction on-device. Calling item() for every policy used to
        # force one CUDA synchronization per sample; Averages converts this
        # single batch result to a float once.
        return {
            "perplexity": entropies.exp().mean(),
            (
                "normalized_nucleus_size_threshold_"
                f"{self.nucleus_threshold:g}".replace(".", "_")
            ): nucleus_sizes.mean(),
        }


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

    def begin_epoch(self, num_batches):
        pass

    def finish(self):
        return {}


class BatchUpdates(Updates, reusable=True):
    def __init__(self, learner, samples):
        super().__init__(learner, samples)
        self.lr_scale = 1.0
        self.unscaled_lrs = None

    @property
    def epochs(self):
        return self.learner.epochs

    def start(self):
        self.unscaled_lrs = [
            group["lr"] for group in self.learner.optimizer.param_groups
        ]

    def begin_epoch(self, num_batches):
        if self.learner.base_batches is None or not self.learner.normalize_lr:
            self.learner.base_batches = num_batches
        self.lr_scale = (
            self.learner.base_batches / num_batches
            if self.learner.normalize_lr
            else 1.0
        )
        for lr, group in zip(
            self.unscaled_lrs,
            self.learner.optimizer.param_groups,
        ):
            group["lr"] = lr * self.lr_scale

    def begin_batch(self):
        self.learner.optimizer.zero_grad(set_to_none=True)

    def objective(self, objective, batch):
        return objective

    def end_batch(self):
        return {"gradient_norm": self.learner._finish_batch()}

    def finish(self):
        for lr, group in zip(
            self.unscaled_lrs,
            self.learner.optimizer.param_groups,
        ):
            group["lr"] = lr
        return {"lr_scale": self.lr_scale}


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
    any opponent. Used as a context manager, the learner also owns cooperative
    Ctrl-Z handling. ``offload_modules`` adds reference or target networks that
    must move with the optimized model, and :meth:`safe_point` marks
    experiment-level boundaries where pausing is safe.
    """

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        losses: Sequence[Loss],
        *,
        batch_size: int,
        device: str | torch.device,
        epochs: int = 1,
        gradient_clip: float | None = None,
        augmentations: Sequence[Callable] = (),
        batch_metrics: Sequence[Callable] = (),
        normalize_lr: bool = False,
        offload_modules: Sequence[torch.nn.Module] = (),
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
        self._pause = CudaOffloadPause(
            (model, *offload_modules),
            optimizer,
            device=device,
        )
        # Keep the reference across train() calls so changing rollout sizes
        # does not change the intended update magnitude.
        self.base_batches = None

    def __enter__(self):
        self._pause.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._pause.__exit__(exc_type, exc_value, traceback)

    def safe_point(self):
        """Honor a pending pause at an experiment-safe boundary."""

        self._pause.service()

    def state_dict(self):
        return {
            "base_batches": self.base_batches,
            "losses": [loss.state_dict() for loss in self.losses],
        }

    def load_state_dict(self, state):
        loss_states = state["losses"]
        if len(loss_states) != len(self.losses):
            raise ValueError(
                "checkpoint loss count does not match the configured learner"
            )
        self.base_batches = state["base_batches"]
        for loss, loss_state in zip(self.losses, loss_states):
            loss.load_state_dict(loss_state)

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
            updates.begin_epoch(num_batches)
            for raw_batch in chunk(epoch_samples, self.batch_size):
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
        metrics.add({"lr": self.optimizer.param_groups[0]["lr"]})
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


class ScheduleShape:
    """Interpolation shape mapping normalized progress from zero to one."""

    def __call__(self, progress: float) -> float:
        raise NotImplementedError


class LinearScheduleShape(ScheduleShape):
    def __call__(self, progress: float) -> float:
        return progress


class CosineScheduleShape(ScheduleShape):
    def __call__(self, progress: float) -> float:
        return 0.5 * (1.0 - math.cos(math.pi * progress))


SCHEDULE_SHAPES = {
    "linear": LinearScheduleShape(),
    "cosine": CosineScheduleShape(),
}


class WarmupDecay:
    """Optimizer warmup followed by a shaped decay to a floor."""

    def __init__(
        self,
        optimizer,
        *,
        steps: int,
        warmup: int | None = None,
        min_scale: float = 0.0,
        shape: ScheduleShape,
    ):
        self.optimizer = optimizer
        self.steps = steps
        self.warmup = min(100, steps * 0.05) if warmup is None else warmup
        self.min_scale = min_scale
        self.shape = shape
        self.initial_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, step: int) -> float:
        if self.warmup > 0 and step < self.warmup:
            scale = step / self.warmup
        else:
            decay_steps = max(self.steps - self.warmup, 1)
            progress = (step - self.warmup) / decay_steps
            progress = min(max(progress, 0.0), 1.0)
            scale = self.min_scale + (1 - self.min_scale) * (
                1 - self.shape(progress)
            )
            scale = max(scale, self.min_scale)
        for initial_lr, group in zip(self.initial_lrs, self.optimizer.param_groups):
            group["lr"] = initial_lr * scale
        return self.optimizer.param_groups[0]["lr"]


class LinearWarmupDecay(WarmupDecay):
    """Backward-compatible linear warmup/decay scheduler."""

    def __init__(self, optimizer, *, steps, warmup=None, min_scale=0.0):
        super().__init__(
            optimizer,
            steps=steps,
            warmup=warmup,
            min_scale=min_scale,
            shape=SCHEDULE_SHAPES["linear"],
        )


class CosineWarmupDecay(WarmupDecay):
    """Warm up linearly, then decay with a half cosine."""

    def __init__(self, optimizer, *, steps, warmup=None, min_scale=0.0):
        super().__init__(
            optimizer,
            steps=steps,
            warmup=warmup,
            min_scale=min_scale,
            shape=SCHEDULE_SHAPES["cosine"],
        )
