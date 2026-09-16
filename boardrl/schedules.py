"""Reusable schedules driven by normalized training progress."""

from __future__ import annotations

import math


class ScheduleShape:
    """Map normalized progress from zero to one to a schedule value."""

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


class Scheduler:
    """Convert global normalized progress into a scheduled scalar.

    ``start``, ``end``, and ``warmup`` are positions in the input progress
    domain, not step counts. Before ``start`` the schedule holds
    ``start_value``. If ``warmup`` is nonzero, it linearly ramps from zero to
    ``start_value`` first. Between ``start`` and ``end`` the configured shape
    interpolates from ``start_value`` to ``end_value``.
    """

    def __init__(
        self,
        *,
        start: float = 0.0,
        end: float = 1.0,
        warmup: float = 0.0,
        shape: ScheduleShape | str = "linear",
        curve: float = 1.0,
        start_value: float = 0.0,
        end_value: float = 1.0,
    ):
        if not 0.0 <= start <= 1.0:
            raise ValueError("schedule start must be in [0, 1]")
        if not 0.0 <= end <= 1.0:
            raise ValueError("schedule end must be in [0, 1]")
        if end < start:
            raise ValueError("schedule end must not precede its start")
        if not 0.0 <= warmup <= 1.0:
            raise ValueError("schedule warmup must be in [0, 1]")
        if curve <= 0.0:
            raise ValueError("schedule curve must be positive")

        if isinstance(shape, str):
            try:
                shape = SCHEDULE_SHAPES[shape]
            except KeyError as exc:
                raise ValueError(f"unknown schedule shape: {shape}") from exc

        self.start = start
        self.end = end
        self.warmup = warmup
        self.shape = shape
        self.curve = curve
        self.start_value = start_value
        self.end_value = end_value

    @classmethod
    def from_steps(
        cls,
        *,
        total_steps: int,
        start_step: int = 0,
        end_step: int | None = None,
        warmup_steps: int = 0,
        **kwargs,
    ):
        """Build a normalized scheduler from integer step boundaries."""

        denominator = max(total_steps - 1, 1)
        if end_step is None:
            end_step = total_steps - 1
        normalize = lambda step: min(max(step / denominator, 0.0), 1.0)
        return cls(
            start=normalize(start_step),
            end=normalize(end_step),
            warmup=normalize(warmup_steps),
            **kwargs,
        )

    def to_schedule(self, progress: float) -> float:
        progress = min(max(float(progress), 0.0), 1.0)
        decay_start = max(self.start, self.warmup)

        if self.warmup > 0.0 and progress < self.warmup:
            return self.start_value * progress / self.warmup
        if progress < decay_start:
            return self.start_value
        if progress >= self.end or self.end == decay_start:
            return self.end_value

        local = (progress - decay_start) / (self.end - decay_start)
        local = min(max(local, 0.0), 1.0) ** self.curve
        shaped = self.shape(local)
        return self.start_value + (self.end_value - self.start_value) * shaped
