"""Metrics collection and display independent of training algorithms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import singledispatch
from pathlib import Path

import torch


@dataclass(frozen=True)
class Range:
    values: Sequence[float]


@singledispatch
def _flatten_value(value, name):
    yield name, value


@_flatten_value.register(dict)
def _(value, name):
    for child_name, child in value.items():
        full_name = f"{name}.{child_name}" if name else str(child_name)
        yield from _flatten_value(child, full_name)


def _flatten(values: Mapping):
    yield from _flatten_value(dict(values), "")


@singledispatch
def _console_value(value):
    return value


@_console_value.register(Range)
def _(value):
    xs = list(value.values)
    return sum(xs) / len(xs) if xs else float("nan")


class Console:
    """Compact terminal metrics display."""

    def log(self, step: int, values: Mapping[str, object]) -> None:
        rendered = []
        for name, value in _flatten(values):
            value = _console_value(value)
            if torch.is_tensor(value):
                value = value.detach().item()
            rendered.append(f"{name}={value}")
        print(f"step {step}: " + "  ".join(rendered))


@singledispatch
def _trackio_value(value):
    return value


@_trackio_value.register(Range)
def _(value):
    # Keep the scalar behavior used by Console while leaving histogram support
    # available for a future visualization-specific sink.
    return _trackio_value(_console_value(value))


@_trackio_value.register(torch.Tensor)
def _(value):
    value = value.detach().cpu()
    return value.item() if value.numel() == 1 else value.tolist()


def _flatten_trackio(values: Mapping):
    for name, value in _flatten(values):
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, child in enumerate(value):
                yield f"{name}.{index}", _trackio_value(child)
        else:
            yield name, _trackio_value(value)


class Trackio:
    """Adapter for a Trackio run using the common metric sink interface."""

    def __init__(self, run):
        self.run = run

    def log(self, step: int, values: Mapping[str, object]) -> None:
        self.run.log(dict(_flatten_trackio(values)), step=step)

    def finish(self) -> None:
        self.run.finish()


def make_trackio(
    *,
    project: str | None,
    name: str | None = None,
    config: Mapping[str, object] | None = None,
) -> Trackio | None:
    """Create a Trackio sink only when a project was explicitly requested."""
    if project is None:
        return None

    try:
        import trackio
    except ImportError as exc:
        raise RuntimeError(
            "Trackio logging requires the optional dependency; run `uv sync --extra trackio`"
        ) from exc

    kwargs = {
        "project": project,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in (config or {}).items()
        },
    }
    if name is not None:
        kwargs["name"] = name
    return Trackio(trackio.init(**kwargs))


class MetricLogger:
    """Fan metrics out to any number of display/storage sinks."""

    def __init__(self, *sinks):
        self.sinks = sinks

    def log(self, step: int, **values) -> None:
        for sink in self.sinks:
            sink.log(step, values)

    def game(self, step: int, metrics, *, histories: bool = False) -> None:
        if histories:
            metrics.print_short_history()
        self.log(step, game=metrics.metrics())


class GameMetrics:
    """A game supplies metric values to the experiment's configured sinks."""


def rollout_metrics(results) -> dict[str, object]:
    """Game-independent rollout statistics."""
    if not results:
        return {"games": 0, "samples": 0}
    return {
        "games": len(results),
        "samples": results.num_samples(),
        "win_rate": [
            results.win_rate(player, by="strategy")
            for player in range(results.num_players())
        ],
        "avg_reward": [
            results.my_avg_reward(player, by="strategy")
            for player in range(results.num_players())
        ],
        "points": Range(results.my_points(0, by="strategy")),
    }
