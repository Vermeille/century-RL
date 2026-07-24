"""Metrics collection and display independent of training algorithms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import singledispatch
from html import escape

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

class Visdom:
    """Adapter for the existing Visdom/offline visualizer."""

    def __init__(self, visualizer):
        self.visualizer = visualizer

    def log(self, step: int, values: Mapping[str, object]) -> None:
        for name, value in _flatten(values):
            _push_visdom(value, self.visualizer, name, step)

    def text(self, name: str, value: str) -> None:
        self.visualizer.html(name, f"<pre>{escape(value)}</pre>")

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
    """A game only supplies values; every sink gets display support for free."""

    def metrics_to_visdom(self, visualizer, step):
        Visdom(visualizer).log(step, self.metrics())


@singledispatch
def _push_visdom(value, visualizer, name, step):
    visualizer.push(name, value, step)


@_push_visdom.register(Range)
def _(value, visualizer, name, step):
    visualizer.push_range(name, list(value.values), step)


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
