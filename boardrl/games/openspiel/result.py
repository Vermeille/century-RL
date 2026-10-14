"""Exact NashConv and exploitability in the game's native utility units."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NashConvResult:
    nash_conv: float
    exploitability: float
    player_improvements: tuple[float, ...]
    evaluated_states: int
