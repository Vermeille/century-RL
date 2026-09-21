"""Head-to-head dominance graph and cycle detection."""

from __future__ import annotations

import math
from typing import Iterable

from .ratings import Match, pair_aggregates


MAXIMUM_REPORTED_CYCLES = 100


def confirmed_win_graph(
    matches: Iterable[Match],
    policy_ids: Iterable[str],
    *,
    minimum_games: int = 32,
) -> dict[str, set[str]]:
    """Build edges for head-to-head wins whose 95% Wilson bound exceeds 50%."""

    ids = sorted(set(policy_ids))
    graph = {policy_id: set() for policy_id in ids}
    z = 1.96
    for match in pair_aggregates(matches).values():
        if (
            match.games < minimum_games
            or match.first not in graph
            or match.second not in graph
        ):
            continue
        probability = match.score / match.games
        denominator = 1.0 + z**2 / match.games
        center = probability + z**2 / (2.0 * match.games)
        margin = z * math.sqrt(
            probability * (1.0 - probability) / match.games
            + z**2 / (4.0 * match.games**2)
        )
        lower = (center - margin) / denominator
        upper = (center + margin) / denominator
        if lower > 0.5:
            graph[match.first].add(match.second)
        elif upper < 0.5:
            graph[match.second].add(match.first)
    return graph


def find_directed_cycles(
    graph: dict[str, set[str]],
    *,
    maximum: int = MAXIMUM_REPORTED_CYCLES,
) -> list[tuple[str, ...]]:
    """Return rotation-canonical simple cycles, including cycles longer than three."""

    cycles: set[tuple[str, ...]] = set()
    for start in sorted(graph):
        path = [start]
        visited = {start}

        def search(node: str):
            for successor in sorted(graph.get(node, ())):
                if successor == start and len(path) >= 3:
                    cycles.add(tuple(path))
                    if len(cycles) >= maximum:
                        return
                elif successor > start and successor not in visited:
                    visited.add(successor)
                    path.append(successor)
                    search(successor)
                    path.pop()
                    visited.remove(successor)
                    if len(cycles) >= maximum:
                        return

        search(start)
        if len(cycles) >= maximum:
            break
    return sorted(cycles, key=lambda cycle: (len(cycle), cycle))


def confirmed_cycles(
    matches: Iterable[Match], policy_ids: Iterable[str]
) -> list[tuple[str, ...]]:
    return find_directed_cycles(confirmed_win_graph(matches, policy_ids))
