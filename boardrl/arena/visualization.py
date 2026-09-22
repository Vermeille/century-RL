"""Matplotlib figures published by the arena."""

from __future__ import annotations

import math
from typing import Iterable

from .ratings import OPPONENT_ID, RANDOM_ID, RatingFit, pair_aggregates
from .store import ArenaStore


def plot_rating_curve(
    store: ArenaStore,
    fit: RatingFit,
    *,
    opponent_name: str,
    include_checkpoints: bool = True,
):
    import matplotlib.pyplot as plt

    policies = (
        store.policies(checkpoints_only=True) if include_checkpoints else []
    )
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    if policies:
        steps = [policy.step for policy in policies]
        ratings = [fit.ratings[policy.id] for policy in policies]
        deviations = [fit.deviations[policy.id] for policy in policies]
        lower = [
            rating - 1.96 * deviation
            for rating, deviation in zip(ratings, deviations)
        ]
        upper = [
            rating + 1.96 * deviation
            for rating, deviation in zip(ratings, deviations)
        ]
        axis.fill_between(steps, lower, upper, alpha=0.2, label="95% uncertainty")
        axis.plot(steps, ratings, marker="o", linewidth=1.5, label="agent")
    axis.axhline(0.0, color="black", linestyle="--", label="random = 0")
    opponent_rating = fit.ratings.get(OPPONENT_ID, 0.0)
    axis.axhline(
        opponent_rating,
        color="tab:orange",
        linestyle=":",
        label=f"evaluation opponent ({opponent_name}) = {opponent_rating:.1f}",
    )
    axis.set_title("Bayesian arena rating")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Skill (100 points = 2× expected-score odds)")
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    return figure


def _wilson_excludes_half(score: float, games: int, *, z: float = 1.96) -> bool:
    """Whether a 95% Wilson interval for the score rate excludes 50%."""

    if games <= 0:
        return False
    probability = score / games
    denominator = 1.0 + z**2 / games
    center = probability + z**2 / (2.0 * games)
    margin = z * math.sqrt(
        probability * (1.0 - probability) / games
        + z**2 / (4.0 * games**2)
    )
    lower = (center - margin) / denominator
    upper = (center + margin) / denominator
    return lower > 0.5 or upper < 0.5


def plot_win_rate_matrix(
    store: ArenaStore,
    fit: RatingFit,
    *,
    reference_id: str,
):
    """Observed head-to-head win rates with BT completion for unplayed pairs."""

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Circle, Rectangle

    policy_ids = [
        RANDOM_ID,
        *([OPPONENT_ID] if reference_id == OPPONENT_ID else []),
        *(
            policy.id
            for policy in store.policies(
                checkpoints_only=True, playable_only=True
            )
        ),
    ]
    aggregates = pair_aggregates(store.matches())
    size = len(policy_ids)

    cell_inches = 0.34
    margin_inches = 2.3
    side = max(4.0, min(13.0, size * cell_inches + margin_inches))
    figure, axis = plt.subplots(figsize=(side, side), constrained_layout=True)

    cmap = LinearSegmentedColormap.from_list(
        "arena_win_rate",
        ["#3b82f6", "#d1d5db", "#ef4444"],
    )
    norm = Normalize(vmin=0.0, vmax=1.0)

    axis.set_xlim(0, size)
    axis.set_ylim(size, 0)
    axis.set_aspect("equal")

    for row, first in enumerate(policy_ids):
        for column, second in enumerate(policy_ids):
            if row == column:
                axis.add_patch(
                    Rectangle(
                        (column, row),
                        1,
                        1,
                        facecolor="#f3f4f6",
                        edgecolor="#e5e7eb",
                        linewidth=0.35,
                    )
                )
                continue

            pair = tuple(sorted((first, second)))
            match = aggregates.get(pair)
            if match is not None and match.games > 0:
                first_score = (
                    match.score
                    if match.first == first
                    else match.games - match.score
                )
                win_rate = first_score / match.games
                confident = _wilson_excludes_half(first_score, match.games)
                axis.add_patch(
                    Rectangle(
                        (column, row),
                        1,
                        1,
                        facecolor=cmap(norm(win_rate)),
                        edgecolor="black" if confident else "#ffffff",
                        linewidth=1.15 if confident else 0.35,
                    )
                )
                continue

            predicted = fit.expected_score(first, second)
            axis.add_patch(
                Circle(
                    (column + 0.5, row + 0.5),
                    radius=0.19,
                    facecolor=cmap(norm(predicted)),
                    edgecolor="#4b5563",
                    linewidth=0.45,
                )
            )

    labels = [
        "random"
        if policy_id == RANDOM_ID
        else "opponent"
        if policy_id == OPPONENT_ID
        else policy_id.replace("checkpoint-", "s")
        for policy_id in policy_ids
    ]
    ticks = np.arange(size) + 0.5
    axis.set_xticks(ticks, labels=labels, rotation=90)
    axis.set_yticks(ticks, labels=labels)
    axis.tick_params(
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        length=0,
        labelsize=7,
    )
    for spine in axis.spines.values():
        spine.set_visible(False)

    axis.set_title(
        "Head-to-head win rates\n"
        "square = observed · dot = inferred · black border = >95% confidence",
        fontsize=10,
    )
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = figure.colorbar(
        scalar,
        ax=axis,
        fraction=0.035,
        pad=0.025,
    )
    colorbar.set_label("Row policy win rate", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    colorbar.set_ticks([0.0, 0.5, 1.0], labels=["0%", "50%", "100%"])
    return figure


def _policy_label(policy_id: str) -> str:
    if policy_id == RANDOM_ID:
        return "random"
    if policy_id == OPPONENT_ID:
        return "opponent"
    return policy_id.replace("checkpoint-", "s")


def _strongly_connected_components(
    graph: dict[str, set[str]],
) -> list[tuple[str, ...]]:
    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(node: str):
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for successor in sorted(graph.get(node, ())):
            if successor not in indices:
                visit(successor)
                lowlinks[node] = min(lowlinks[node], lowlinks[successor])
            elif successor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[successor])

        if lowlinks[node] == indices[node]:
            component = []
            while True:
                popped = stack.pop()
                on_stack.remove(popped)
                component.append(popped)
                if popped == node:
                    break
            components.append(tuple(sorted(component)))

    for node in sorted(graph):
        if node not in indices:
            visit(node)
    return components


def _edge_without_path(
    graph: dict[int, set[int]],
    start: int,
    target: int,
) -> bool:
    stack = [start]
    visited = {start}
    while stack:
        node = stack.pop()
        for successor in graph.get(node, ()):
            if node == start and successor == target:
                continue
            if successor == target:
                return False
            if successor not in visited:
                visited.add(successor)
                stack.append(successor)
    return True


def _transitive_reduction_dag(graph: dict[int, set[int]]) -> dict[int, set[int]]:
    reduced = {node: set() for node in graph}
    for node, successors in graph.items():
        for successor in successors:
            if _edge_without_path(graph, node, successor):
                reduced[node].add(successor)
    return reduced


def _find_any_cycle(
    graph: dict[str, set[str]],
    component: set[str],
) -> tuple[str, ...] | None:
    for start in sorted(component):
        path = [start]
        visited = {start}

        def dfs(node: str):
            for successor in sorted(graph.get(node, ())):
                if successor not in component:
                    continue
                if successor == start and len(path) >= 2:
                    return tuple(path)
                if successor not in visited:
                    visited.add(successor)
                    path.append(successor)
                    cycle = dfs(successor)
                    if cycle is not None:
                        return cycle
                    path.pop()
                    visited.remove(successor)
            return None

        cycle = dfs(start)
        if cycle is not None:
            return cycle
    return None


def _choose_witness_cycle(
    component: set[str],
    graph: dict[str, set[str]],
    cycles: Iterable[tuple[str, ...]],
) -> tuple[str, ...] | None:
    candidates = [
        cycle
        for cycle in cycles
        if cycle and set(cycle).issubset(component)
    ]
    if candidates:
        return min(candidates, key=lambda cycle: (len(cycle), cycle))
    return _find_any_cycle(graph, component)


def plot_dominance_graph(
    store: ArenaStore,
    fit: RatingFit,
    graph: dict[str, set[str]],
    cycles: Iterable[tuple[str, ...]],
    *,
    reference_id: str,
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    policies = store.policies(checkpoints_only=True, playable_only=True)
    if not policies:
        figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
        axis.text(
            0.5,
            0.5,
            "No playable checkpoints in the arena pool yet",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_title("Dominance structure")
        axis.axis("off")
        return figure

    policy_ids = [policy.id for policy in policies]
    policy_set = set(policy_ids)
    graph = {
        policy_id: {
            other for other in graph.get(policy_id, set()) if other in policy_set
        }
        for policy_id in policy_ids
    }

    steps = {policy.id: float(policy.step or 0) for policy in policies}
    ratings = {policy_id: fit.ratings.get(policy_id, 0.0) for policy_id in policy_ids}
    deviations = {
        policy_id: fit.deviations.get(policy_id, 0.0) for policy_id in policy_ids
    }

    step_values = list(steps.values())
    rating_values = list(ratings.values())
    if RANDOM_ID in fit.ratings:
        rating_values.append(fit.ratings[RANDOM_ID])
    if reference_id == OPPONENT_ID and OPPONENT_ID in fit.ratings:
        rating_values.append(fit.ratings[OPPONENT_ID])

    min_step = min(step_values)
    max_step = max(step_values)
    min_rating = min(rating_values)
    max_rating = max(rating_values)
    step_span = max(1.0, max_step - min_step)
    rating_span = max(1.0, max_rating - min_rating)
    x_pad = max(1.0, step_span * 0.06)
    y_pad = max(15.0, rating_span * 0.12)

    figure, axis = plt.subplots(figsize=(10.5, 6.6), constrained_layout=True)

    def draw_reference_line(policy_id: str, *, linestyle: str, color: str, text: str):
        if policy_id not in fit.ratings:
            return
        rating = fit.ratings[policy_id]
        axis.axhline(
            rating,
            color=color,
            linestyle=linestyle,
            linewidth=1.1,
            zorder=0,
        )
        axis.text(
            max_step + x_pad * 0.6,
            rating,
            text,
            color=color,
            fontsize=8,
            va="bottom",
            ha="left",
        )

    draw_reference_line(RANDOM_ID, linestyle="--", color="#64748b", text="random")
    if reference_id == OPPONENT_ID:
        draw_reference_line(
            OPPONENT_ID,
            linestyle=":",
            color="#d97706",
            text="opponent",
        )

    components = _strongly_connected_components(graph)
    component_index = {
        node: index
        for index, component in enumerate(components)
        for node in component
    }

    condensed = {index: set() for index in range(len(components))}
    for winner, losers in graph.items():
        for loser in losers:
            source = component_index[winner]
            target = component_index[loser]
            if source != target:
                condensed[source].add(target)

    reduced = _transitive_reduction_dag(condensed)
    positions = {
        policy_id: (steps[policy_id], ratings[policy_id]) for policy_id in policy_ids
    }

    cycle_edges: set[tuple[str, str]] = set()
    nontrivial_components = [
        set(component) for component in components if len(component) > 1
    ]

    for component in nontrivial_components:
        xs = [positions[node][0] for node in component]
        ys = [positions[node][1] for node in component]
        rect = Rectangle(
            (min(xs) - x_pad * 0.45, min(ys) - y_pad * 0.45),
            max(x_pad * 0.9, max(xs) - min(xs) + x_pad * 0.9),
            max(y_pad * 0.9, max(ys) - min(ys) + y_pad * 0.9),
            facecolor="#f97316",
            edgecolor="#ea580c",
            linewidth=1.2,
            alpha=0.08,
            zorder=0,
        )
        axis.add_patch(rect)
        witness = _choose_witness_cycle(component, graph, cycles)
        if witness is None:
            continue
        for index, winner in enumerate(witness):
            loser = witness[(index + 1) % len(witness)]
            cycle_edges.add((winner, loser))

    def draw_arrow(
        start_id: str,
        end_id: str,
        *,
        color: str,
        linewidth: float,
        alpha: float,
        curvature: float,
        zorder: int,
    ):
        start = positions[start_id]
        end = positions[end_id]
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=linewidth,
            color=color,
            alpha=alpha,
            connectionstyle=f"arc3,rad={curvature}",
            shrinkA=10,
            shrinkB=10,
            zorder=zorder,
        )
        axis.add_patch(arrow)

    for source_component, target_components in reduced.items():
        for target_component in target_components:
            candidates = [
                (winner, loser)
                for winner in components[source_component]
                for loser in components[target_component]
                if loser in graph.get(winner, ())
            ]
            if not candidates:
                continue
            winner, loser = min(
                candidates,
                key=lambda edge: (
                    abs(positions[edge[0]][0] - positions[edge[1]][0]),
                    abs(positions[edge[0]][1] - positions[edge[1]][1]),
                    edge,
                ),
            )
            draw_arrow(
                winner,
                loser,
                color="#94a3b8",
                linewidth=1.1,
                alpha=0.8,
                curvature=0.03,
                zorder=1,
            )

    for winner, loser in sorted(cycle_edges):
        draw_arrow(
            winner,
            loser,
            color="#c2410c",
            linewidth=2.0,
            alpha=0.95,
            curvature=0.12,
            zorder=2,
        )

    for policy_id in policy_ids:
        x, y = positions[policy_id]
        radius = 1.96 * deviations[policy_id]
        axis.vlines(
            x,
            y - radius,
            y + radius,
            color="#cbd5e1",
            linewidth=1.0,
            zorder=2,
        )

    latest_id = max(policy_ids, key=lambda policy_id: (steps[policy_id], policy_id))
    best_id = max(policy_ids, key=lambda policy_id: (ratings[policy_id], policy_id))
    cycle_nodes = {node for edge in cycle_edges for node in edge}

    for policy_id in policy_ids:
        x, y = positions[policy_id]
        facecolor = "#ef4444" if policy_id in cycle_nodes else "#2563eb"
        axis.scatter(
            [x],
            [y],
            s=40,
            color=facecolor,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axis.annotate(
            _policy_label(policy_id),
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="#0f172a",
            zorder=4,
        )

    latest_x, latest_y = positions[latest_id]
    axis.scatter(
        [latest_x],
        [latest_y],
        s=115,
        facecolors="none",
        edgecolors="#111827",
        linewidth=1.3,
        zorder=5,
    )

    best_x, best_y = positions[best_id]
    axis.scatter(
        [best_x],
        [best_y],
        s=135,
        marker="*",
        color="#f59e0b",
        edgecolor="#78350f",
        linewidth=0.8,
        zorder=6,
    )

    axis.set_xlim(min_step - x_pad, max_step + x_pad * 1.5)
    axis.set_ylim(min_rating - y_pad, max_rating + y_pad)
    axis.set_xlabel("Training step")
    axis.set_ylabel("Bayesian rating")
    axis.set_title(
        "Dominance structure over training\n"
        "gray arrows = reduced confirmed dominance · orange boxes/arrows = non-transitive SCC witnesses",
        fontsize=10,
    )
    axis.grid(alpha=0.18)
    return figure
