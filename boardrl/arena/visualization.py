"""Matplotlib figures published by the arena."""

from __future__ import annotations

import math
from typing import Iterable

from .ratings import OPPONENT_ID, RANDOM_ID, RatingFit, residuals
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


def plot_residuals(
    store: ArenaStore,
    fit: RatingFit,
    *,
    reference_id: str,
):
    import matplotlib.pyplot as plt
    import numpy as np

    policies = [
        RANDOM_ID,
        *([OPPONENT_ID] if reference_id == OPPONENT_ID else []),
        *(
            policy.id
            for policy in store.policies(
                checkpoints_only=True, playable_only=True
            )
        ),
    ]
    values = residuals(store.matches(), fit)
    matrix = np.full((len(policies), len(policies)), np.nan)
    for row, first in enumerate(policies):
        matrix[row, row] = 0.0
        for column, second in enumerate(policies):
            pair = tuple(sorted((first, second)))
            if pair in values:
                value = values[pair]
                matrix[row, column] = value if first == pair[0] else -value
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-3, vmax=3)
    labels = [
        "random"
        if policy == RANDOM_ID
        else "opponent"
        if policy == OPPONENT_ID
        else policy.replace("checkpoint-", "s")
        for policy in policies
    ]
    axis.set_xticks(range(len(labels)), labels=labels, rotation=90)
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_title("Observed − transitive-model matchup residual (z)")
    figure.colorbar(image, ax=axis, label="standardized residual")
    return figure


def plot_dominance_graph(
    store: ArenaStore,
    graph: dict[str, set[str]],
    cycles: Iterable[tuple[str, ...]],
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    connected = {
        policy_id
        for winner, losers in graph.items()
        for policy_id in (winner, *losers)
        if losers or any(winner in others for others in graph.values())
    }
    figure, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    if not connected:
        axis.text(
            0.5,
            0.5,
            "No statistically confirmed head-to-head edges yet",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_title("Confirmed dominance graph")
        axis.axis("off")
        return figure

    nodes = sorted(
        connected,
        key=lambda policy_id: (store.policy(policy_id).step or -1, policy_id),
    )
    positions = {
        policy_id: (
            math.cos(2.0 * math.pi * index / len(nodes)),
            math.sin(2.0 * math.pi * index / len(nodes)),
        )
        for index, policy_id in enumerate(nodes)
    }
    cycle_edges = {
        (cycle[index], cycle[(index + 1) % len(cycle)])
        for cycle in cycles
        for index in range(len(cycle))
        if cycle[index] in positions
        and cycle[(index + 1) % len(cycle)] in positions
    }
    cycle_nodes = {policy_id for edge in cycle_edges for policy_id in edge}

    for winner in nodes:
        for loser in sorted(graph[winner]):
            if loser not in positions:
                continue
            edge = (winner, loser)
            color = "#c2410c" if edge in cycle_edges else "#94a3b8"
            arrow = FancyArrowPatch(
                positions[winner],
                positions[loser],
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=2.0 if edge in cycle_edges else 1.0,
                color=color,
                alpha=0.9 if edge in cycle_edges else 0.65,
                connectionstyle="arc3,rad=0.08",
                shrinkA=18,
                shrinkB=18,
            )
            axis.add_patch(arrow)

    for policy_id in nodes:
        x, y = positions[policy_id]
        if policy_id == RANDOM_ID:
            color, label = "#475569", "random"
        elif policy_id == OPPONENT_ID:
            color, label = "#d97706", "opponent"
        else:
            color = "#ef4444" if policy_id in cycle_nodes else "#3b82f6"
            label = policy_id.replace("checkpoint-", "s")
        axis.scatter([x], [y], s=420, color=color, edgecolor="white", zorder=3)
        axis.text(x, y, label, ha="center", va="center", color="white", zorder=4)

    axis.set_title(
        "Confirmed dominance graph\n"
        "arrow: winner → loser; orange/red edges participate in a cycle"
    )
    axis.set_xlim(-1.35, 1.35)
    axis.set_ylim(-1.35, 1.35)
    axis.set_aspect("equal")
    axis.axis("off")
    return figure
