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
