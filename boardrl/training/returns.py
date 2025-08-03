"""Utilities for computing rewards, returns and scores for game logs."""

from __future__ import annotations

from functools import partial
from typing import Iterable

import torch


def discount(rews: Iterable, discount_factor: float) -> float:
    """Return the discounted sum of rewards."""
    return sum(discount_factor**i * r.reward for i, r in enumerate(rews))


def rescale(history: list, scale: float) -> None:
    """Scale ``current_diff_points`` of each log by ``scale``."""

    for log in history:
        log.current_diff_points *= scale


def set_next(history: list) -> None:
    """Link each log in ``history`` to its successor via ``next`` attribute."""

    for i, log in enumerate(history[:-1]):
        log.next = history[i + 1]


def set_rewards(history: list) -> None:
    """Populate ``reward`` fields from ``current_diff_points`` differences."""

    history[-1].reward = 0
    for i in range(len(history) - 1):
        history[i].reward = (
            history[i + 1].current_diff_points - history[i].current_diff_points
        )


def entropy_reward(history: list, strength: float) -> None:
    """Add an entropy bonus to each step in ``history``.

    The bonus encourages exploration by penalising low-entropy action
    distributions.
    """

    for log in history[:-1]:
        log.reward += (
            strength * -torch.log_softmax(log.action_distribution, dim=0)[log.action_idx]
        )


def set_returns(history: list, discount_factor: float) -> None:
    """Compute discounted returns for each log.

    If the history is non-terminal (the last log has ``final`` False) the
    ``returns`` field of each step is set to ``NaN``.
    """

    if history[-1].final:
        for i in range(len(history) - 1):
            history[i].returns = discount(history[i:], discount_factor)
    else:
        for i in range(len(history) - 1):
            history[i].returns = float("nan")


def set_score(history: list) -> None:
    """Attach the final score to each log if the game terminated."""

    if history[-1].final:
        for log in history:
            log.score = history[-1].current_diff_points
    else:
        for log in history:
            log.score = float("nan")


def compute_returns(
    games,
    discount_factor: float,
    *,
    entropy_reward_scale: float | None = None,
    reward_rescale: float | None = None,
) -> None:
    """Compute rewards, returns and scores for a batch of games.

    The function mutates the histories in ``games`` in-place. Steps are applied
    declaratively via a list of helper functions that operate on each history.
    """

    for game in games:
        for history in game:
            if len(history) == 0:
                continue

            fns = []
            if reward_rescale is not None:
                fns.append(partial(rescale, scale=reward_rescale))
            fns.extend([set_next, set_rewards])
            if entropy_reward_scale is not None:
                fns.append(partial(entropy_reward, strength=entropy_reward_scale))
            fns.append(partial(set_returns, discount_factor=discount_factor))
            fns.append(set_score)

            for fn in fns:
                fn(history)

