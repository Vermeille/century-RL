"""Run metadata parsing, checkpoint capture, and bounded pool selection."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import torch

from boardrl.checkpoints import atomic_torch_save

from .ratings import RatingFit
from .store import Policy


def read_run_arguments(path: Path) -> dict[str, object]:
    text = path.read_text()
    marker = "Arguments\n=========\n"
    if marker not in text:
        raise ValueError(f"{path} does not contain RunInfo arguments")
    arguments, _ = json.JSONDecoder().raw_decode(text.split(marker, 1)[1])
    return arguments


def capture_agent_checkpoint(
    source: Path,
    destination: Path,
    *,
    expected_game: str,
) -> tuple[str, int]:
    payload = torch.load(source, weights_only=False, map_location="cpu")
    metadata = payload.get("metadata", {})
    if metadata.get("trainer") != "adversarial-advshape":
        raise ValueError(f"{source} is not an adversarial-advshape checkpoint")
    if metadata.get("game") != expected_game:
        raise ValueError(
            f"{source} game {metadata.get('game')!r} does not match {expected_game!r}"
        )
    if "agent" not in payload.get("models", {}):
        raise KeyError(f"{source} does not contain models['agent']")
    step = int(payload["step"])
    compact = {
        "step": step,
        "models": {"agent": payload["models"]["agent"]},
        "model_specs": {"agent": payload["model_specs"]["agent"]},
        "optimizers": {},
        "states": {},
        "metadata": {
            "trainer": "arena",
            "source_trainer": "adversarial-advshape",
            "game": expected_game,
            "source": str(source),
        },
    }
    atomic_torch_save(compact, destination)
    return f"checkpoint-{step}", step


def choose_retained_pool(
    policies: Iterable[Policy],
    fit: RatingFit,
    pair_residuals: dict[tuple[str, str], float],
    *,
    limit: int,
) -> set[str]:
    playable = [policy for policy in policies if policy.path is not None]
    if len(playable) <= limit:
        return {policy.id for policy in playable}

    protected = {policy.id for policy in playable if policy.placement_batches < 3}
    if len(protected) >= limit:
        return {
            policy.id
            for policy in sorted(
                (policy for policy in playable if policy.id in protected),
                key=lambda policy: (policy.step or 0, policy.id),
            )[:limit]
        }
    selected = set(protected)
    quota = min(8, max(limit // 4, 1))

    def add_ranked(items, key, count):
        added = 0
        for policy in sorted(items, key=key, reverse=True):
            if len(selected) >= limit:
                return
            if policy.id not in selected:
                selected.add(policy.id)
                added += 1
                if added >= count:
                    return

    add_ranked(playable, lambda policy: (policy.step or -1, policy.id), quota)
    add_ranked(
        playable,
        lambda policy: (fit.deviations.get(policy.id, float("inf")), policy.id),
        quota,
    )

    per_policy_residual = {policy.id: 0.0 for policy in playable}
    for pair, value in pair_residuals.items():
        for policy_id in pair:
            if policy_id in per_policy_residual:
                per_policy_residual[policy_id] = max(
                    per_policy_residual[policy_id], abs(value)
                )
    add_ranked(
        playable,
        key=lambda policy: (per_policy_residual[policy.id], policy.id),
        count=quota,
    )

    remaining = [policy for policy in playable if policy.id not in selected]
    steps = [policy.step or 0 for policy in playable]
    ratings = [fit.ratings.get(policy.id, 0.0) for policy in playable]
    step_span = max(max(steps) - min(steps), 1)
    rating_span = max(max(ratings) - min(ratings), 1.0)

    def point(policy):
        return (
            ((policy.step or 0) - min(steps)) / step_span,
            (fit.ratings.get(policy.id, 0.0) - min(ratings)) / rating_span,
        )

    while remaining and len(selected) < limit:
        selected_points = [point(policy) for policy in playable if policy.id in selected]

        def distance(policy):
            x, y = point(policy)
            if not selected_points:
                return float("inf")
            return min(math.hypot(x - sx, y - sy) for sx, sy in selected_points)

        chosen = max(remaining, key=lambda policy: (distance(policy), policy.id))
        selected.add(chosen.id)
        remaining.remove(chosen)
    return selected
