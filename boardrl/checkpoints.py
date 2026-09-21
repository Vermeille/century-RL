"""Checkpoint lifecycle for one or many trainable models."""

from __future__ import annotations

import os
import random
import re
import uuid
from pathlib import Path

import torch


def atomic_torch_save(payload, path: str | Path) -> Path:
    """Publish a torch payload without exposing a partially written file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


class Checkpoints:
    def __init__(self, directory, *, prefix="step", keep: int | None = None):
        self.directory = Path(directory)
        self.prefix = prefix
        self.keep = keep

    @property
    def paths(self) -> list[Path]:
        return sorted(
            self.directory.glob(f"{self.prefix}-*.pth"),
            key=self._step,
        )

    @staticmethod
    def _step(path: Path) -> int:
        match = re.search(r"(\d+)$", path.stem)
        return int(match.group(1)) if match else -1

    @property
    def latest(self) -> Path | None:
        paths = self.paths
        return paths[-1] if paths else None

    def sample(self, *, exclude_latest=False, rng=random) -> Path:
        paths = self.paths[:-1] if exclude_latest else self.paths
        if not paths:
            raise LookupError("no checkpoints are available")
        return rng.choice(paths)

    def save(
        self,
        step: int,
        models,
        *,
        optimizers=None,
        states=None,
        metadata=None,
    ) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        optimizer_map = {} if optimizers is None else optimizers
        state_map = {} if states is None else states
        payload = {
            "step": step,
            "models": {name: model.state_dict() for name, model in models.items()},
            "model_specs": {
                name: model.spec() for name, model in models.items()
            },
            "optimizers": {
                name: optimizer.state_dict()
                for name, optimizer in optimizer_map.items()
            },
            "states": {
                name: stateful.state_dict()
                for name, stateful in state_map.items()
            },
            "metadata": dict(metadata or {}),
        }
        path = self.directory / f"{self.prefix}-{step}.pth"
        atomic_torch_save(payload, path)
        self._prune(step)
        return path

    def load(
        self,
        path=None,
        *,
        models=None,
        optimizers=None,
        states=None,
        map_location="cpu",
    ):
        path = Path(path) if path is not None else self.latest
        if path is None:
            raise LookupError("no checkpoint is available")
        payload = torch.load(path, weights_only=False, map_location=map_location)
        for name, model in (models or {}).items():
            model.load_state_dict(payload["models"][name])
        for name, optimizer in (optimizers or {}).items():
            optimizer.load_state_dict(payload["optimizers"][name])
        saved_states = payload.get("states", {})
        for name, stateful in (states or {}).items():
            if name not in saved_states:
                raise KeyError(
                    f"checkpoint {path} does not contain required state {name!r}"
                )
            stateful.load_state_dict(saved_states[name])
        return payload

    def _prune(self, current_step: int):
        if self.keep is None:
            return

        # A checkpoint directory can be reused after an interrupted run. In
        # that case it may contain checkpoints from a later iteration of the
        # previous run. They must not win retention simply because their step
        # number is larger than the current run's step. Treat checkpoints
        # beyond the current iteration as stale and remove them first.
        paths = self.paths
        for path in paths:
            if self._step(path) > current_step:
                path.unlink()

        current_paths = [path for path in paths if self._step(path) <= current_step]
        for path in current_paths[:-self.keep]:
            path.unlink()
