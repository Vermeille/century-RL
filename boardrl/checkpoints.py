"""Checkpoint lifecycle for one or many trainable models."""

from __future__ import annotations

from pathlib import Path
import random
import re

import torch


class Checkpoints:
    def __init__(self, directory, *, prefix="step", keep: int | None = None):
        self.directory = Path(directory)
        self.prefix = prefix
        self.keep = keep

    @property
    def paths(self) -> list[Path]:
        def key(path):
            match = re.search(r"(\d+)$", path.stem)
            return int(match.group(1)) if match else -1
        return sorted(self.directory.glob(f"{self.prefix}-*.pth"), key=key)

    @property
    def latest(self) -> Path | None:
        paths = self.paths
        return paths[-1] if paths else None

    def sample(self, *, exclude_latest=False, rng=random) -> Path:
        paths = self.paths[:-1] if exclude_latest else self.paths
        if not paths:
            raise LookupError("no checkpoints are available")
        return rng.choice(paths)

    def save(self, step: int, models, *, optimizers=None, metadata=None) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        optimizer_map = {} if optimizers is None else optimizers
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
            "metadata": dict(metadata or {}),
        }
        path = self.directory / f"{self.prefix}-{step}.pth"
        torch.save(payload, path)
        self._prune()
        return path

    def load(self, path=None, *, models=None, optimizers=None, map_location="cpu"):
        path = Path(path) if path is not None else self.latest
        if path is None:
            raise LookupError("no checkpoint is available")
        payload = torch.load(path, weights_only=False, map_location=map_location)
        for name, model in (models or {}).items():
            model.load_state_dict(payload["models"][name])
        for name, optimizer in (optimizers or {}).items():
            optimizer.load_state_dict(payload["optimizers"][name])
        return payload

    def _prune(self):
        if self.keep is None:
            return
        for path in self.paths[:-self.keep]:
            path.unlink()
