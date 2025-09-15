import os
import random
from typing import Iterable

from boardrl.rl.model import load_model
from boardrl.utils.batchprocessor import BatchProcessor


def _recent_models(topk):
    import psutil

    process_start_time = psutil.Process().create_time()

    files_in_directory = []
    for root, _, files in os.walk("."):
        for f in files:
            if f.endswith(".pth"):
                files_in_directory.append(os.path.join(root, f))

    recent_files = [
        f for f in files_in_directory if os.path.getmtime(f) > process_start_time
    ]

    recent_files_with_times = [(f, os.path.getmtime(f)) for f in recent_files]
    recent_files_with_times.sort(key=lambda x: x[1], reverse=True)

    return [f[0] for f in recent_files_with_times[:topk]]


class ModelPool:
    """Utility to resolve model specifications to ``BatchProcessor`` instances."""

    def __init__(
        self,
        base_model: BatchProcessor,
        batch_size: int,
        timeout: float,
        reference_model: BatchProcessor | None = None,
    ):
        self.base_model = base_model
        self.reference_model = reference_model
        self.batch_size = batch_size
        self.timeout = timeout
        self.cache: dict[str, BatchProcessor] = {}

    def _load(self, path: str) -> BatchProcessor:
        model = load_model(path)
        model.eval()
        return BatchProcessor(self.batch_size, model, timeout=self.timeout)

    def _resolve_path(self, spec: str) -> str:
        if spec.startswith("recent-"):
            try:
                topk = int(spec.split("-", 1)[1])
            except ValueError as exc:  # pragma: no cover - defensive programming
                raise ValueError(f"invalid recent model spec: {spec}") from exc
            candidates = _recent_models(topk)
            if not candidates:
                raise ValueError("no recent model files found")
            return random.choice(candidates)
        return spec

    def __call__(self, spec: str | None):
        if spec in (None, "this"):
            if self.base_model is None:
                raise ValueError("model='this' requires a provided model")
            return self.base_model
        if spec == "reference":
            if self.reference_model is None:
                raise ValueError("model='reference' requires a provided model")
            return self.reference_model
        path = self._resolve_path(spec)
        if not os.path.exists(path):
            raise ValueError(f"model file '{path}' does not exist")
        if path not in self.cache:
            self.cache[path] = self._load(path)
        return self.cache[path]
