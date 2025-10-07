import os
import random
from collections import OrderedDict

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
        *,
        max_cache_size: int | None = 16,
    ):
        self.base_model = base_model
        self.reference_model = reference_model
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_cache_size = max_cache_size
        self.cache: OrderedDict[str, BatchProcessor] = OrderedDict()
        self._pinned_keys: set[str] = set()

        if self.base_model is not None:
            self.cache["this"] = self.base_model
            self._pinned_keys.add("this")
        if self.reference_model is not None:
            self.cache["reference"] = self.reference_model
            self._pinned_keys.add("reference")

        if (
            self.max_cache_size is not None
            and self.max_cache_size < len(self._pinned_keys)
        ):
            raise ValueError(
                "max_cache_size must be at least the number of persistent models"
            )

    def _evict(self) -> None:
        if self.max_cache_size is None:
            return
        while len(self.cache) > self.max_cache_size:
            oldest_key = next(iter(self.cache))
            if oldest_key in self._pinned_keys:
                # Keep persistent models in the cache by treating them as recently used.
                self.cache.move_to_end(oldest_key)
                continue
            self.cache.pop(oldest_key)

    def _load(self, path: str) -> BatchProcessor:
        model = load_model(path)
        model.eval()
        return BatchProcessor(
            self.batch_size, model, timeout=self.timeout, model_name=path
        )

    def __call__(self, spec: str | None):
        if spec in (None, "this"):
            if "this" not in self.cache:
                raise ValueError("model='this' requires a provided model")
            self.cache.move_to_end("this")
            return self.cache["this"]
        if spec == "reference":
            if "reference" not in self.cache:
                raise ValueError("model='reference' requires a provided model")
            self.cache.move_to_end("reference")
            return self.cache["reference"]
        if not os.path.exists(spec):
            raise ValueError(f"model file '{spec}' does not exist")
        if spec not in self.cache:
            self.cache[spec] = self._load(spec)
        self.cache.move_to_end(spec)
        self._evict()
        return self.cache[spec]
