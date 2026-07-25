"""Python model presets and model-copy utilities."""

from __future__ import annotations

import copy
from collections.abc import Callable

import torch

from boardrl.rl.model import Model


architectures: dict[str, Callable[..., Model]] = {}


def architecture(name):
    def register(factory):
        architectures[name] = factory
        return factory

    return register


def make(name, **overrides):
    return architectures[name](**overrides)


@architecture("toy")
def toy(**overrides):
    return Model(**({"dim": 16, "num_layers": 1, "head_size": 2} | overrides))


@architecture("cnn")
def cnn(**overrides):
    return Model(**({"dim": 64, "num_layers": 5, "backbone": "cnn"} | overrides))


@architecture("shared-patch")
def shared_patch(**overrides):
    return shared_patch_small(**overrides)


def shared_patch_model(scale, **overrides):
    spec = {
        "backbone": "patch_transformer_cnn",
        "backbone_kwargs": {"patch_size": 4},
    } | scale
    return Model(**(spec | overrides))


@architecture("shared-patch-tiny")
def shared_patch_tiny(**overrides):
    scale = {"dim": 32, "num_layers": 2, "num_heads": 4, "head_size": 8}
    return shared_patch_model(scale, **overrides)


@architecture("shared-patch-small")
def shared_patch_small(**overrides):
    scale = {"dim": 64, "num_layers": 4, "num_heads": 4, "head_size": 16}
    return shared_patch_model(scale, **overrides)


@architecture("shared-patch-medium")
def shared_patch_medium(**overrides):
    scale = {"dim": 128, "num_layers": 4, "num_heads": 8, "head_size": 16}
    return shared_patch_model(scale, **overrides)


@architecture("shared-patch-large")
def shared_patch_large(**overrides):
    scale = {"dim": 256, "num_layers": 6, "num_heads": 8, "head_size": 32}
    return shared_patch_model(scale, **overrides)


@architecture("cnn-large")
def cnn_large(**overrides):
    return Model(**({"dim": 64, "num_layers": 8, "backbone": "cnn"} | overrides))


@architecture("small")
def small(**overrides):
    return Model(
        **({"dim": 128, "num_layers": 8, "num_heads": 16, "head_size": 8} | overrides)
    )


@architecture("extra-small")
def extra_small(**overrides):
    spec = {"dim": 64, "num_layers": 8, "num_heads": 16, "head_size": 4}
    return Model(**(spec | overrides))


@architecture("medium")
def medium(**overrides):
    return Model(
        **({"dim": 256, "num_layers": 8, "num_heads": 16, "head_size": 16} | overrides)
    )


@architecture("large")
def large(**overrides):
    return Model(
        **({"dim": 512, "num_layers": 8, "num_heads": 16, "head_size": 32} | overrides)
    )


@architecture("extra-large")
def extra_large(**overrides):
    spec = {"dim": 128, "num_layers": 2, "num_heads": 16, "head_size": 48}
    return Model(**(spec | overrides))


@architecture("minimal-lstm")
def minimal_lstm(**overrides):
    return Model(**({"dim": 32, "num_layers": 1, "backbone": "lstm"} | overrides))


@architecture("minimal-gated-cnn")
def minimal_gated_cnn(**overrides):
    spec = {"dim": 128, "num_layers": 8, "backbone": "gated_cnn"}
    return Model(**(spec | overrides))


@torch.no_grad()
def copy_weights(target, source) -> None:
    target.load_state_dict(source.state_dict())


class ExponentialMovingAverage:
    """A model copy updated with exponential moving averages."""

    def __init__(self, model, decay: float):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, source) -> None:
        for average, current in zip(
            self.model.state_dict().values(), source.state_dict().values()
        ):
            average.lerp_(current, 1 - self.decay)
        self.model.eval()
        self.updates += 1
