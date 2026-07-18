import os
import sys
import math
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pydantic import ValidationError

from boardrl.config import Config
from boardrl.rl.model.model import Model


def test_config_defaults():
    cfg = Config.from_dict({})
    assert cfg.device == "cpu"
    assert cfg.seed is None
    assert cfg.train.optimizer == "AdamW"
    assert cfg.train.gradient_clip_norm is None
    assert cfg.train.lr == 1e-4
    assert cfg.train.betas == (0.9, 0.999)
    assert cfg.train.weight_decay == 0.01
    assert cfg.train.reference_model_update == "False"
    assert cfg.train.losses == []
    assert cfg.self_play.num_games == 4
    assert cfg.self_play.max_len == 500
    assert cfg.pit.every == 4
    assert cfg.pit.num_games == 32
    assert cfg.pit.max_len == 220
    assert math.isinf(cfg.train.iterations)
    assert cfg.net.backbone == "transformer"
    assert cfg.net.shared_backbone is True


def test_config_overrides():
    raw = {
        "device": "cuda",
        "seed": 123,
        "net": {"backbone": "lstm"},
        "train": {
            "lr": 0.5,
            "gradient_clip_norm": 12.5,
            "betas": [0.8, 0.9],
            "weight_decay": 0.1,
            "iterations": 5,
            "losses": ["custom_value", "custom_policy"],
            "reference_model_update": "epoch % 2 == 0",
        },
        "self_play": {"num_games": 7, "strategies": ["random", "best"]},
        "pit": {"every": 1},
    }
    cfg = Config.from_dict(raw)
    assert cfg.device == "cuda"
    assert cfg.seed == 123
    assert cfg.train.lr == 0.5
    assert cfg.train.gradient_clip_norm == 12.5
    assert cfg.train.betas == (0.8, 0.9)
    assert cfg.train.weight_decay == 0.1
    assert cfg.train.iterations == 5
    assert cfg.train.losses == ["custom_value", "custom_policy"]
    assert cfg.train.reference_model_update == "epoch % 2 == 0"
    assert cfg.self_play.num_games == 7
    assert cfg.self_play.strategies == ["random", "best"]
    assert cfg.pit.every == 1
    # Ensure unspecified pit fields keep defaults
    assert cfg.pit.num_games == 32
    assert cfg.pit.max_len == 220
    assert cfg.net.backbone == "lstm"


def test_config_invalid_top_level_key():
    with pytest.raises(ValidationError):
        Config.from_dict({"bogus": 1})


def test_config_invalid_nested_key():
    with pytest.raises(ValidationError):
        Config.from_dict({"train": {"unknown": 1}})


def test_netconfig_head_params():
    cfg = Config.from_dict({"net": {"dim": 60, "num_layers": 1, "num_heads": 3}})
    assert cfg.net.num_heads == 3
    assert cfg.net.head_size is None
    model = Model(**cfg.net.model_dump())
    assert model.backbone.num_heads == 3
    assert model.backbone.head_size == 20

    cfg2 = Config.from_dict({"net": {"dim": 60, "num_layers": 1, "head_size": 10}})
    assert cfg2.net.head_size == 10
    assert cfg2.net.num_heads is None
    model2 = Model(**cfg2.net.model_dump())
    assert model2.backbone.head_size == 10
    assert model2.backbone.num_heads == 6

    cfg3 = Config.from_dict(
        {"net": {"dim": 60, "num_layers": 1, "head_size": 10, "num_heads": 5}}
    )
    model3 = Model(**cfg3.net.model_dump())
    assert model3.backbone.head_size == 10
    assert model3.backbone.num_heads == 5


def test_dual_backbone_config_and_state_mapping():
    shared = Model(dim=16, num_layers=1, backbone="cnn", shared_backbone=True)
    dual = Model(dim=16, num_layers=1, backbone="cnn", shared_backbone=False)

    assert hasattr(shared, "backbone")
    assert not hasattr(dual, "backbone")
    assert hasattr(dual, "policy_backbone")
    assert hasattr(dual, "value_backbone")

    shared_state = shared.state_dict()
    dual_state = dual.state_dict()
    mapped_to_dual = dual._normalize_backbone_state_dict(shared_state)
    mapped_to_shared = shared._normalize_backbone_state_dict(dual_state)

    assert any(k.startswith("policy_backbone.") for k in mapped_to_dual)
    assert any(k.startswith("value_backbone.") for k in mapped_to_dual)
    assert any(k.startswith("backbone.") for k in mapped_to_shared)
    assert not any(k.startswith("value_backbone.") for k in mapped_to_shared)
