import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import Config, LossConfig


def test_config_defaults():
    cfg = Config.from_dict({})
    assert cfg.device == "cpu"
    assert cfg.train.optimizer == "AdamW"
    assert cfg.train.lr == 1e-4
    assert isinstance(cfg.train.loss, LossConfig)
    assert cfg.train.loss.value == "value_mse_loss"
    assert cfg.train.loss.policy == "policy_gradient_loss"
    assert cfg.self_play.num_games == 4
    assert cfg.self_play.max_len == 500
    assert cfg.pit.every == 4
    assert cfg.pit.num_games == 32
    assert cfg.pit.max_len == 220


def test_config_overrides():
    raw = {
        "device": "cuda",
        "train": {
            "lr": 0.5,
            "loss": {"value": "custom_value", "policy": "custom_policy"},
        },
        "self_play": {"num_games": 7, "strategies": ["random", "best"]},
        "pit": {"every": 1},
    }
    cfg = Config.from_dict(raw)
    assert cfg.device == "cuda"
    assert cfg.train.lr == 0.5
    assert cfg.train.loss.value == "custom_value"
    assert cfg.train.loss.policy == "custom_policy"
    assert cfg.self_play.num_games == 7
    assert cfg.self_play.strategies == ["random", "best"]
    assert cfg.pit.every == 1
    # Ensure unspecified pit fields keep defaults
    assert cfg.pit.num_games == 32
    assert cfg.pit.max_len == 220
