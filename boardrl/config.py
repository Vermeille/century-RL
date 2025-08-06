"""Configuration dataclasses used throughout training.

The project previously relied on ``EasyDict`` to parse YAML configuration
files.  While convenient, this provided little structure and offered no
defaults within the code base itself.  This module defines a small set of
dataclasses that document the expected configuration shape and provide sane
default values.  ``Config.from_dict`` can be used to construct the hierarchy
from a raw ``dict`` loaded from YAML.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class LossConfig:
    value: str = "value_mse_loss"
    policy: str = "policy_gradient_loss"


@dataclass
class TrainConfig:
    optimizer: str = "AdamW"
    gradient_epochs: int = 1
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.01
    batch_size: int = 12
    show_every: int = 10
    save_every: int = 100
    iterations: float = float("inf")
    discount_factor: float = 0.99
    loss: LossConfig = field(default_factory=LossConfig)
    entropy_reward_scale: Optional[float] = None
    reward_rescale: Optional[float] = None
    only_players: Optional[List[int]] = None
    only_strategies: Optional[List[int]] = None


@dataclass
class SelfPlayConfig:
    num_games: int = 4
    max_len: int = 500
    strategies: List[str] = field(default_factory=list)
    batch_size: Optional[int] = None


@dataclass
class PitConfig(SelfPlayConfig):
    every: int = 4
    num_games: int = 32
    max_len: int = 220


@dataclass
class NetConfig:
    dim: int = 64
    num_layers: int = 2
    head_size: int = 33


@dataclass
class Config:
    device: str = "cpu"
    tag: str = ""
    game: str = "century"
    model: Optional[str] = None
    net: NetConfig = field(default_factory=NetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    pit: PitConfig = field(default_factory=PitConfig)
    visdom_url: str = "offline"
    visdom_port: int = 8097

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Config":
        """Create a :class:`Config` from a plain dictionary."""

        train_dict = d.get("train", {})
        loss_dict = train_dict.get("loss", {})
        train_kwargs = {k: v for k, v in train_dict.items() if k != "loss"}
        if "betas" in train_kwargs:
            train_kwargs["betas"] = tuple(train_kwargs["betas"])
        train = TrainConfig(**train_kwargs)
        train.loss = LossConfig(**loss_dict)

        self_play = SelfPlayConfig(**d.get("self_play", {}))
        pit = PitConfig(**d.get("pit", {}))
        net = NetConfig(**d.get("net", {}))

        return Config(
            device=d.get("device", "cpu"),
            tag=d.get("tag", ""),
            game=d.get("game", "century"),
            model=d.get("model"),
            train=train,
            self_play=self_play,
            pit=pit,
            net=net,
            visdom_url=d.get("visdom_url", ""),
            visdom_port=d.get("visdom_port", 8097),
        )
