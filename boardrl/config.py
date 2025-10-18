"""Configuration models used throughout training.

The project previously relied on ``EasyDict`` to parse YAML configuration
files. While convenient, this provided little structure and offered no
defaults within the code base itself.  This module defines a small set of
Pydantic models that document the expected configuration shape, provide sane
default values and automatically validate input dictionaries, rejecting any
unknown keys. ``Config.from_dict`` can be used to construct the hierarchy
from a raw ``dict`` loaded from YAML.
"""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ConfigDict


class StrictModel(BaseModel):
    """Base model that forbids extra keys."""

    model_config = ConfigDict(extra="forbid")


class TrainConfig(StrictModel):
    optimizer: str = "AdamW"
    gradient_epochs: int = 1
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    batch_size: int = 12
    show_every: int = 10
    save_every: int = 100
    iterations: float = float("inf")
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    losses: List[str | dict] = Field(default_factory=list)
    entropy_reward_scale: Optional[float] = None
    reward_rescale: Optional[float] = None
    only_players: Optional[List[int]] = None
    only_strategies: Optional[List[int]] = None
    reference_model_update: Optional[str] = "False"


class SelfPlayConfig(StrictModel):
    num_games: int = 4
    max_len: int = 500
    strategies: List[dict | str] = Field(default_factory=list)
    batch_size: Optional[int] = None
    rotate: bool = True


class PitConfig(SelfPlayConfig):
    every: int = 4
    num_games: int = 32
    max_len: int = 220


class NetConfig(StrictModel):
    dim: int = 64
    num_layers: int = 2
    head_size: Optional[int] = None
    num_heads: Optional[int] = None
    backbone: str = "transformer"


class Config(StrictModel):
    device: str = "cpu"
    tag: str = ""
    game: str = "century"
    model: Optional[str] = None
    net: NetConfig = Field(default_factory=NetConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    self_play: SelfPlayConfig = Field(default_factory=SelfPlayConfig)
    pit: PitConfig = Field(default_factory=PitConfig)
    visdom_url: str = "offline"
    visdom_port: int = 8097

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Config":
        """Create a :class:`Config` from a plain dictionary."""

        if "self_play" in d:
            d["self_play"].setdefault("batch_size", d["self_play"]["num_games"])
        return Config.model_validate(d)
