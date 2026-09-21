from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Protocol

import torch


class RolloutEndpoint(Protocol):
    state: str
    terminal: bool
    truncated: bool


State = str | list[str]
Moves = list[str] | list[list[str]]
ActionIndex = int | torch.Tensor
Policy = torch.Tensor | list[torch.Tensor]
Scalar = float | torch.Tensor
Flag = bool | torch.Tensor


def collate(xs):
    if isinstance(xs[0], (int, float)):
        return torch.tensor(xs)
    if isinstance(xs[0], torch.Tensor):
        try:
            return torch.stack(xs, dim=0)
        except RuntimeError:
            return xs
    return xs


@dataclass(slots=True, repr=False, eq=False)
class TrainingSample:
    """Explicit schema for one training transition or a collated batch."""

    state: State | None = None
    moves: Moves | None = None
    action_idx: ActionIndex | None = None
    action_distribution: Policy | None = None
    score: Scalar | None = None
    reward: Scalar | None = None
    returns: Scalar | None = None
    next: TrainingSample | RolloutEndpoint | None = None
    terminal: Flag = False
    truncated: Flag = False

    reference_policy: Policy | None = None
    reference_value: Scalar | None = None
    reference_value_stddev: Scalar | None = None
    reference_max_q: Scalar | None = None
    next_reference_value: Scalar | None = None
    next_reference_max_q: Scalar | None = None
    advantage: Scalar | None = None
    td_lambda: Scalar | None = None
    gae: Scalar | None = None
    normalized_gae: Scalar | None = None

    @classmethod
    def collate(cls, samples: list[TrainingSample]) -> TrainingSample:
        if not samples:
            raise ValueError("cannot collate an empty sample list")

        values = {}
        for field in fields(cls):
            if field.name == "next":
                continue
            field_values = [getattr(sample, field.name) for sample in samples]
            if all(value is None for value in field_values):
                continue
            if any(value is None for value in field_values):
                raise ValueError(
                    f"cannot collate partially populated field '{field.name}'"
                )
            values[field.name] = collate(field_values)
        return cls(**values)

    def to(self, *args, **kwargs):
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                setattr(self, field.name, value.to(*args, **kwargs))
            elif (
                isinstance(value, list)
                and value
                and isinstance(value[0], torch.Tensor)
            ):
                setattr(
                    self,
                    field.name,
                    [tensor.to(*args, **kwargs) for tensor in value],
                )
        return self

    def __repr__(self):
        out = ["TrainingSample:"]
        for field in fields(self):
            if field.name == "next":
                continue
            value = getattr(self, field.name)
            if value is not None:
                out.append(f"{field.name}: {value}")
        return "\n".join(out)
