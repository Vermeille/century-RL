from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch


class _Unset:
    __slots__ = ()


_UNSET = _Unset()

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


@dataclass(slots=True, init=False, repr=False, eq=False)
class TrainingSample:
    """Explicit schema for one training transition or a collated batch.

    Fields are declared up front, but optional pipeline annotations remain
    genuinely absent until produced. This preserves the existing ``hasattr``
    semantics used by augmentations while rejecting unknown sample fields.
    """

    state: State
    moves: Moves
    action_idx: ActionIndex
    action_distribution: Policy
    score: Scalar
    reward: Scalar
    returns: Scalar
    next: object | None
    terminal: Flag
    truncated: Flag

    reference_policy: Policy
    reference_value: Scalar
    reference_value_stddev: Scalar
    reference_max_q: Scalar
    next_reference_value: Scalar
    next_reference_max_q: Scalar
    advantage: Scalar
    td_lambda: Scalar
    gae: Scalar
    normalized_gae: Scalar

    def __init__(
        self,
        *,
        state: State | _Unset = _UNSET,
        moves: Moves | _Unset = _UNSET,
        action_idx: ActionIndex | _Unset = _UNSET,
        action_distribution: Policy | _Unset = _UNSET,
        score: Scalar | _Unset = _UNSET,
        reward: Scalar | _Unset = _UNSET,
        returns: Scalar | _Unset = _UNSET,
        next: object | None | _Unset = _UNSET,
        terminal: Flag | _Unset = _UNSET,
        truncated: Flag | _Unset = _UNSET,
        reference_policy: Policy | _Unset = _UNSET,
        reference_value: Scalar | _Unset = _UNSET,
        reference_value_stddev: Scalar | _Unset = _UNSET,
        reference_max_q: Scalar | _Unset = _UNSET,
        next_reference_value: Scalar | _Unset = _UNSET,
        next_reference_max_q: Scalar | _Unset = _UNSET,
        advantage: Scalar | _Unset = _UNSET,
        td_lambda: Scalar | _Unset = _UNSET,
        gae: Scalar | _Unset = _UNSET,
        normalized_gae: Scalar | _Unset = _UNSET,
    ):
        values = {
            "state": state,
            "moves": moves,
            "action_idx": action_idx,
            "action_distribution": action_distribution,
            "score": score,
            "reward": reward,
            "returns": returns,
            "next": next,
            "terminal": terminal,
            "truncated": truncated,
            "reference_policy": reference_policy,
            "reference_value": reference_value,
            "reference_value_stddev": reference_value_stddev,
            "reference_max_q": reference_max_q,
            "next_reference_value": next_reference_value,
            "next_reference_max_q": next_reference_max_q,
            "advantage": advantage,
            "td_lambda": td_lambda,
            "gae": gae,
            "normalized_gae": normalized_gae,
        }
        for name, value in values.items():
            if value is not _UNSET:
                setattr(self, name, value)

    @classmethod
    def collate(cls, samples: list[TrainingSample]) -> TrainingSample:
        if not samples:
            raise ValueError("cannot collate an empty sample list")

        present = [
            field.name
            for field in fields(cls)
            if field.name != "next" and hasattr(samples[0], field.name)
        ]
        return cls(
            **{
                name: collate([getattr(sample, name) for sample in samples])
                for name in present
            }
        )

    def to(self, *args, **kwargs):
        for field in fields(self):
            name = field.name
            if not hasattr(self, name):
                continue
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(*args, **kwargs))
            elif (
                isinstance(value, list)
                and value
                and isinstance(value[0], torch.Tensor)
            ):
                setattr(self, name, [x.to(*args, **kwargs) for x in value])
        return self

    def __repr__(self):
        out = ["TrainingSample:"]
        for field in fields(self):
            name = field.name
            if name == "next" or not hasattr(self, name):
                continue
            out.append(f"{name}: {getattr(self, name)}")
        return "\n".join(out)
