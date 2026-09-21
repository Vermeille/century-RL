from copy import copy
import random

import torch


def _reorder(value, order, field_name):
    if len(value) != len(order):
        raise ValueError(
            f"{field_name} has length {len(value)}, expected {len(order)}"
        )
    if isinstance(value, torch.Tensor):
        return value[order]
    return type(value)(value[i] for i in order)


def shuffle_actions(samples):
    """Return copies of samples with randomized legal-action order.

    The model identifies actions by the order of lines beginning with ``@``.
    Keep the action lines in their existing slots, but permute their contents
    and apply the same permutation to all metadata indexed by those actions.
    """

    augmented = []
    for sample in samples:
        sample = copy(sample)
        augmented.append(sample)
        if sample.state is None:
            raise ValueError("sample state is required for action shuffling")
        lines = sample.state.splitlines(keepends=True)
        action_positions = [
            position for position, line in enumerate(lines) if line.startswith("@")
        ]
        if len(action_positions) < 2:
            continue

        action_order = list(range(len(action_positions)))
        random.shuffle(action_order)
        action_lines = []
        action_endings = []
        for position in action_positions:
            line = lines[position]
            for ending in ("\r\n", "\n", "\r"):
                if line.endswith(ending):
                    line = line[: -len(ending)]
                    break
            else:
                ending = ""
            action_lines.append(line)
            action_endings.append(ending)
        for position, old_idx, ending in zip(
            action_positions, action_order, action_endings
        ):
            lines[position] = action_lines[old_idx] + ending
        sample.state = "".join(lines)

        if sample.action_idx is not None:
            old_action_idx = sample.action_idx
            if not 0 <= old_action_idx < len(action_positions):
                raise ValueError(
                    f"action_idx {old_action_idx} is out of range for "
                    f"{len(action_positions)} actions"
                )
            sample.action_idx = action_order.index(old_action_idx)

        if sample.moves is not None:
            sample.moves = _reorder(sample.moves, action_order, "moves")
        if sample.action_distribution is not None:
            sample.action_distribution = _reorder(
                sample.action_distribution,
                action_order,
                "action_distribution",
            )
        if sample.reference_policy is not None:
            sample.reference_policy = _reorder(
                sample.reference_policy,
                action_order,
                "reference_policy",
            )
    return augmented
