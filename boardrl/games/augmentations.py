from copy import copy
import random

import torch


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

        old_action_idx = sample.action_idx
        if not 0 <= old_action_idx < len(action_positions):
            raise ValueError(
                f"action_idx {old_action_idx} is out of range for "
                f"{len(action_positions)} actions"
            )
        sample.action_idx = action_order.index(old_action_idx)

        for field in ("moves", "action_distribution", "reference_policy"):
            if not hasattr(sample, field):
                continue
            value = getattr(sample, field)
            if len(value) != len(action_order):
                raise ValueError(
                    f"{field} has length {len(value)}, expected "
                    f"{len(action_order)}"
                )
            if isinstance(value, torch.Tensor):
                value = value[action_order]
            else:
                value = type(value)(value[i] for i in action_order)
            setattr(sample, field, value)
    return augmented
