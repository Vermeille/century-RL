from copy import copy
import random
import re


_CARD_LINE = re.compile(r"^([HD])(\d+)(?: (.*))?$")
_HAND_MOVE = re.compile(r"^(@?)H(\d+)(.*)$")


def _split_line_ending(line):
    for ending in ("\r\n", "\n", "\r"):
        if line.endswith(ending):
            return line[: -len(ending)], ending
    return line, ""


def _owned_region(lines):
    start = None
    for i, line in enumerate(lines):
        body, _ = _split_line_ending(line)
        if body == "_Me" or body.startswith("_Me "):
            start = i + 1
            break

    if start is None:
        return range(0)

    end = len(lines)
    for i in range(start, len(lines)):
        body, _ = _split_line_ending(lines[i])
        if body == "_Moves":
            end = i
            break
    return range(start, end)


def _shuffle_card_group(lines, positions, prefix):
    slots = []
    for position in positions:
        body, ending = _split_line_ending(lines[position])
        match = _CARD_LINE.fullmatch(body)
        if match is None or match.group(1) != prefix:
            continue
        slots.append((position, int(match.group(2)), match.group(3), ending))

    if len(slots) < 2:
        return {}

    cards = [(index, payload) for _, index, payload, _ in slots]
    random.shuffle(cards)

    old_to_new = {}
    for (position, new_index, _, ending), (old_index, payload) in zip(slots, cards):
        line = f"{prefix}{new_index}"
        if payload is not None:
            line += " " + payload
        lines[position] = line + ending
        old_to_new[old_index] = new_index

    return old_to_new


def _remap_hand_move(move, old_to_new):
    match = _HAND_MOVE.fullmatch(move)
    if match is None:
        return move

    old_index = int(match.group(2))
    if old_index not in old_to_new:
        return move

    return f"{match.group(1)}H{old_to_new[old_index]}{match.group(3)}"


def shuffle_hand(samples):
    """Shuffle Century's current player's hand representation.

    Hand ordering has no game semantics. Hand indices do appear in legal moves,
    so every ``H{i}`` move is renamed to follow the shuffled card. Legal-action
    order itself is unchanged, therefore action-indexed training metadata does
    not need to be permuted.
    """

    augmented = []
    for sample in samples:
        sample = copy(sample)
        augmented.append(sample)
        if sample.state is None:
            raise ValueError("sample state is required for Century hand shuffling")

        lines = sample.state.splitlines(keepends=True)
        old_to_new = _shuffle_card_group(lines, _owned_region(lines), "H")

        if old_to_new:
            for i, line in enumerate(lines):
                body, ending = _split_line_ending(line)
                if body.startswith("@H"):
                    lines[i] = _remap_hand_move(body, old_to_new) + ending

            if sample.moves is not None:
                sample.moves = [
                    _remap_hand_move(move, old_to_new) for move in sample.moves
                ]

        sample.state = "".join(lines)

    return augmented


def shuffle_discard(samples):
    """Shuffle Century's current player's discard representation.

    Discard ordering has no game semantics and discarded cards are not
    referenced by legal moves, so only the ``D{i}`` card descriptors need to be
    permuted.
    """

    augmented = []
    for sample in samples:
        sample = copy(sample)
        augmented.append(sample)
        if sample.state is None:
            raise ValueError("sample state is required for Century discard shuffling")

        lines = sample.state.splitlines(keepends=True)
        _shuffle_card_group(lines, _owned_region(lines), "D")
        sample.state = "".join(lines)

    return augmented
