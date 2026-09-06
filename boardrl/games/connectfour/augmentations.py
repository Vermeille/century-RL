from copy import copy
import random


def _split_line_ending(line):
    for ending in ("\r\n", "\n", "\r"):
        if line.endswith(ending):
            return line[: -len(ending)], ending
    return line, ""


def _board_width(lines):
    widths = {
        len(body) - 2
        for line in lines
        for body, _ending in [_split_line_ending(line)]
        if body.startswith("|") and body.endswith("|")
    }
    if len(widths) != 1:
        raise ValueError(
            "expected Connect Four state to contain board rows with one width"
        )
    return widths.pop()


def _mirror_move(move, width):
    column = int(move)
    if not 0 <= column < width:
        raise ValueError(f"Connect Four column {column} is out of range")
    return str(width - 1 - column)


def horizontal_symmetry(samples):
    """Randomly mirror Connect Four samples left-to-right.

    Each sample is mirrored independently with probability 0.5. The board
    width is inferred from the serialized ``|...|`` rows. Action lines and
    ``moves`` are transformed in place by semantic column mapping
    ``column -> width - 1 - column``. Their ordering is intentionally unchanged,
    so ``action_idx`` and action-indexed distributions remain aligned without
    any permutation.
    """

    augmented = []
    for sample in samples:
        sample = copy(sample)
        augmented.append(sample)

        if random.random() >= 0.5:
            continue

        lines = sample.state.splitlines(keepends=True)
        width = _board_width(lines)
        for idx, line in enumerate(lines):
            body, ending = _split_line_ending(line)
            if (
                body.startswith("|")
                and body.endswith("|")
                and len(body) == width + 2
            ):
                body = f"|{body[1:-1][::-1]}|"
            elif body.startswith("@"):
                body = f"@{_mirror_move(body[1:], width)}"
            lines[idx] = body + ending
        sample.state = "".join(lines)

        if hasattr(sample, "moves"):
            moves = sample.moves
            sample.moves = type(moves)(
                _mirror_move(move, width) for move in moves
            )

    return augmented
