from copy import copy
import random
import re


SIZE = 5
_BOARD_ROW = re.compile(r"^([1-5]) ((?:[0-4][.#OX](?: [0-4][.#OX]){4}))$")
_COORD = re.compile(r"\b([a-e])([1-5])\b")


def _split_line_ending(line):
    for ending in ("\r\n", "\n", "\r"):
        if line.endswith(ending):
            return line[: -len(ending)], ending
    return line, ""


def _transform_coord(coord, transform):
    x = ord(coord[0]) - ord("a")
    y = int(coord[1]) - 1
    x, y = transform(x, y)
    return f"{chr(ord('a') + x)}{y + 1}"


def _transform_move(move, transform):
    return _COORD.sub(
        lambda match: _transform_coord(match.group(0), transform),
        move,
    )


def _transform_state(state, transform):
    lines = state.splitlines(keepends=True)
    board_rows = {}

    for index, line in enumerate(lines):
        body, ending = _split_line_ending(line)
        match = _BOARD_ROW.fullmatch(body)
        if match:
            board_rows[int(match.group(1)) - 1] = (
                index,
                match.group(2).split(" "),
                ending,
            )

    if len(board_rows) != SIZE:
        raise ValueError("expected Santorini state to contain exactly five board rows")

    transformed = [[None for _ in range(SIZE)] for _ in range(SIZE)]
    for y in range(SIZE):
        _index, cells, _ending = board_rows[y]
        for x, cell in enumerate(cells):
            xx, yy = transform(x, y)
            transformed[yy][xx] = cell

    for y in range(SIZE):
        index, _cells, ending = board_rows[y]
        lines[index] = f"{y + 1} " + " ".join(transformed[y]) + ending

    for index, line in enumerate(lines):
        body, ending = _split_line_ending(line)
        if body.startswith("@"):
            lines[index] = f"@{_transform_move(body[1:], transform)}{ending}"

    return "".join(lines)


def _apply_transform(samples, transform_factory):
    augmented = []
    for sample in samples:
        sample = copy(sample)
        augmented.append(sample)

        transform = transform_factory()
        if transform is None:
            continue

        sample.state = _transform_state(sample.state, transform)
        if hasattr(sample, "moves"):
            moves = sample.moves
            sample.moves = type(moves)(
                _transform_move(move, transform) for move in moves
            )

    return augmented


def _horizontal_transform(x, y):
    return SIZE - 1 - x, y


def _vertical_transform(x, y):
    return x, SIZE - 1 - y


def _rotation_transform(turns):
    def transform(x, y):
        for _ in range(turns):
            x, y = SIZE - 1 - y, x
        return x, y

    return transform


def horizontal_symmetry(samples):
    """Randomly mirror Santorini samples left-to-right."""

    return _apply_transform(
        samples,
        lambda: _horizontal_transform if random.random() < 0.5 else None,
    )


def vertical_symmetry(samples):
    """Randomly mirror Santorini samples top-to-bottom."""

    return _apply_transform(
        samples,
        lambda: _vertical_transform if random.random() < 0.5 else None,
    )


def rotation_symmetry(samples):
    """Randomly rotate Santorini samples by 0, 90, 180, or 270 degrees."""

    def transform_factory():
        turns = random.randrange(4)
        return None if turns == 0 else _rotation_transform(turns)

    return _apply_transform(samples, transform_factory)
