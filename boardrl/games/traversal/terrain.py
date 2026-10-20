"""Offline sampling of terrain collected from the official Traversal API."""

from dataclasses import dataclass
from functools import cache
from importlib.resources import files
import json
import random


SIZE = 10
SOURCE = "https://api.traversalgame.com/api/boards/generate"
TERRAIN_TYPES = (
    "mountains", "valleys", "highlands", "woodlands", "desert", "dunes",
    "jungle", "meadows", "canyons", "rivers", "ridges", "petrified",
    "everest", "mariana", "hell",
)
Grid = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Terrain:
    name: str
    heights: Grid


@dataclass(frozen=True)
class TerrainTransform:
    quarter_turns: int = 0
    reflected: bool = False

    def apply(self, board: Grid) -> Grid:
        if self.reflected:
            board = tuple(row[::-1] for row in board)
        for _ in range(self.quarter_turns % 4):
            board = tuple(zip(*board[::-1]))
        return board


TRANSFORMS = tuple(
    TerrainTransform(turns, reflected)
    for reflected in (False, True)
    for turns in range(4)
)


class TerrainLibrary:
    def __init__(self):
        data = json.loads(
            files(__package__).joinpath("terrain_samples.json").read_text(encoding="utf-8")
        )
        self.boards = {
            name: tuple(tuple(tuple(row) for row in board) for board in boards)
            for name, boards in data["boards"].items()
        }
        self.boards["flat"] = (tuple((0,) * SIZE for _ in range(SIZE)),)

    def sample(self, mode: str, rng) -> Terrain:
        name = rng.choice(TERRAIN_TYPES) if mode == "random" else mode
        try:
            pool = self.boards[name]
        except KeyError:
            choices = ", ".join(("random", "flat", *TERRAIN_TYPES))
            raise ValueError(f"Unknown Traversal terrain {mode!r}; choose {choices}") from None
        board = rng.choice(pool)
        return Terrain(name, rng.choice(TRANSFORMS).apply(board))


@cache
def terrain_library() -> TerrainLibrary:
    return TerrainLibrary()


def sample_terrain(mode: str, seed: int) -> Terrain:
    rng = random if seed == -1 else random.Random(seed)
    return terrain_library().sample(mode, rng)
