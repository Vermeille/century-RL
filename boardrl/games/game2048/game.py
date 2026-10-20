"""Single-player 2048, optionally continuing beyond the target tile."""

import random
from copy import copy
from dataclasses import dataclass


@dataclass(frozen=True)
class Slide:
    """Read each line from the destination edge to share one merge algorithm."""

    lines: tuple[tuple[int, ...], ...]

    def apply(self, board: list[int]) -> tuple[list[int], int]:
        result = [0] * 16
        score = 0
        for line in self.lines:
            tiles = [board[index] for index in line if board[index]]
            merged = []
            index = 0
            while index < len(tiles):
                value = tiles[index]
                if index + 1 < len(tiles) and tiles[index + 1] == value:
                    value *= 2
                    score += value
                    index += 1
                merged.append(value)
                index += 1
            for destination, value in zip(line, merged):
                result[destination] = value
        return result, score


ROWS = tuple(tuple(row * 4 + col for col in range(4)) for row in range(4))
COLUMNS = tuple(tuple(row * 4 + col for row in range(4)) for col in range(4))
SLIDES = {
    "up": Slide(COLUMNS),
    "right": Slide(tuple(tuple(reversed(row)) for row in ROWS)),
    "down": Slide(tuple(tuple(reversed(column)) for column in COLUMNS)),
    "left": Slide(ROWS),
}


class Game2048:
    num_players = 1
    target = 2048

    def __init__(self, num_players: int = 1, seed: int = -1, nolimits: bool = False):
        if num_players != self.num_players:
            raise ValueError(f"2048 requires exactly {self.num_players} player(s)")
        self.rng = random.Random(None if seed < 0 else seed)
        self.nolimits = nolimits
        self.board = [0] * 16
        self.score = 0
        self.turn = 0
        self._spawn_tile()
        self._spawn_tile()
        self._refresh_moves()

    def _spawn_tile(self) -> None:
        empty = [index for index, value in enumerate(self.board) if value == 0]
        destination = self.rng.choice(empty)
        self.board[destination] = 2 if self.rng.random() < 0.9 else 4

    def _refresh_moves(self) -> None:
        self.moves = [] if self._target_ends_game() else [
            name for name, slide in SLIDES.items()
            if slide.apply(self.board)[0] != self.board
        ]

    def play_str(self, move: str) -> None:
        if move not in self.moves:
            raise ValueError(f"Illegal 2048 move: {move!r}")
        self._slide(move)
        self._spawn_tile()
        self._refresh_moves()

    def _slide(self, move: str) -> None:
        self.board, gained = SLIDES[move].apply(self.board)
        self.score += gained
        self.turn += 1

    def play_idx(self, index: int) -> None:
        self.play_str(self.moves[index])

    def current_player(self) -> int:
        return 0

    def round(self) -> int:
        return self.turn

    def ended(self) -> bool:
        return not self.moves

    def won(self) -> bool:
        return max(self.board) >= self.target

    def _target_ends_game(self) -> bool:
        return not self.nolimits and self.won()

    def winner(self) -> int | None:
        return 0 if self.ended() and self.won() else None

    def points_for(self, player: int) -> int:
        if player != 0:
            raise ValueError("2048 has only player 0")
        return self.score

    def points(self) -> int:
        return self.points_for(self.current_player())

    diff_points_for = points_for
    diff_points = points

    def display(self, force: int = -1) -> str:
        if force != -1 and force not in range(self.num_players):
            raise ValueError(f"Invalid 2048 player: {force}")
        status = "won" if self._target_ends_game() else "stuck" if self.ended() else "playing"
        lines = [
            "2048", f"Score: {self.score}", f"Moves: {self.turn}",
            f"Target: {self.target}", f"Nolimits: {self.nolimits}",
            f"Status: {status}", "Board:",
        ]
        lines.extend(
            " ".join(str(self.board[i]) if self.board[i] else "." for i in row)
            for row in ROWS
        )
        return "\n".join(lines)

    def display_with_moves(self) -> str:
        return "\n".join([self.display(), *(f"@{move}" for move in self.moves)])

    def copy(self, randomize: bool = False) -> "Game2048":
        game = copy(self)
        game.board = self.board.copy()
        game.moves = self.moves.copy()
        game.score = self.score
        game.turn = self.turn
        game.rng = random.Random()
        if not randomize:
            game.rng.setstate(self.rng.getstate())
        return game

    def simulate_to_end(self) -> None:
        while not self.ended():
            self.play_str(random.choice(self.moves))
