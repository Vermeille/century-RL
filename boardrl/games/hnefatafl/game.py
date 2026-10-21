"""11x11 Fetlar Hnefatafl, following the panel's 2018 printed rules.

Source: https://aagenielsen.dk/fetlar_rules2018visual.pdf
Player 0 attacks; player 1 defends. Coordinates a1..kB start at the top left;
ranks A and B denote rows 10 and 11. Moves name the source and changed axis.
A configurable ply cap adjudicates otherwise indefinite games. Repetition
draws are disabled so observations need no history.
"""

from dataclasses import dataclass
import random


SIZE = 11
THRONE = (5, 5)
CORNERS = frozenset(((0, 0), (0, 10), (10, 0), (10, 10)))
RESTRICTED = CORNERS | {THRONE}
CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
FILES = "abcdefghijk"
RANKS = "123456789AB"
Position = tuple[int, int]


def coordinate(position: Position) -> str:
    row, col = position
    return FILES[col] + RANKS[row]


def position(text: str) -> Position:
    if len(text) != 2 or text[0] not in FILES or text[1] not in RANKS:
        raise ValueError(f"Invalid Hnefatafl coordinate: {text}")
    return RANKS.index(text[1]), FILES.index(text[0])


def on_board(position: Position) -> bool:
    return all(0 <= value < SIZE for value in position)


@dataclass(frozen=True)
class Warrior:
    player: int
    symbol: str
    royal = False

    def can_stop(self, position: Position) -> bool:
        return position not in RESTRICTED

    def captured(self, board, position: Position, direction: Position) -> bool:
        row, col = position
        dr, dc = direction
        anchor = row + dr, col + dc
        supporter = board.get(anchor)
        if supporter is not None:
            return supporter.player != self.player
        return anchor in RESTRICTED


@dataclass(frozen=True)
class King(Warrior):
    royal = True

    def can_stop(self, position: Position) -> bool:
        return True

    def captured(self, board, position: Position, direction: Position) -> bool:
        # The 2018 panel rules require four actual attackers, including beside
        # the throne. Neither the throne nor the board edge substitutes for one.
        row, col = position
        return all(board.get((row + dr, col + dc)) == ATTACKER for dr, dc in CARDINAL)


ATTACKER = Warrior(0, "A")
DEFENDER = Warrior(1, "D")
KING = King(1, "K")


@dataclass(frozen=True)
class Move:
    source: Position
    destination: Position

    def text(self) -> str:
        target = coordinate(self.destination)
        changed_axis = target[0] if self.source[0] == self.destination[0] else target[1]
        return coordinate(self.source) + changed_axis

    @classmethod
    def from_text(cls, text: str):
        if len(text) != 3:
            raise ValueError(f"Invalid Hnefatafl move: {text}")
        source = position(text[:2])
        if text[2] in FILES:
            destination = source[0], FILES.index(text[2])
        elif text[2] in RANKS:
            destination = RANKS.index(text[2]), source[1]
        else:
            raise ValueError(f"Invalid Hnefatafl destination: {text[2]}")
        return cls(source, destination)


class Hnefatafl:
    def __init__(self, num_players: int = 2, max_plies: int = 500):
        if num_players != 2:
            raise ValueError("Hnefatafl supports exactly two players")
        if max_plies < 1:
            raise ValueError("max_plies must be positive")
        self.num_players = num_players
        self.max_plies = max_plies
        self.board: dict[Position, Warrior] = {}
        for offset in range(3, 8):
            for position in ((0, offset), (10, offset), (offset, 0), (offset, 10)):
                self.board[position] = ATTACKER
        for position in ((1, 5), (9, 5), (5, 1), (5, 9)):
            self.board[position] = ATTACKER
        for row in range(3, 8):
            for col in range(3, 8):
                if 0 < abs(row - 5) + abs(col - 5) <= 2:
                    self.board[row, col] = DEFENDER
        self.board[THRONE] = KING
        self.turn = 0
        self.captured = [0, 0]  # Removed soldiers of each side; excludes the king.
        self.end_reason = "ongoing"
        self._winner = None
        self._refresh_moves()

    def current_player(self):
        return self.turn % self.num_players

    def round(self):
        return self.turn // self.num_players

    def king_position(self):
        return next((position for position, piece in self.board.items() if piece.royal), None)

    def _piece_moves(self, source: Position, piece: Warrior):
        row, col = source
        for dr, dc in CARDINAL:
            target = row + dr, col + dc
            while on_board(target) and target not in self.board:
                if piece.can_stop(target):
                    yield Move(source, target)
                target = target[0] + dr, target[1] + dc

    def _finish(self, reason: str, winner=None):
        self.end_reason = reason
        self._winner = winner

    def _refresh_moves(self):
        self._legal_moves = {}
        if not self.ended():
            self._legal_moves = {
                move.text(): move
                for position, piece in sorted(self.board.items())
                if piece.player == self.current_player()
                for move in self._piece_moves(position, piece)
            }
            if not self._legal_moves:
                self._finish("stalemate")
        self.moves = list(self._legal_moves)

    def play_str(self, move: str):
        if self.ended() or move not in self._legal_moves:
            raise ValueError(f"Illegal Hnefatafl move: {move}")
        selected = self._legal_moves[move]
        piece = self.board.pop(selected.source)
        self.board[selected.destination] = piece
        row, col = selected.destination
        victims = []
        for dr, dc in CARDINAL:
            position = row + dr, col + dc
            victim = self.board.get(position)
            if (
                victim is not None and victim.player != piece.player
                and victim.captured(self.board, position, (dr, dc))
            ):
                victims.append(position)
        # Resolve all sandwiches from the same post-move board.
        for position in victims:
            victim = self.board.pop(position)
            self.captured[victim.player] += int(not victim.royal)
        self.turn += 1
        king = self.king_position()
        if king is None:
            self._finish("king_captured", 0)
        elif king in CORNERS:
            self._finish("king_escaped", 1)
        elif self.turn >= self.max_plies:
            self._finish("move_limit")
        self._refresh_moves()

    def play_idx(self, index):
        self.play_str(self.moves[index])

    def ended(self):
        return self.end_reason != "ongoing"

    def winner(self):
        return self._winner

    def points_for(self, player):
        if player not in (0, 1):
            raise ValueError(f"Invalid player: {player}")
        if self._winner is None:
            return 0
        return 1 if player == self._winner else -1

    def points(self):
        return self.points_for(self.current_player())

    diff_points_for = points_for
    diff_points = points

    def display(self, force=-1):
        viewer = self.current_player() if force == -1 else force
        if viewer not in (0, 1):
            raise ValueError(f"Invalid viewer: {viewer}")
        lines = [
            f"V:{viewer} P:{self.current_player()} Ply:{self.turn}/{self.max_plies}",
            f"C:{self.captured[0]} {self.captured[1]}",
        ]
        for row in range(SIZE):
            cells = []
            for col in range(SIZE):
                position = row, col
                piece = self.board.get(position)
                cells.append(
                    piece.symbol if piece is not None
                    else "T" if position == THRONE
                    else "X" if position in CORNERS else "."
                )
            lines.append("".join(cells))
        if self.ended():
            lines.append(f"End:{self.end_reason}")
        return "\n".join(lines)

    def display_with_moves(self):
        return self.display() + "".join(f"\n@{move}" for move in self.moves)

    def copy(self):
        game = Hnefatafl.__new__(Hnefatafl)
        game.num_players = self.num_players
        game.max_plies = self.max_plies
        game.board = self.board.copy()
        game.turn = self.turn
        game.captured = self.captured.copy()
        game.end_reason = self.end_reason
        game._winner = self._winner
        game._legal_moves = self._legal_moves.copy()
        game.moves = self.moves.copy()
        return game

    def simulate_to_end(self):
        while not self.ended():
            self.play_str(random.choice(self.moves))
