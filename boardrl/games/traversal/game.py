"""Traversal rules: https://www.traversalgame.com/rules.

Text coordinates a0..j9 run from the top left. Player 0 starts at the bottom
and exits above row 0; player 1 travels in the opposite direction.
"""

from dataclasses import dataclass
from heapq import heappop, heappush
import random

from boardrl.games.traversal.terrain import SIZE, sample_terrain


MOVE_LIMIT = 200
HEIGHT_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"
CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
Position = tuple[int, int]


def coordinate(position: Position) -> str:
    row, col = position
    return f"{chr(ord('a') + col)}{row}"


def position(text: str) -> Position:
    if len(text) != 2 or text[0] not in "abcdefghij" or text[1] not in "0123456789":
        raise ValueError(f"Invalid Traversal coordinate: {text}")
    return int(text[1]), ord(text[0]) - ord("a")


def player_positions(board, player):
    return sorted(square for square, piece in board.items() if piece.player == player)


def on_board(position: Position) -> bool:
    return all(0 <= value < SIZE for value in position)


def neighbors(position: Position):
    row, col = position
    for dr, dc in CARDINAL:
        target = row + dr, col + dc
        if on_board(target):
            yield target


class PieceType:
    """Immutable rule collaborators shared by pieces and copies."""

    name = ""
    symbol = ""
    movement = 0
    strength = 0
    first_finish_wins = False

    def special_targets(self, source: Position, occupied):
        return ()

    def attack_bonus(self, cost: int) -> int:
        return 0


class Explorer(PieceType):
    name = "Explorer"
    symbol = "E"
    movement = 5
    strength = 1

    def special_targets(self, source: Position, occupied):
        row, col = source
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            yield row + dr, col + dc
        for dr, dc in CARDINAL:
            middle = row + dr, col + dc
            # Leaps ignore terrain, but the general prohibition on crossing
            # pieces still applies (as in the official local game engine).
            if on_board(middle) and middle not in occupied:
                yield row + 2 * dr, col + 2 * dc


class Vanguard(PieceType):
    name = "Vanguard"
    symbol = "V"
    movement = 4
    strength = 2

    def attack_bonus(self, cost: int) -> int:
        return int(cost == self.movement)


class Sentinel(PieceType):
    name = "Sentinel"
    symbol = "S"
    movement = 3
    strength = 3
    first_finish_wins = True


EXPLORER = Explorer()
VANGUARD = Vanguard()
SENTINEL = Sentinel()
STARTING_TYPES = (SENTINEL, VANGUARD, EXPLORER, EXPLORER, VANGUARD, SENTINEL)


@dataclass(frozen=True)
class Piece:
    player: int
    kind: PieceType


@dataclass(frozen=True)
class Move:
    source: Position
    destination: Position | None
    cost: int

    def text(self, source_index: int) -> str:
        target = "-" if self.destination is None else coordinate(self.destination)
        return f"{source_index}{target}"


class Traversal:
    def __init__(self, num_players: int = 2, terrain: str = "random", seed: int = -1):
        if num_players != 2:
            raise ValueError("Traversal supports exactly two players")
        self.num_players = num_players
        self.terrain = terrain
        self.seed = seed
        selected = sample_terrain(terrain, seed)
        self.terrain_name = selected.name
        self.heights = selected.heights
        self.board = {
            (row, col): Piece(player, kind)
            for player, row in ((0, SIZE - 1), (1, 0))
            for col, kind in enumerate(STARTING_TYPES, start=2)
        }
        self.turn = 0
        self.scores = [0, 0]
        self.finished: list[Piece] = []
        self.captured: list[Piece] = []
        self.history: list[str] = []
        self._repetition_moves: list[str] = []
        self._positions: list[tuple] = []
        self._ended = False
        self._winner = None
        self.end_reason = "ongoing"
        self._refresh_moves()

    def current_player(self):
        return self.turn % self.num_players

    def round(self):
        return self.turn // self.num_players

    def movement_cost(self, source: Position, target: Position) -> int:
        row, col = source
        next_row, next_col = target
        return 1 + abs(self.heights[row][col] - self.heights[next_row][next_col])

    @staticmethod
    def goal_row(player: int) -> int:
        return (SIZE - 1) * player

    def attack_strengths(self, source: Position, target: Position, cost: int):
        attacker, defender = self.board[source], self.board[target]
        height_difference = (
            self.heights[source[0]][source[1]] - self.heights[target[0]][target[1]]
        )
        height_bonus = int(height_difference > 0) - int(height_difference < 0)
        attack = attacker.kind.strength + height_bonus + attacker.kind.attack_bonus(cost)
        defense = defender.kind.strength
        for position in neighbors(target):
            ally = self.board.get(position)
            if ally is None or position == source:
                continue
            if ally.player == attacker.player:
                attack += ally.kind.strength
            else:
                defense += 1
        return attack, defense

    def _piece_moves(self, source: Position, piece: Piece):
        # Dijkstra records the cheapest route, including the last step onto an
        # enemy. An affordable detour must never manufacture a charge bonus.
        costs = {source: 0}
        attacks: dict[Position, int] = {}
        frontier = [(0, source)]
        exit_cost = None
        while frontier:
            cost, position = heappop(frontier)
            if cost != costs[position]:
                continue
            if position[0] == self.goal_row(piece.player) and cost < piece.kind.movement:
                candidate = cost + 1
                exit_cost = candidate if exit_cost is None else min(exit_cost, candidate)
            for target in neighbors(position):
                next_cost = cost + self.movement_cost(position, target)
                if next_cost > piece.kind.movement:
                    continue
                occupant = self.board.get(target)
                if occupant is not None:
                    if occupant.player != piece.player:
                        attacks[target] = min(attacks.get(target, next_cost), next_cost)
                    continue
                if next_cost < costs.get(target, piece.kind.movement + 1):
                    costs[target] = next_cost
                    heappush(frontier, (next_cost, target))

        moves = {
            target: Move(source, target, cost)
            for target, cost in costs.items()
            if target != source
        }
        for target, cost in attacks.items():
            attack, defense = self.attack_strengths(source, target, cost)
            if attack >= defense:
                moves[target] = Move(source, target, cost)

        for target in piece.kind.special_targets(source, self.board):
            if not on_board(target):
                # Explorer specials may cross the goal edge, just as in the
                # official implementation; crossing the sides/home is illegal.
                goal = -1 if piece.player == 0 else SIZE
                if target[0] == goal and 0 <= target[1] < SIZE:
                    exit_cost = piece.kind.movement
                continue
            if target in moves:
                continue
            occupant = self.board.get(target)
            if occupant is not None:
                if occupant.player == piece.player:
                    continue
                attack, defense = self.attack_strengths(source, target, piece.kind.movement)
                if attack < defense:
                    continue
            moves[target] = Move(source, target, piece.kind.movement)

        for target in sorted(moves):
            yield moves[target]
        if exit_cost is not None:
            # All exit columns produce the same state, so expose one action.
            yield Move(source, None, exit_cost)

    def _refresh_moves(self):
        self._actions = {}
        if not self._ended:
            for index, square in enumerate(player_positions(self.board, self.current_player())):
                for move in self._piece_moves(square, self.board[square]):
                    self._actions[move.text(index)] = move
            # The published rules do not adjudicate immobilization. A stuck
            # position is a stalemate draw, so the rollout contract stays total.
            if not self._actions:
                self._finish("stalemate", None)
        self.moves = list(self._actions)

    def _finish(self, reason: str, winner):
        self._ended = True
        self._winner = winner
        self.end_reason = reason

    def _score_winner(self):
        if self.scores[0] == self.scores[1]:
            return None
        return int(self.scores[1] > self.scores[0])

    def _adjudicate_move_limit(self):
        if self.scores[0] != self.scores[1]:
            self._finish("move_limit_score", self._score_winner())
            return
        # More advanced wins; among equally advanced pieces, stronger wins.
        best = [(-SIZE, 0), (-SIZE, 0)]
        for (row, _), piece in self.board.items():
            distance = abs(row - self.goal_row(piece.player))
            best[piece.player] = max(best[piece.player], (-distance, piece.kind.strength))
        winner = None if best[0] == best[1] else int(best[1] > best[0])
        self._finish(
            "move_limit_advancement" if winner is not None else "move_limit_draw",
            winner,
        )

    def _position_key(self):
        return (
            self.current_player(),
            tuple(
                (position, piece.player, piece.kind.symbol)
                for position, piece in sorted(self.board.items())
            ),
            tuple((piece.player, piece.kind.symbol) for piece in self.finished),
        )

    def _repeated_cycle(self):
        # Match the site's local engine: three identical consecutive cycles
        # with the same side to move (cycle periods must therefore be even).
        count = len(self._positions)
        return any(
            self._positions[-period:] == self._positions[-2 * period:-period]
            == self._positions[-3 * period:-2 * period]
            for period in range(2, count // 3 + 1, 2)
        )

    def play_str(self, move: str):
        if self._ended or move not in self._actions:
            raise ValueError(f"Illegal Traversal move: {move}")
        action = self._actions[move]
        piece = self.board.pop(action.source)
        if action.destination is None:
            self._repetition_moves.clear()
            self.finished.append(piece)
            self.scores[piece.player] += piece.kind.strength
            if len(self.finished) == 1 and piece.kind.first_finish_wins:
                self._finish("sentinel", piece.player)
            elif len(self.finished) == 2:
                self._finish("two_finished", self._score_winner())
        else:
            captured = self.board.pop(action.destination, None)
            if captured is not None:
                self._repetition_moves.clear()
                self.captured.append(captured)
            self.board[action.destination] = piece
            if captured is None:
                # An undo entry identifies the moved piece in the resulting
                # position and its old square. Three characters suffice and
                # reconstruct the complete ordered quiet-position history.
                index = player_positions(self.board, piece.player).index(action.destination)
                self._repetition_moves.append(f"{index}{coordinate(action.source)}")

        if not self._ended:
            remaining = [
                [p.kind.strength for p in self.board.values() if p.player == player]
                for player in range(2)
            ]
            if not all(remaining):
                # The official local engine ends an eliminated side's game by
                # crediting each side's strongest remaining piece to its score.
                for player in range(2):
                    self.scores[player] += max(remaining[player], default=0)
                self._finish("elimination", self._score_winner())

        self.history.append(move)
        self.turn += 1
        if not self._ended and self.turn >= MOVE_LIMIT:
            self._adjudicate_move_limit()
        if not self._ended:
            self._positions.append(self._position_key())
            if self._repeated_cycle():
                self._finish("repetition", None)
        self._refresh_moves()

    def play_idx(self, index: int):
        self.play_str(self.moves[index])

    def ended(self):
        return self._ended

    def winner(self):
        return self._winner

    def points_for(self, player: int):
        return self.scores[player]

    def points(self):
        return self.points_for(self.current_player())

    def diff_points_for(self, player: int):
        if self._ended and self._winner is None:
            return 0
        difference = self.scores[player] - self.scores[1 - player]
        if self._ended and difference == 0:
            return 0.01 if player == self._winner else -0.01
        return difference

    def diff_points(self):
        return self.diff_points_for(self.current_player())

    def display(self, force: int = -1):
        player = self.current_player() if force == -1 else force
        if player not in (0, 1):
            raise ValueError("Traversal viewer must be player 0 or 1")
        finished = "".join(f"{p.player}{p.kind.symbol}" for p in self.finished) or "-"
        lines = [f"T:{self.turn} V:{player} M:{self.terrain_name} F:{finished}"]
        lines.extend("".join(HEIGHT_DIGITS[height] for height in row) for row in self.heights)
        for owner in range(2):
            pieces = " ".join(
                f"{self.board[square].kind.symbol}{coordinate(square)}"
                for square in player_positions(self.board, owner)
            )
            lines.append(f"{owner}:{pieces}")
        if self._ended:
            # Elimination can award points beyond finished-piece strengths.
            lines.append(f"S:{self.scores[0]},{self.scores[1]} E:{self.end_reason} W:{self._winner if self._winner is not None else 'draw'}")
        else:
            # Captures and finishes permanently change material, preventing
            # repetition across that boundary. The current board and these
            # reversible moves reconstruct every still-relevant position.
            # Turn distinguishes the omitted initial position (no material
            # change yet) from the counted position after a material change.
            lines.append("R:" + ("".join(self._repetition_moves) or "-"))
        return "\n".join(lines) + "\n"

    def display_with_moves(self):
        return self.display() + "\n".join(f"@{move}" for move in self.moves)

    def copy(self):
        other = Traversal.__new__(Traversal)
        other.num_players = self.num_players
        other.terrain = self.terrain
        other.terrain_name = self.terrain_name
        other.seed = self.seed
        other.heights = self.heights
        other.board = self.board.copy()
        other.turn = self.turn
        other.scores = self.scores.copy()
        other.finished = self.finished.copy()
        other.captured = self.captured.copy()
        other.history = self.history.copy()
        other._repetition_moves = self._repetition_moves.copy()
        other._positions = self._positions.copy()
        other._ended = self._ended
        other._winner = self._winner
        other.end_reason = self.end_reason
        other._actions = self._actions.copy()
        other.moves = self.moves.copy()
        return other

    def simulate_to_end(self):
        while not self.ended():
            self.play_str(random.choice(self.moves))
