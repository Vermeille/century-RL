import random


_EMPTY = 0
_PLAYER0 = 1
_PLAYER1 = 2


class _ColumnView:
    __slots__ = ("game", "x")

    def __init__(self, game: "ConnectFour", x: int):
        self.game = game
        self.x = x

    def __getitem__(self, y: int):
        raw = self.game._cell(self.x, y)
        return None if raw == _EMPTY else raw - 1

    def __setitem__(self, y: int, value):
        self.game._board[self.game._index(self.x, y)] = _EMPTY if value is None else value + 1

    def __len__(self):
        return self.game.height


class _BoardView:
    __slots__ = ("game",)

    def __init__(self, game: "ConnectFour"):
        self.game = game

    def __getitem__(self, x: int):
        return _ColumnView(self.game, x)

    def __len__(self):
        return self.game.width


class ConnectFour:
    __slots__ = (
        "num_players",
        "width",
        "height",
        "_board",
        "heights",
        "turn",
        "moves",
        "_winner",
    )

    def __init__(self, num_players=2):
        assert num_players == 2
        self.num_players = num_players
        self.width = 7
        self.height = 6
        self._board = bytearray(self.width * self.height)
        self.heights = bytearray(self.width)
        self.turn = 0
        self.moves = [str(i) for i in range(self.width)]
        self._winner = None

    @property
    def board(self):
        """Compatibility 2D view; the retained state itself is ``_board`` bytes."""
        return _BoardView(self)

    def _index(self, x: int, y: int) -> int:
        return x * self.height + y

    def _cell(self, x: int, y: int) -> int:
        return self._board[self._index(x, y)]

    def _set_cell(self, x: int, y: int, player: int) -> None:
        self._board[self._index(x, y)] = player + 1

    def copy(self):
        g = ConnectFour.__new__(ConnectFour)
        g.num_players = self.num_players
        g.width = self.width
        g.height = self.height
        g._board = self._board.copy()
        g.heights = self.heights.copy()
        g.turn = self.turn
        g.moves = self.moves
        g._winner = self._winner
        return g

    def round(self):
        return self.turn // 2

    def current_player(self):
        return self.turn % 2

    def display(self, force=-1):
        if force == -1:
            p = self.current_player()
        else:
            assert force in [0, 1]
            p = force

        rep = (" ", "O", "X")
        lines = []
        for y in range(self.height - 1, -1, -1):
            lines.append("".join(rep[self._cell(x, y)] for x in range(self.width)))
        lines.append("-" * self.width)
        lines.append(">" + rep[p + 1])
        return "\n".join(lines) + "\n"

    def display_with_moves(self):
        board = self.display()
        return board + "\n".join(f"@{m}" for m in self.moves)

    def play_str(self, mov):
        assert not self.ended()
        col = int(mov)
        assert 0 <= col < self.width
        assert self.heights[col] < self.height

        self._set_cell(col, self.heights[col], self.current_player())
        self.heights[col] += 1
        self.moves = [
            str(i) for i in range(self.width) if self.heights[i] < self.height
        ]
        self.turn += 1
        self._winner = self._check_winner()

    def _check_winner_at(self, x, y):
        cell = self._cell(x, y)
        if cell == _EMPTY:
            return None

        if x <= self.width - 4:
            if all(self._cell(x + i, y) == cell for i in range(4)):
                return cell - 1

        if y <= self.height - 4:
            if all(self._cell(x, y + i) == cell for i in range(4)):
                return cell - 1

        if x <= self.width - 4 and y <= self.height - 4:
            if all(self._cell(x + i, y + i) == cell for i in range(4)):
                return cell - 1

        if x >= 3 and y <= self.height - 4:
            if all(self._cell(x - i, y + i) == cell for i in range(4)):
                return cell - 1

        return None

    def _check_winner(self):
        for x in range(self.width):
            for y in range(self.height):
                if (winner := self._check_winner_at(x, y)) is not None:
                    return winner
        return None

    def winner(self):
        return self._winner

    def ended(self):
        return self.winner() is not None or self.turn == self.width * self.height

    def points_for(self, me):
        if (winner := self.winner()) is None:
            return 0
        return 1 if winner == me else -1

    def points(self):
        return self.points_for(self.current_player())

    def simulate_to_end(self):
        while not self.ended():
            self.play_str(random.choice(self.moves))

    diff_points = points
    diff_points_for = points_for

    def play_idx(self, idx):
        return self.play_str(self.moves[idx])


if __name__ == "__main__":
    g = ConnectFour()
    print(g.display_with_moves())
    g.play_str("3")
    print(g.ended())
    print(g.current_player())
    print(g.display_with_moves())