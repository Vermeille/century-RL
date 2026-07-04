import random


class ConnectFour:
    def __init__(self, num_players=2):
        assert num_players == 2
        self.num_players = num_players
        self.width = 7
        self.height = 6
        self.board = [[None for _ in range(self.height)] for _ in range(self.width)]
        self.heights = [0 for _ in range(self.width)]
        self.turn = 0
        self.moves = [str(i) for i in range(self.width)]
        self._winner = None

    def copy(self):
        g = ConnectFour()
        g.board = [col[:] for col in self.board]
        g.heights = self.heights[:]
        g.turn = self.turn
        g.moves = self.moves[:]
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

        # Absolute encoding; the leading marker below indicates the side to move.
        rep = {None: " ", 0: "O", 1: "X"}
        lines = [">" + rep[p]]
        for y in range(self.height - 1, -1, -1):
            line = "|"
            for x in range(self.width):
                line += rep[self.board[x][y]]
            line += "|"
            lines.append(line)
        lines.append("-" * (self.width + 2))
        return "\n".join(lines) + "\n"

    def display_with_moves(self):
        board = self.display()
        moves = [m for m in self.moves]
        return board + "\n".join([f"@{m}" for m in moves])

    def play_str(self, mov):
        assert not self.ended()
        col = int(mov)
        assert 0 <= col < self.width
        assert self.heights[col] < self.height

        self.board[col][self.heights[col]] = self.current_player()
        self.heights[col] += 1
        self.moves = [
            str(i) for i in range(self.width) if self.heights[i] < self.height
        ]
        self.turn += 1
        self._winner = self._check_winner()

    def _check_winner_at(self, x, y):
        if self.board[x][y] is None:
            return None

        # Check horizontal
        if x <= self.width - 4:
            if all(self.board[x + i][y] == self.board[x][y] for i in range(4)):
                return self.board[x][y]

        # Check vertical
        if y <= self.height - 4:
            if all(self.board[x][y + i] == self.board[x][y] for i in range(4)):
                return self.board[x][y]

        # Check diagonal up-right
        if x <= self.width - 4 and y <= self.height - 4:
            if all(self.board[x + i][y + i] == self.board[x][y] for i in range(4)):
                return self.board[x][y]

        # Check diagonal up-left
        if x >= 3 and y <= self.height - 4:
            if all(self.board[x - i][y + i] == self.board[x][y] for i in range(4)):
                return self.board[x][y]

        return None

    def _check_winner(self):
        for x in range(self.width):
            for y in range(self.height):
                if (w := self._check_winner_at(x, y)) is not None:
                    return w
        return None

    def winner(self):
        return self._winner

    def ended(self):
        return self.winner() is not None or self.turn == self.width * self.height

    def points_for(self, me):
        if (winner := self.winner()) is None:
            return 0
        else:
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
    g.play_str("3")  # Middle column
    print(g.ended())
    print(g.current_player())
    print(g.display_with_moves())
    g.play_str("3")  # Middle column
    print(g.current_player())
    print(g.ended())
    print(g.display_with_moves())
    print(g.display(0))
    print(g.display(1))
