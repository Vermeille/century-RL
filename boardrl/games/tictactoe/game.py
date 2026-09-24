import random


EMPTY = 0
PLAYER_OFFSET = 1


class TicTacToe:
    __slots__ = ("num_players", "board", "turn", "moves")

    def __init__(self, num_players: int = 2) -> None:
        assert num_players == 2
        self.num_players = num_players
        self.board = bytearray(9)
        self.turn = 0
        self.moves = [str(i) for i in range(9)]

    def copy(self):
        g = TicTacToe.__new__(TicTacToe)
        g.num_players = self.num_players
        g.board = self.board.copy()
        g.turn = self.turn
        g.moves = self.moves
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

        rep = {EMPTY: " ", 1: "O", 2: "X"}
        lines = [">" + rep[p + PLAYER_OFFSET]]
        for row in [self.board[i * 3 : (i + 1) * 3] for i in range(3)]:
            lines.append("".join(rep[cell] for cell in row))
        return "\n".join(lines)

    def display_with_moves(self):
        board = self.display()
        return board + "\nMoves\n" + "\n".join([f"@{i}" for i in self.moves])

    def play_str(self, mov):
        assert not self.ended()
        idx = int(mov)
        assert self.board[idx] == EMPTY
        self.board[idx] = self.current_player() + PLAYER_OFFSET
        self.moves = [str(i) for i in range(9) if self.board[i] == EMPTY]
        self.turn += 1

    def ended(self):
        return self.winner() is not None or self.turn == 9

    def winner(self):
        for i in range(3):
            if self.board[i] == self.board[i + 3] == self.board[i + 6] != EMPTY:
                return self.board[i] - PLAYER_OFFSET
            row = i * 3
            if self.board[row] == self.board[row + 1] == self.board[row + 2] != EMPTY:
                return self.board[row] - PLAYER_OFFSET
        if self.board[0] == self.board[4] == self.board[8] != EMPTY:
            return self.board[0] - PLAYER_OFFSET
        if self.board[2] == self.board[4] == self.board[6] != EMPTY:
            return self.board[2] - PLAYER_OFFSET
        return None

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
