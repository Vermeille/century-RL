import random
from boardrl.utils import fast_sample


class TicTacToe:
    # Specific
    def __init__(self, num_players=2):
        assert num_players == 2
        self.board = [None for _ in range(9)]
        self.turn = 0
        self.moves = [str(i) for i in range(9)]

    def copy(self):
        g = TicTacToe()
        g.board = self.board[:]
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

        rep = {None: " ", 0: "O", 1: "X"}
        lines = [">" + rep[p]]
        for row in [self.board[i * 3 : (i + 1) * 3] for i in range(3)]:
            lines.append("|" + "".join([rep[r] for r in row]) + "|")
        return "\n".join(lines)

    def display_with_moves(self):
        board = self.display()
        return board + "\nMoves\n" + "\n".join([f"@{i}" for i in self.moves])

    def play_str(self, mov):
        assert not self.ended()
        idx = int(mov)
        assert self.board[idx] is None
        self.board[idx] = self.current_player()
        self.moves = [str(i) for i in range(9) if self.board[i] is None]
        self.turn += 1

    def ended(self):
        return self.winner() is not None or self.turn == 9

    def winner(self):
        for i in range(3):
            if self.board[i] == self.board[i + 3] == self.board[i + 6]:
                if self.board[i] is not None:
                    return self.board[i]
            if self.board[i * 3] == self.board[i * 3 + 1] == self.board[i * 3 + 2]:
                if self.board[i * 3] is not None:
                    return self.board[i * 3]
        if self.board[0] == self.board[4] == self.board[8]:
            if self.board[0] is not None:
                return self.board[0]
        if self.board[2] == self.board[4] == self.board[6]:
            if self.board[2] is not None:
                return self.board[2]
        return None

    def points_for(self, me):
        if (winner := self.winner()) is None:
            return 0
        else:
            return 1 if winner == me else -1

    # Predefined
    def points(self):
        return self.points_for(self.current_player())

    def simulate_to_end(self):
        while not self.ended():
            self.play_str(random.choice(self.moves))

    diff_points = points
    diff_points_for = points_for

    def play_idx(self, idx):
        return self.play_str(self.moves[idx])

    def play_distribution(self, x):
        idx = fast_sample(x)
        move = self.moves[idx]
        return self.play_str(move)


if __name__ == "__main__":
    g = TicTacToe()
    # print(g.display_with_moves())
    print(g.ended())
    g.play_str("0")
    print(g.display_with_moves())
    g.play_str("1")
    print(g.display_with_moves())
    g.play_str("3")
    print(g.display_with_moves())
    g.play_str("5")
    print(g.display_with_moves())
    g.play_str("6")
    print(g.display_with_moves())
    print(g.ended())
    print(g.points())
