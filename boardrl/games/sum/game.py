import random


class Sum:
    def __init__(self, num_players=2):
        self.num_players = num_players
        self.current_player_ = 0
        self.scores = [0, 0]
        self.moves = [str(i) for i in range(10)]
        self.turn = 0
        self.round_ = 0
        self.make_board()

    def make_board(self):
        self.a = random.randint(0, 9)
        self.b = random.randint(0, 9)

    def round(self):
        return self.round_

    def copy(self):
        g = Sum()
        g.a = self.a
        g.b = self.b
        g.scores = self.scores[:]
        g.moves = self.moves[:]
        return g

    def current_player(self):
        return self.current_player_

    def display(self, force=-1):
        if force == -1:
            p = self.current_player()
        else:
            assert force in [0, 1]
            p = force

        lines = [f"{self.scores[p]} | {self.scores[1 - p]}"]
        lines.append(f"{self.a} {self.b}")
        return "\n".join(lines) + "\n"

    def display_with_moves(self):
        board = self.display()
        moves = [m for m in self.moves]
        return board + "\n".join([f"@{m}" for m in moves])

    def play_str(self, mov):
        assert not self.ended()
        self.turn += 1
        mov = int(mov)
        if mov != int((self.a + self.b) // 2):
            self.current_player_ = (self.current_player_ + 1) % self.num_players
            if self.current_player_ == 0:
                self.round_ += 1
            return
        self.round_ += 1
        self.scores[self.current_player_] += 1
        self.make_board()

    def ended(self):
        return 3 in self.scores

    def diff_points(self):
        return self.diff_points_for(self.current_player())

    def diff_points_for(self, me):
        return self.scores[me] - self.scores[1 - me]

    def simulate_to_end(self):
        while not self.ended():
            self.play_str(random.choice(self.moves))

    def points_for(self, me):
        return self.scores[me] - self.scores[1 - me]

    def points(self):
        return self.points_for(self.current_player())

    def play_idx(self, idx):
        return self.play_str(self.moves[idx])


if __name__ == "__main__":
    g = Sum()
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
