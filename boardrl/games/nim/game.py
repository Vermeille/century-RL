import random


class Nim:
    def __init__(self, num_players: int = 2, num_stones: int = 21, max_pick: int = 3):
        assert num_players == 2, "Nim supports exactly two players"
        self.num_players = num_players
        self.num_stones = num_stones
        self.max_pick = max_pick
        self.turn = 0
        self._winner = None
        self.moves = self._legal_moves()

    def _legal_moves(self):
        return [
            str(i)
            for i in range(1, min(self.max_pick, self.num_stones) + 1)
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def copy(self):
        g = Nim(self.num_players, self.num_stones, self.max_pick)
        g.turn = self.turn
        g._winner = self._winner
        g.moves = self.moves[:]
        return g

    def current_player(self):
        return self.turn % self.num_players

    def round(self):
        return self.turn // self.num_players

    # ------------------------------------------------------------------
    # Display utilities
    # ------------------------------------------------------------------
    def display(self, force: int = -1):
        if force == -1:
            player = self.current_player()
        else:
            assert force in [0, 1]
            player = force
        lines = []
        if self._winner is not None:
            lines.append("WIN" if self._winner == player else "LOST")
        lines.append(str(self.num_stones))
        return "\n".join(lines)

    def display_with_moves(self):
        board = self.display()
        moves = "\n".join(f"@{m}" for m in self.moves)
        return f"{board}\nMoves\n{moves}"

    # ------------------------------------------------------------------
    # Gameplay
    # ------------------------------------------------------------------
    def play_str(self, mov: str):
        assert not self.ended(), "Game already ended"
        pick = int(mov)
        assert 1 <= pick <= min(self.max_pick, self.num_stones)
        self.num_stones -= pick

        if self.num_stones == 0:
            self._winner = self.current_player()

        self.turn += 1
        self.moves = self._legal_moves()

    def play_idx(self, idx):
        return self.play_str(self.moves[idx])

    def ended(self):
        return self._winner is not None

    def winner(self):
        return self._winner

    def points_for(self, me: int):
        if not self.ended():
            return 0
        return 1 if self._winner == me else -1

    def points(self):
        return self.points_for(self.current_player())

    def diff_points_for(self, me: int):
        return self.points_for(me)

    def diff_points(self):
        return self.points_for(self.current_player())

    def simulate_to_end(self):
        while not self.ended():
            self.play_str(random.choice(self.moves))
