from typing import List
import random


class RockPaperScissors:
    """Single-round Rock Paper Scissors game with hidden decisions."""

    def __init__(
        self,
        num_players: int = 2,
        v_rock: float = 1,
        v_paper: float = 1,
        v_scissors: float = 1,
    ):
        assert num_players == 2, "RockPaperScissors supports exactly two players"
        self.num_players = num_players
        self.turn = 0
        self.moves = ["rock", "paper", "scissors"]
        self.values = [v_rock, v_paper, v_scissors]
        self._choices: list[str | None] = [None, None]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def copy(self):
        g = RockPaperScissors()
        g.turn = self.turn
        g._choices = self._choices[:]
        return g

    def round(self):
        return self.turn // self.num_players

    def current_player(self):
        return self.turn % self.num_players

    # ------------------------------------------------------------------
    # Display utilities
    # ------------------------------------------------------------------
    def _repr_choice(self, choice):
        if choice is None:
            return "?"
        return choice[0].upper()

    def display(self, force: int = -1):
        if force == -1:
            player = self.current_player()
        else:
            assert force in [0, 1]
            player = force
        mine = self._choices[player]
        opp = self._choices[1 - player]
        # hide opponent choice until both have played
        opp_repr = self._repr_choice(opp if self.turn >= 2 else None)
        return f">{self._repr_choice(mine)}|{opp_repr}"

    def display_with_moves(self):
        board = self.display()
        moves = "\n".join(f"@{m}" for m in self.moves)
        return f"{board}\n{moves}"

    # ------------------------------------------------------------------
    # Gameplay
    # ------------------------------------------------------------------
    def play_str(self, mov: str):
        assert not self.ended(), "Game already ended"
        assert mov in self.moves
        self._choices[self.current_player()] = mov
        self.turn += 1

    def play_idx(self, idx: int):
        self.play_str(self.moves[idx])

    def ended(self):
        return self.turn >= 2

    def _result(self):
        a, b = self._choices
        if a is None or b is None:
            return 0
        if a == b:
            return 0
        wins = {
            ("rock", "scissors"): self.values[0],
            ("paper", "rock"): self.values[1],
            ("scissors", "paper"): self.values[2],
        }
        return wins.get((a, b), -wins.get((b, a), 0))

    def points_for(self, me: int):
        if not self.ended():
            return 0
        res = self._result()
        return res if me == 0 else -res

    def points(self):
        return self.points_for(self.current_player())

    def diff_points_for(self, me: int):
        return self.points_for(me)

    diff_points = points

    def simulate_to_end(self):
        while not self.ended():
            self.play_str(random.choice(self.moves))
