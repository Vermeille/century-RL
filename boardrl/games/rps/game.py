import random


MOVES = ["rock", "paper", "scissors"]
MOVE_ID = {move: i + 1 for i, move in enumerate(MOVES)}
DISPLAY_BY_ID = ("?", "R", "P", "S")
WINNING_PAIRS = {(1, 3), (2, 1), (3, 2)}


class RockPaperScissors:
    """Single-round Rock Paper Scissors game with hidden decisions."""

    __slots__ = ("num_players", "turn", "moves", "values", "_choices")

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
        self.moves = MOVES
        self.values = (v_rock, v_paper, v_scissors)
        self._choices = bytearray(2)

    def copy(self):
        g = RockPaperScissors.__new__(RockPaperScissors)
        g.num_players = self.num_players
        g.turn = self.turn
        g.moves = self.moves
        g.values = self.values
        g._choices = self._choices.copy()
        return g

    def round(self):
        return self.turn // self.num_players

    def current_player(self):
        return self.turn % self.num_players

    def _repr_choice(self, choice_id):
        return DISPLAY_BY_ID[choice_id]

    def display(self, force: int = -1):
        if force == -1:
            player = self.current_player()
        else:
            assert force in [0, 1]
            player = force
        mine = self._choices[player]
        opp = self._choices[1 - player] if self.turn >= 2 else 0
        return f">{self._repr_choice(mine)}|{self._repr_choice(opp)}"

    def display_with_moves(self):
        board = self.display()
        moves = "\n".join(f"@{m}" for m in self.moves)
        return f"{board}\n{moves}"

    def play_str(self, mov: str):
        assert not self.ended(), "Game already ended"
        choice_id = MOVE_ID.get(mov, 0)
        assert choice_id
        self._choices[self.current_player()] = choice_id
        self.turn += 1

    def play_idx(self, idx: int):
        self.play_str(self.moves[idx])

    def ended(self):
        return self.turn >= 2

    def _result(self):
        a, b = self._choices
        if not a or not b or a == b:
            return 0
        if (a, b) in WINNING_PAIRS:
            return self.values[a - 1]
        return -self.values[b - 1]

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
