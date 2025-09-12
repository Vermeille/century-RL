import random


class GuessNumber:
    def __init__(self, num_symbols: int = 2, secret: int | None = None):
        """Two player communication game.

        Player 1 chooses symbols to hint the secret number. Player 2 guesses
        numbers. The game ends once the secret is guessed.

        Args:
            num_symbols: Number of symbols available to player 1.
            secret: Optional secret number for deterministic tests. If ``None``
                a random number in ``[0, 100]`` is sampled.
        """
        assert num_symbols > 0
        self.num_symbols = num_symbols
        self.secret = secret if secret is not None else random.randint(0, 100)
        self.turn = 0
        self.round_ = 0
        self.finished = False
        self.current_symbol: str | None = None
        self.history: list[tuple[str, int]] = []
        self.moves = self._moves_for_current_player()

    # Helpers -----------------------------------------------------------------
    def _moves_for_current_player(self) -> list[str]:
        if self.finished:
            return []
        if self.current_player() == 0:
            return [chr(ord("A") + i) for i in range(self.num_symbols)]
        else:
            return [str(i) for i in range(101)]

    # Interface required by framework ----------------------------------------
    def copy(self) -> "GuessNumber":
        g = GuessNumber(self.num_symbols, secret=self.secret)
        g.turn = self.turn
        g.round_ = self.round_
        g.finished = self.finished
        g.current_symbol = self.current_symbol
        g.history = self.history[:]
        g.moves = self.moves[:]
        return g

    def round(self) -> int:
        return self.round_

    def current_player(self) -> int:
        return self.turn % 2

    def display(self, force: int = -1) -> str:
        if force == -1:
            p = self.current_player()
        else:
            assert force in [0, 1]
            p = force

        lines: list[str] = []
        if p == 0:
            lines.append(f"Secret: {self.secret}")
        lines.append("History:")
        for sym, guess in self.history:
            lines.append(f"{sym} {guess}")
        return "\n".join(lines)

    def display_with_moves(self) -> str:
        board = self.display()
        moves = [f"@{m}" for m in self.moves]
        return board + ("\n" + "\n".join(moves) if moves else "")

    def play_str(self, mov: str) -> None:
        assert not self.ended()
        if self.current_player() == 0:
            assert mov in self.moves
            self.current_symbol = mov
            self.turn += 1
        else:
            guess = int(mov)
            assert 0 <= guess <= 100
            assert self.current_symbol is not None
            self.history.append((self.current_symbol, guess))
            self.turn += 1
            self.round_ += 1
            if guess == self.secret:
                self.finished = True
            self.current_symbol = None
        self.moves = self._moves_for_current_player()

    def play_idx(self, idx: int) -> None:
        self.play_str(self.moves[idx])

    def ended(self) -> bool:
        return self.finished

    def winner(self) -> int | None:
        return 0 if self.finished else None

    def points_for(self, me: int) -> int:
        return -self.round_

    def points(self) -> int:
        return self.points_for(self.current_player())

    diff_points = points
    diff_points_for = points_for

    def simulate_to_end(self) -> None:
        while not self.ended():
            self.play_str(random.choice(self.moves))
