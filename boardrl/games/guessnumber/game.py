import random


class GuessNumber:
    def __init__(
        self,
        max_number: int = 10,
        num_symbols: int = 2,
        secret: int | None = None,
        num_players: int = 2,
    ):
        """Two player communication game.

        Player 1 chooses symbols to hint the secret number. Player 2 guesses
        numbers. The game ends once the secret is guessed.

        Args:
            num_symbols: Number of symbols available to player 1.
            secret: Optional secret number for deterministic tests. If ``None``
                a random number in ``[0, max_number]`` is sampled.
        """
        assert num_symbols > 0
        assert num_players == 2
        self.num_players = num_players
        self.num_symbols = num_symbols
        self.secret = secret or random.randint(1, max_number)
        self.max_number = max_number
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
            return [str(i) for i in range(1, self.max_number + 1)]

    # Interface required by framework ----------------------------------------
    def copy(self) -> "GuessNumber":
        g = GuessNumber(self.num_symbols, secret=self.secret)
        g.max_number = self.max_number
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

        if p == 1 and self.current_symbol:
            lines.append(f"{self.current_symbol} ?")

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
        else:
            guess = int(mov)
            assert 1 <= guess <= self.max_number
            assert self.current_symbol is not None
            self.history.append((self.current_symbol, guess))
            self.round_ += 1
            if guess == self.secret:
                self.finished = True
            self.current_symbol = None
        self.turn += 1
        self.moves = self._moves_for_current_player()

    def play_idx(self, idx: int) -> None:
        self.play_str(self.moves[idx])

    def ended(self) -> bool:
        return self.finished

    def won(self) -> bool:
        return self.finished

    def points_for(self, me: int) -> int:
        # return 1 if self.ended() else 0
        return -self.round_

    def points(self) -> int:
        return self.points_for(self.current_player())

    diff_points = points
    diff_points_for = points_for

    def simulate_to_end(self) -> None:
        while not self.ended():
            self.play_str(random.choice(self.moves))


if __name__ == "__main__":
    g = GuessNumber(secret=4)
    print(g.display_with_moves())
    g.play_idx(0)
    print("--", g.points())
    print(g.display_with_moves())
    g.play_idx(0)
    print("--", g.points())
    print(g.display_with_moves())
    print("--", g.points())
