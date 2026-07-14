from typing import Tuple
import random


def card_points(c: int) -> int:
    p = 0
    p += 2 if c % 5 == 0 else 0
    p += 3 if c % 10 == 0 else 0
    p += 5 if (c // 11) * 11 == c else 0
    return p


class Take5:
    def __init__(self, num_players: int = 4, num_stacks: int = 4, num_cards: int = 104):
        self.num_players = num_players
        self.num_stacks = num_stacks
        self.players: list[list[int]] = [[] for _ in range(num_players)]
        cards: list[int] = list(range(1, num_cards + 1))
        self.stacks: list[list[int]] = [[] for _ in range(num_stacks)]
        self.table: list[Tuple[int, int]] = []
        self.points_: list[int] = [0 for _ in range(num_players)]
        random.shuffle(cards)
        for p in self.players:
            for _ in range(10):
                p.append(cards.pop())

        for s in self.stacks:
            s.append(cards.pop())

        self.current_player_ = 0
        self.round_ = 0
        self.phase = 0
        self.moves = self._moves()

    def _moves(self) -> list[str]:
        if self.ended():
            return []
        if self.phase == 0:
            return [str(c) for c in self.players[self.current_player_]]
        return [f"S{i}" for i in range(len(self.stacks))]

    def copy(self) -> "Take5":
        game = Take5.__new__(Take5)
        game.num_players = self.num_players
        game.num_stacks = self.num_stacks
        game.players = [player[:] for player in self.players]
        game.stacks = [stack[:] for stack in self.stacks]
        game.table = self.table[:]
        game.points_ = self.points_[:]
        game.current_player_ = self.current_player_
        game.round_ = self.round_
        game.phase = self.phase
        game.moves = self.moves[:]
        return game

    def round(self) -> int:
        return self.round_

    def current_player(self) -> int:
        return self.current_player_

    def display(self, force=-1) -> str:
        if force == -1:
            p = self.current_player()
        else:
            assert force in range(self.num_players)
            p = force
        phase = "Put\n" if self.phase == 0 else f"Take {self.table[0][1]}\n"
        return phase + f"Hand: {' '.join(map(str, self.players[p]))}\n" + "\n".join(
            [f"S{i}: {' '.join(map(str, s))}\n" for i, s in enumerate(self.stacks)]
        )

    def display_with_moves(self) -> str:
        board = self.display()
        if self.phase == 0:
            moves = [f"@{c}" for c in self.players[self.current_player_]]
        else:
            moves = [f"@S{i}" for i in range(len(self.stacks))]
        return board + "\n".join(moves)

    def solve(self):
        self.phase = 1
        self.table.sort(key=lambda x: x[1])
        while self.table:
            player, card = self.table[0]
            valid_stacks = [s for s in self.stacks if s[-1] < card]
            if len(valid_stacks) == 0:
                self.current_player_ = player
                self.moves = self._moves()
                return
            else:
                self.table.pop(0)
                fitting_stack = max(valid_stacks, key=lambda s: s[-1])
                if len(fitting_stack) == 5:
                    self.points_[player] -= sum(card_points(c) for c in fitting_stack)
                    fitting_stack.clear()
                fitting_stack.append(card)

        self.phase = 0
        self.current_player_ = 0
        self.round_ += 1
        self.moves = self._moves()

    def play_str(self, move: str) -> None:
        assert not self.ended()
        if self.phase == 0:
            c = int(move)
            assert move in self.moves
            self.players[self.current_player_].remove(c)
            self.table.append((self.current_player_, c))
            self.current_player_ = (self.current_player_ + 1) % self.num_players
            if self.current_player_ == 0:
                self.solve()
            else:
                self.moves = self._moves()
        else:
            assert move in self.moves
            assert self.current_player_ == self.table[0][0]
            stack = self.stacks[int(move[1:])]
            self.points_[self.current_player_] -= sum(card_points(c) for c in stack)
            stack.clear()
            stack.append(self.table[0][1])
            self.table.pop(0)
            self.solve()

    def ended(self) -> bool:
        return self.phase == 0 and all(len(p) == 0 for p in self.players)

    def winner(self) -> int | None:
        if not self.ended():
            return None
        return max(range(self.num_players), key=self.points_for)

    def points(self) -> int:
        return self.points_for(self.current_player_)

    def points_for(self, player: int) -> int:
        return self.points_[player]

    def diff_points(self) -> int:
        return self.points()

    def diff_points_for(self, player: int) -> int:
        return self.points_for(player)

    def simulate_to_end(self) -> None:
        while not self.ended():
            self.play_str(random.choice(self.moves))

    def play_idx(self, idx: int) -> None:
        self.play_str(self.moves[idx])
