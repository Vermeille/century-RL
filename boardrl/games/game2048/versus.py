"""Adversarial 2048: the second player places each randomly valued tile."""

from boardrl.games.game2048.game import Game2048, SLIDES


SPAWN_CELLS = {
    f"spawn {chr(97 + col)}{row + 1}": row * 4 + col
    for row in range(4) for col in range(4)
}


class Sliding:
    player = 0
    name = "slide"

    def moves(self, game):
        return [
            name for name, slide in SLIDES.items()
            if slide.apply(game.board)[0] != game.board
        ]

    def play(self, game, move):
        game._slide(move)
        if not game._target_ends_game():
            game.pending_tile = 2 if game.rng.random() < 0.9 else 4
            game.phase = PLACING


class Placing:
    player = 1
    name = "spawn"

    def moves(self, game):
        return [move for move, index in SPAWN_CELLS.items() if game.board[index] == 0]

    def play(self, game, move):
        game.board[SPAWN_CELLS[move]] = game.pending_tile
        game.pending_tile = 0
        game.phase = SLIDING


SLIDING = Sliding()
PLACING = Placing()


class Versus2048(Game2048):
    num_players = 2

    def __init__(self, num_players: int = 2, seed: int = -1, nolimits: bool = False):
        self.phase = SLIDING
        self.pending_tile = 0
        super().__init__(num_players=num_players, seed=seed, nolimits=nolimits)

    def _refresh_moves(self):
        self.moves = [] if self._target_ends_game() else self.phase.moves(self)

    def play_str(self, move):
        if move not in self.moves:
            raise ValueError(f"Illegal 2048 move: {move!r}")
        self.phase.play(self, move)
        self._refresh_moves()

    def current_player(self):
        return self.phase.player

    def winner(self):
        if not self.ended():
            return None
        return 0 if self.won() else 1

    def points_for(self, player):
        if player not in (0, 1):
            raise ValueError(f"Invalid 2048 player: {player}")
        return (self.score, -self.score)[player]

    diff_points_for = points_for

    def display(self, force=-1):
        viewer = self.current_player() if force == -1 else force
        header = [
            "Mode: versus", f"Viewer: {viewer}", f"Player: {self.current_player()}",
            f"Phase: {self.phase.name}", f"Next tile: {self.pending_tile}",
        ]
        if self.ended():
            header.append(f"Winner: {self.winner()}")
        return "\n".join([*header, super().display(force)])
