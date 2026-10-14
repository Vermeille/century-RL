"""Exact evaluation for Abrupt 3x3 Phantom Tic-Tac-Toe."""

from boardrl.games.openspiel.exp_a_spiel import ExpASpielEvaluator
from boardrl.games.openspiel.abrupt_phantom_ttt.game import AbruptPhantomTicTacToe


class AbruptPhantomTicTacToeEvaluator(ExpASpielEvaluator):
    game_key = ("phantom_ttt", "abrupt")
    traverser_name = "AbruptPtttTraverser"
    infosets = 23_320_000

    def __init__(self, game, *, traverser=None):
        super().__init__(
            AbruptPhantomTicTacToe(spiel_game=game, resolve_chance=False),
            traverser=traverser,
        )

    def _create_traverser(self, eas):
        return eas.AbruptPtttTraverser()
