"""Exact evaluation for Classical 3x3 Phantom Tic-Tac-Toe."""

from boardrl.games.openspiel.exp_a_spiel import ExpASpielEvaluator
from boardrl.games.openspiel.phantom_ttt.game import PhantomTicTacToe


class PhantomTicTacToeEvaluator(ExpASpielEvaluator):
    game_key = ("phantom_ttt", "classical")
    traverser_name = "PtttTraverser"
    infosets = 6_000_000

    def __init__(self, game, *, traverser=None):
        super().__init__(
            PhantomTicTacToe(spiel_game=game, resolve_chance=False), traverser=traverser
        )

    def _create_traverser(self, eas):
        return eas.PtttTraverser()
