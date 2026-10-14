"""Exact evaluation for Abrupt 3x3 Phantom Tic-Tac-Toe."""

from boardrl.games.openspiel.exp_a_spiel import ExpASpielEvaluator


class AbruptPhantomTicTacToeEvaluator(ExpASpielEvaluator):
    game_key = ("phantom_ttt", "abrupt")
    traverser_name = "AbruptPtttTraverser"
    infosets = 23_320_000

    def _create_traverser(self, eas):
        return eas.AbruptPtttTraverser()
