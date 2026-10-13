"""Exact evaluation for Classical 3x3 Phantom Tic-Tac-Toe."""

from boardrl.games.openspiel.exp_a_spiel import ExpASpielEvaluator


class PhantomTicTacToeEvaluator(ExpASpielEvaluator):
    game_key = ("phantom_ttt", "classical")
    traverser_name = "PtttTraverser"
    infosets = 6_000_000
