"""Exact evaluation for Abrupt 3x3 Dark Hex."""

from boardrl.games.openspiel.exp_a_spiel import ExpASpielEvaluator


class AbruptDarkHexEvaluator(ExpASpielEvaluator):
    game_key = ("dark_hex", "adh")
    traverser_name = "AbruptDhTraverser"
    infosets = 27_340_000
