"""Exact evaluation for Classical 3x3 Dark Hex."""

from boardrl.games.openspiel.exp_a_spiel import ExpASpielEvaluator


class DarkHexEvaluator(ExpASpielEvaluator):
    game_key = ("dark_hex", "cdh")
    traverser_name = "DhTraverser"
    infosets = 6_080_000

    def _create_traverser(self, eas):
        return eas.DhTraverser()
