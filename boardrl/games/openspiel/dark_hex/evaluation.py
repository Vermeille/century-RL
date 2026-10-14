"""Exact evaluation for Classical 3x3 Dark Hex."""

from boardrl.games.openspiel.exp_a_spiel import ExpASpielEvaluator
from boardrl.games.openspiel.dark_hex.game import DarkHex


class DarkHexEvaluator(ExpASpielEvaluator):
    text_type = DarkHex.text_type
    game_key = ("dark_hex", "cdh")
    traverser_name = "DhTraverser"
    infosets = 6_080_000

    def _create_traverser(self, eas):
        return eas.DhTraverser()
