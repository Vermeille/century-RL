"""Exact evaluation for Abrupt 3x3 Dark Hex."""

from boardrl.games.openspiel.exp_a_spiel import ExpASpielEvaluator
from boardrl.games.openspiel.abrupt_dark_hex.game import AbruptDarkHex


class AbruptDarkHexEvaluator(ExpASpielEvaluator):
    text_type = AbruptDarkHex.text_type
    game_key = ("dark_hex", "adh")
    traverser_name = "AbruptDhTraverser"
    infosets = 27_340_000

    def _create_traverser(self, eas):
        return eas.AbruptDhTraverser()
