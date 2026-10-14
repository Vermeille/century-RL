"""Exact evaluation for Abrupt 3x3 Dark Hex."""

from boardrl.games.openspiel.exp_a_spiel import ExpASpielEvaluator
from boardrl.games.openspiel.abrupt_dark_hex.game import AbruptDarkHex


class AbruptDarkHexEvaluator(ExpASpielEvaluator):
    game_key = ("dark_hex", "adh")
    traverser_name = "AbruptDhTraverser"
    infosets = 27_340_000

    def __init__(self, game, *, traverser=None):
        super().__init__(
            AbruptDarkHex(spiel_game=game, resolve_chance=False), traverser=traverser
        )

    def _create_traverser(self, eas):
        return eas.AbruptDhTraverser()
