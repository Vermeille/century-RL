"""Exact Leduc evaluation using the Leduc game adapter."""

from boardrl.games.openspiel.evaluation import OpenSpielTreeEvaluator
from boardrl.games.openspiel.leduc.game import Leduc


class LeducEvaluator(OpenSpielTreeEvaluator):
    def _create_game(self):
        return Leduc(spiel_game=self.game, resolve_chance=False)
