"""Exact Leduc evaluation with its native player-visible text."""

from boardrl.games.openspiel.evaluation import OpenSpielTreeEvaluator
from boardrl.games.openspiel.leduc.game import Leduc


class LeducEvaluator(OpenSpielTreeEvaluator):
    text_type = Leduc.text_type
