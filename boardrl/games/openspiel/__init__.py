"""Shared OpenSpiel integration."""


def make_model_evaluator(game, *, progress=True):
    # Registration happens while boardrl.evaluation is still being imported.
    from boardrl.games.openspiel.evaluation import OpenSpielModelEvaluator

    return OpenSpielModelEvaluator(game, progress=progress)
