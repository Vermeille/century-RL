from boardrl.games.strategies import strategy_from_string


class GameDesc:
    def __init__(self, game_class, strategy_from_string, metrics_class):
        self.make_game = game_class
        self.strategy_from_string = strategy_from_string
        self.make_metrics = metrics_class


def century():
    import pyximport

    pyximport.install(setup_args={"script_args": ["--cython-cplus"]})
    from boardrl.games.century.strategies import (
        strategy_from_string as century_strategy_from_string,
    )
    from boardrl.games.century.engine import Century
    from boardrl.games.century.metrics import Metrics

    century_strategy_from_string.update(strategy_from_string)
    return GameDesc(Century, century_strategy_from_string, Metrics)
