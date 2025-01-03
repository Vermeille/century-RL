from boardrl.utils import RegisterByName
from boardrl.games.strategies import strategy_from_string


class GameDesc:
    def __init__(self, game_class, strategy_from_string, metrics_class):
        self.make_game = game_class
        self.strategy_from_string = strategy_from_string
        self.make_metrics = metrics_class


games_library = RegisterByName()


@games_library.register("century")
class Century:
    def __call__(self):
        import pyximport

        pyximport.install(setup_args={"script_args": ["--cython-cplus"]})
        from boardrl.games.century.strategies import (
            strategy_from_string as century_strategy_from_string,
        )
        from boardrl.games.century.engine import Century
        from boardrl.games.century.metrics import Metrics

        century_strategy_from_string.update(strategy_from_string)
        return GameDesc(Century, century_strategy_from_string, Metrics)


@games_library.register("tictactoe")
class TicTacToe:
    def __call__(self):
        from boardrl.games.tictactoe.game import TicTacToe
        from boardrl.games.tictactoe.metrics import Metrics

        return GameDesc(TicTacToe, strategy_from_string, Metrics)
