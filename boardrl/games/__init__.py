from boardrl.utils import RegisterByName
from boardrl.games.strategies import strategy_from_string


class GameDesc:
    def __init__(self, game_class, strategy_from_string, metrics_class):
        self.make_game = game_class
        self.strategy_from_string = strategy_from_string
        self.make_metrics = metrics_class


games_library = RegisterByName()


@games_library.register("century")
class Century(GameDesc):
    def __init__(self):
        import pyximport

        pyximport.install(setup_args={"script_args": ["--cython-cplus"]})
        from boardrl.games.century.strategies import (
            strategy_from_string as century_strategy_from_string,
        )
        from boardrl.games.century.engine import Century
        from boardrl.games.century.metrics import Metrics

        strats = century_strategy_from_string.copy().update(strategy_from_string)
        super().__init__(Century, strats, Metrics)


@games_library.register("tictactoe")
class TicTacToe(GameDesc):
    def __init__(self):
        from boardrl.games.tictactoe.game import TicTacToe
        from boardrl.games.tictactoe.metrics import Metrics

        super().__init__(TicTacToe, strategy_from_string, Metrics)


@games_library.register("connectfour")
class ConnectFour(GameDesc):
    def __init__(self):
        from boardrl.games.connectfour.game import ConnectFour
        from boardrl.games.connectfour.metrics import Metrics

        super().__init__(ConnectFour, strategy_from_string, Metrics)


@games_library.register("sum")
class Sum(GameDesc):
    def __init__(self):
        from boardrl.games.sum.game import Sum
        from boardrl.games.sum.metrics import Metrics
        from boardrl.games.sum.strategies import (
            strategy_from_string as sum_strategy_from_string,
        )

        strats = sum_strategy_from_string.copy().update(strategy_from_string)
        super().__init__(Sum, strats, Metrics)


@games_library.register("thegame")
class TheGame(GameDesc):
    def __init__(self):
        from boardrl.games.thegame.game import TheGame
        from boardrl.games.thegame.metrics import Metrics
        from boardrl.games.thegame.strategy import (
            strategy_from_string as thegame_strategy_from_string,
        )

        strats = thegame_strategy_from_string.copy().update(strategy_from_string)
        super().__init__(TheGame, strats, Metrics)
