from boardrl.utils import RegisterByName
from boardrl.games.strategies import strategy_from_string
from functools import partial

# Guard Century imports so it's only available when Cython/pyximport works
CenturyGame = None
try:
    import pyximport  # type: ignore

    pyximport.install(setup_args={"script_args": ["--cython-cplus"]})
    # If pyximport is available, try importing the compiled/compilable engine
    from boardrl.games.century.engine import Century as CenturyGame  # type: ignore
except Exception:
    CenturyGame = None

from boardrl.games.thegame.game import TheGame as TheGameGame
from boardrl.games.guessnumber.game import GuessNumber as GuessNumberGame


class GameDesc:
    def __init__(self, game_class, strategy_from_string, metrics_class):
        self.make_game = game_class
        self.strategy_from_string = strategy_from_string
        self.make_metrics = metrics_class


games_library = RegisterByName()


# Only register Century when the engine is importable
if CenturyGame is not None:
    @games_library.register("century", args_from=CenturyGame)
    class Century(GameDesc):
        def __init__(self, *args, **kwargs):
            from boardrl.games.century.strategies import (
                strategy_from_string as century_strategy_from_string,
            )
            from boardrl.games.century.metrics import Metrics

            strats = century_strategy_from_string.copy().update(strategy_from_string)
            super().__init__(partial(CenturyGame, *args, **kwargs), strats, Metrics)


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


@games_library.register("thegame", args_from=TheGameGame)
class TheGame(GameDesc):
    def __init__(self, *args, **kwargs):
        from boardrl.games.thegame.metrics import Metrics
        from boardrl.games.thegame.strategies import (
            strategy_from_string as thegame_strategy_from_string,
        )

        strats = thegame_strategy_from_string.copy().update(strategy_from_string)
        super().__init__(partial(TheGameGame, *args, **kwargs), strats, Metrics)


@games_library.register("guessnumber", args_from=GuessNumberGame)
class GuessNumber(GameDesc):
    def __init__(self, num_symbols: int = 2, secret: int | None = None):
        from boardrl.games.guessnumber.metrics import Metrics

        super().__init__(
            partial(GuessNumberGame, num_symbols=num_symbols, secret=secret),
            strategy_from_string,
            Metrics,
        )


@games_library.register("rps")
class RPS(GameDesc):
    def __init__(self):
        from boardrl.games.rps.game import RockPaperScissors
        from boardrl.games.rps.metrics import Metrics

        super().__init__(RockPaperScissors, strategy_from_string, Metrics)

