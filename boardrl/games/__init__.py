from functools import partial
from importlib import import_module
from importlib.util import find_spec

from boardrl.utils import RegisterByName
from boardrl.games.augmentations import shuffle_actions
from boardrl.games.strategies import strategy_from_string
from boardrl.games.connectfour.augmentations import (
    horizontal_symmetry as connectfour_horizontal_symmetry,
)
from boardrl.games.connectfour.game import ConnectFour as ConnectFourGame
from boardrl.games.connectfour.strategies import (
    strategy_from_string as connectfour_strategy_from_string,
)
from boardrl.games.guessnumber.game import GuessNumber as GuessNumberGame
from boardrl.games.hanabi.game import Hanabi as HanabiGame
from boardrl.games.nim.game import Nim as NimGame
from boardrl.games.nim.strategies import (
    strategy_from_string as nim_strategy_from_string,
)
from boardrl.games.regicide.game import Regicide as RegicideGame
from boardrl.games.rps.game import RockPaperScissors as RockPaperScissorsGame
from boardrl.games.santorini.augmentations import (
    horizontal_symmetry as santorini_horizontal_symmetry,
    rotation_symmetry as santorini_rotation_symmetry,
    vertical_symmetry as santorini_vertical_symmetry,
)
from boardrl.games.santorini.game import Santorini as SantoriniGame
from boardrl.games.skullking.game import SkullKing as SkullKingGame
from boardrl.games.splendor.game import Splendor as SplendorGame
from boardrl.games.splendor.semantics import SplendorOutcome
from boardrl.games.sum.game import Sum as SumGame
from boardrl.games.sum.strategies import (
    strategy_from_string as sum_strategy_from_string,
)
from boardrl.games.take5.game import Take5 as Take5Game
from boardrl.games.thegame.augmentations import shuffle_hand
from boardrl.games.thegame.game import TheGame as TheGameGame
from boardrl.games.thegame.strategies import (
    strategy_from_string as thegame_strategy_from_string,
)
from boardrl.games.tictactoe.game import TicTacToe as TicTacToeGame
from boardrl.games.semantics import (
    CompetitiveOutcome,
    CooperativeOutcome,
    OutcomeScores,
    PointScores,
    TerminalOutcomeRewards,
    PointDeltaRewards,
)


class GameDesc:
    def __init__(
        self,
        game_class,
        strategy_from_string,
        metrics_class,
        augmentations=(),
        *,
        scores,
        outcome,
        rewards,
        model_evaluation=None,
    ):
        self.make_game = game_class
        self.strategy_from_string = strategy_from_string
        self.make_metrics = metrics_class
        self.augmentations = tuple(augmentations)
        self.scores = scores
        self.outcome = outcome
        self.rewards = rewards
        self.coop = outcome.cooperative
        self.model_evaluation = model_evaluation

    def make_model_evaluator(self, *, progress=True, sampled=False):
        from boardrl.evaluation import SampledModelEvaluator

        factory = (
            SampledModelEvaluator
            if sampled
            else self.model_evaluation or SampledModelEvaluator
        )
        return factory(self, progress=progress)


games_library = RegisterByName()


def _metrics_for(game_class):
    package = game_class.__module__.rsplit(".", 1)[0]
    return import_module(f"{package}.metrics").Metrics


def _no_registry_args():
    pass


def register_game(
    name,
    game_class,
    *,
    strategies=None,
    augmentations=(),
    scores,
    outcome,
    rewards,
    args_from=None,
    metrics_from=None,
    model_evaluation=None,
):
    """Register a game; metrics_from can select a shared metrics module."""

    def descriptor(**game_kwargs):
        game_strategies = strategy_from_string
        if strategies is not None:
            game_strategies = strategies.copy().update(strategy_from_string)
        make_game = partial(game_class, **game_kwargs) if game_kwargs else game_class
        return GameDesc(
            make_game,
            game_strategies,
            _metrics_for(metrics_from or game_class),
            augmentations=augmentations,
            scores=scores,
            outcome=outcome,
            rewards=rewards,
            model_evaluation=model_evaluation,
        )

    descriptor.__name__ = f"{name}_game_desc"
    return games_library.register(
        name,
        args_from=args_from or _no_registry_args,
    )(descriptor)


@games_library.register("century")
class Century(GameDesc):
    # Century is defined in Cython so inspect can't find the signature
    # so we have to repeat the arguments here and it sucks
    def __init__(self, goal_cards: int = -1, num_players: int = 2):
        from boardrl.games.century.augmentations import shuffle_discard, shuffle_hand
        from boardrl.games.century.strategies import (
            strategy_from_string as century_strategy_from_string,
        )
        from boardrl.games.century.metrics import Metrics
        import pyximport  # type: ignore

        pyximport.install(setup_args={"script_args": ["--cython-cplus"]})
        from boardrl.games.century.engine import Century as CenturyGame  # type: ignore

        strats = century_strategy_from_string.copy().update(strategy_from_string)
        super().__init__(
            partial(CenturyGame, goal_cards=goal_cards, num_players=num_players),
            strats,
            Metrics,
            augmentations=(shuffle_actions, shuffle_hand, shuffle_discard),
            scores=PointScores(),
            outcome=CompetitiveOutcome(),
            rewards=TerminalOutcomeRewards(),
        )


if find_spec("pyspiel") is not None:
    from boardrl.games.openspiel import make_model_evaluator as openspiel_evaluator
    from boardrl.games.openspiel.abrupt_dark_hex.game import AbruptDarkHex
    from boardrl.games.openspiel.abrupt_phantom_ttt.game import AbruptPhantomTicTacToe
    from boardrl.games.openspiel.dark_hex.game import DarkHex
    from boardrl.games.openspiel.game import OpenSpiel
    from boardrl.games.openspiel.leduc.game import Leduc
    from boardrl.games.openspiel.liars_dice.game import LiarsDice
    from boardrl.games.openspiel.othello.game import Othello
    from boardrl.games.openspiel.phantom_ttt.game import PhantomTicTacToe

    register_game(
        "leduc",
        Leduc,
        augmentations=(shuffle_actions,),
        scores=PointScores(),
        outcome=CompetitiveOutcome(),
        rewards=PointDeltaRewards(scale=0.1),
        metrics_from=OpenSpiel,
        model_evaluation=openspiel_evaluator,
    )

    register_game(
        "liars_dice",
        LiarsDice,
        augmentations=(shuffle_actions,),
        scores=OutcomeScores(),
        outcome=CompetitiveOutcome(),
        rewards=TerminalOutcomeRewards(),
        metrics_from=OpenSpiel,
        model_evaluation=openspiel_evaluator,
    )

    register_game(
        "dark_hex",
        DarkHex,
        augmentations=(shuffle_actions,),
        scores=OutcomeScores(),
        outcome=CompetitiveOutcome(),
        rewards=TerminalOutcomeRewards(),
        metrics_from=OpenSpiel,
        model_evaluation=openspiel_evaluator,
    )

    register_game(
        "abrupt_dark_hex",
        AbruptDarkHex,
        augmentations=(shuffle_actions,),
        scores=OutcomeScores(),
        outcome=CompetitiveOutcome(),
        rewards=TerminalOutcomeRewards(),
        metrics_from=OpenSpiel,
        model_evaluation=openspiel_evaluator,
    )

    register_game(
        "phantom_ttt",
        PhantomTicTacToe,
        augmentations=(shuffle_actions,),
        scores=OutcomeScores(),
        outcome=CompetitiveOutcome(),
        rewards=TerminalOutcomeRewards(),
        metrics_from=OpenSpiel,
        model_evaluation=openspiel_evaluator,
    )

    register_game(
        "abrupt_phantom_ttt",
        AbruptPhantomTicTacToe,
        augmentations=(shuffle_actions,),
        scores=OutcomeScores(),
        outcome=CompetitiveOutcome(),
        rewards=TerminalOutcomeRewards(),
        metrics_from=OpenSpiel,
        model_evaluation=openspiel_evaluator,
    )

    register_game(
        "othello",
        Othello,
        augmentations=(shuffle_actions,),
        scores=OutcomeScores(),
        outcome=CompetitiveOutcome(),
        rewards=TerminalOutcomeRewards(),
        metrics_from=OpenSpiel,
    )


register_game(
    "tictactoe",
    TicTacToeGame,
    augmentations=(shuffle_actions,),
    scores=OutcomeScores(),
    outcome=CompetitiveOutcome(),
    rewards=TerminalOutcomeRewards(),
)

register_game(
    "connectfour",
    ConnectFourGame,
    strategies=connectfour_strategy_from_string,
    augmentations=(shuffle_actions, connectfour_horizontal_symmetry),
    scores=OutcomeScores(),
    outcome=CompetitiveOutcome(),
    rewards=TerminalOutcomeRewards(),
)

register_game(
    "sum",
    SumGame,
    strategies=sum_strategy_from_string,
    augmentations=(shuffle_actions,),
    scores=PointScores(),
    outcome=CompetitiveOutcome(),
    rewards=TerminalOutcomeRewards(),
)

register_game(
    "thegame",
    TheGameGame,
    strategies=thegame_strategy_from_string,
    augmentations=(shuffle_actions, shuffle_hand),
    scores=PointScores(),
    outcome=CooperativeOutcome(),
    rewards=PointDeltaRewards(scale=0.1),
    args_from=TheGameGame,
)

register_game(
    "guessnumber",
    GuessNumberGame,
    augmentations=(shuffle_actions,),
    scores=PointScores(),
    outcome=CooperativeOutcome(),
    rewards=TerminalOutcomeRewards(),
    args_from=GuessNumberGame,
)

register_game(
    "rps",
    RockPaperScissorsGame,
    augmentations=(shuffle_actions,),
    scores=OutcomeScores(),
    outcome=CompetitiveOutcome(),
    rewards=TerminalOutcomeRewards(),
    args_from=RockPaperScissorsGame,
)

register_game(
    "nim",
    NimGame,
    strategies=nim_strategy_from_string,
    augmentations=(shuffle_actions,),
    scores=OutcomeScores(),
    outcome=CompetitiveOutcome(),
    rewards=TerminalOutcomeRewards(),
    args_from=NimGame,
)

register_game(
    "take5",
    Take5Game,
    augmentations=(shuffle_actions,),
    scores=PointScores(),
    outcome=CompetitiveOutcome(),
    rewards=TerminalOutcomeRewards(),
    args_from=Take5Game,
)

register_game(
    "skullking",
    SkullKingGame,
    augmentations=(shuffle_actions,),
    scores=PointScores(),
    outcome=CompetitiveOutcome(),
    rewards=TerminalOutcomeRewards(),
    args_from=SkullKingGame,
)

register_game(
    "regicide",
    RegicideGame,
    augmentations=(shuffle_actions,),
    scores=PointScores(),
    outcome=CooperativeOutcome(),
    rewards=PointDeltaRewards(scale=0.01),
    args_from=RegicideGame,
)

register_game(
    "hanabi",
    HanabiGame,
    augmentations=(shuffle_actions,),
    scores=PointScores(),
    outcome=CooperativeOutcome(),
    rewards=PointDeltaRewards(scale=0.1),
    args_from=HanabiGame,
)

register_game(
    "santorini",
    SantoriniGame,
    augmentations=(
        shuffle_actions,
        santorini_horizontal_symmetry,
        santorini_vertical_symmetry,
        santorini_rotation_symmetry,
    ),
    scores=OutcomeScores(),
    outcome=CompetitiveOutcome(),
    rewards=TerminalOutcomeRewards(),
)

register_game(
    "splendor",
    SplendorGame,
    augmentations=(shuffle_actions,),
    scores=PointScores(),
    outcome=SplendorOutcome(),
    rewards=TerminalOutcomeRewards(),
    args_from=SplendorGame,
)
