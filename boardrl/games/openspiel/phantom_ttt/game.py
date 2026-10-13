from boardrl.games.openspiel.game import OpenSpiel


class PhantomTicTacToe(OpenSpiel):
    def __init__(self, num_players: int = 2, **kwargs):
        super().__init__(
            "phantom_ttt(gameversion=classical,obstype=reveal-nothing)",
            num_players=num_players,
            information_state=True,
            **kwargs,
        )

    @staticmethod
    def evaluation_backend(game):
        from boardrl.games.openspiel.phantom_ttt.evaluation import PhantomTicTacToeEvaluator

        return PhantomTicTacToeEvaluator(game)
