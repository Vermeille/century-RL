from boardrl.games.openspiel.game import OpenSpiel


class Leduc(OpenSpiel):
    def __init__(self, num_players: int = 2, **kwargs):
        super().__init__(
            "leduc_poker", num_players=num_players, information_state=True, **kwargs
        )

    @staticmethod
    def evaluation_backend(game):
        from boardrl.games.openspiel.evaluation import OpenSpielTreeEvaluator

        return OpenSpielTreeEvaluator(game)
