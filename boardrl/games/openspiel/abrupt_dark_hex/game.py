from boardrl.games.openspiel.game import OpenSpiel


class AbruptDarkHex(OpenSpiel):
    def __init__(self, num_players: int = 2, **kwargs):
        super().__init__(
            "dark_hex(board_size=3,gameversion=adh,obstype=reveal-nothing)",
            num_players=num_players,
            information_state=True,
            **kwargs,
        )

    @staticmethod
    def evaluation_backend(game):
        from boardrl.games.openspiel.abrupt_dark_hex.evaluation import AbruptDarkHexEvaluator

        return AbruptDarkHexEvaluator(game)
