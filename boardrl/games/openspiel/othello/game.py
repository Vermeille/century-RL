from boardrl.games.openspiel.game import OpenSpiel


class Othello(OpenSpiel):
    def __init__(self, num_players: int = 2, **kwargs):
        super().__init__(
            "othello", num_players=num_players, information_state=False, **kwargs
        )
