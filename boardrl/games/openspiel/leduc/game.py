from boardrl.games.openspiel.game import OpenSpiel
from boardrl.games.openspiel.leduc.representation import LeducStateText


class Leduc(OpenSpiel):
    text_type = LeducStateText

    def __init__(self, num_players: int = 2, **kwargs):
        super().__init__(
            "leduc_poker", num_players=num_players, information_state=True, **kwargs
        )
