from boardrl.games.openspiel.game import OpenSpiel


class DarkHex(OpenSpiel):
    def __init__(self, num_players: int = 2, **kwargs):
        super().__init__(
            "dark_hex(board_size=3,gameversion=cdh,obstype=reveal-nothing)",
            num_players=num_players,
            information_state=True,
            **kwargs,
        )
