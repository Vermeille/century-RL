from boardrl.games.openspiel.game import OpenSpiel
from boardrl.games.openspiel.dark_hex.representation import DarkHexStateText


class AbruptDarkHex(OpenSpiel):
    text_type = DarkHexStateText

    def __init__(self, num_players: int = 2, **kwargs):
        super().__init__(
            "dark_hex(board_size=3,gameversion=adh,obstype=reveal-nothing)",
            num_players=num_players,
            information_state=True,
            **kwargs,
        )
