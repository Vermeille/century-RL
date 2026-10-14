from boardrl.games.openspiel.dark_hex.game import DarkHex


class AbruptDarkHex(DarkHex):
    def __init__(self, num_players: int = 2, **kwargs):
        super().__init__(num_players=num_players, gameversion="adh", **kwargs)
