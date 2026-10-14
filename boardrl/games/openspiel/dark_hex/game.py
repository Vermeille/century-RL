from boardrl.games.openspiel.game import OpenSpiel


class DarkHex(OpenSpiel):
    def __init__(self, num_players: int = 2, *, gameversion="cdh", **kwargs):
        super().__init__(
            f"dark_hex(board_size=3,gameversion={gameversion},obstype=reveal-nothing)",
            num_players=num_players,
            information_state=True,
            **kwargs,
        )

    def _display_state(self, player: int) -> str:
        # The paper's information tensor excludes the opponent's attempt count.
        lines = super()._display_state(player).split("\n")
        rows = self._game.get_parameters()["num_rows"]
        return "\n".join(lines[:rows] + lines[rows + 1 :])
