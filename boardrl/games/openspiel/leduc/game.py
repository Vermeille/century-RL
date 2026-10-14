from boardrl.games.openspiel.game import OpenSpiel


class Leduc(OpenSpiel):
    def __init__(self, num_players: int = 2, *, information_state=True, **kwargs):
        super().__init__(
            "leduc_poker",
            num_players=num_players,
            information_state=information_state,
            **kwargs,
        )

    def _display_player_view(self, text: str, player: int) -> str:
        # Native Observer identifies the viewer; Player identifies the actor.
        return text
