from boardrl.games.openspiel.game import OpenSpiel


class LiarsDice(OpenSpiel):
    """Two players, each rolling two five-sided dice (2D5F)."""

    def __init__(self, num_players: int = 2, **kwargs):
        super().__init__(
            "liars_dice(players=2,numdice=2,dice_sides=5,bidding_rule=reset-face)",
            num_players=num_players,
            information_state=True,
            **kwargs,
        )
