from boardrl.games.semantics import CompetitiveOutcome


class VersusOutcome(CompetitiveOutcome):
    """Reaching 2048 or blocking the slider decides victory, independently of score."""

    def utility(self, game, player: int, terminal: bool) -> float:
        if not terminal:
            return 0.0
        return 1.0 if game.winner() == player else -1.0
