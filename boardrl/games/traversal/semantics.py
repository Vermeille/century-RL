from boardrl.games.semantics import CompetitiveOutcome


class TraversalOutcome(CompetitiveOutcome):
    """Preserve advancement tiebreaks and repetition draws independently of points."""

    def utility(self, game, player: int, terminal: bool) -> float:
        if not terminal or game.winner() is None:
            return 0.0
        return 1.0 if game.winner() == player else -1.0
