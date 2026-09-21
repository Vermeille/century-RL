from boardrl.games.semantics import CompetitiveOutcome


class SplendorOutcome(CompetitiveOutcome):
    """Terminal utility including Splendor's fewest-development-card tiebreak."""

    def utility(self, game, player: int, terminal: bool) -> float:
        if not terminal or game.stalemate():
            return 0.0
        winners = game.winners()
        if len(winners) != 1:
            return 0.0 if player in winners else -1.0
        return 1.0 if player == winners[0] else -1.0
