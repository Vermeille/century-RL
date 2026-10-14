"""Dark Hex information-state text without the opponent's hidden attempt count."""

from boardrl.games.openspiel.representation import (
    ObservationText,
    PlayerViewText,
    TerminalStateText,
)


class DarkHexInformationStateText:
    """Remove OpenSpiel's extra global clock from its reveal-nothing string.

    The tensor used by arXiv:2502.08938 contains player identity, the private
    board and ordered own attempts, but no count of the opponent's failures.
    """

    def __init__(self, rows):
        self.rows = rows

    def display(self, state, player):
        lines = state.information_state_string(player).split("\n")
        return self.private_board(
            lines[: self.rows], "\n".join(lines[self.rows + 1 :]), player
        )

    def private_board(self, rows, history, player):
        return "\n".join(rows) + "\n" + history


class DarkHexStateText(TerminalStateText):
    def __init__(self, game, information_state):
        source = (
            DarkHexInformationStateText(game.get_parameters()["num_rows"])
            if information_state
            else ObservationText()
        )
        super().__init__(PlayerViewText(source))
