"""Dark Hex information-state text without the opponent's hidden attempt count."""


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
