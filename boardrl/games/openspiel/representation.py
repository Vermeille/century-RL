"""Composable player-visible text for OpenSpiel game adapters."""


class InformationStateText:
    def display(self, state, player):
        return state.information_state_string(player)

    def private_board(self, rows, history, player):
        return "\n".join(rows) + "\n" + history


class ObservationText:
    def display(self, state, player):
        return state.observation_string(player)


class PlayerViewText:
    """Supply a player tag for games whose native text has none."""

    def __init__(self, source):
        self.source = source

    @staticmethod
    def for_player(text, player):
        return f"[Player: {player}]\n{text}"

    def display(self, state, player):
        return self.for_player(self.source.display(state, player), player)

    def private_board(self, rows, history, player):
        return self.for_player(self.source.private_board(rows, history, player), player)


class TerminalStateText:
    """Keep the normal player view and append its settled result at game end."""

    def __init__(self, source):
        self.source = source

    def display(self, state, player):
        text = self.source.display(state, player)
        if not state.is_terminal():
            return text
        returns = state.returns()
        payoff = returns[player]
        other = max(
            (value for seat, value in enumerate(returns) if seat != player),
            default=payoff,
        )
        result = "win" if payoff > other else "loss" if payoff < other else "draw"
        return f"{text}\nGame: ended\nResult: {result}\nPayoff: {payoff:g}"

    def private_board(self, rows, history, player):
        return self.source.private_board(rows, history, player)


class PlayerStateText(TerminalStateText):
    """Native player view with identity and terminal outcome."""

    def __init__(self, game, information_state):
        source = InformationStateText() if information_state else ObservationText()
        super().__init__(PlayerViewText(source))
