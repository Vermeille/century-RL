"""Leduc preserves OpenSpiel's native actor and observer fields."""

from boardrl.games.openspiel.representation import (
    InformationStateText,
    ObservationText,
    TerminalStateText,
)


class LeducStateText(TerminalStateText):
    def __init__(self, game, information_state):
        source = InformationStateText() if information_state else ObservationText()
        super().__init__(source)
