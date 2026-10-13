"""Preserve native Leduc text and correct its player tag at termination."""

import re


class LeducStateText:
    def __init__(self, source):
        self.source = source

    def display(self, state, player):
        text = self.source.display(state, player)
        # OpenSpiel leaves its internal last actor in both players' final views.
        if state.is_terminal():
            return re.sub(r"\[Player: -?\d+\]", f"[Player: {player}]", text)
        return text
