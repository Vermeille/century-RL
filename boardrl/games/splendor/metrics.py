from boardrl.games.semantics import PointScores
from boardrl.metrics import GameMetrics, GroupedTraceMetrics, TraceMetrics


class Metrics(GameMetrics):
    def __init__(self, data):
        self.data = data

    def print_short_history(self):
        for game in self.data:
            terminal = game.by_seat[0][-1]
            print(terminal.state, terminal.my_points)
            print()

    def metrics(self):
        terminal_games = sum(game.by_seat[0][-1].terminal for game in self.data)
        return {
            "terminal_rate": terminal_games / len(self.data),
            "avg_game_actions": sum(
                max(len(trace) - 1, 0)
                for game in self.data
                for trace in game.by_seat
            )
            / len(self.data),
            "strategy": GroupedTraceMetrics(
                self.data.by_strategy, TraceMetrics, scores=PointScores()
            ).metrics(),
        }
