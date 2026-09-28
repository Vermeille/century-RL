from boardrl.games.semantics import OutcomeScores
from boardrl.metrics import GameMetrics, GroupedTraceMetrics


class Metrics(GameMetrics):
    def __init__(self, data):
        self.data = data

    def print_short_history(self):
        for game in self.data:
            for trace in game.by_seat:
                moves = [record.moves[record.action_idx] for record in trace[:-1]]
                print(f"p{trace.seat_id}: {' '.join(moves)} -> {trace[-1].current_diff_points}")

    def metrics(self):
        if not self.data:
            return {"games": 0}
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
                self.data.by_strategy, scores=OutcomeScores()
            ).metrics(),
        }
