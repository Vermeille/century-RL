import torch
import crayons  # type: ignore[import-untyped]
from boardrl.metrics import GameMetrics, GroupedTraceMetrics, TraceMetrics


class StrategyMetrics(TraceMetrics):
    """Policy metrics that only make sense for a stable strategy identity."""

    def winning_move_probability(self) -> float:
        winning_traces = [
            trace
            for trace in self.traces
            if trace[-1].current_diff_points > 0
        ]
        if not winning_traces:
            return float("nan")
        probabilities = [
            torch.softmax(trace[-2].action_distribution, 0)[
                trace[-2].action_idx
            ].item()
            for trace in winning_traces
        ]
        return sum(probabilities) / len(probabilities)

    def metrics(self):
        return super().metrics() | {
            "winning_games": sum(
                trace[-1].current_diff_points > 0 for trace in self.traces
            ),
            "avg_winning_move_probability": self.winning_move_probability(),
        }


class Metrics(GameMetrics):
    def __init__(self, data):
        self.data = data

    def print_short_history(self):
        for players in self.data:
            for player in players:
                print(
                    "".join(hh.moves[hh.action_idx] for hh in player[:-1]),
                    player[-1].current_diff_points,
                )
            print(
                players[0][-1]
                .state.replace("O", str(crayons.green("O")))
                .replace("X", str(crayons.red("X"))),
                players[0][-1].my_points,
            )
            print()

    def metrics(self):
        terminal_games = sum(
            game.by_seat[0][-1].terminal for game in self.data
        )
        draws = sum(
            game.by_seat[0][-1].terminal
            and game.by_seat[0][-1].current_diff_points == 0
            for game in self.data
        )
        return {
            "terminal_rate": terminal_games / len(self.data),
            "draw_rate": draws / len(self.data),
            "avg_game_actions": sum(
                max(len(trace) - 1, 0)
                for game in self.data
                for trace in game.by_seat
            )
            / len(self.data),
            "strategy": GroupedTraceMetrics(
                self.data.by_strategy, StrategyMetrics
            ).metrics(),
            "seat": GroupedTraceMetrics(self.data.by_seat).metrics(),
        }
