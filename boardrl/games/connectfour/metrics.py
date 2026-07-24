import torch
import crayons  # type: ignore[import-untyped]
from boardrl.metrics import GameMetrics


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
        ratio_complete = sum(" " not in h[0][-1].state for h in self.data) / len(
            self.data
        )
        avg_len = sum(len(h) for h in self.data.all_traces()) / (self.data.num_traces())
        winning_games = [
            p for players in self.data for p in players if p[-1].current_diff_points > 0
        ]
        winning_probability = (
            sum(
                    torch.softmax(w[-2].action_distribution, 0)[w[-2].action_idx]
                    for w in winning_games
                )
            / len(winning_games)
            if winning_games
            else 0.0
        )
        return {
            "ratio_complete": ratio_complete,
            "avg_len": avg_len,
            "collapse": self.data.collapse(),
            "avg_winning_move_probability": winning_probability,
        }
