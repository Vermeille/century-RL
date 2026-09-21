import torch
from boardrl.metrics import GameMetrics
from boardrl.rollouts import Rollouts


class Metrics(GameMetrics):
    def __init__(self, data: Rollouts) -> None:
        self.data = data

    def print_short_history(self):
        for game in self.data:
            for player in game:
                moves = [hh.moves[hh.action_idx] for hh in player[:-1]]
                print(moves)
            print("--")

    def _chosen_move_probabilities(self) -> list[float]:
        max_steps = max(len(game.by_strategy[0]) - 1 for game in self.data)
        by_step: list[list[float]] = [[] for _ in range(max_steps)]
        for game in self.data:
            player = game.by_strategy[0]
            for step, rec in enumerate(player[:-1]):
                probs = torch.softmax(rec.action_distribution, dim=0)
                by_step[step].append(probs[rec.action_idx].item())

        return [
            sum(step_probs) / len(step_probs)
            for step_probs in by_step
            if step_probs
        ]

    def metrics(self):
        avg_len = sum(len(h) for p in self.data for h in p) / (
            len(self.data) * len(self.data[0])
        )
        return {
            "avg_len": avg_len,
            "sensitivity": self.data.sensitivity(),
            "chosen_move_probability_by_match": self._chosen_move_probabilities(),
        }
