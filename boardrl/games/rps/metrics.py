import torch
from boardrl.metrics import GameMetrics


class Metrics(GameMetrics):
    def __init__(self, data):
        self.data = data

    def print_short_history(self):
        for game in self.data:
            for player in game:
                rec = player[0]
                probs = torch.softmax(rec.action_distribution, dim=0)
                print(
                    f"{rec.moves[rec.action_idx]} {probs.tolist()} {player[-1].my_points}"
                )
            print("--")

    def metrics(self):
        totals = {"rock": [], "paper": [], "scissors": []}
        for game in self.data:
            for player in game:
                rec = player[0]
                probs = torch.softmax(rec.action_distribution, dim=0).tolist()
                for move, prob in zip(rec.moves, probs):
                    totals[move].append(prob)
        avg_probs = [
            sum(totals["rock"]) / len(totals["rock"]) if totals["rock"] else 0.0,
            sum(totals["paper"]) / len(totals["paper"]) if totals["paper"] else 0.0,
            sum(totals["scissors"]) / len(totals["scissors"])
            if totals["scissors"]
            else 0.0,
        ]
        return {
            "move_probabilities": avg_probs,
            "collapse": self.data.collapse(),
        }
