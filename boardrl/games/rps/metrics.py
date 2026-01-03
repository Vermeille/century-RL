import torch


class Metrics:
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

    def metrics_to_visdom(self, viz, epoch):
        totals = {"rock": [], "paper": [], "scissors": []}
        for game in self.data.only_strategy([0]):
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
        viz.push("move_probabilities", avg_probs, epoch)
        viz.push("collapse", self.data.collapse(), epoch)
