import torch


class Metrics:
    def __init__(self, data):
        self.data = data

    def print_short_history(self):
        for game in self.data:
            for player in game:
                moves = [hh.moves[hh.action_idx] for hh in player[:-1]]
                print(moves)
            print("--")

    def _chosen_move_probabilities(self):
        by_step: list[list[float]] = [
            [] for _ in range(int(self.data[0].by_seat[0][0].state.split("\n")[0]))
        ]
        for game in self.data:
            player = game.by_strategy[0]
            for rec in player[:-1]:
                match_num = int(rec.state.split("\n")[0])
                probs = torch.softmax(rec.action_distribution, dim=0)
                for i, p in enumerate(probs):
                    by_step[match_num - i - 1].append(p.item())

        return [sum(step_probs) / len(step_probs) for step_probs in by_step]

    def metrics_to_visdom(self, viz, epoch):
        avg_len = sum(len(h) for p in self.data for h in p) / (
            len(self.data) * len(self.data[0])
        )
        viz.push("avg_len", avg_len, epoch)
        viz.push("collapse", self.data.collapse(), epoch)

        move_probs = self._chosen_move_probabilities()
        print(move_probs)
        if move_probs:
            viz.visdom(
                "line",
                Y=torch.tensor(move_probs, dtype=torch.float32),
                X=torch.arange(len(move_probs), dtype=torch.float32),
                opts=dict(
                    title="chosen_move_probability_by_match",
                    # legend=["avg"],
                    xlabel="ply",
                    ylabel="probability",
                ),
                win="chosen_move_probability_by_match",
            )
