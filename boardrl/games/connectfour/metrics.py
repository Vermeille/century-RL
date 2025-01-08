import crayons


class Metrics:
    def __init__(self, data):
        self.data = data

    def print_short_history(self):
        for h1, h2 in zip(self.data[::2], self.data[1::2]):
            print(
                "".join(hh.moves[hh.action_idx] for hh in h1[:-1]),
                h1[-1].current_diff_points,
            )
            print(
                "".join(hh.moves[hh.action_idx] for hh in h2[:-1]),
                h2[-1].current_diff_points,
            )
            print(
                h1[-1]
                .state.replace("O", str(crayons.green("O")))
                .replace("X", str(crayons.red("X"))),
                h1[-1].my_points,
            )
            print()

    def metrics_to_visdom(self, viz, epoch):
        ratio_complete = sum(" " not in h[-1].state for h in self.data) / len(self.data)
        viz.push("ratio_complete", ratio_complete, epoch)
        avg_len = sum(len(h) for h in self.data) / len(self.data)
        viz.push("avg_len", avg_len, epoch)
