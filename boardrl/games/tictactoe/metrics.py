import crayons


class Metrics:
    def __init__(self, data):
        self.data = data

    def print_short_history(self):
        for players in self.data:
            h = players.by_strategy[0]
            print(
                h[-1]
                .state.replace("O", str(crayons.green("O")))
                .replace("X", str(crayons.red("X"))),
                h[-1].my_points,
            )
            print()

    def metrics_to_visdom(self, viz, epoch):
        ratio_complete = sum(
            " " not in players[0][-1].state for players in self.data
        ) / len(self.data)
        viz.push("ratio_complete", ratio_complete, epoch)
        viz.push("collapse", self.data.collapse(), epoch)
