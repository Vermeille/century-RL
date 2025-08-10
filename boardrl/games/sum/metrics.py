class Metrics:
    def __init__(self, data):
        self.data = data

    def print_short_history(self):
        for game in self.data:
            for player in game:
                print([hh.my_points for hh in player[:-1]])
            print("--")

    def metrics_to_visdom(self, viz, epoch):
        avg_len = sum(len(h) for p in self.data for h in p) / (
            len(self.data) * len(self.data[0])
        )
        viz.push("avg_len", avg_len, epoch)
        viz.push("collapse", self.data.collapse(), epoch)
