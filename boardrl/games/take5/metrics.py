class Metrics:
    def __init__(self, data):
        self.data = data

    def print_short_history(self):
        for game in self.data:
            for player in game:
                print([record.moves[record.action_idx] for record in player[:-1]])
            print("--")

    def metrics_to_visdom(self, viz, epoch):
        avg_len = sum(len(history) for game in self.data for history in game) / (
            len(self.data) * self.data.num_players()
        )
        viz.push("avg_len", avg_len, epoch)
        viz.push("collapse", self.data.collapse(), epoch)
