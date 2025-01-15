class Metrics:
    def __init__(self, data):
        self.data = data

    def print_short_history(self):
        for h1, h2 in zip(self.data[::2], self.data[1::2]):
            print([hh.my_points for hh in h1[:-1]], [hh.my_points for hh in h2[:-1]])

    def metrics_to_visdom(self, viz, epoch):
        avg_len = sum(len(h) for h in self.data) / len(self.data)
        viz.push("avg_len", avg_len, epoch)
