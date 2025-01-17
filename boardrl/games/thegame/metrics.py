class Metrics:
    def __init__(self, data):
        self.data = data

    def print_short_history(self):
        pass

    def metrics_to_visdom(self, viz, epoch):
        avg_points = sum(p[0][-1].score for p in self.data) / len(self.data)
        viz.push("avg_points", avg_points, epoch)
