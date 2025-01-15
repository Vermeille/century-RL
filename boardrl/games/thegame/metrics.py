class Metrics:
    def __init__(self, data):
        self.data = data

    def print_short_history(self):
        pass

    def metrics_to_visdom(self, viz, epoch):
        avg_points = sum(h[-1].my_points for h in self.data) / len(self.data)
        viz.push("avg_points", avg_points, epoch)
