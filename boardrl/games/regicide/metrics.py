from boardrl.metrics import GameMetrics, Range


class Metrics(GameMetrics):
    def __init__(self, data):
        self.data = data

    def metrics(self):
        points = [game[0][-1].my_points for game in self.data]
        rounds = [game[0][-1].round for game in self.data]
        return {
            "points": Range(points),
            "rounds": Range(rounds),
        }
