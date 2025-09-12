class Metrics:
    def __init__(self, data):
        self.data = data

    def print_short_history(self):
        for game in self.data:
            for player in game:
                print([hh.my_points for hh in player[:-1]])
            print("--")
