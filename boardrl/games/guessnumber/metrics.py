from boardrl.metrics import GameMetrics


class Metrics(GameMetrics):
    def __init__(self, data):
        self.data = data

    def print_short_history(self):
        for game in self.data:
            print(game[0][0].state.split("\n")[0])
            for player in game:
                print([hh.moves[hh.action_idx] for hh in player[:-1]])
            print("--")

    def metrics(self):
        return {
            "num_rounds": sum(len(trace) for trace in self.data.all_traces())
            / self.data.num_traces()
        }
