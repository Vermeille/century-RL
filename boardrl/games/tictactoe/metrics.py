import crayons  # type: ignore[import-untyped]
from boardrl.metrics import GameMetrics


class Metrics(GameMetrics):
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

    def metrics(self):
        ratio_complete = sum(
            " " not in players[0][-1].state for players in self.data
        ) / len(self.data)
        return {
            "ratio_complete": ratio_complete,
            "sensitivity": self.data.sensitivity(),
        }
