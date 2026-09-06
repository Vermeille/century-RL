from itertools import accumulate

from boardrl.games.regicide.game import ENEMY_HP
from boardrl.metrics import GameMetrics, Range


ENEMY_DEFEAT_THRESHOLDS = tuple(
    accumulate(
        [ENEMY_HP[rank] for rank in ("J",) * 4 + ("Q",) * 4 + ("K",) * 4]
    )
)


def enemies_defeated(points: float) -> int:
    return sum(points >= threshold for threshold in ENEMY_DEFEAT_THRESHOLDS)


class Metrics(GameMetrics):
    def __init__(self, data):
        self.data = data

    def metrics(self):
        points = [game[0][-1].my_points for game in self.data]
        rounds = [game[0][-1].round for game in self.data]
        return {
            "points": Range(points),
            "enemies_defeated": Range([enemies_defeated(score) for score in points]),
            "rounds": Range(rounds),
        }
