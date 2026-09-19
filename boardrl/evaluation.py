"""Evaluation mechanics without opponent-selection policy."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from boardrl.rl.eval.selfplay import SelfPlayResults
from boardrl.rollouts import RolloutRunner
from boardrl.games.semantics import CompetitiveOutcome


@dataclass(frozen=True)
class Evaluation:
    names: tuple[str, ...]
    rollouts: SelfPlayResults
    outcome: object = field(default_factory=CompetitiveOutcome)

    def win_rate(self, player=0) -> float:
        return self.outcome.win_rate(self.rollouts, player)

    def avg_points(self, player=0) -> float:
        return self.rollouts.by_strategy.group(player).avg_points()

    @property
    def games(self) -> int:
        return len(self.rollouts)


class Evaluator:
    def __init__(self, make_game, *, progress=True, outcome=None, coop: bool = False):
        self.runner = RolloutRunner(
            make_game, progress=progress, outcome=outcome, coop=coop
        )
        self.outcome = self.runner.outcome

    def compare(
        self,
        players,
        *,
        names=None,
        games: int,
        max_steps: int,
        rotate: bool = True,
    ) -> Evaluation:
        names = tuple(names or (f"player-{i}" for i in range(len(players))))
        results = self.runner.play(
            players,
            games=games,
            max_steps=max_steps,
            rotate=rotate,
            description="evaluation",
        )
        return Evaluation(names, results, outcome=self.outcome)


class Scoreboard:
    """Optional cumulative head-to-head metrics for experiment code."""

    def __init__(self) -> None:
        self._scores: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(lambda: [0.0, 0.0])
        )

    def record(self, evaluation: Evaluation) -> None:
        names = evaluation.names
        for game in evaluation.rollouts:
            points = [trace[-1].current_diff_points for trace in game.by_strategy]
            for i, first in enumerate(names):
                for j, second in enumerate(names):
                    if i == j:
                        continue
                    score = 1.0 if points[i] > points[j] else 0.5 if points[i] == points[j] else 0.0
                    aggregate = self._scores[first][second]
                    aggregate[0] += score
                    aggregate[1] += 1

    def win_rate(self, first, second) -> float:
        wins, games = self._scores[first][second]
        return wins / games if games else 0.0

    def as_dict(self):
        return {
            first: {
                second: self.win_rate(first, second)
                for second in opponents
            }
            for first, opponents in self._scores.items()
        }
