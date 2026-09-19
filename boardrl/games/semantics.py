"""Composable score, outcome, and training-reward semantics for games."""

from __future__ import annotations

from functools import partial


class Outcome:
    def utility(self, game, player: int, terminal: bool) -> float:
        raise NotImplementedError

    def won(self, game, terminal: bool):
        raise NotImplementedError

    def win_rate(self, rollouts, player: int) -> float:
        raise NotImplementedError


class CompetitiveOutcome(Outcome):
    cooperative = False

    def utility(self, game, player: int, terminal: bool) -> float:
        if not terminal:
            return 0.0
        mine = game.points_for(player)
        opponents = (
            game.points_for(other)
            for other in range(game.num_players)
            if other != player
        )
        best_opponent = max(opponents, default=mine)
        return float(mine > best_opponent) - float(mine < best_opponent)

    def won(self, game, terminal: bool):
        return None

    def win_rate(self, rollouts, player: int) -> float:
        return rollouts.by_strategy.group(player).win_rate()


class CooperativeOutcome(Outcome):
    cooperative = True

    def utility(self, game, player: int, terminal: bool) -> float:
        if not terminal:
            return 0.0
        return 1.0 if game.won() else -1.0

    def won(self, game, terminal: bool) -> bool:
        return game.won() if terminal else False

    def win_rate(self, rollouts, player: int) -> float:
        return rollouts.objective_win_rate()


class Scores:
    def trace_metrics(self, traces) -> dict[str, object]:
        raise NotImplementedError

    def evaluation_metrics(self, evaluation) -> dict[str, object]:
        raise NotImplementedError


class OutcomeScores(Scores):
    def trace_metrics(self, traces) -> dict[str, object]:
        return {}

    def evaluation_metrics(self, evaluation) -> dict[str, object]:
        return {}


class PointScores(Scores):
    def trace_metrics(self, traces) -> dict[str, object]:
        from boardrl.metrics import Range

        return {"points": Range(traces.points())}

    def evaluation_metrics(self, evaluation) -> dict[str, object]:
        from boardrl.metrics import Range

        return {"points": Range(evaluation.rollouts.my_points(0))}


class Rewards:
    points_value_head: bool

    def make_returns(self, discount, *, entropy_bonus=None):
        raise NotImplementedError

    def evaluation_score(self, evaluation) -> float:
        raise NotImplementedError


class TerminalOutcomeRewards(Rewards):
    points_value_head = False

    def make_returns(self, discount, *, entropy_bonus=None):
        from boardrl.training.postprocess import ComputeReturns
        from boardrl.training.returns import set_episodic_rewards

        return ComputeReturns(
            discount,
            entropy_bonus=entropy_bonus,
            reward_fn=set_episodic_rewards,
        )

    def evaluation_score(self, evaluation) -> float:
        return evaluation.win_rate()


class PointDeltaRewards(Rewards):
    points_value_head = True

    def __init__(self, scale: float):
        self.scale = scale

    def make_returns(self, discount, *, entropy_bonus=None):
        from boardrl.training.postprocess import ComputeReturns
        from boardrl.training.returns import set_rewards

        return ComputeReturns(
            discount,
            entropy_bonus=entropy_bonus,
            reward_fn=partial(set_rewards, scale=self.scale),
        )

    def evaluation_score(self, evaluation) -> float:
        return evaluation.avg_points()
