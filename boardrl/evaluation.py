"""Evaluation mechanics without opponent-selection policy."""

from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import dataclass, field
import random

import torch

from boardrl.games.semantics import CompetitiveOutcome
from boardrl.rollouts import Inference, RolloutRunner, Rollouts


@dataclass(frozen=True)
class ModelEvaluation:
    metrics: dict[str, object]
    primary_metric: str
    primary_value: float
    selection_score: float
    temperature: float

    def checkpoint_metadata(self):
        return {
            "evaluation_metric": self.primary_metric,
            "evaluation_value": self.primary_value,
            "evaluation_score": self.selection_score,
            "evaluation_temperature": self.temperature,
        }


class ModelEvaluator:
    """Evaluate one policy and compare checkpoints in the same metric."""

    metric_name: str

    def evaluate(self, inference, **kwargs) -> ModelEvaluation:
        with inference.evaluating():
            return self._evaluate(inference, **kwargs)

    def _evaluate(self, inference, **kwargs) -> ModelEvaluation:
        raise NotImplementedError

    def _saved_score(self, state, model_name, temperature):
        metadata = state["metadata"]
        if (
            metadata.get("evaluation_metric") == self.metric_name
            and metadata.get("evaluation_temperature") == temperature
        ):
            return metadata["evaluation_score"]
        return None

    def restore_best_score(self, checkpoints, inference, *, model_name, **kwargs):
        if checkpoints.latest is None:
            return float("-inf")
        state = checkpoints.load(map_location="cpu")
        score = self._saved_score(state, model_name, kwargs.get("temperature", 1.0))
        if score is not None:
            return score
        # Reevaluate legacy or differently configured results without touching
        # the live model, optimizer, or random initialization state.
        model = copy.deepcopy(inference.model)
        model.load_state_dict(state["models"][model_name])
        device = next(model.parameters()).device
        devices = [device.index] if device.type == "cuda" else []
        random_state = random.getstate()
        try:
            with torch.random.fork_rng(devices=devices):
                result = self.evaluate(
                    Inference(model, batch_size=inference.batch_size), **kwargs
                )
        finally:
            random.setstate(random_state)
        return result.selection_score


class SampledModelEvaluator(ModelEvaluator):
    """Preserve the game's existing opponent or self-play evaluation."""

    def __init__(self, game, *, progress=True):
        self.game = game
        self.metric_name = game.rewards.evaluation_metric
        self._opponents = {}
        self.evaluator = Evaluator(
            game.make_game, progress=progress, outcome=game.outcome
        )

    def _saved_score(self, state, model_name, temperature):
        metadata = state["metadata"]
        if "evaluation_metric" not in metadata and "evaluation_points" in metadata:
            # Legacy scores used one key; the value head identifies the metric.
            points_based = state["model_specs"][model_name].get("points_based")
            if {True: "points", False: "win_rate"}.get(points_based) == self.metric_name:
                return metadata["evaluation_points"]
        return super()._saved_score(state, model_name, temperature)

    def _opponent(self, spec):
        if spec not in self._opponents:
            self._opponents[spec] = self.game.strategy_from_string(spec)
        return self._opponents[spec]

    def _evaluate(
        self,
        inference,
        *,
        games,
        max_steps,
        opponent=None,
        names=None,
        temperature=1.0,
        rotate=True,
    ) -> ModelEvaluation:
        player = inference.policy(temperature=temperature)
        other = player if opponent is None else self._opponent(opponent)
        result = self.evaluator.compare(
            [player, other],
            names=names,
            games=games,
            max_steps=max_steps,
            rotate=rotate,
        )
        score = self.game.rewards.evaluation_score(result)
        return ModelEvaluation(
            {
                "win_rate": result.win_rate(),
                **self.game.scores.evaluation_metrics(result),
            },
            self.metric_name,
            score,
            score,
            temperature,
        )


@dataclass(frozen=True)
class Evaluation:
    names: tuple[str, ...]
    rollouts: Rollouts
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
                    score = (
                        1.0
                        if points[i] > points[j]
                        else 0.5 if points[i] == points[j] else 0.0
                    )
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
