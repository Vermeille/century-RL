"""Bayesian Bradley-Terry rating model used by the arena."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch


RANDOM_ID = "random"
OPPONENT_ID = "evaluation-opponent"
RATING_POINTS_PER_ODDS_DOUBLING = 100.0
LOGISTIC_SCALE = math.log(2.0) / RATING_POINTS_PER_ODDS_DOUBLING


@dataclass(frozen=True)
class Match:
    first: str
    second: str
    games: int
    score: float


@dataclass(frozen=True)
class RatingFit:
    ratings: dict[str, float]
    deviations: dict[str, float]
    covariance: dict[tuple[str, str], float]

    def expected_score(self, first: str, second: str) -> float:
        difference = self.ratings[first] - self.ratings[second]
        return 1.0 / (1.0 + 2.0 ** (-difference / RATING_POINTS_PER_ODDS_DOUBLING))

    def pair_variance(self, first: str, second: str) -> float:
        return max(
            0.0,
            self.covariance.get((first, first), 0.0)
            + self.covariance.get((second, second), 0.0)
            - 2.0 * self.covariance.get((first, second), 0.0),
        )


def fit_ratings(
    policy_ids: Iterable[str],
    matches: Iterable[Match],
    *,
    anchor: str = RANDOM_ID,
    prior_deviation: float = 600.0,
) -> RatingFit:
    """Fit an anchored Bradley-Terry posterior with a Gaussian prior."""

    ids = sorted(set(policy_ids))
    if anchor not in ids:
        ids.insert(0, anchor)
    unknown = [policy_id for policy_id in ids if policy_id != anchor]
    index = {policy_id: position for position, policy_id in enumerate(unknown)}
    records = [match for match in matches if match.games > 0]

    if not unknown:
        return RatingFit({anchor: 0.0}, {anchor: 0.0}, {(anchor, anchor): 0.0})

    ratings = torch.zeros(len(unknown), dtype=torch.float64)
    prior_precision = 1.0 / prior_deviation**2

    def value_gradient_hessian(values):
        objective = 0.5 * prior_precision * values.square().sum()
        gradient = prior_precision * values.clone()
        hessian = torch.eye(len(unknown), dtype=torch.float64) * prior_precision

        for match in records:
            first = 0.0 if match.first == anchor else values[index[match.first]]
            second = 0.0 if match.second == anchor else values[index[match.second]]
            log_odds = LOGISTIC_SCALE * (first - second)
            probability = torch.sigmoid(log_odds)
            objective = objective + match.games * torch.nn.functional.softplus(
                log_odds
            ) - match.score * log_odds
            difference_gradient = LOGISTIC_SCALE * (
                match.games * probability - match.score
            )
            weight = (
                LOGISTIC_SCALE**2
                * match.games
                * probability
                * (1.0 - probability)
            )
            first_index = index.get(match.first)
            second_index = index.get(match.second)
            if first_index is not None:
                gradient[first_index] += difference_gradient
                hessian[first_index, first_index] += weight
            if second_index is not None:
                gradient[second_index] -= difference_gradient
                hessian[second_index, second_index] += weight
            if first_index is not None and second_index is not None:
                hessian[first_index, second_index] -= weight
                hessian[second_index, first_index] -= weight
        return objective, gradient, hessian

    for _ in range(100):
        objective, gradient, hessian = value_gradient_hessian(ratings)
        update = torch.linalg.solve(hessian, gradient)
        if float(update.abs().max()) < 1e-8:
            break
        scale = 1.0
        while scale > 1e-6:
            candidate = ratings - scale * update
            candidate_objective, _, _ = value_gradient_hessian(candidate)
            if candidate_objective <= objective:
                ratings = candidate
                break
            scale *= 0.5
        else:
            break

    _, _, hessian = value_gradient_hessian(ratings)
    posterior_covariance = torch.linalg.inv(hessian)
    fitted = {anchor: 0.0} | {
        policy_id: float(ratings[position])
        for policy_id, position in index.items()
    }
    deviations = {anchor: 0.0} | {
        policy_id: math.sqrt(max(float(posterior_covariance[position, position]), 0.0))
        for policy_id, position in index.items()
    }
    covariance: dict[tuple[str, str], float] = {}
    for first in ids:
        for second in ids:
            if anchor in (first, second):
                covariance[first, second] = 0.0
            else:
                covariance[first, second] = float(
                    posterior_covariance[index[first], index[second]]
                )
    return RatingFit(fitted, deviations, covariance)


def pair_aggregates(matches: Iterable[Match]) -> dict[tuple[str, str], Match]:
    aggregate: dict[tuple[str, str], list[float]] = {}
    for match in matches:
        first, second = sorted((match.first, match.second))
        games, score = aggregate.setdefault((first, second), [0.0, 0.0])
        aggregate[(first, second)][0] = games + match.games
        aggregate[(first, second)][1] = score + (
            match.score if match.first == first else match.games - match.score
        )
    return {
        pair: Match(pair[0], pair[1], int(values[0]), values[1])
        for pair, values in aggregate.items()
    }


def residuals(matches: Iterable[Match], fit: RatingFit) -> dict[tuple[str, str], float]:
    result = {}
    for pair, match in pair_aggregates(matches).items():
        observed = (match.score + 0.5) / (match.games + 1.0)
        expected = fit.expected_score(match.first, match.second)
        standard_error = math.sqrt(
            max(expected * (1.0 - expected) / max(match.games, 1), 1e-12)
        )
        result[pair] = (observed - expected) / standard_error
    return result
