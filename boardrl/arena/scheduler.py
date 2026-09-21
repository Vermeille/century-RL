"""Placement and background matchup scheduling."""

from __future__ import annotations

import itertools

from .ratings import OPPONENT_ID, RANDOM_ID, RatingFit, pair_aggregates, residuals
from .store import ArenaStore


class MatchScheduler:
    def __init__(
        self,
        store: ArenaStore,
        *,
        reference_id: str = OPPONENT_ID,
        placement_batches: int = 3,
    ):
        self.store = store
        self.reference_id = reference_id
        self.placement_batches = placement_batches

    def placement(self, fit: RatingFit) -> tuple[str, str, str] | None:
        waiting = [
            policy
            for policy in self.store.policies(checkpoints_only=True, playable_only=True)
            if policy.placement_batches < self.placement_batches
        ]
        if not waiting:
            return None
        policy = min(waiting, key=lambda item: (item.step or 0, item.id))
        batch = policy.placement_batches
        active = [
            candidate
            for candidate in self.store.policies(
                checkpoints_only=True, playable_only=True
            )
            if candidate.id != policy.id and candidate.placement_batches >= 1
        ]
        if batch == 0 or not active:
            opponent = self.reference_id
        elif batch == 1:
            opponent = min(
                active,
                key=lambda candidate: (
                    abs(fit.ratings[policy.id] - fit.ratings[candidate.id]),
                    candidate.id,
                ),
            ).id
        else:
            opponent = max(
                active,
                key=lambda candidate: (
                    abs((policy.step or 0) - (candidate.step or 0)),
                    candidate.id,
                ),
            ).id
        return policy.id, opponent, f"placement:{policy.id}:{batch}"

    def idle(self, fit: RatingFit) -> tuple[str, str, str] | None:
        checkpoints = self.store.policies(checkpoints_only=True, playable_only=True)
        if not checkpoints:
            return None
        base_policies = [RANDOM_ID]
        if self.reference_id != RANDOM_ID:
            base_policies.append(self.reference_id)
        candidates = [*base_policies, *(policy.id for policy in checkpoints)]
        pairs = [
            pair
            for pair in itertools.combinations(candidates, 2)
            if len(base_policies) == 1 or set(pair) != set(base_policies)
        ]
        idle_batch = self.store.get_int("idle_batches")
        aggregates = pair_aggregates(self.store.matches())
        pair_residuals = residuals(self.store.matches(), fit)

        if (idle_batch + 1) % 5 == 0:
            suspicious_pair = max(
                pairs,
                key=lambda pair: abs(
                    pair_residuals.get(tuple(sorted(pair)), 0.0)
                ),
            )
            suspicious_residual = abs(
                pair_residuals.get(tuple(sorted(suspicious_pair)), 0.0)
            )

            def audit_priority(pair):
                ordered = tuple(sorted(pair))
                direct = aggregates.get(ordered)
                games = direct.games if direct else 0
                first = self.store.policy(pair[0])
                second = self.store.policy(pair[1])
                distance = abs((first.step or 0) - (second.step or 0))
                balanced = 1.0 - abs(fit.expected_score(*pair) - 0.5) * 2.0
                return -games, distance, balanced, ordered

            if suspicious_residual >= 2.0:
                first, second = suspicious_pair
                purpose = "confirmation"
            else:
                first, second = max(pairs, key=audit_priority)
                purpose = "audit"
        else:

            def information_priority(pair):
                probability = fit.expected_score(*pair)
                information = probability * (1.0 - probability)
                variance = fit.pair_variance(*pair)
                games = self.store.games_between(*pair)
                return variance * information / (1.0 + games / 32.0), tuple(
                    sorted(pair)
                )

            first, second = max(pairs, key=information_priority)
            purpose = "refinement"
        self.store.set_int("idle_batches", idle_batch + 1)
        return first, second, purpose
