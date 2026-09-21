"""Event detection, alerts, and complete Trackio publication."""

from __future__ import annotations

import math
import time
from typing import Iterable

from .cycles import (
    MAXIMUM_REPORTED_CYCLES,
    confirmed_cycles,
    confirmed_win_graph,
    find_directed_cycles,
)
from .events import ArenaEvent, TrackioTelemetry, render_event_log
from .ratings import (
    OPPONENT_ID,
    RANDOM_ID,
    RATING_POINTS_PER_ODDS_DOUBLING,
    RatingFit,
    pair_aggregates,
)
from .store import ArenaStore
from .visualization import (
    plot_dominance_graph,
    plot_rating_curve,
    plot_residuals,
)


class ArenaReporter:
    """Turns rating state into persisted events and Trackio observability."""

    def __init__(
        self,
        store: ArenaStore,
        *,
        telemetry: TrackioTelemetry | None,
        reference_id: str,
        opponent_name: str,
        placement_batches: int,
    ):
        self.store = store
        self.telemetry = telemetry
        self.reference_id = reference_id
        self.opponent_name = opponent_name
        self.placement_batches = placement_batches

    @staticmethod
    def _rating_interval(fit: RatingFit, policy_id: str) -> str:
        rating = fit.ratings[policy_id]
        radius = 1.96 * fit.deviations[policy_id]
        return (
            f"{rating:.1f} (95% posterior interval "
            f"[{rating - radius:.1f}, {rating + radius:.1f}])"
        )

    def _record_checkpoint_events(self, fit: RatingFit):
        completed = sorted(
            (
                policy
                for policy in self.store.policies(checkpoints_only=True)
                if policy.placement_batches >= self.placement_batches
            ),
            key=lambda policy: (policy.step or 0, policy.id),
        )
        last_assessed = self.store.get_int("last_assessed_checkpoint_step", -1)
        best_id = self.store.get_text("best_checkpoint")
        if best_id not in fit.ratings:
            best_id = None

        for policy in completed:
            step = policy.step or 0
            if step <= last_assessed:
                continue
            if best_id is None or fit.ratings[policy.id] > fit.ratings[best_id]:
                previous = best_id
                text = (
                    f"Checkpoint step {step} established a new best posterior rating "
                    f"of {self._rating_interval(fit, policy.id)}."
                )
                if previous is not None:
                    improvement = fit.ratings[policy.id] - fit.ratings[previous]
                    text += (
                        f" Previous best was {previous.replace('checkpoint-', 'step ')} "
                        f"at {fit.ratings[previous]:.1f}; the posterior-mean gain is "
                        f"{improvement:.1f} points."
                    )
                self.store.record_event(
                    ArenaEvent(
                        f"new-best:{policy.id}",
                        "new_best",
                        f"New arena best at step {step}",
                        text,
                        "info",
                        step,
                        time.time(),
                    )
                )
                best_id = policy.id
                self.store.set_text("best_checkpoint", best_id)
            last_assessed = step
            self.store.set_int("last_assessed_checkpoint_step", step)

        if best_id is None:
            return
        best_step = self.store.policy(best_id).step or 0
        for policy in completed:
            step = policy.step or 0
            if step <= best_step:
                continue
            drop = fit.ratings[best_id] - fit.ratings[policy.id]
            deviation = math.sqrt(fit.pair_variance(best_id, policy.id))
            lower_drop = drop - 1.96 * deviation
            if drop >= 100.0 and lower_drop > 0.0:
                odds = 2.0 ** (drop / RATING_POINTS_PER_ODDS_DOUBLING)
                self.store.record_event(
                    ArenaEvent(
                        f"regression:{policy.id}:{best_id}",
                        "regression",
                        f"Strong regression at step {step}",
                        f"Checkpoint step {step} rates {drop:.1f} points below "
                        f"the prior best {best_id.replace('checkpoint-', 'step ')}; "
                        f"the 95% lower bound on the drop is {lower_drop:.1f} points. "
                        f"The posterior means imply about {odds:.1f}:1 expected-score "
                        "odds for the prior best.",
                        "warn",
                        step,
                        time.time(),
                    )
                )

    def _record_cycle_events(
        self,
        cycles: Iterable[tuple[str, ...]],
        *,
        step: int,
    ):
        aggregates = pair_aggregates(self.store.matches())
        for cycle in cycles:
            route = " → ".join((*cycle, cycle[0]))
            evidence = []
            for index, winner in enumerate(cycle):
                loser = cycle[(index + 1) % len(cycle)]
                match = aggregates[tuple(sorted((winner, loser)))]
                score = (
                    match.score
                    if match.first == winner
                    else match.games - match.score
                )
                evidence.append(
                    f"{winner} beats {loser} at {score / match.games:.1%} "
                    f"over {match.games} games"
                )
            self.store.record_event(
                ArenaEvent(
                    "cycle:" + "->".join(cycle),
                    "cycle",
                    f"Confirmed policy cycle ({len(cycle)} policies)",
                    f"Confirmed head-to-head edges form {route}. This can indicate "
                    "cycling learning dynamics that a scalar rating cannot represent. "
                    + "; ".join(evidence)
                    + ".",
                    "warn",
                    step,
                    time.time(),
                )
            )

    def _publish_pending_alerts(self):
        if self.telemetry is None:
            return
        pending = self.store.pending_alerts()
        cycle_events = [event for event in pending if event.kind == "cycle"]
        for event in (event for event in pending if event.kind != "cycle"):
            try:
                self.telemetry.alert(event)
            except Exception as exc:
                print(
                    f"arena: Trackio alert failed for {event.key}: {exc}",
                    flush=True,
                )
            else:
                self.store.mark_alerted(event.key)
        if not cycle_events:
            return
        if len(cycle_events) == 1:
            alert = cycle_events[0]
        else:
            suffix = (
                f" The search is capped at {MAXIMUM_REPORTED_CYCLES} cycles."
                if len(cycle_events) >= MAXIMUM_REPORTED_CYCLES
                else ""
            )
            alert = ArenaEvent(
                "cycle-summary",
                "cycle",
                f"{len(cycle_events)} new confirmed policy cycles",
                "Multiple confirmed cycles appeared in the directed head-to-head "
                f"graph.{suffix} Inspect arena/event_log for their routes.",
                "warn",
                max(event.step for event in cycle_events),
                time.time(),
            )
        try:
            self.telemetry.alert(alert)
        except Exception as exc:
            print(f"arena: Trackio cycle alert failed: {exc}", flush=True)
        else:
            for event in cycle_events:
                self.store.mark_alerted(event.key)

    def publish_calibration(self, fit: RatingFit):
        """Publish the reference-only curve before any checkpoint is rated."""

        if (
            self.telemetry is None
            or OPPONENT_ID not in fit.ratings
            or self.store.get_int("calibration_published")
        ):
            return
        figure = plot_rating_curve(
            self.store,
            fit,
            opponent_name=self.opponent_name,
            include_checkpoints=False,
        )
        try:
            try:
                self.telemetry.log(
                    {
                        "arena/rating_curve": figure,
                    },
                    step=0,
                )
            except Exception as exc:
                print(
                    f"arena: Trackio calibration publication failed: {exc}",
                    flush=True,
                )
            else:
                self.store.set_int("calibration_published", 1)
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)

    def publish(self, fit: RatingFit):
        checkpoint_policies = self.store.policies(checkpoints_only=True)
        if not checkpoint_policies:
            return
        matches = self.store.matches()
        active_ids = [
            policy.id
            for policy in self.store.policies(
                checkpoints_only=True, playable_only=True
            )
        ]
        completed_ids = [
            policy.id
            for policy in checkpoint_policies
            if policy.placement_batches >= self.placement_batches
        ]
        cycle_ids = [RANDOM_ID, *completed_ids]
        coverage_ids = [RANDOM_ID, *active_ids]
        if self.reference_id == OPPONENT_ID:
            cycle_ids.append(OPPONENT_ID)
            coverage_ids.append(OPPONENT_ID)
        cycles = confirmed_cycles(matches, cycle_ids)
        active_graph = confirmed_win_graph(matches, coverage_ids)
        active_cycles = find_directed_cycles(active_graph)
        latest_step = max(policy.step or 0 for policy in checkpoint_policies)
        self._record_checkpoint_events(fit)
        self._record_cycle_events(cycles, step=latest_step)
        total_games = sum(match.games for match in matches)
        already_published = (
            self.store.get_int("last_published_step", -1) == latest_step
            and self.store.get_int("last_published_games", -1) == total_games
        )
        if self.telemetry is None:
            print(
                f"arena: publish step={latest_step} "
                f"opponent={fit.ratings.get(OPPONENT_ID, 0.0):.1f} "
                f"games={total_games} pool={len(active_ids)}",
                flush=True,
            )
            return
        self._publish_pending_alerts()
        if already_published and not self.store.pending_alerts():
            return
        print(
            f"arena: publish step={latest_step} "
            f"opponent={fit.ratings.get(OPPONENT_ID, 0.0):.1f} "
            f"games={total_games} pool={len(active_ids)}",
            flush=True,
        )
        rating_figure = plot_rating_curve(
            self.store,
            fit,
            opponent_name=self.opponent_name,
        )
        residual_figure = plot_residuals(
            self.store,
            fit,
            reference_id=self.reference_id,
        )
        dominance_figure = plot_dominance_graph(
            self.store,
            active_graph,
            active_cycles,
        )
        try:
            try:
                self.telemetry.log(
                    {
                        "arena/rating_curve": rating_figure,
                        "arena/nontransitivity": residual_figure,
                        "arena/dominance_graph": dominance_figure,
                        "arena/event_log": self.telemetry.html(
                            render_event_log(self.store.events())
                        ),
                    },
                    step=latest_step,
                )
            except Exception as exc:
                print(f"arena: Trackio publication failed: {exc}", flush=True)
            else:
                self.store.set_int("last_published_step", latest_step)
                self.store.set_int("last_published_games", total_games)
        finally:
            import matplotlib.pyplot as plt

            plt.close(rating_figure)
            plt.close(residual_figure)
            plt.close(dominance_figure)
