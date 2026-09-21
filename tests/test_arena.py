from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from boardrl.arena import (
    OPPONENT_ID,
    RANDOM_ID,
    ArenaStore,
    Match,
    MatchScheduler,
    Policy,
    RatingFit,
    capture_agent_checkpoint,
    choose_retained_pool,
    confirmed_cycles,
    confirmed_win_graph,
    find_directed_cycles,
    fit_ratings,
)


def test_rating_scale_doubles_expected_score_odds_every_100_points():
    fit = RatingFit(
        ratings={"a": 100.0, "b": 0.0},
        deviations={"a": 0.0, "b": 0.0},
        covariance={},
    )

    assert fit.expected_score("a", "b") == pytest.approx(2 / 3)
    fit.ratings["a"] = 200.0
    assert fit.expected_score("a", "b") == pytest.approx(4 / 5)


def test_bayesian_rating_is_anchored_order_independent_and_allows_negatives():
    matches = [
        Match("strong", RANDOM_ID, 300, 240.0),
        Match("weak", RANDOM_ID, 300, 60.0),
        Match("strong", "weak", 300, 270.0),
    ]

    forward = fit_ratings([RANDOM_ID, "strong", "weak"], matches)
    reverse = fit_ratings(
        ["weak", RANDOM_ID, "strong"],
        [Match(match.first, match.second, match.games, match.score) for match in reversed(matches)],
    )

    assert forward.ratings[RANDOM_ID] == 0.0
    assert forward.ratings["strong"] > 0.0
    assert forward.ratings["weak"] < 0.0
    assert forward.ratings == pytest.approx(reverse.ratings)


def test_draws_are_half_a_point_and_perfect_records_remain_finite():
    drawn = fit_ratings(
        [RANDOM_ID, "drawn"],
        [Match("drawn", RANDOM_ID, 64, 32.0)],
    )
    perfect = fit_ratings(
        [RANDOM_ID, "perfect"],
        [Match("perfect", RANDOM_ID, 64, 64.0)],
    )

    assert drawn.ratings["drawn"] == pytest.approx(0.0, abs=1e-9)
    assert math.isfinite(perfect.ratings["perfect"])
    assert math.isfinite(perfect.deviations["perfect"])


def test_more_games_shrink_posterior_uncertainty():
    small = fit_ratings(
        [RANDOM_ID, "agent"],
        [Match("agent", RANDOM_ID, 32, 24.0)],
    )
    large = fit_ratings(
        [RANDOM_ID, "agent"],
        [Match("agent", RANDOM_ID, 320, 240.0)],
    )

    assert large.deviations["agent"] < small.deviations["agent"]


def test_cycle_detection_requires_well_sampled_confirmed_edges():
    cycle = [
        Match("a", "b", 64, 48.0),
        Match("b", "c", 64, 48.0),
        Match("c", "a", 64, 48.0),
    ]

    assert confirmed_cycles(cycle, ["a", "b", "c"]) == [("a", "b", "c")]
    assert confirmed_cycles(cycle[:2], ["a", "b", "c"]) == []


def test_directed_cycle_detection_finds_cycles_longer_than_triangles():
    graph = confirmed_win_graph(
        [
            Match("a", "b", 64, 48.0),
            Match("b", "c", 64, 48.0),
            Match("c", "d", 64, 48.0),
            Match("d", "a", 64, 48.0),
        ],
        ["a", "b", "c", "d"],
    )

    assert find_directed_cycles(graph) == [("a", "b", "c", "d")]


def test_checkpoint_capture_keeps_only_agent_and_validates_metadata(tmp_path):
    source = tmp_path / "step-12.pth"
    destination = tmp_path / "pool" / "agent-step-12.pth"
    torch.save(
        {
            "step": 12,
            "models": {"agent": {"weight": torch.tensor([1])}, "environment": {}},
            "model_specs": {"agent": {"dim": 1}, "environment": {"dim": 2}},
            "optimizers": {"agent": {"large": "state"}},
            "states": {"agent_learner": {"large": "state"}},
            "metadata": {"trainer": "adversarial-advshape", "game": "connectfour"},
        },
        source,
    )

    policy_id, step = capture_agent_checkpoint(
        source, destination, expected_game="connectfour"
    )
    compact = torch.load(destination, weights_only=False)

    assert (policy_id, step) == ("checkpoint-12", 12)
    assert compact["models"].keys() == {"agent"}
    assert compact["model_specs"].keys() == {"agent"}
    assert compact["optimizers"] == {}
    assert compact["states"] == {}


def test_pool_retention_is_bounded_deterministic_and_covers_priorities(tmp_path):
    policies = []
    ratings = {RANDOM_ID: 0.0}
    deviations = {RANDOM_ID: 0.0}
    covariance = {(RANDOM_ID, RANDOM_ID): 0.0}
    for step in range(40):
        path = tmp_path / f"{step}.pth"
        path.touch()
        policy_id = f"checkpoint-{step}"
        policies.append(Policy(policy_id, "checkpoint", step, path, True, 3))
        ratings[policy_id] = float(step)
        deviations[policy_id] = 1000.0 if step == 4 else float(step)
        covariance[policy_id, policy_id] = deviations[policy_id] ** 2
    fit = RatingFit(ratings, deviations, covariance)
    residuals = {("checkpoint-2", "checkpoint-3"): 12.0}

    first = choose_retained_pool(policies, fit, residuals, limit=32)
    second = choose_retained_pool(reversed(policies), fit, residuals, limit=32)

    assert first == second
    assert len(first) == 32
    assert "checkpoint-39" in first
    assert "checkpoint-4" in first
    assert "checkpoint-2" in first or "checkpoint-3" in first


def _add_placed_policy(store: ArenaStore, tmp_path: Path, policy_id: str, step: int):
    path = tmp_path / f"{policy_id}.pth"
    path.touch()
    store.add_policy(policy_id, kind="checkpoint", step=step, path=path)
    for batch in range(3):
        store.record_match(
            batch_id=f"{policy_id}-{batch}",
            first=policy_id,
            second=RANDOM_ID,
            games=32,
            score=16.0,
            seed=batch,
            purpose=f"placement:{policy_id}:{batch}",
        )


def test_scheduler_places_against_reference_nearest_then_step_distant(tmp_path):
    store = ArenaStore(tmp_path / "arena.sqlite")
    store.add_policy(RANDOM_ID, kind="strategy")
    store.add_policy(OPPONENT_ID, kind="strategy")
    _add_placed_policy(store, tmp_path, "checkpoint-10", 10)
    _add_placed_policy(store, tmp_path, "checkpoint-90", 90)
    waiting_path = tmp_path / "waiting.pth"
    waiting_path.touch()
    store.add_policy("checkpoint-100", kind="checkpoint", step=100, path=waiting_path)
    scheduler = MatchScheduler(store)
    fit = RatingFit(
        ratings={
            RANDOM_ID: 0.0,
            OPPONENT_ID: 0.0,
            "checkpoint-10": -200.0,
            "checkpoint-90": 5.0,
            "checkpoint-100": 0.0,
        },
        deviations={},
        covariance={},
    )

    first = scheduler.placement(fit)
    assert first == ("checkpoint-100", OPPONENT_ID, "placement:checkpoint-100:0")
    store.record_match(
        batch_id="waiting-0",
        first=first[0],
        second=first[1],
        games=32,
        score=16.0,
        seed=0,
        purpose=first[2],
    )
    assert scheduler.placement(fit)[1] == "checkpoint-90"
    store.record_match(
        batch_id="waiting-1",
        first="checkpoint-100",
        second="checkpoint-90",
        games=32,
        score=16.0,
        seed=1,
        purpose="placement:checkpoint-100:1",
    )
    assert scheduler.placement(fit)[1] == "checkpoint-10"
    store.close()


def test_random_evaluation_opponent_reuses_zero_reference(tmp_path):
    store = ArenaStore(tmp_path / "arena.sqlite")
    store.add_policy(RANDOM_ID, kind="strategy")
    path = tmp_path / "agent.pth"
    path.touch()
    store.add_policy("checkpoint-1", kind="checkpoint", step=1, path=path)
    scheduler = MatchScheduler(store, reference_id=RANDOM_ID)
    fit = fit_ratings([RANDOM_ID, "checkpoint-1"], [])

    assert scheduler.placement(fit)[1] == RANDOM_ID
    store.close()


def test_idle_scheduler_uses_four_refinements_then_an_audit(tmp_path):
    store = ArenaStore(tmp_path / "arena.sqlite")
    store.add_policy(RANDOM_ID, kind="strategy")
    store.add_policy(OPPONENT_ID, kind="strategy")
    _add_placed_policy(store, tmp_path, "checkpoint-10", 10)
    _add_placed_policy(store, tmp_path, "checkpoint-90", 90)
    fit = fit_ratings(
        [policy.id for policy in store.policies()],
        store.matches(),
    )
    scheduler = MatchScheduler(store)

    purposes = [scheduler.idle(fit)[2] for _ in range(5)]

    assert purposes == ["refinement"] * 4 + ["audit"]
    store.close()


def test_replayed_match_batch_is_recorded_only_once(tmp_path):
    store = ArenaStore(tmp_path / "arena.sqlite")
    store.add_policy(RANDOM_ID, kind="strategy")
    path = tmp_path / "agent.pth"
    path.touch()
    store.add_policy("checkpoint-1", kind="checkpoint", step=1, path=path)
    arguments = {
        "batch_id": "deterministic-batch",
        "first": "checkpoint-1",
        "second": RANDOM_ID,
        "games": 32,
        "score": 16.0,
        "seed": 123,
        "purpose": "placement:checkpoint-1:0",
    }

    assert store.record_match(**arguments)
    assert not store.record_match(**arguments)
    assert store.policy("checkpoint-1").placement_batches == 1
    assert store.games_between("checkpoint-1", RANDOM_ID) == 32
    store.close()


def test_idle_scheduler_confirms_large_pairwise_residual(tmp_path):
    store = ArenaStore(tmp_path / "arena.sqlite")
    store.add_policy(RANDOM_ID, kind="strategy")
    store.add_policy(OPPONENT_ID, kind="strategy")
    _add_placed_policy(store, tmp_path, "checkpoint-10", 10)
    _add_placed_policy(store, tmp_path, "checkpoint-90", 90)
    store.record_match(
        batch_id="surprise",
        first="checkpoint-10",
        second="checkpoint-90",
        games=32,
        score=32.0,
        seed=9,
        purpose="refinement",
    )
    store.set_int("idle_batches", 4)
    fit = RatingFit(
        ratings={
            RANDOM_ID: 0.0,
            OPPONENT_ID: 0.0,
            "checkpoint-10": 0.0,
            "checkpoint-90": 0.0,
        },
        deviations={},
        covariance={},
    )

    first, second, purpose = MatchScheduler(store).idle(fit)

    assert {first, second} == {"checkpoint-10", "checkpoint-90"}
    assert purpose == "confirmation"
    store.close()
