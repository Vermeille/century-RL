import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from boardrl.rollouts import GameTrace, PlayerTrace, Rollouts


def load_trainer(filename, module_name):
    path = Path(__file__).parents[1] / "trainers" / filename
    spec = importlib.util.spec_from_file_location(f"trainers.{module_name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


adversarial_advshape = load_trainer(
    "adversarial-advshape.py",
    "adversarial_advshape",
)


def trace(strategy_id, points):
    player = PlayerTrace(seat_id=strategy_id, strategy_id=strategy_id)
    player.append(SimpleNamespace())
    player.append(SimpleNamespace(terminal=True, current_diff_points=points))
    return player


def test_defaults_to_requested_advantage_thresholds():
    args = adversarial_advshape.build_parser().parse_args([])

    assert args.tag == "adversarial-advshape"
    assert args.agent_threshold == 0.98
    assert args.environment_threshold == 0.7


def test_trackio_run_name_is_unique_for_each_process(monkeypatch):
    calls = []
    monkeypatch.setattr(
        adversarial_advshape,
        "make_trackio",
        lambda **kwargs: calls.append(kwargs) or SimpleNamespace(finish=lambda: None),
    )
    monkeypatch.setattr(adversarial_advshape, "_run", lambda args, sink: None)

    args = adversarial_advshape.build_parser().parse_args(
        ["--trackio", "--tag", "same-tag"]
    )
    adversarial_advshape.run(args)

    assert calls[0]["name"] == args.trackio_run_name
    assert calls[0]["name"].startswith("same-tag-")


def test_prepare_reuses_complete_rollout_predictions():
    args = SimpleNamespace(
        inference_batch_size=8,
        discount=1.0,
        gae_lambda=0.0,
        value_lambda=1.0,
    )

    prepare = adversarial_advshape.make_prepare(
        object(), args, strategy_id=0, threshold=0.7
    )
    reference_targets = next(
        step
        for step in prepare.steps
        if isinstance(step, adversarial_advshape.ReferenceTargets)
    )

    assert reference_targets.reuse_rollout_predictions


def test_advantage_shaping_uses_strategy_batch_win_rate():
    games = Rollouts(
        [
            GameTrace([trace(0, 1.0), trace(1, -1.0)]),
            GameTrace([trace(0, -1.0), trace(1, 1.0)]),
            GameTrace([trace(0, -1.0), trace(1, 1.0)]),
            GameTrace([trace(0, -1.0), trace(1, 1.0)]),
        ]
    )
    sample = SimpleNamespace(gae=2.0, normalized_gae=4.0)
    scale = adversarial_advshape.ScaleAdvantages()
    pipeline = adversarial_advshape.AdvantageShapingPipeline(
        lambda value: [sample],
        strategy_id=0,
        threshold=0.75,
        scale=scale,
    )

    result = pipeline(games)

    assert result == [sample]
    assert scale.factor == pytest.approx(0.5)
    assert sample.gae == pytest.approx(1.0)
    assert sample.normalized_gae == pytest.approx(2.0)


def test_advantage_shaping_reverses_above_threshold():
    games = Rollouts(
        [GameTrace([trace(0, 1.0), trace(1, -1.0)])]
    )
    sample = SimpleNamespace(gae=2.0, normalized_gae=-3.0)
    scale = adversarial_advshape.ScaleAdvantages()
    pipeline = adversarial_advshape.AdvantageShapingPipeline(
        lambda value: [sample],
        strategy_id=0,
        threshold=0.7,
        scale=scale,
    )

    pipeline(games)

    assert scale.factor == pytest.approx(-0.3)
    assert sample.gae == pytest.approx(-0.6)
    assert sample.normalized_gae == pytest.approx(0.9)
