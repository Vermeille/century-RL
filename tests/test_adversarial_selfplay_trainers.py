import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from boardrl.arena import capture_agent_checkpoint
from boardrl.rl.model.loss import (
    BootstrapValueLogProbLoss,
    EntropyBonus,
    KLPenalty,
    PolicyGradientLoss,
)
from boardrl.training import ReferenceTargets, ToSamples
from trainers import arena


def load_trainer(filename, module_name):
    path = Path(__file__).parents[1] / "trainers" / filename
    spec = importlib.util.spec_from_file_location(f"trainers.{module_name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


adversarial_selfplay = load_trainer(
    "adversarial_selfplay.py",
    "adversarial_selfplay",
)
adversarial_ppo = load_trainer(
    "adversarial-ppo.py",
    "adversarial_ppo",
)
adversarial_mmd = load_trainer(
    "adversarial-mmd.py",
    "adversarial_mmd",
)


def toy_model():
    return torch.nn.Linear(1, 1)


def toy_game():
    return SimpleNamespace(augmentations=())


def test_baseline_defaults_are_distinct():
    ppo_args = adversarial_ppo.build_parser().parse_args(["--device", "cpu"])
    mmd_args = adversarial_mmd.build_parser().parse_args(["--device", "cpu"])

    assert ppo_args.tag == "adversarial-ppo"
    assert mmd_args.tag == "adversarial-mmd"
    assert mmd_args.mmd_coefficient == pytest.approx(0.05)
    assert mmd_args.mmd_reference_timesteps == 10_000_000


def test_shared_selfplay_prepare_uses_both_seats_and_rollout_reference():
    args = SimpleNamespace(
        inference_batch_size=8,
        discount=1.0,
        gae_lambda=0.0,
        value_lambda=1.0,
    )

    prepare = adversarial_selfplay.make_prepare(object(), args)

    assert isinstance(prepare.steps[0], ToSamples)
    assert isinstance(prepare.steps[1], ReferenceTargets)
    assert prepare.steps[1].reuse_rollout_predictions


def test_raw_ppo_has_no_exploration_or_kl_regularizer():
    args = adversarial_ppo.build_parser().parse_args(["--device", "cpu"])
    learner, _ = adversarial_ppo.make_learner(toy_model(), toy_game(), args)

    assert [type(loss) for loss in learner.losses] == [
        PolicyGradientLoss,
        BootstrapValueLogProbLoss,
    ]


def test_mmd_uses_ppo_entropy_and_reverse_kl():
    args = adversarial_mmd.build_parser().parse_args(["--device", "cpu"])
    learner, _ = adversarial_mmd.make_learner(toy_model(), toy_game(), args)

    assert [type(loss) for loss in learner.losses] == [
        PolicyGradientLoss,
        EntropyBonus,
        KLPenalty,
        BootstrapValueLogProbLoss,
    ]


def test_mmd_schedule_matches_deep_multiagent_recipe():
    alpha, eta, kl_strength = adversarial_mmd.mmd_parameters(10_000_000)
    assert alpha == pytest.approx(0.05)
    assert eta == pytest.approx(0.05)
    assert kl_strength == pytest.approx(20.0)

    alpha, eta, kl_strength = adversarial_mmd.mmd_parameters(2_500_000)
    assert alpha == pytest.approx(0.1)
    assert eta == pytest.approx(0.1)
    assert kl_strength == pytest.approx(10.0)


def test_mmd_timestep_state_survives_checkpoint_round_trip():
    args = adversarial_mmd.build_parser().parse_args(["--device", "cpu"])
    learner, _ = adversarial_mmd.make_learner(toy_model(), toy_game(), args)
    learner.mmd_timesteps = 123_456

    restored, _ = adversarial_mmd.make_learner(toy_model(), toy_game(), args)
    restored.load_state_dict(learner.state_dict())

    assert restored.mmd_timesteps == 123_456


@pytest.mark.parametrize("trainer", ["adversarial-ppo", "adversarial-mmd"])
def test_arena_captures_baseline_checkpoints(tmp_path, trainer):
    source = tmp_path / "step-12.pth"
    destination = tmp_path / "pool" / "agent-step-12.pth"
    torch.save(
        {
            "step": 12,
            "models": {"agent": {"weight": torch.tensor([1])}},
            "model_specs": {"agent": {"dim": 1}},
            "optimizers": {"agent": {"large": "state"}},
            "states": {"agent_learner": {"large": "state"}},
            "metadata": {"trainer": trainer, "game": "connectfour"},
        },
        source,
    )

    policy_id, step = capture_agent_checkpoint(
        source, destination, expected_game="connectfour"
    )
    compact = torch.load(destination, weights_only=False)

    assert (policy_id, step) == ("checkpoint-12", 12)
    assert compact["metadata"]["source_trainer"] == trainer


def test_arena_checkpoint_directory_loads_selected_trainer():
    path = arena.effective_checkpoint_directory(
        ["--game", "connectfour", "--mmd-coefficient", "0.1"],
        trainer_name="adversarial-mmd",
    )

    assert "adversarial-mmd" in path.parts
