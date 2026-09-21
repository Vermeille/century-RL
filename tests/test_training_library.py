from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from boardrl.checkpoints import Checkpoints
from boardrl.games import games_library
from boardrl.games.strategies import RandomStrategy
from boardrl.models import toy
from boardrl.rl.model.loss import ImitationCELoss, ScheduledPerplexity
from boardrl.rl.model import NormalValueDistribution, PolicyValue, load_model
from boardrl.rollouts import RolloutRunner
from boardrl.training import (
    ComputeReturns,
    DoubleQTargets,
    Learner,
    Pipeline,
    ReplayBuffer,
    Select,
    Scheduler,
    ToSamples,
    TrainingSample,
)
from boardrl.training.learner import (
    BatchUpdates,
    PolicyMetrics,
    RolloutUpdate,
    normalized_nucleus_size,
)


def test_rollout_pipeline_is_composable():
    game = games_library("tictactoe")
    results = RolloutRunner(game.make_game, progress=False).play(
        [RandomStrategy(), RandomStrategy()], games=2, max_steps=2
    )

    samples = Pipeline(
        ComputeReturns(1.0), Select(strategies=[0]), ToSamples()
    )(results)

    assert samples
    assert all(sample.next is not None for sample in samples)


def test_rollout_factory_owns_matchmaking():
    game = games_library("tictactoe")
    calls = []

    def lineup(index):
        calls.append(index)
        return [RandomStrategy(), RandomStrategy()]

    results = RolloutRunner(game.make_game, progress=False).play(
        lineup, games=3, max_steps=1
    )

    assert calls == [0, 1, 2]
    assert len(results) == 3


def test_replay_buffer_retains_old_transitions_with_a_fixed_capacity():
    replay = ReplayBuffer(capacity=3, samples_per_update=3)
    old = [TrainingSample(state=f"old-{i}") for i in range(3)]
    new = TrainingSample(state="new")

    replay(old)
    sampled = replay([new])

    assert len(replay.samples) == 3
    assert old[0] not in replay.samples
    assert set(sampled) == set(replay.samples)


def test_double_q_targets_select_online_action_and_evaluate_with_target():
    class FixedQ(torch.nn.Module):
        def __init__(self, policy, value):
            super().__init__()
            self.policy = torch.tensor(policy)
            self.value = torch.tensor(value)

        def forward(self, states):
            batch = len(states)
            return PolicyValue(
                [self.policy.clone() for _ in states],
                NormalValueDistribution(
                    self.value.repeat(batch),
                    torch.ones(batch),
                ),
            )

    online = FixedQ([0.0, 2.0], 0.0)
    target = FixedQ([5.0, 1.0], 10.0)
    nonterminal = TrainingSample(
        next=SimpleNamespace(terminal=False, state="next\n@a\n@b")
    )
    terminal = TrainingSample(next=SimpleNamespace(terminal=True))

    DoubleQTargets(online, target, batch_size=8)([nonterminal, terminal])

    # Online selects action 1. Its target-network Q is 10 + (1 - mean(5, 1)) = 8.
    assert nonterminal.next_reference_max_q == 8.0
    assert terminal.next_reference_max_q == 0.0


def test_checkpoints_support_multiple_models(tmp_path):
    current = toy()
    average = toy()
    optimizer = torch.optim.AdamW(current.parameters())
    checkpoints = Checkpoints(tmp_path, keep=2)

    path = checkpoints.save(
        4,
        {"current": current, "average": average},
        optimizers={"current": optimizer},
        metadata={"algorithm": "nfsp"},
    )
    restored_current = toy()
    restored_average = toy()
    payload = checkpoints.load(
        path,
        models={"current": restored_current, "average": restored_average},
    )

    assert payload["step"] == 4
    assert payload["metadata"]["algorithm"] == "nfsp"
    for expected, restored in zip(current.parameters(), restored_current.parameters()):
        assert torch.equal(expected, restored)


def test_checkpoint_pruning_is_relative_to_current_iteration(tmp_path):
    model = toy()

    # Simulate stale checkpoints left by a previous run that reached a later
    # iteration before the current run restarted in the same directory.
    Checkpoints(tmp_path).save(1575, {"current": model})
    Checkpoints(tmp_path).save(1800, {"current": model})

    checkpoints = Checkpoints(tmp_path, keep=2)
    checkpoints.save(25, {"current": model})

    assert [path.name for path in checkpoints.paths] == ["step-25.pth"]

    checkpoints.save(50, {"current": model})
    checkpoints.save(75, {"current": model})

    assert [path.name for path in checkpoints.paths] == [
        "step-50.pth",
        "step-75.pth",
    ]


def test_checkpoints_restore_training_state(tmp_path):
    class State:
        def __init__(self, value):
            self.value = value

        def state_dict(self):
            return {"value": self.value}

        def load_state_dict(self, state):
            self.value = state["value"]

    model = toy()
    saved = State(42)
    restored = State(0)
    checkpoints = Checkpoints(tmp_path)

    path = checkpoints.save(1, {"current": model}, states={"controller": saved})
    checkpoints.load(path, states={"controller": restored})

    assert restored.value == 42


def test_checkpoint_rejects_resume_without_required_training_state(tmp_path):
    class State:
        def load_state_dict(self, state):
            pass

    model = toy()
    checkpoints = Checkpoints(tmp_path)
    path = checkpoints.save(1, {"current": model})

    with pytest.raises(KeyError, match="required state 'controller'"):
        checkpoints.load(path, states={"controller": State()})


def test_checkpoint_can_be_loaded_as_a_rollout_model(tmp_path):
    model = toy()
    path = Checkpoints(tmp_path).save(1, {"current": model})

    loaded = load_model(path)

    assert loaded.spec() == model.spec()


def test_load_model_can_explicitly_stay_on_cpu(tmp_path, monkeypatch):
    model = toy()
    path = Checkpoints(tmp_path).save(1, {"current": model})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    loaded = load_model(path, device="cpu")

    assert next(loaded.parameters()).device.type == "cpu"


def test_atomic_checkpoint_failure_preserves_previous_file(tmp_path, monkeypatch):
    model = toy()
    checkpoints = Checkpoints(tmp_path)
    path = checkpoints.save(1, {"current": model})
    previous = path.read_bytes()

    def interrupted_save(payload, temporary):
        Path(temporary).write_bytes(b"partial")
        raise RuntimeError("interrupted")

    monkeypatch.setattr(torch, "save", interrupted_save)

    with pytest.raises(RuntimeError, match="interrupted"):
        checkpoints.save(1, {"current": model})

    assert path.read_bytes() == previous
    assert not list(tmp_path.glob(".*.tmp"))


def test_learner_accepts_pure_imitation_samples_without_returns():
    model = toy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    learner = Learner(
        model,
        optimizer,
        [ImitationCELoss()],
        batch_size=1,
        device="cpu",
    )
    sample = TrainingSample(
        state="board\n@left\n@right",
        action_idx=0,
        action_distribution=torch.tensor([2.0, -1.0]),
        next=None,
    )

    result = learner.train([sample])

    assert result.samples == 1
    assert result.batches == 1


def test_normalized_learner_handles_partial_batch():
    model = toy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    learner = Learner(
        model,
        optimizer,
        [ImitationCELoss()],
        batch_size=4,
        device="cpu",
        normalize_lr=True,
    )
    sample = TrainingSample(
        state="board\n@left\n@right",
        action_idx=0,
        action_distribution=torch.tensor([2.0, -1.0]),
        next=None,
    )

    result = learner.train([sample])

    assert result.samples == 1
    assert result.batches == 1
    assert result.metrics["lr"] == pytest.approx(1e-4)


def test_learner_reports_optimizer_learning_rate():
    model = toy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.param_groups[0]["lr"] = 2e-4
    learner = Learner(
        model,
        optimizer,
        [ImitationCELoss()],
        batch_size=1,
        device="cpu",
    )
    sample = TrainingSample(
        state="board\n@left\n@right",
        action_idx=0,
        action_distribution=torch.tensor([2.0, -1.0]),
        next=None,
    )

    result = learner.train([sample])

    assert result.metrics["lr"] == pytest.approx(2e-4)


def test_learner_applies_lr_schedule_from_training_progress():
    model = toy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    learner = Learner(
        model,
        optimizer,
        [ImitationCELoss()],
        batch_size=1,
        device="cpu",
        lr_schedule=Scheduler(start_value=1.0, end_value=0.0),
    )
    sample = TrainingSample(
        state="board\n@left\n@right",
        action_idx=0,
        action_distribution=torch.tensor([2.0, -1.0]),
        next=None,
    )

    first = learner.train([sample], progress=0.0)
    assert first.metrics["lr"] == pytest.approx(1e-3)

    second = learner.train([sample], progress=0.25)
    assert second.metrics["lr"] == pytest.approx(0.75e-3)

    skipped = learner.train([], progress=0.5)
    assert skipped.samples == 0
    assert learner.optimizer.param_groups[0]["lr"] == pytest.approx(0.75e-3)


def test_policy_metrics_average_per_sample_perplexity_on_device():
    policy = [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 1.0, 2.0])]

    result = PolicyMetrics()(policy, value=None, batch=None)
    expected = torch.stack(
        [
            (-(logits.softmax(0) * logits.log_softmax(0)).sum()).exp()
            for logits in policy
        ]
    ).mean()

    assert torch.is_tensor(result["perplexity"])
    assert torch.allclose(result["perplexity"], expected)


def test_normalized_nucleus_size_maps_deterministic_and_uniform_policies():
    probs = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )

    result = normalized_nucleus_size(probs)

    assert torch.allclose(result, torch.tensor([0.0, 1.0]))


def test_policy_metrics_reports_normalized_nucleus_size():
    policy = [torch.tensor([10.0, 0.0, 0.0]), torch.zeros(2)]

    result = PolicyMetrics()(policy, value=None, batch=None)

    assert torch.allclose(
        result["normalized_nucleus_size_threshold_0_95"],
        torch.tensor(0.5),
        atol=1e-4,
    )


def test_learner_restores_its_losses_and_batch_baseline():
    model = toy()
    saved_loss = ScheduledPerplexity(start=0.5, init_strength=0.1, ppl_beta=0.0)
    saved = Learner(
        model,
        torch.optim.AdamW(model.parameters()),
        [saved_loss],
        batch_size=4,
        device="cpu",
        normalize_lr=True,
    )
    saved.base_batches = 17
    saved_loss.update_strength(measured_ppl=0.0, target_ppl=0.5)

    restored_model = toy()
    restored_loss = ScheduledPerplexity(start=0.5, init_strength=0.1)
    restored = Learner(
        restored_model,
        torch.optim.AdamW(restored_model.parameters()),
        [restored_loss],
        batch_size=4,
        device="cpu",
        normalize_lr=True,
    )

    restored.load_state_dict(saved.state_dict())

    assert restored.base_batches == 17
    assert restored_loss.entropy.strength == saved_loss.entropy.strength
    assert restored_loss.ppl_ema == saved_loss.ppl_ema


def test_lr_equalizer_only_scales_minibatch_updates():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    learner = SimpleNamespace(
        base_batches=None,
        normalize_lr=True,
        epochs=1,
        optimizer=optimizer,
    )
    updates = BatchUpdates(learner, [object()])

    updates.start()
    updates.begin_epoch(4)
    assert updates.objective(torch.tensor(1.0), []) == 1.0
    assert optimizer.param_groups[0]["lr"] == 0.01

    updates.begin_epoch(2)
    assert updates.objective(torch.tensor(1.0), []) == 1.0
    assert optimizer.param_groups[0]["lr"] == 0.02

    metrics = updates.finish()
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert metrics == {"lr_scale": 2.0}

    rollout = RolloutUpdate(
        SimpleNamespace(normalize_lr=True),
        [object(), object(), object(), object()],
    )
    rollout.begin_epoch(2)
    assert rollout.objective(torch.tensor(1.0), [object(), object()]) == 0.5
