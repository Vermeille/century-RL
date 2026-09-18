import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from boardrl.rl.eval.selfplay import GameTrace, PlayerTrace, SelfPlayResults
from boardrl.rl.model.loss import ScheduledPerplexity
from boardrl.training import Pipeline, Select, ToSamples, TrainingSample


def load_trainer(filename, module_name):
    path = Path(__file__).parents[1] / "trainers" / filename
    spec = importlib.util.spec_from_file_location(f"trainers.{module_name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


adversarial_threshold = load_trainer(
    "adversarial-threshold.py",
    "adversarial_threshold",
)


class SampleRecord:
    def __init__(self, state):
        self.state = state

    def training_sample(self):
        return TrainingSample(state=self.state)


def _trace(seat_id, strategy_id, state):
    trace = PlayerTrace(seat_id=seat_id, strategy_id=strategy_id)
    trace.append(SampleRecord(state))
    trace.append(SimpleNamespace(terminal=True, current_diff_points=0.0))
    return trace


def test_adversarial_threshold_defaults_to_two_policy_connect_four():
    args = adversarial_threshold.build_parser().parse_args([])

    assert args.game == "connectfour"
    assert args.tag == "adversarial-threshold"
    assert args.perplexity_schedule_shape == "cosine"
    assert args.environment_perplexity_start == 3.0
    assert args.environment_perplexity_end == 1.0
    assert args.stop_win_rate_threshold == 0.7
    assert args.restart_win_rate_threshold == 0.5


def test_environment_perplexity_schedule_is_configurable():
    args = adversarial_threshold.build_parser().parse_args(
        [
            "--environment-perplexity-start",
            "2.75",
            "--environment-perplexity-end",
            "1.25",
        ]
    )

    assert args.environment_perplexity_start == 2.75
    assert args.environment_perplexity_end == 1.25


def test_win_rate_hysteresis_thresholds_are_configurable():
    args = adversarial_threshold.build_parser().parse_args(
        [
            "--stop-win-rate-threshold",
            "0.8",
            "--restart-win-rate-threshold",
            "0.4",
        ]
    )

    assert args.stop_win_rate_threshold == 0.8
    assert args.restart_win_rate_threshold == 0.4


def test_environment_schedule_changes_only_environment_learner_args(monkeypatch):
    args = adversarial_threshold.build_parser().parse_args([])
    received = []

    def fake_make_learner(model, game, learner_args, *, offload_modules):
        assert len(offload_modules) == 1
        received.append(learner_args)
        return SimpleNamespace(losses=[object(), object()]), object()

    monkeypatch.setattr(adversarial_threshold.coop, "make_learner", fake_make_learner)

    adversarial_threshold.make_strategy_learner("one", "ref-one", "game", args)
    adversarial_threshold.make_strategy_learner(
        "two",
        "ref-two",
        "game",
        args,
        perplexity_start=args.environment_perplexity_start,
        perplexity_end=args.environment_perplexity_end,
    )

    assert received[0].perplexity_start == args.perplexity_start
    assert received[0].perplexity_end == args.perplexity_end
    assert received[1].perplexity_start == args.environment_perplexity_start
    assert received[1].perplexity_end == args.environment_perplexity_end
    assert args.perplexity_start != args.perplexity_end


def test_rollout_samples_are_split_by_strategy_identity():
    args = SimpleNamespace(
        learner_batch_size=8,
        inference_batch_size=8,
        discount=1.0,
        gae_lambda=0.1,
        value_lambda=0.9,
    )

    agent = object()
    agent_reference = object()
    environment = object()
    environment_reference = object()
    prepare_agent = adversarial_threshold.make_prepare(agent, agent_reference, args, 0)
    prepare_environment = adversarial_threshold.make_prepare(
        environment,
        environment_reference,
        args,
        1,
    )

    assert prepare_agent.steps[0].strategies == [0]
    assert prepare_agent.steps[0].seats is None
    assert prepare_agent.steps[2].model is agent
    assert prepare_agent.steps[3].model is agent_reference
    assert prepare_environment.steps[0].strategies == [1]
    assert prepare_environment.steps[0].seats is None
    assert prepare_environment.steps[2].model is environment
    assert prepare_environment.steps[3].model is environment_reference


def test_select_strategy_filters_training_samples_after_rotation():
    games = SelfPlayResults(
        [
            GameTrace(
                [
                    _trace(0, 0, "game-0-agent"),
                    _trace(1, 1, "game-0-environment"),
                ]
            ),
            GameTrace(
                [
                    _trace(0, 1, "game-1-environment"),
                    _trace(1, 0, "game-1-agent"),
                ]
            ),
        ]
    )

    samples = Pipeline(Select(strategies=[0]), ToSamples())(games)

    assert [sample.state for sample in samples] == [
        "game-0-agent",
        "game-1-agent",
    ]
    assert all("environment" not in sample.state for sample in samples)


def test_batch_win_rate_controller_trains_policy_above_threshold(monkeypatch):
    games = SelfPlayResults(
        [
            GameTrace(
                [
                    _trace(0, 0, "agent"),
                    _trace(1, 1, "environment"),
                ]
            )
        ]
    )
    games[0].by_strategy[0][-1].current_diff_points = 1.0
    games[0].by_strategy[1][-1].current_diff_points = -1.0
    controller = adversarial_threshold.BatchWinRateController(0.7, 0.5)
    prepared = object()
    result = SimpleNamespace(metrics={"loss": 1.0}, samples=1)
    calls = []
    learner = SimpleNamespace(
        train=lambda samples, *, progress, reference_frozen, reference_released: calls.append(
            (samples, progress, reference_frozen, reference_released)
        )
        or result,
    )
    monkeypatch.setattr(
        adversarial_threshold,
        "copy_weights",
        lambda reference, model: (_ for _ in ()).throw(AssertionError()),
    )

    update = controller.update(
        games,
        0,
        model=None,
        reference=None,
        prepare=lambda value, **kwargs: prepared,
        learner=learner,
        progress=0.0,
    )

    assert update.win_rate == 1.0
    assert update.updated
    assert update.paused
    assert update.metrics == {"loss": 1.0}
    assert calls == [(prepared, 0.0, True, False)]


def test_batch_win_rate_controller_updates_policy_at_threshold(monkeypatch):
    games = SelfPlayResults(
        [
            GameTrace(
                [
                    _trace(0, 0, "agent"),
                    _trace(1, 1, "environment"),
                ]
            )
        ]
    )
    games[0].by_strategy[0][-1].current_diff_points = 1.0
    games[0].by_strategy[1][-1].current_diff_points = -1.0
    copied = []
    prepared = object()
    result = SimpleNamespace(metrics={"loss": 1.0})
    learner = SimpleNamespace(
        train=lambda samples, *, progress, reference_frozen, reference_released: result,
    )
    monkeypatch.setattr(
        adversarial_threshold,
        "copy_weights",
        lambda reference, model: copied.append((reference, model)),
    )
    controller = adversarial_threshold.BatchWinRateController(1.0, 0.5)

    update = controller.update(
        games,
        0,
        model="model",
        reference="reference",
        prepare=lambda value, **kwargs: prepared,
        learner=learner,
        progress=0.5,
    )

    assert update.win_rate == 1.0
    assert update.updated
    assert not update.paused
    assert update.metrics == {"loss": 1.0}
    assert copied == [("reference", "model")]


def test_releasing_reference_resets_perplexity_controller(monkeypatch):
    games = SelfPlayResults(
        [GameTrace([_trace(0, 0, "agent"), _trace(1, 1, "environment")])]
    )
    games[0].by_strategy[0][-1].current_diff_points = 1.0
    games[0].by_strategy[1][-1].current_diff_points = -1.0
    calls = []
    result = SimpleNamespace(metrics={}, samples=1)
    learner = SimpleNamespace(
        train=lambda samples, *, progress, reference_frozen, reference_released: calls.append(
            (reference_frozen, reference_released)
        )
        or result,
    )
    monkeypatch.setattr(adversarial_threshold, "copy_weights", lambda reference, model: None)
    controller = adversarial_threshold.BatchWinRateController(0.7, 0.5)
    controller.paused_strategy_ids.add(1)

    update = controller.update(
        games,
        1,
        model=None,
        reference=None,
        prepare=lambda value, **kwargs: value,
        learner=learner,
    )

    assert not update.paused
    assert calls == [(False, True)]


def test_perplexity_handoff_holds_strength_while_reference_is_frozen():
    perplexity = ScheduledPerplexity(
        start=1.0,
        init_strength=0.1,
        baseline_ratio=0.2,
        ppl_beta=0.0,
    )
    perplexity.regularizer.strength = 0.2
    original_baseline = perplexity.baseline_strength
    result = object()

    def train(samples, *, progress):
        del samples, progress
        perplexity.update_strength(measured_ppl=2.0, target_ppl=1.0)
        return result

    learner = adversarial_threshold.PerplexityHandoffLearner(
        SimpleNamespace(train=train),
        perplexity,
    )

    actual = learner.train(
        [],
        progress=0.5,
        reference_frozen=True,
        reference_released=False,
    )

    assert actual is result
    assert perplexity.regularizer.strength == 0.2
    assert perplexity.baseline_strength == original_baseline


def test_perplexity_handoff_resets_ema_on_reference_release():
    perplexity = ScheduledPerplexity(
        start=3.0,
        init_strength=0.1,
        ppl_beta=0.99,
    )
    perplexity.ppl_ema = 5.0
    observations = []

    def train(samples, *, progress):
        del samples, progress
        observations.append((perplexity.ppl_ema, perplexity.ppl_beta))
        perplexity.update_strength(measured_ppl=2.0, target_ppl=3.0)

    learner = adversarial_threshold.PerplexityHandoffLearner(
        SimpleNamespace(train=train),
        perplexity,
    )

    learner.train(
        [],
        progress=0.5,
        reference_frozen=False,
        reference_released=True,
    )

    assert observations == [(None, 0.0)]
    assert perplexity.ppl_ema == 2.0
    assert perplexity.regularizer.strength > perplexity.init_strength
    assert perplexity.ppl_beta == 0.99


def test_update_controller_signal_encodes_reference_refresh_side():
    active = SimpleNamespace(paused=False)
    paused = SimpleNamespace(paused=True)

    assert adversarial_threshold.update_controller_signal(paused, active) == -1
    assert adversarial_threshold.update_controller_signal(active, active) == 0
    assert adversarial_threshold.update_controller_signal(active, paused) == 1


def test_frozen_reference_overwrites_only_kl_policy():
    class Prediction:
        def __init__(self, policy):
            self.policy = [policy]

    class Predictions(list):
        def unbatched(self):
            return self

    class Reference:
        def __init__(self):
            self.training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = mode
            return self

        def __call__(self, states):
            return Predictions(
                Prediction(torch.tensor([float(state), -float(state)]))
                for state in states
            )

    samples = [
        TrainingSample(
            state=1,
            action_distribution=torch.tensor([9.0, 9.0]),
            reference_policy=torch.tensor([8.0, 8.0]),
            reference_value=3.0,
            gae=4.0,
        ),
        TrainingSample(
            state=2,
            action_distribution=torch.tensor([9.0, 9.0]),
            reference_policy=torch.tensor([8.0, 8.0]),
            reference_value=5.0,
            gae=6.0,
        ),
    ]
    reference = Reference()

    result = adversarial_threshold.FrozenReferencePolicy(reference, batch_size=1)(samples)

    assert result is samples
    assert torch.equal(samples[0].action_distribution, torch.tensor([9.0, 9.0]))
    assert torch.equal(samples[0].reference_policy, torch.tensor([1.0, -1.0]))
    assert torch.equal(samples[1].action_distribution, torch.tensor([9.0, 9.0]))
    assert torch.equal(samples[1].reference_policy, torch.tensor([2.0, -2.0]))
    assert [sample.reference_value for sample in samples] == [3.0, 5.0]
    assert [sample.gae for sample in samples] == [4.0, 6.0]
    assert reference.training


def test_current_reference_preserves_rollout_distribution_without_inference():
    class Reference:
        training = False

        def __call__(self, states):
            raise AssertionError("the current reference must not be evaluated twice")

    samples = [
        TrainingSample(
            reference_policy=torch.tensor([1.0, -1.0]),
            action_distribution=torch.tensor([9.0, 9.0]),
        )
    ]
    targets = adversarial_threshold.FrozenReferencePolicy(Reference(), batch_size=1)
    targets.reference_is_current = True

    result = targets(samples)

    assert result is samples
    assert torch.equal(samples[0].action_distribution, torch.tensor([9.0, 9.0]))
    assert torch.equal(samples[0].reference_policy, torch.tensor([1.0, -1.0]))


def test_batch_win_rate_controller_has_hysteresis():
    controller = adversarial_threshold.BatchWinRateController(0.7, 0.5)

    assert not controller.should_update(0, 0.8)
    assert controller.is_paused(0)
    assert not controller.should_update(0, 0.6)
    assert not controller.should_update(0, 0.5)
    assert controller.should_update(0, 0.49)
    assert not controller.is_paused(0)


def test_batch_win_rate_controller_restores_paused_policies():
    controller = adversarial_threshold.BatchWinRateController(0.7, 0.5)
    controller.should_update(1, 0.8)
    controller.schedule_progress[0] = 0.25
    controller.completed_strategy_ids.add(1)
    restored = adversarial_threshold.BatchWinRateController(0.7, 0.5)

    restored.load_state_dict(controller.state_dict())

    assert not restored.is_paused(0)
    assert restored.is_paused(1)
    assert restored.progress(0) == 0.25
    assert restored.is_complete(1)


def test_batch_win_rate_controller_freezes_completed_policy(monkeypatch):
    games = SelfPlayResults(
        [GameTrace([_trace(0, 0, "agent"), _trace(1, 1, "environment")])]
    )
    games[0].by_strategy[0][-1].current_diff_points = 1.0
    games[0].by_strategy[1][-1].current_diff_points = -1.0
    calls = []
    result = SimpleNamespace(metrics={}, samples=1)
    learner = SimpleNamespace(
        train=lambda samples, *, progress, reference_frozen, reference_released: calls.append(progress)
        or result,
    )
    monkeypatch.setattr(adversarial_threshold, "copy_weights", lambda reference, model: None)
    controller = adversarial_threshold.BatchWinRateController(
        1.0,
        0.5,
        progress_step=1.0,
        complete_on_first_update=True,
    )

    first = controller.update(
        games,
        0,
        model=None,
        reference=None,
        prepare=lambda value, **kwargs: value,
        learner=learner,
    )
    second = controller.update(
        games,
        0,
        model=None,
        reference=None,
        prepare=lambda value, **kwargs: value,
        learner=learner,
    )

    assert first.updated
    assert controller.is_complete(0)
    assert not second.updated
    assert calls == [0.0]


def test_batch_win_rate_controller_trains_through_progress_one(monkeypatch):
    games = SelfPlayResults(
        [GameTrace([_trace(0, 0, "agent"), _trace(1, 1, "environment")])]
    )
    games[0].by_strategy[0][-1].current_diff_points = 1.0
    games[0].by_strategy[1][-1].current_diff_points = -1.0
    calls = []
    result = SimpleNamespace(metrics={}, samples=1)
    learner = SimpleNamespace(
        train=lambda samples, *, progress, reference_frozen, reference_released: calls.append(progress)
        or result,
    )
    monkeypatch.setattr(adversarial_threshold, "copy_weights", lambda reference, model: None)
    controller = adversarial_threshold.BatchWinRateController(
        1.0,
        0.5,
        progress_step=0.5,
    )

    for _ in range(3):
        controller.update(
            games,
            0,
            model=None,
            reference=None,
            prepare=lambda value, **kwargs: value,
            learner=learner,
        )

    assert calls == [0.0, 0.5, 1.0]
    assert controller.is_complete(0)


def test_completed_environment_forces_agent_update(monkeypatch):
    games = SelfPlayResults(
        [GameTrace([_trace(0, 0, "agent"), _trace(1, 1, "environment")])]
    )
    games[0].by_strategy[0][-1].current_diff_points = 1.0
    games[0].by_strategy[1][-1].current_diff_points = -1.0
    result = SimpleNamespace(metrics={}, samples=1)
    learner = SimpleNamespace(
        train=lambda samples, *, progress, reference_frozen, reference_released: result
    )
    monkeypatch.setattr(adversarial_threshold, "copy_weights", lambda reference, model: None)
    controller = adversarial_threshold.BatchWinRateController(0.7, 0.5)
    controller.paused_strategy_ids.add(0)
    controller.completed_strategy_ids.add(1)

    update = controller.update(
        games,
        0,
        model=None,
        reference=None,
        prepare=lambda value, **kwargs: value,
        learner=learner,
        force=controller.is_complete(1),
    )

    assert update.updated
    assert not update.paused
    assert not controller.is_paused(0)
