from types import SimpleNamespace

from trainers import adversarial2
from boardrl.rl.eval.selfplay import GameTrace, PlayerTrace, SelfPlayResults
from boardrl.training import Pipeline, Select, ToSamples, TrainingSample


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


def test_adversarial2_defaults_to_two_policy_connect_four():
    args = adversarial2.build_parser().parse_args([])

    assert args.game == "connectfour"
    assert args.tag == "adversarial2"
    assert args.perplexity_schedule_shape == "cosine"
    assert args.environment_perplexity_start == 3.0
    assert args.environment_perplexity_end == 1.0
    assert args.stop_win_rate_threshold == 0.7
    assert args.restart_win_rate_threshold == 0.5


def test_environment_perplexity_schedule_is_configurable():
    args = adversarial2.build_parser().parse_args(
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
    args = adversarial2.build_parser().parse_args(
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
    args = adversarial2.build_parser().parse_args([])
    received = []

    def fake_make_learner(model, reference, game, learner_args):
        received.append(learner_args)
        return model, object()

    monkeypatch.setattr(adversarial2.coop, "make_learner", fake_make_learner)

    adversarial2.make_strategy_learner("one", "ref-one", "game", args)
    adversarial2.make_strategy_learner(
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

    prepare_agent = adversarial2.make_prepare(object(), args, 0)
    prepare_environment = adversarial2.make_prepare(object(), args, 1)

    assert prepare_agent.steps[0].strategies == [0]
    assert prepare_agent.steps[0].seats is None
    assert prepare_environment.steps[0].strategies == [1]
    assert prepare_environment.steps[0].seats is None


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


def test_batch_win_rate_controller_skips_policy_above_threshold():
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
    controller = adversarial2.BatchWinRateController(0.7, 0.5)

    update = controller.update(
        games,
        0,
        model=None,
        reference=None,
        prepare=None,
        learner=None,
        progress=0.0,
    )

    assert update.win_rate == 1.0
    assert not update.updated
    assert update.paused
    assert update.metrics == {}


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
        train=lambda samples, *, progress: result,
    )
    monkeypatch.setattr(
        adversarial2,
        "copy_weights",
        lambda reference, model: copied.append((reference, model)),
    )
    controller = adversarial2.BatchWinRateController(1.0, 0.5)

    update = controller.update(
        games,
        0,
        model="model",
        reference="reference",
        prepare=lambda value: prepared,
        learner=learner,
        progress=0.5,
    )

    assert update.win_rate == 1.0
    assert update.updated
    assert not update.paused
    assert update.metrics == {"loss": 1.0}
    assert copied == [("reference", "model")]


def test_batch_win_rate_controller_has_hysteresis():
    controller = adversarial2.BatchWinRateController(0.7, 0.5)

    assert not controller.should_update(0, 0.8)
    assert controller.is_paused(0)
    assert not controller.should_update(0, 0.6)
    assert not controller.should_update(0, 0.5)
    assert controller.should_update(0, 0.49)
    assert not controller.is_paused(0)


def test_batch_win_rate_controller_restores_paused_policies():
    controller = adversarial2.BatchWinRateController(0.7, 0.5)
    controller.should_update(1, 0.8)
    restored = adversarial2.BatchWinRateController(0.7, 0.5)

    restored.load_state_dict(controller.state_dict())

    assert not restored.is_paused(0)
    assert restored.is_paused(1)
