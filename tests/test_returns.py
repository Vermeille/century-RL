import math
from functools import partial
from types import SimpleNamespace

import torch

from boardrl.training import TrainingSample
from boardrl.training.returns import compute_returns, set_episodic_rewards, set_rewards
from boardrl.training.returns import annotate_with_model


class Log:
    """Minimal log object for testing ``compute_returns``."""

    def __init__(
        self,
        current_diff_points,
        *,
        terminal=False,
        truncated=False,
        episodic_utility=0.0,
    ):
        self.current_diff_points = current_diff_points
        self.terminal = terminal
        self.truncated = truncated
        self.episodic_utility = episodic_utility


def make_history(points, *, terminal=True, truncated=False):
    """Utility to build a history from ``points``."""

    logs = [Log(p) for p in points[:-1]]
    logs.append(Log(points[-1], terminal=terminal, truncated=truncated))
    return logs


def test_reward_scale_does_not_modify_points_or_score():
    history = make_history([0.0, 1.0, 2.0])
    games = [[history]]

    compute_returns(games, 1.0, reward_fn=partial(set_rewards, scale=0.5))

    assert history[0].reward == 0.5
    assert history[1].reward == 0.5
    assert history[0].returns == 1.0
    assert history[1].returns == 0.5
    assert [log.current_diff_points for log in history] == [0.0, 1.0, 2.0]
    assert history[0].score == 2.0


def test_terminal_vs_non_terminal():
    # Terminal case
    terminal_hist = make_history([0.0, 1.0, 2.0], terminal=True)
    compute_returns([[terminal_hist]], 1.0)
    assert terminal_hist[0].returns == 2.0
    assert terminal_hist[1].returns == 1.0
    assert terminal_hist[0].score == 2.0

    # Non-terminal case
    non_term_hist = make_history([0.0, 1.0, 2.0], terminal=False)
    compute_returns([[non_term_hist]], 1.0)
    assert math.isnan(non_term_hist[0].returns)
    assert math.isnan(non_term_hist[0].score)


def test_episodic_rewards_preserve_point_scores():
    history = make_history([0.0, 4.0, 17.0])
    history[-1].episodic_utility = 1.0

    compute_returns([[history]], 1.0, reward_fn=set_episodic_rewards)

    assert history[0].reward == 0.0
    assert history[1].reward == 1.0
    assert history[0].returns == 1.0
    assert history[0].score == 17.0


def test_truncated_is_not_a_terminal_return_or_score():
    history = make_history([0.0, 1.0, 2.0], terminal=False, truncated=True)
    compute_returns([[history]], 1.0)

    assert all(math.isnan(log.returns) for log in history[:-1])
    assert all(math.isnan(log.score) for log in history)


def test_annotate_with_model_can_use_cached_rollout_references():
    def unexpected_model_call(_):
        raise AssertionError("cached rollout references should avoid model eval")

    end = SimpleNamespace(terminal=True, truncated=False)
    second = TrainingSample(
        state="s1",
        reward=2.0,
        next=end,
        terminal=False,
        truncated=False,
        reference_policy=torch.tensor([0.2, 0.3]),
        reference_value=0.5,
        reference_value_stddev=1.5,
        reference_max_q=0.6,
    )
    first = TrainingSample(
        state="s0",
        reward=1.0,
        next=second,
        terminal=False,
        truncated=False,
        reference_policy=torch.tensor([0.1, 0.4]),
        reference_value=0.25,
        reference_value_stddev=1.25,
        reference_max_q=0.7,
    )

    annotate_with_model(
        unexpected_model_call,
        [first, second],
        bs=2,
        gamma=1.0,
        gae_lambda=0.5,
        value_lambda=1.0,
        use_cached_rollout=True,
    )

    assert first.td_lambda == 3.0
    assert second.td_lambda == 2.0
    assert first.gae == 2.0
    assert second.gae == 1.5
    normalized = torch.tensor([first.normalized_gae, second.normalized_gae])
    assert torch.isclose(normalized.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(normalized.std(), torch.tensor(1.0), atol=1e-3)
    assert first.reference_value_stddev == 1.25
    assert second.reference_value_stddev == 1.5
    assert end.reference_value == 0


def test_annotate_with_model_bootstraps_truncated_endpoint():
    class Prediction:
        def __init__(self, value):
            self.policy = [torch.tensor([0.2, 0.3])]
            self.value = torch.distributions.Normal(
                torch.tensor(value), torch.tensor(0.5)
            )

        def q_value(self):
            return torch.tensor([[value for value in (0.6, 0.7)]])

    class Model:
        def __call__(self, states):
            return SimpleNamespace(
                unbatched=lambda: [
                    Prediction({"s0": 10.0, "s1": 11.0, "cutoff": 12.0}[state])
                    for state in states
                ]
            )

    end = SimpleNamespace(state="cutoff", terminal=False, truncated=True)
    last = TrainingSample(
        state="s1", reward=2.0, next=end, terminal=False, truncated=False
    )
    first = TrainingSample(
        state="s0", reward=1.0, next=last, terminal=False, truncated=False
    )

    annotate_with_model(
        Model(),
        [first, last],
        bs=2,
        gamma=1.0,
        gae_lambda=1.0,
        value_lambda=1.0,
    )

    assert end.reference_value == 12.0
    assert end.reference_value_stddev == 0.5
    assert end.td_lambda == 12.0
    assert last.td_lambda == 14.0
    assert first.td_lambda == 15.0
