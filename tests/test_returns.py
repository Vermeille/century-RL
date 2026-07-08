import math
from types import SimpleNamespace

import torch

from boardrl.training import TrainingSample
from boardrl.training.returns import compute_returns
from boardrl.training.returns import annotate_with_model


class Log:
    """Minimal log object for testing ``compute_returns``."""

    def __init__(self, current_diff_points, final=False):
        self.current_diff_points = current_diff_points
        self.final = final


def make_history(points, final=True):
    """Utility to build a history from ``points``."""

    logs = [Log(p) for p in points[:-1]]
    logs.append(Log(points[-1], final=final))
    return logs


def test_reward_rescale():
    history = make_history([0.0, 1.0, 2.0])
    games = [[history]]

    compute_returns(games, 1.0, reward_rescale=0.5)

    assert history[0].reward == 0.5
    assert history[1].reward == 0.5
    assert history[0].returns == 1.0
    assert history[1].returns == 0.5


def test_terminal_vs_non_terminal():
    # Terminal case
    terminal_hist = make_history([0.0, 1.0, 2.0], final=True)
    compute_returns([[terminal_hist]], 1.0)
    assert terminal_hist[0].returns == 2.0
    assert terminal_hist[1].returns == 1.0
    assert terminal_hist[0].score == 2.0

    # Non-terminal case
    non_term_hist = make_history([0.0, 1.0, 2.0], final=False)
    compute_returns([[non_term_hist]], 1.0)
    assert math.isnan(non_term_hist[0].returns)
    assert math.isnan(non_term_hist[0].score)


def test_annotate_with_model_can_use_cached_rollout_references():
    def unexpected_model_call(_):
        raise AssertionError("cached rollout references should avoid model eval")

    end = SimpleNamespace(final=True)
    second = TrainingSample(
        state="s1",
        reward=2.0,
        next=end,
        final=False,
        reference_policy=torch.tensor([0.2, 0.3]),
        reference_value=0.5,
        reference_max_q=0.6,
    )
    first = TrainingSample(
        state="s0",
        reward=1.0,
        next=second,
        final=False,
        reference_policy=torch.tensor([0.1, 0.4]),
        reference_value=0.25,
        reference_max_q=0.7,
    )

    annotate_with_model(
        unexpected_model_call,
        [first, second],
        bs=2,
        gamma=1.0,
        lmbda=1.0,
        use_cached_rollout=True,
    )

    assert first.td_lambda == 3.0
    assert second.td_lambda == 2.0
    assert first.gae == 2.75
    assert second.gae == 1.5
    assert end.reference_value == 0
