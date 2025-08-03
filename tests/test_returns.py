import math

from boardrl.training.returns import compute_returns


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
