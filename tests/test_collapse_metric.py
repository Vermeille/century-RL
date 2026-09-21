from types import SimpleNamespace

import pytest
import torch

from boardrl.games.tictactoe.metrics import Metrics
from boardrl.rollouts import GameTrace, PlayerTrace, Rollouts


def make_results(policies):
    games = []
    for moves, probabilities in policies:
        players = [PlayerTrace(seat_id=i, strategy_id=i) for i in range(2)]
        for player in players:
            player.append(
                SimpleNamespace(
                    action_idx=0,
                    moves=moves,
                    action_distribution=torch.tensor(probabilities).log(),
                )
            )
            player.append(
                SimpleNamespace(
                    final=True,
                    current_diff_points=0.0,
                    my_points=0.0,
                    state=" ",
                )
            )
        games.append(GameTrace(players))
    return Rollouts(games)


@pytest.mark.parametrize("probabilities", ([1.0, 0.0], [0.5, 0.5]))
def test_sensitivity_is_zero_for_state_independent_policy(probabilities):
    results = make_results([(["a", "b"], probabilities)] * 4)

    assert results.sensitivity() == pytest.approx([0.0, 0.0], abs=1e-6)


def test_sensitivity_is_one_for_balanced_state_dependent_one_hot_policy():
    results = make_results(
        [
            (["a", "b"], [1.0, 0.0]),
            (["a", "b"], [0.0, 1.0]),
        ]
    )

    assert results.sensitivity() == pytest.approx([1.0, 1.0], abs=1e-6)


def test_sensitivity_preserves_partial_policy_information():
    results = make_results(
        [
            (["a", "b"], [1.0, 0.0]),
            (["a", "b"], [0.5, 0.5]),
        ]
    )

    assert results.sensitivity() == pytest.approx(
        [0.311278, 0.311278],
        abs=1e-6,
    )


def test_sensitivity_aligns_probabilities_by_move_string():
    results = make_results(
        [
            (["a", "b"], [0.9, 0.1]),
            (["b", "a"], [0.1, 0.9]),
        ]
    )

    assert results.sensitivity() == pytest.approx([0.0, 0.0], abs=1e-6)


def test_metrics_exposes_sensitivity():
    results = make_results([(["a", "b"], [1.0, 0.0])])

    assert Metrics(results).metrics()["sensitivity"] == [0.0, 0.0]
