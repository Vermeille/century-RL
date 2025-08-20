import pytest
import torch
from types import SimpleNamespace
from unittest.mock import MagicMock

from boardrl.games.rps.game import RockPaperScissors
from boardrl.games.rps.metrics import Metrics
from boardrl.rl.eval.selfplay import PlayerTrace, GameTrace, SelfPlayResults
from boardrl.utils import Visualizer


def make_results():
    moves = ["rock", "paper", "scissors"]
    players = [PlayerTrace(seat_id=i, strategy_id=i) for i in range(2)]
    dist0 = torch.log(torch.tensor([0.2, 0.5, 0.3]))
    dist1 = torch.log(torch.tensor([0.3, 0.3, 0.4]))
    players[0].append(
        SimpleNamespace(
            action_distribution=dist0,
            action_idx=1,
            moves=moves,
            current_diff_points=0,
            my_points=0,
            final=False,
            player=0,
            round=0,
        )
    )
    players[1].append(
        SimpleNamespace(
            action_distribution=dist1,
            action_idx=2,
            moves=moves,
            current_diff_points=0,
            my_points=0,
            final=False,
            player=1,
            round=0,
        )
    )
    end0 = SimpleNamespace(
        final=True, round=0, player=0, state="", my_points=0, current_diff_points=0
    )
    end1 = SimpleNamespace(
        final=True, round=0, player=1, state="", my_points=0, current_diff_points=0
    )
    players[0].append(end0)
    players[1].append(end1)
    return SelfPlayResults([GameTrace(players)])


def test_hidden_decisions():
    g = RockPaperScissors()
    g.play_str("rock")
    assert "R" in g.display(force=0)
    assert "R" not in g.display(force=1)
    g.play_str("scissors")
    assert g.points_for(0) == 1


def test_metrics_probabilities():
    res = make_results()
    metrics = Metrics(res)
    viz = Visualizer("test", "offline", 0)
    spy = MagicMock(wraps=viz.push)
    viz.push = spy
    metrics.metrics_to_visdom(viz, 0)
    prob_call = next(c for c in spy.call_args_list if c[0][0] == "move_probabilities")
    assert prob_call[0][1] == pytest.approx([0.25, 0.4, 0.35], abs=1e-6)
