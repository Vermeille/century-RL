from types import SimpleNamespace

import pytest

from boardrl.rl.eval.selfplay import PlayerTrace, GameTrace, SelfPlayResults
from boardrl.games.tictactoe.metrics import Metrics


def make_results(seqs, moves=None):
    games = []
    for g_idx, seq in enumerate(seqs):
        players = [PlayerTrace(seat_id=i, strategy_id=i) for i in range(2)]
        move_list = (
            moves[g_idx]
            if moves is not None and g_idx < len(moves)
            else [str(i) for i in range(max(seq) + 1 if seq else 1)]
        )
        for idx, action in enumerate(seq):
            player = idx % 2
            round_num = idx // 2
            players[player].append(
                SimpleNamespace(
                    action_idx=action,
                    round=round_num,
                    player=player,
                    moves=move_list,
                )
            )
        for pid in range(2):
            players[pid].append(
                SimpleNamespace(final=True, round=len(seq) // 2, player=pid, state=" ")
            )
        games.append(GameTrace(players))
    return SelfPlayResults(games)


def test_collapse_extremes():
    same = make_results([[0, 1, 2, 3], [0, 1, 2, 3]])
    assert same.collapse() == pytest.approx([1.0, 1.0])

    different = make_results([[0, 1], [2, 3], [4, 5]])
    assert different.collapse() == pytest.approx([0.0, 0.0], abs=1e-6)


def test_collapse_partial():
    partial = make_results([[0, 1], [0, 1], [2, 3]])
    assert partial.collapse() == pytest.approx([1 / 3, 1 / 3])


def test_collapse_padding():
    varying = make_results([[0], [0, 1, 2], [0, 1, 2, 3, 4]])
    assert varying.collapse() == pytest.approx([5 / 9, 1 / 3])


def test_metrics_pushes_collapse():
    res = make_results([[0, 1]])
    metrics = Metrics(res)

    assert metrics.metrics()["collapse"] == [1.0, 1.0]


def test_collapse_uses_action_strings():
    res = make_results([[0, 1], [0, 1]], moves=[["a", "b"], ["c", "d"]])
    assert res.collapse() == pytest.approx([0.0, 0.0], abs=1e-6)
