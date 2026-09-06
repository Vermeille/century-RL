from types import SimpleNamespace

import pytest

from boardrl.games.skullking.metrics import Metrics


class DummyResults(list):
    def num_players(self):
        return len(self[0])

    def collapse(self):
        return [0.0] * self.num_players()


def _state(round_number, phase, tricks):
    return "\n".join(
        [
            f"Round: {round_number}/2",
            f"Phase: {phase}",
            "Current player: 0",
            "Scores: 0 0",
            "Bids: -",
            f"Tricks: {' '.join(map(str, tricks))}",
            "Lead: -",
            "Top: _:E",
            "Hand: -",
        ]
    )


def _bid(round_number, bid, previous_tricks):
    return SimpleNamespace(
        state=_state(round_number, "bid", previous_tricks),
        moves=[str(i) for i in range(round_number + 1)],
        action_idx=bid,
    )


def _end(tricks):
    return SimpleNamespace(state=_state(2, "ended", tricks))


def test_skullking_bidding_metrics():
    # Round 1: bids [0, 1], tricks [0, 1] -> both exact.
    # Round 2: bids [0, 2], tricks [1, 1] -> both miss by one.
    player0 = [_bid(1, 0, [0, 0]), _bid(2, 0, [0, 1]), _end([1, 1])]
    player1 = [_bid(1, 1, [0, 0]), _bid(2, 2, [0, 1]), _end([1, 1])]
    values = Metrics(DummyResults([[player0, player1]])).metrics()

    assert values["won_bet_ratio"] == pytest.approx(0.5)
    assert values["avg_abs_bet_error"] == pytest.approx(0.5)
    assert values["zero_bet_ratio"] == pytest.approx(0.5)
    assert values["zero_bet_win_ratio"] == pytest.approx(0.5)
    assert values["overbet_ratio"] == pytest.approx(0.25)
    assert values["underbet_ratio"] == pytest.approx(0.25)


def test_equal_distribution_is_round_number_divided_by_players():
    # For two players, equal bids are 0.5 in round 1 and 1.0 in round 2.
    # Actual bids are [0, 1] then [0, 2], so the signed deviations cancel
    # while the mean absolute deviation is (0.5 + 0.5 + 1 + 1) / 4 = 0.75.
    player0 = [_bid(1, 0, [0, 0]), _bid(2, 0, [0, 1]), _end([1, 1])]
    player1 = [_bid(1, 1, [0, 0]), _bid(2, 2, [0, 1]), _end([1, 1])]
    values = Metrics(DummyResults([[player0, player1]])).metrics()

    assert values["bet_diff_equal"] == pytest.approx(0.0)
    assert values["abs_bet_diff_equal"] == pytest.approx(0.75)


def test_skullking_metrics_ignore_unfinished_rounds():
    # A truncated game may contain bids for a round whose tricks are unknown.
    # Round 1 is still recoverable from the round-2 bid observations.
    player0 = [
        _bid(1, 0, [0, 0]),
        _bid(2, 1, [0, 1]),
        SimpleNamespace(state=_state(2, "play", [0, 0])),
    ]
    player1 = [
        _bid(1, 1, [0, 0]),
        _bid(2, 1, [0, 1]),
        SimpleNamespace(state=_state(2, "play", [0, 0])),
    ]
    values = Metrics(DummyResults([[player0, player1]])).metrics()

    assert values["won_bet_ratio"] == pytest.approx(1.0)
    assert values["zero_bet_ratio"] == pytest.approx(0.5)


def test_skullking_metrics_self_play_runs():
    from boardrl.games.skullking.game import SkullKing
    from boardrl.games.strategies import strategy_from_string
    from boardrl.rl.eval.selfplay import self_play

    results = self_play(
        lambda num_players: SkullKing(num_players=num_players, num_rounds=2),
        [strategy_from_string("random"), strategy_from_string("random")],
        n_games=1,
        max_len=20,
        rotate=False,
        desc="",
    )

    values = Metrics(results).metrics()
    assert 0.0 <= values["won_bet_ratio"] <= 1.0
    assert 0.0 <= values["zero_bet_ratio"] <= 1.0
