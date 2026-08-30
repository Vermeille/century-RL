from types import SimpleNamespace
import pytest
import torch

from boardrl.games.thegame.game import MESSAGE_MOVES
from boardrl.games.thegame.metrics import Metrics


class DummyResults(list):
    def collapse(self):
        return [1.0]

    def num_players(self):
        return 1


def _record(state, moves, action_idx, action_distribution=None):
    if action_distribution is None:
        action_distribution = torch.zeros(len(moves))
    return SimpleNamespace(
        state=state,
        moves=moves,
        action_idx=action_idx,
        action_distribution=action_distribution,
        final=False,
    )


def _state(action, *, cards=20):
    return "\n".join(
        [
            f"Round: 0, Action: {action}",
            "Piles: 1 1 100 100",
            f"Cards: {cards}",
            "Hand: 10 20",
        ]
    )


def _metrics_for(records):
    end = SimpleNamespace(final=True, my_points=50)
    return Metrics(DummyResults([[records + [end]]])).metrics()


def test_thegame_metrics():
    # First move uses the 10 rule and is the lowest cost option
    state1 = "\n".join(
        [
            "Round: 0, Action: 0",
            "Piles: 20 1 100 100",
            "Cards: 0",
            "Hand: 10 30",
        ]
    )
    rec1 = _record(state1, ["10->0", "30->2"], 0)

    # Second move does not use the 10 rule and is not the cheapest
    state2 = "\n".join(
        [
            "Round: 0, Action: 1",
            "Piles: 10 1 60 100",
            "Cards: 0",
            "Hand: 30 50",
        ]
    )
    rec2 = _record(state2, ["30->2", "50->2"], 0)

    end = SimpleNamespace(final=True, my_points=50)

    player = [rec1, rec2, end]
    game = [player]
    results = DummyResults([game])
    metrics = Metrics(results)
    values = metrics.metrics()

    assert values["avg_cost"] == 10.0
    assert values["ratio_lowest_cost"] == 0.5
    assert list(values["points"].values) == [50]
    assert list(values["ten_rule_moves"].values) == [1, 0]


def test_ten_rule_moves_is_per_card_and_excludes_non_card_actions():
    ten_rule_state = _state(0).replace(
        "Piles: 1 1 100 100", "Piles: 21 1 100 100"
    )
    records = [
        _record(ten_rule_state, ["11->0"], 0),
        _record(_state(1), ["x"], 0),
        _record(_state(2), MESSAGE_MOVES, 0),
    ]

    ten_rule_moves = _metrics_for(records)["ten_rule_moves"]

    assert list(ten_rule_moves.values) == [1]


def test_thegame_metrics_self_play_runs():
    from boardrl.games.thegame.game import TheGame
    from boardrl.games.thegame.strategies import strategy_from_string
    from boardrl.rl.eval.selfplay import self_play

    make_game = lambda num_players: TheGame(num_players=num_players)
    strat = lambda: strategy_from_string("lowest_cost")
    results = self_play(
        make_game, [strat, strat], n_games=1, max_len=10, rotate=False, desc=""
    )
    metrics = Metrics(results)
    assert metrics.metrics()


def test_plays_before_x_is_a_range():
    records = [_record(_state(action), ["x"], 0) for action in range(2, 22)]

    plays_before_x = _metrics_for(records)["plays_before_x"]

    assert list(plays_before_x.values) == list(range(20))


@pytest.mark.parametrize(
    "action,cards,expected",
    [
        (2, 20, 0),
        (5, 20, 3),
        (1, 0, 0),
        (4, 0, 3),
    ],
)
def test_plays_before_x_subtracts_the_rule_mandatory_plays(
    action, cards, expected
):
    records = [_record(_state(action, cards=cards), ["x"], 0)]

    plays_before_x = _metrics_for(records)["plays_before_x"]

    assert list(plays_before_x.values) == [expected]


def test_plays_before_x_needs_no_trace_before_a_resumed_midturn_state():
    records = [_record(_state(5, cards=20), ["x"], 0)]

    plays_before_x = _metrics_for(records)["plays_before_x"]

    assert list(plays_before_x.values) == [3]


def test_message_information_detects_state_dependent_one_hot_protocol():
    records = []
    for message_idx in range(len(MESSAGE_MOVES)):
        logits = torch.full((len(MESSAGE_MOVES),), -20.0)
        logits[message_idx] = 20.0
        records.append(_record(_state(2), MESSAGE_MOVES, message_idx, logits))

    information = _metrics_for(records)["message_information"]

    assert information == pytest.approx(1.0)


@pytest.mark.parametrize(
    "logits",
    [
        torch.tensor([20.0] + [-20.0] * 9),
        torch.zeros(10),
    ],
)
def test_message_information_detects_state_independent_policies(logits):
    records = [_record(_state(2), MESSAGE_MOVES, 0, logits) for _ in range(10)]

    information = _metrics_for(records)["message_information"]

    assert information == pytest.approx(0.0, abs=1e-6)
