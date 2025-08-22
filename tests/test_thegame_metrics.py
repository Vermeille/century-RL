from types import SimpleNamespace
from unittest.mock import MagicMock

from boardrl.games.thegame.metrics import Metrics


class DummyResults(list):
    def collapse(self):
        return [1.0]


def _record(state, moves, action_idx):
    return SimpleNamespace(state=state, moves=moves, action_idx=action_idx, final=False)


def test_thegame_metrics_to_visdom():
    # First move uses the 10 rule and is the lowest cost option
    state1 = "\n".join([
        "Round: 0, Action: 0",
        "Piles: asc:20, asc:1, desc:100, desc:100",
        "Cards: 0",
        "Hand: 10 30",
    ])
    rec1 = _record(state1, ["10->0", "30->2"], 0)

    # Second move does not use the 10 rule and is not the cheapest
    state2 = "\n".join([
        "Round: 0, Action: 1",
        "Piles: asc:10, asc:1, desc:60, desc:100",
        "Cards: 0",
        "Hand: 30 50",
    ])
    rec2 = _record(state2, ["30->2", "50->2"], 0)

    end = SimpleNamespace(final=True, my_points=50)

    player = [rec1, rec2, end]
    game = [player]
    results = DummyResults([game])
    metrics = Metrics(results)
    viz = SimpleNamespace(push=MagicMock())
    metrics.metrics_to_visdom(viz, 0)

    viz.push.assert_any_call("avg_points", 50, 0)
    viz.push.assert_any_call("avg_cost", 10.0, 0)
    viz.push.assert_any_call("ratio_lowest_cost", 0.5, 0)
    viz.push.assert_any_call("ten_rule_moves", 1, 0)


def test_thegame_metrics_self_play_runs():
    from boardrl.games.thegame.game import TheGame
    from boardrl.games.thegame.strategies import strategy_from_string
    from boardrl.rl.eval.selfplay import self_play

    make_game = lambda num_players: TheGame(num_players=num_players)
    strat = strategy_from_string("lowest_cost")
    results = self_play(make_game, [strat, strat], n_games=1, max_len=10, rotate=False, desc="")
    metrics = Metrics(results)
    viz = SimpleNamespace(push=MagicMock())
    metrics.metrics_to_visdom(viz, 0)
    assert viz.push.call_count > 0
