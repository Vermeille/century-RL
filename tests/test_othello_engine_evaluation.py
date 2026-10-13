import asyncio
import json

import pytest

pyspiel = pytest.importorskip("pyspiel")

import evaluate_othello_engine as cli
from boardrl.games import games_library
from boardrl.games.openspiel.othello.game import Othello
from boardrl.games.openspiel.othello.strategies import GTPStrategy
from boardrl.models import make_for_game


class NativeEngine:
    """A deterministic in-process GTP opponent with independent native state."""

    def __init__(self, command=()):
        self.clear_board()

    def clear_board(self):
        self.state = pyspiel.load_game("othello").new_initial_state()

    def play(self, color, move):
        player = self.state.current_player()
        assert color == "bw"[player]
        actions = {
            self.state.action_to_string(player, action): action
            for action in self.state.legal_actions()
        }
        self.state.apply_action(actions[move])

    def genmove(self, color):
        assert color == "bw"[self.state.current_player()]
        action = self.state.legal_actions()[0]
        move = self.state.action_to_string(self.state.current_player(), action)
        self.state.apply_action(action)
        return move

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_engine_strategy_restores_interleaved_positions_and_rejects_illegal_move():
    first = Othello()
    first.play_idx(0)
    second = Othello()
    second.play_idx(-1)
    second.play_idx(0)
    engine = NativeEngine()
    strategy = GTPStrategy(engine)
    for game in (first, second, first):
        policy, _ = asyncio.run(strategy(game))
        assert policy.argmax().item() == 0
        assert engine.state.history()[:-1] == game._state.history()
    engine.genmove = lambda color: "a1"
    with pytest.raises(ValueError, match="illegal move"):
        asyncio.run(strategy(Othello()))


def test_cli_uses_standard_evaluator_with_paired_openings(monkeypatch, capsys):
    model = make_for_game("toy", games_library("othello"))
    monkeypatch.setattr(cli, "load_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(cli, "GTPEngine", NativeEngine)
    compare = cli.Evaluator.compare
    pairs = []

    def record(self, players, **options):
        first = self.runner.make_game(num_players=2)
        second = self.runner.make_game(num_players=2)
        assert first._state.history() == second._state.history()
        assert len(first._state.history()) == 4
        assert options["games"] == 2 and options["rotate"] is True
        result = compare(self, players, **options)
        pairs.append(result)
        return result

    monkeypatch.setattr(cli.Evaluator, "compare", record)
    cli.main(["unused.pth", "--engine", "fake", "--pairs", "2", "--opening-plies", "4"])
    report = json.loads(capsys.readouterr().out)
    assert report["games"] == 4 == report["wins"] + report["draws"] + report["losses"]
    assert report["score"] == (report["wins"] + 0.5 * report["draws"]) / 4
    assert len(pairs) == 2
    assert model.training
    for result in pairs:
        assert [game.by_strategy[0].seat_id for game in result.rollouts] == [0, 1]
        assert all(trace[-1].terminal for game in result.rollouts for trace in game)
