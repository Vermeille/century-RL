import math
import random

import pytest

pyspiel = pytest.importorskip("pyspiel")

from boardrl.games.openspiel.evaluation import OpenSpielModelEvaluator
from boardrl.games import games_library
from boardrl.games.augmentations import shuffle_actions
from boardrl.games.openspiel.game import OpenSpielGame
from boardrl.games.semantics import OutcomeScores, TerminalOutcomeRewards
from boardrl.rollouts import RolloutRunner
from boardrl.training import ToSamples

PRESETS = {
    "liars_dice": (
        "liars_dice",
        {"numdice": 2, "dice_sides": 5, "players": 2, "bidding_rule": "reset-face"},
    ),
    "dark_hex": (
        "dark_hex",
        {
            "board_size": 3,
            "num_rows": 3,
            "num_cols": 3,
            "gameversion": "cdh",
            "obstype": "reveal-nothing",
        },
    ),
    "abrupt_dark_hex": (
        "dark_hex",
        {
            "board_size": 3,
            "num_rows": 3,
            "num_cols": 3,
            "gameversion": "adh",
            "obstype": "reveal-nothing",
        },
    ),
    "phantom_ttt": (
        "phantom_ttt",
        {"gameversion": "classical", "obstype": "reveal-nothing"},
    ),
    "abrupt_phantom_ttt": (
        "phantom_ttt",
        {"gameversion": "abrupt", "obstype": "reveal-nothing"},
    ),
}
BOARD_PRESETS = tuple(name for name in PRESETS if name != "liars_dice")


def play_actions(game, actions):
    for action in actions:
        game.play_str(game.moves[game.action_ids.index(action)])


@pytest.mark.parametrize("name", PRESETS)
def test_presets_select_native_rules_and_training_semantics(name):
    desc = games_library(name)
    game = desc.make_game(num_players=2)
    native_name, parameters = PRESETS[name]
    assert game._game.get_type().short_name == native_name
    assert game._game.get_parameters() == parameters
    assert game.num_players == 2
    assert game.information_state is True
    assert not game._state.is_chance_node()
    assert game.current_player() == 0
    assert len(game.moves) == (20 if name == "liars_dice" else 9)
    assert type(desc.scores) is OutcomeScores
    assert type(desc.rewards) is TerminalOutcomeRewards
    assert desc.rewards.points_value_head is False
    assert desc.coop is False
    assert desc.augmentations == (shuffle_actions,)
    assert type(desc.make_model_evaluator(progress=False)) is OpenSpielModelEvaluator
    with pytest.raises(ValueError, match="players"):
        desc.make_game(num_players=3)


@pytest.mark.parametrize("name", PRESETS)
def test_complete_games_match_native_states_actions_and_payoffs(name):
    for seed in range(8):
        rng = random.Random(seed)
        game = games_library(name).make_game()
        native = game._state.clone()
        initial = game.copy()
        initial_text = initial.display_with_moves()
        for step in range(game._game.max_game_length() + 1):
            assert game.ended() == native.is_terminal()
            assert game.action_ids == tuple(native.legal_actions())
            assert game.round() == step // 2
            for player in range(2):
                native_text = native.information_state_string(player)
                if name in ("dark_hex", "abrupt_dark_hex"):
                    lines = native_text.split("\n")
                    del lines[3]
                    native_text = "\n".join(lines)
                view = game.display(force=player).split("\nGame: ended\n", 1)[0]
                assert view == f"[Player: {player}]\n{native_text}"
                assert game.points_for(player) == native.returns()[player]
            if game.ended():
                break
            assert game.current_player() == native.current_player()
            assert game.moves == [
                native.action_to_string(native.current_player(), action)
                for action in native.legal_actions()
            ]
            assert [
                line[1:]
                for line in game.display_with_moves().splitlines()
                if line.startswith("@")
            ] == game.moves
            index = rng.randrange(len(game.moves))
            native.apply_action(game.action_ids[index])
            game.play_idx(index)
        assert game.ended()
        assert game.moves == []
        assert sum(game.points_for(player) for player in range(2)) == 0.0
        assert initial.display_with_moves() == initial_text
        assert initial.moves is not game.moves
        assert not initial.ended()
        with pytest.raises(ValueError, match="terminal"):
            game.play_idx(0)
        with pytest.raises(ValueError, match="illegal move"):
            initial.play_str("not a move")
        assert initial.display_with_moves() == initial_text
        assert sum(initial.simulate_to_end()) == 0.0
        assert initial.ended() and initial.moves == []


@pytest.mark.parametrize("name", BOARD_PRESETS)
def test_players_have_distinct_opening_information_states(name):
    game = games_library(name).make_game()
    first = game.display_with_moves()
    game.play_idx(0)
    assert game.current_player() == 1
    assert game.display_with_moves() != first


@pytest.mark.parametrize("name", BOARD_PRESETS)
def test_hidden_placements_do_not_leak_through_text_or_legal_moves(name):
    first = games_library(name).make_game()
    second = first.copy()
    play_actions(first, [0])
    play_actions(second, [8])
    assert first.current_player() == second.current_player() == 1
    assert first.display_with_moves() == second.display_with_moves()
    assert first.action_ids == second.action_ids == tuple(range(9))
    assert first.display(force=0) != second.display(force=0)


@pytest.mark.parametrize("name", BOARD_PRESETS)
def test_collision_reveals_cell_and_only_abrupt_variants_lose_turn(name):
    game = games_library(name).make_game()
    play_actions(game, [0])
    before = str(game._state)
    own_view = game.display(force=1)
    assert 0 in game.action_ids
    play_actions(game, [0])
    assert str(game._state) == before
    assert not game.ended()
    assert game.display(force=1) != own_view
    assert game.display(force=1).splitlines()[1] == "x.."
    assert game.current_player() == (0 if name.startswith("abrupt") else 1)
    assert 0 not in game.action_ids
    assert game.points_for(0) == game.points_for(1) == 0.0


@pytest.mark.parametrize("name", BOARD_PRESETS)
def test_information_state_preserves_order_of_own_attempts(name):
    first = games_library(name).make_game()
    second = first.copy()
    play_actions(first, [0, 4, 1, 5])
    play_actions(second, [1, 4, 0, 5])
    assert first.current_player() == second.current_player() == 0
    assert first._state.observation_string(0) == second._state.observation_string(0)
    assert first.moves == second.moves
    assert first.display_with_moves() != second.display_with_moves()


@pytest.mark.parametrize(
    "name,actions,winner",
    [
        ("dark_hex", [0, 1, 3, 2, 6], 0),
        ("abrupt_dark_hex", [0, 1, 3, 2, 6], 0),
        ("phantom_ttt", [0, 3, 1, 4, 2], 0),
        ("abrupt_phantom_ttt", [0, 3, 1, 4, 2], 0),
        ("dark_hex", [0, 3, 1, 4, 6, 5], 1),
        ("abrupt_dark_hex", [0, 3, 1, 4, 6, 5], 1),
        ("phantom_ttt", [0, 3, 1, 4, 8, 5], 1),
        ("abrupt_phantom_ttt", [0, 3, 1, 4, 8, 5], 1),
    ],
)
def test_board_winning_lines(name, actions, winner):
    game = games_library(name).make_game()
    play_actions(game, actions[:-1])
    losing_view = game.display(force=1 - winner)
    play_actions(game, actions[-1:])
    assert game.ended() and game.moves == []
    assert game.winner() == winner
    for player, result, payoff in ((winner, "win", 1), (1 - winner, "loss", -1)):
        assert game.points_for(player) == payoff
        assert game.display(force=player).startswith(f"[Player: {player}]\n")
        assert game.display(force=player).endswith(
            f"\nGame: ended\nResult: {result}\nPayoff: {payoff}"
        )
    # The opponent's winning move stays hidden, but the loss is now observable.
    assert game.display(force=1 - winner) != losing_view
    assert game.display(force=1 - winner).split("\nGame: ended\n", 1)[0] == losing_view


@pytest.mark.parametrize("name", ["phantom_ttt", "abrupt_phantom_ttt"])
def test_phantom_tictactoe_draw(name):
    game = games_library(name).make_game()
    play_actions(game, [0, 1, 2, 4, 3, 5, 7, 6, 8])
    assert game.ended() and game.moves == []
    assert game.winner() is None
    assert [game.points_for(player) for player in range(2)] == [0.0, 0.0]
    for player in range(2):
        assert game.display(force=player).startswith(f"[Player: {player}]\n")
        assert game.display(force=player).endswith(
            "\nGame: ended\nResult: draw\nPayoff: 0"
        )


def dice_state(rolls):
    native = games_library("liars_dice").make_game()._game.new_initial_state()
    for roll in rolls:
        assert native.is_chance_node()
        native.apply_action(roll - 1)
    assert not native.is_chance_node()
    return OpenSpielGame.from_state(native, information_state=True)


def test_liars_dice_hides_opponents_dice_and_preserves_bids():
    first = dice_state([1, 2, 3, 5])
    second = dice_state([1, 2, 4, 4])
    assert first.display_with_moves() == second.display_with_moves()
    assert first.display() == "[Player: 0]\n12"
    assert first.display(force=1) == "[Player: 1]\n35"
    assert second.display(force=1) == "[Player: 1]\n44"
    assert "Liar" not in first.moves
    first.play_str("1-4")
    assert "1-4" not in first.moves
    assert "1-5" in first.moves and "2-1" in first.moves
    assert "Liar" in first.moves
    first.play_str("2-1")
    assert first.display() == "[Player: 0]\n12 1-4 2-1"


@pytest.mark.parametrize("bid,winner", [("2-1", 0), ("3-1", 1), ("1-5", 0), ("2-5", 1)])
def test_liars_dice_challenge_counts_fives_as_wild(bid, winner):
    game = dice_state([1, 2, 3, 5])
    game.play_str(bid)
    game.play_str("Liar")
    assert game.ended() and game.moves == []
    assert game.winner() == winner
    assert game.points_for(winner) == 1.0
    assert game.points_for(1 - winner) == -1.0


@pytest.mark.parametrize("name", PRESETS)
@pytest.mark.parametrize("max_steps", [1, 40])
def test_rollouts_rewards_samples_and_payoff_metrics(name, max_steps):
    desc = games_library(name)
    strategy = desc.strategy_from_string("random")
    games = RolloutRunner(desc.make_game, progress=False, outcome=desc.outcome).play(
        [strategy, strategy],
        games=4,
        max_steps=max_steps,
        rotate=True,
    )
    desc.rewards.make_returns(1.0)(games)
    terminal = max_steps == 40
    for index, game in enumerate(games):
        assert sum(trace[-1].my_points for trace in game) == 0.0
        for trace in game:
            assert trace.strategy_id == (trace.seat_id + index) % 2
            assert trace[-1].terminal is terminal
            assert trace[-1].truncated is not terminal
            assert trace[-1].state.startswith(f"[Player: {trace.seat_id}]\n")
            if terminal:
                result = {-1.0: "loss", 0.0: "draw", 1.0: "win"}[trace[-1].my_points]
                assert f"\nGame: ended\nResult: {result}\n" in trace[-1].state
            else:
                assert "Game: ended" not in trace[-1].state
                assert "Result:" not in trace[-1].state
            for record in trace[:-1]:
                assert record.player == trace.seat_id
                if terminal:
                    assert record.returns == trace[-1].my_points
                else:
                    assert math.isnan(record.returns)
    samples = ToSamples()(games)
    assert samples
    records = [record for game in games for trace in game for record in trace[:-1]]
    for sample, record in zip(samples, records, strict=True):
        moves = [line[1:] for line in sample.state.splitlines() if line.startswith("@")]
        assert moves == record.moves
        assert sample.action_idx == record.action_idx
        assert len(moves) == len(sample.action_distribution)
    metrics = desc.make_metrics(games).metrics()
    assert metrics["terminal_rate"] == float(terminal)
    for identity in range(2):
        group = metrics["strategy"][str(identity)]
        assert group["payoff_games"] == (4 if terminal else 0)
        assert group["total_payoff"] == sum(
            game.by_strategy[identity][-1].my_points for game in games
        )


@pytest.mark.parametrize("name", PRESETS)
def test_presets_are_playable_in_shared_web_workbench(name):
    from fastapi.testclient import TestClient

    from boardrl.serve import serve

    client = TestClient(serve.app)
    assert name in client.get("/games").json()["games"]
    response = client.get("/reset", params={"game": name})
    assert response.status_code == 200
    response = client.get("/state", params={"game": name})
    assert response.status_code == 200
    before = response.json()
    assert before["name"] == name and not before["ended"]
    assert before["board"].startswith("[Player: 0]\n")
    assert [
        line[1:]
        for line in before["board_with_moves"].splitlines()
        if line.startswith("@")
    ] == before["moves"]
    response = client.post(
        "/do-one", params={"game": name}, json={"action": before["moves"][0]}
    )
    assert response.status_code == 200
    after = response.json()["state"]
    assert after["current_player"] == 1
    assert after["board"].startswith("[Player: 1]\n")
    assert after["can_undo"] is True
    assert len(after["history"]) == len(before["history"]) + 1
