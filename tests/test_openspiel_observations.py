import pytest

pyspiel = pytest.importorskip("pyspiel")

from boardrl.games import games_library
from boardrl.games.openspiel.game import OpenSpielGame
from boardrl.rollouts import EndState


@pytest.mark.parametrize(
    "spec",
    [
        "liars_dice",
        "dark_hex",
        "abrupt_dark_hex",
        "phantom_ttt",
        "abrupt_phantom_ttt",
        "othello",
    ],
)
def test_games_without_native_tags_identify_the_requested_player(spec):
    game = games_library(spec).make_game()
    for _ in range(2):
        current = game.current_player()
        for observer in (0, 1):
            text = game.display(force=observer)
            assert text.startswith(f"[Player: {observer}]\n")
            assert "Game: ended" not in text
            assert "Result:" not in text
        assert game.display() == game.display(force=current)
        game.play_idx(0)


def view_after(game_name, actions, *, information_state=True):
    state = pyspiel.load_game(game_name).new_initial_state()
    for action in actions:
        state.apply_action(action)
    return OpenSpielGame.from_state(state, information_state=information_state)


@pytest.mark.parametrize("information_state", [True, False])
def test_leduc_live_views_preserve_native_identity_fields(information_state):
    game = view_after("leduc_poker", [0, 4], information_state=information_state)
    for _ in range(2):
        for player in (0, 1):
            native = (
                game._state.information_state_string(player)
                if information_state
                else game._state.observation_string(player)
            )
            assert game.display(force=player) == native
            assert f"[Observer: {player}]" in native
            assert f"[Player: {game.current_player()}]" in native
        game.play_str("Call")


@pytest.mark.parametrize("information_state", [True, False])
@pytest.mark.parametrize(
    "actions,payoffs,results,money",
    [
        ([0, 4, 2, 0], (1, -1), ("win", "loss"), "101 99"),
        ([0, 4, 2, 2, 1, 5, 2, 2, 1], (-13, 13), ("loss", "win"), "87 113"),
        ([0, 1, 1, 1, 2, 1, 1], (0, 0), ("draw", "draw"), "100 100"),
    ],
)
def test_leduc_final_views_preserve_native_actor_observer_and_private_cards(
    information_state, actions, payoffs, results, money
):
    game = view_after("leduc_poker", actions, information_state=information_state)
    assert game.ended() and game.moves == []
    for player in (0, 1):
        end = EndState(game, player)
        native = (
            game._state.information_state_string(player)
            if information_state
            else game._state.observation_string(player)
        )
        assert end.state == (
            f"{native}\nGame: ended\nResult: {results[player]}\n"
            f"Payoff: {payoffs[player]}"
        )
        assert end.state.count("[Player:") == 1
        assert end.state.count("[Observer:") == 1
        assert f"[Money: {money}]" in end.state
        assert f"[Private: {actions[player]}]" in end.state
        assert end.terminal and not end.truncated
        assert end.my_points == payoffs[player]
        assert "@" not in end.state


def test_dice_final_text_distinguishes_identical_bids_with_opposite_results():
    name = "liars_dice(numdice=2,dice_sides=5)"
    # Player 0 holds 1,2 and bids two ones. Player 1 either supplies a second
    # one or does not, then challenges. Only the outcome becomes newly visible.
    won = view_after(name, [0, 1, 0, 2])
    lost = view_after(name, [0, 1, 1, 2])
    assert won.display_with_moves() == lost.display_with_moves()
    for game in (won, lost):
        game.play_str("2-1")
        game.play_str("Liar")
    assert won.display(force=0).startswith("[Player: 0]\n12 2-1 Liar\n")
    assert lost.display(force=0).startswith("[Player: 0]\n12 2-1 Liar\n")
    for game, results in ((won, ("win", "loss")), (lost, ("loss", "win"))):
        for player in (0, 1):
            text = EndState(game, player).state
            assert f"Game: ended\nResult: {results[player]}\n" in text
    assert won.display(force=0) != lost.display(force=0)


def test_perfect_information_terminal_view_has_the_same_contract():
    game = view_after("tic_tac_toe", [0, 3, 1, 4, 2], information_state=False)
    assert game.ended()
    for player, result in ((0, "win"), (1, "loss")):
        text = game.display(force=player)
        assert text.startswith(f"[Player: {player}]\n")
        assert f"\nGame: ended\nResult: {result}\n" in text


def test_unresolved_chance_node_preserves_native_observation():
    game = view_after("leduc_poker", [])
    assert game.display() == game._state.information_state_string(0)
    for player in (0, 1):
        assert game.display(force=player) == game._state.information_state_string(
            player
        )
    assert "Game: ended" not in game.display()
