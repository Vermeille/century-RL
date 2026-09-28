import random

from boardrl.games.openspiel.game import OpenSpielGame


class FakeType:
    short_name = "fake_hidden"


class FakeGame:
    def num_players(self):
        return 2

    def get_type(self):
        return FakeType()

    def new_initial_state(self):
        return FakeState(self)


class FakeState:
    def __init__(self, game, phase="chance", private=0, player=0, terminal=False):
        self.game = game
        self.phase = phase
        self.private = private
        self.player = player
        self.terminal = terminal

    def clone(self):
        return FakeState(
            self.game,
            phase=self.phase,
            private=self.private,
            player=self.player,
            terminal=self.terminal,
        )

    def get_game(self):
        return self.game

    def is_chance_node(self):
        return self.phase == "chance"

    def chance_outcomes(self):
        return [(10, 0.0), (11, 1.0)]

    def apply_action(self, action):
        if self.phase == "chance":
            self.private = action
            self.phase = "decision"
            return
        if action not in self.legal_actions():
            raise ValueError("illegal")
        if self.player == 0:
            self.player = 1
        else:
            self.terminal = True

    def is_terminal(self):
        return self.terminal

    def current_player(self):
        return self.player

    def legal_actions(self):
        return [] if self.terminal else [0, 1]

    def action_to_string(self, player, action):
        return ("stay", "switch")[action]

    def information_state_string(self, player):
        own_private = self.private if player == 0 else "?"
        return f"p{player} card={own_private} turn={self.player}"

    def observation_string(self, player):
        return f"public p{player} turn={self.player}"

    def returns(self):
        return [1.0, -1.0] if self.terminal else [0.0, 0.0]


def test_chance_nodes_are_hidden_from_rollout_interface(monkeypatch):
    monkeypatch.setattr(random, "choices", lambda *args, **kwargs: [11])
    game = OpenSpielGame(
        "fake_hidden",
        spiel_game=FakeGame(),
        information_state=True,
    )

    assert not game._state.is_chance_node()
    assert game.current_player() == 0
    assert game.moves == ["stay", "switch"]
    assert "card=11" in game.display()
    assert game.display_with_moves().endswith("@stay\n@switch")


def test_play_copy_and_terminal_returns():
    game = OpenSpielGame(
        "fake_hidden",
        spiel_game=FakeGame(),
        information_state=True,
    )
    copied = game.copy()

    game.play_str("stay")
    assert copied.current_player() == 0
    assert game.current_player() == 1

    game.play_idx(1)
    assert game.ended()
    assert game.winner() == 0
    assert game.points_for(0) == 1.0
    assert game.points_for(1) == -1.0
    assert game.diff_points_for(0) == 2.0


def test_perfect_information_can_use_observation_strings():
    game = OpenSpielGame(
        "fake_public",
        spiel_game=FakeGame(),
        information_state=False,
    )
    assert game.display() == "public p0 turn=0"
