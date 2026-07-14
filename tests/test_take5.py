from boardrl.games import games_library
from boardrl.games.take5.game import Take5, card_points


def test_take5_is_registered_with_constructor_options():
    desc = games_library("take5,num_players=2,num_stacks=2")
    game = desc.make_game()

    assert isinstance(game, Take5)
    assert game.num_players == 2
    assert game.num_stacks == 2
    assert len(game.moves) == 10


def test_take5_resolves_a_round_and_exposes_framework_state():
    game = Take5(num_players=2, num_stacks=1)
    game.players = [[10], [20]]
    game.stacks = [[5]]
    game.table = []
    game.points_ = [0, 0]
    game.current_player_ = 0
    game.round_ = 0
    game.phase = 0
    game.moves = game._moves()

    game.play_idx(0)
    assert game.current_player() == 1
    assert game.moves == ["20"]

    game.play_idx(0)
    assert game.round() == 1
    assert game.phase == 0
    assert game.ended()
    assert game.moves == []
    assert game.copy().points_for(0) == game.points_for(0)


def test_take5_requires_a_take_after_all_hands_are_played():
    game = Take5(num_players=2, num_stacks=1)
    game.players = [[5], [6]]
    game.stacks = [[10]]
    game.table = []
    game.points_ = [0, 0]
    game.current_player_ = 0
    game.phase = 0
    game.moves = game._moves()

    game.play_idx(0)
    game.play_idx(0)

    assert game.phase == 1
    assert not game.ended()
    assert game.current_player() == 0
    assert game.moves == ["S0"]

    game.play_idx(0)
    assert game.ended()
    assert game.points_for(0) == -card_points(10)


def test_take5_random_playout_ends():
    game = Take5()
    game.simulate_to_end()

    assert game.ended()
    assert game.moves == []
