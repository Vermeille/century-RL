from boardrl.games import games_library


def test_skullking_is_registered_with_constructor_options():
    desc = games_library("skullking,num_players=2,num_rounds=1")
    game = desc.make_game()

    assert game.num_players == 2
    assert game.num_rounds == 1
    assert game.moves == ["0", "1"]


def test_skullking_exposes_framework_state_and_random_playout():
    game = games_library("skullking,num_players=2,num_rounds=1").make_game()

    assert "Moves" in game.display_with_moves()
    assert all(
        line.startswith("@") for line in game.display_with_moves().splitlines()[-2:]
    )

    game.simulate_to_end()

    assert game.ended()
    assert game.moves == []
    assert game.winner() in [0, 1]
