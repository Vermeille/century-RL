import math

from boardrl.games import games_library
from boardrl.rl.eval.selfplay import pit, SelfPlayResults


def test_selfplay_oop_indexing_and_filters():
    game_desc = games_library("tictactoe")
    strategies = [
        game_desc.strategy_from_string("random"),
        game_desc.strategy_from_string("random"),
    ]

    results = pit(game_desc.make_game, strategies, n_games=2, max_len=2)
    assert isinstance(results, SelfPlayResults)

    game0 = results[0]
    # seat indexing
    assert game0[0].seat_id == 0
    # strategy indexing
    assert game0.by_strategy[0].strategy_id == 0

    # filtering helpers
    assert len(results.only_player([0])[0]) == 1
    assert len(results.only_strategy([0])[0]) == 1

    # metrics: strategy-indexed
    win_rate = results.win_rate(0)
    assert 0.0 <= win_rate <= 1.0 and not math.isnan(win_rate)
    # metrics: seat-indexed
    seat_win_rate = results.win_rate(0, by="seat")
    assert 0.0 <= seat_win_rate <= 1.0 and not math.isnan(seat_win_rate)


def test_pit_rotate_flag():
    game_desc = games_library("tictactoe")
    strategies = [
        game_desc.strategy_from_string("random"),
        game_desc.strategy_from_string("random"),
    ]

    rotated = pit(game_desc.make_game, strategies, n_games=2, max_len=2, rotate=True)
    static = pit(game_desc.make_game, strategies, n_games=2, max_len=2, rotate=False)

    assert rotated[1][0].strategy_id == 1
    assert static[1][0].strategy_id == 0

