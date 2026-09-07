from boardrl.games.century.strategies import (
    _apply_transform,
    _card_potential_from_string,
    _parse_stock,
    _target_distance,
)


def test_parse_compact_stock_counts():
    assert _parse_stock("2YR3B") == {"Y": 2, "R": 1, "G": 0, "B": 3}


def test_power_card_potential_matches_weighted_cube_economy():
    assert _card_potential_from_string("YY>RR") == 10.0
    assert _card_potential_from_string("YYY>RRR") == 9.0
    assert _card_potential_from_string(">B") == 4.0


def test_transform_applies_ten_cube_trim_with_engine_tie_break():
    out = _apply_transform(_parse_stock("5Y5R"), ">B")
    assert out == {"Y": 4, "R": 5, "G": 0, "B": 1}


def test_target_distance_uses_cube_values():
    stock = _parse_stock("3Y")
    targets = [_parse_stock("2Y2R"), _parse_stock("GB")]
    assert _target_distance(stock, targets) == 4
