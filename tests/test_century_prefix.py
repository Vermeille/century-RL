import pyximport  # type: ignore[import-untyped]

pyximport.install(setup_args={"script_args": ["--cython-cplus"]})

from boardrl.games.century.engine import ActionCard  # type: ignore[import-not-found]


def parse_stock(s: str):
    return ActionCard(s, "").takes()


def test_stock_prefix():
    s = parse_stock("YYRRGB")
    assert s.prefix(2).to_str() == "2Y"
    assert s.prefix(4).to_str() == "2Y2R"
    assert s.prefix(6).to_str() == s.to_str()
