import pyximport
pyximport.install(setup_args={"script_args": ["--cython-cplus"]})

from boardrl.games.century.engine import ActionCard


def parse_stock(s: str):
    return ActionCard(s, "").takes()


def test_stock_prefix():
    s = parse_stock("YYRRGB")
    assert s.prefix(2).to_str_() == "YY"
    assert s.prefix(4).to_str_() == "YYRR"
    assert s.prefix(6).to_str_() == s.to_str_()

