import sys

import pytest

from boardrl.benchmarks.gtp import GTPEngine, GTPError


ENGINE = r'''
import sys
for line in sys.stdin:
    command = line.strip()
    if command == "clear_board":
        print("=", flush=True)
    elif command.startswith("play "):
        print("=", flush=True)
    elif command.startswith("genmove "):
        print("= D3", flush=True)
    elif command == "bad":
        print("? nope", flush=True)
    elif command == "quit":
        print("=", flush=True)
        break
'''


def test_gtp_engine_round_trip():
    with GTPEngine([sys.executable, "-u", "-c", ENGINE]) as engine:
        engine.clear_board()
        engine.play("b", "c4")
        assert engine.genmove("w") == "d3"


def test_gtp_engine_surfaces_protocol_errors():
    with GTPEngine([sys.executable, "-u", "-c", ENGINE]) as engine:
        with pytest.raises(GTPError, match="nope"):
            engine._response("bad")
