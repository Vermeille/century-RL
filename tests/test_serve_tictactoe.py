import importlib
import sys
from contextlib import contextmanager

from fastapi.testclient import TestClient


@contextmanager
def serve_for_game(game: str):
    """Import the FastAPI app for the given game.

    The ``boardrl.serve.serve`` module parses command line arguments at import
    time to determine which game to serve.  For tests we need to re-import it
    with ``--game`` set appropriately, while ensuring we restore the original
    ``sys.argv`` and module state afterwards so other tests remain unaffected.
    """
    module_name = "boardrl.serve.serve"
    # Remove any previous imports so the new game argument is respected.
    sys.modules.pop(module_name, None)
    argv_backup = sys.argv[:]
    sys.argv = ["serve.py", "--game", game]
    try:
        module = importlib.import_module(module_name)
        yield module
    finally:
        # Clean up so other tests can import the default server.
        sys.modules.pop(module_name, None)
        sys.argv = argv_backup


def test_tictactoe_play_one_and_analyze_random_policy():
    with serve_for_game("tictactoe") as serve:
        client = TestClient(serve.app)
        # Ensure a fresh game state
        assert client.get("/reset").status_code == 200

        board_before = client.get("/board").text
        response = client.post("/play-one", json={"strategy": "random"})
        assert response.status_code == 200
        assert response.json()["continue"] is True

        board_after = client.get("/board").text
        # A move from the random policy should alter the board state
        assert board_before != board_after

        analyze = client.get("/analyze", params={"strategy": "random"})
        assert analyze.status_code == 200
        data = analyze.json()
        assert "moves" in data
        assert isinstance(data["moves"], dict)
