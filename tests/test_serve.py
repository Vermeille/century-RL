from pathlib import Path
import subprocess

import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException

from boardrl.serve import serve
from boardrl.serve.serve import Strategies


def run_static_renderer_check(script: str):
    harness = rf"""
const fs = require('fs');
const vm = require('vm');
const code = fs.readFileSync('boardrl/serve/static/app.js', 'utf8');
const noopElement = () => ({{
  addEventListener() {{}},
  replaceChildren() {{}},
  append() {{}},
  disabled: false,
  textContent: '',
  innerHTML: '',
  value: 'thegame',
}});
const sandbox = {{
  document: {{
    addEventListener() {{}},
    querySelector: noopElement,
    querySelectorAll: () => [],
  }},
  fetch: async () => ({{
    ok: true,
    headers: {{ get: () => 'application/json' }},
    json: async () => ({{}}),
  }}),
  setTimeout,
  console,
}};
vm.createContext(sandbox);
vm.runInContext(code.replace(/init\(\);\s*$/, ''), sandbox);
{script}
"""
    result = subprocess.run(
        ["node", "-e", harness],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_populate_strategies_finds_model(tmp_path):
    temp_model = Path(f"{serve.game_name}-example_model.pth")
    temp_model.touch()
    try:
        s = Strategies()
    finally:
        temp_model.unlink()
    assert any(
        f"model=./{temp_model.name}" in name or f"model={temp_model}" in name
        for name in s.strategies
    )
    assert "random" in s.strategies


def test_get_strategy_caching_and_invalid():
    s = Strategies()
    strat1 = s.get_strategy("random")
    strat2 = s.get_strategy("random")
    assert strat1 is strat2
    with pytest.raises(HTTPException):
        s.get_strategy("unknown")


def test_get_strategies_endpoint():
    client = TestClient(serve.app)
    response = client.get("/strategies")
    assert response.status_code == 200
    assert response.json() == serve.strategies.strategies


def test_root_serves_shared_ui():
    client = TestClient(serve.app)
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>BoardRL</title>" in response.text
    assert "/static/app.js" in response.text
    assert 'id="rawState"' in response.text
    assert 'id="analyzeRaw"' in response.text
    assert 'id="playTop"' in response.text
    assert 'id="undo"' in response.text
    assert 'id="redo"' in response.text
    assert 'id="compareStrategy"' in response.text
    assert 'id="gameSpec"' in response.text
    assert 'id="applyGame"' in response.text


def test_static_ui_exposes_keyboard_shortcuts():
    app_js = Path("boardrl/serve/static/app.js").read_text()
    assert "function handleShortcut" in app_js
    assert 'document.addEventListener("keydown", handleShortcut)' in app_js
    assert "state.orderedMoves" in app_js
    assert "function playTopMove" in app_js
    assert "compareStrategy" in app_js
    assert "function gameQuery" in app_js
    assert "gameSpec.value.trim()" in app_js
    assert "applyGameSpec" in app_js


def test_state_endpoint_matches_current_game():
    client = TestClient(serve.app)
    client.get("/reset")
    response = client.get("/state")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == serve.game_name
    assert data["board_with_moves"] == serve.game.display_with_moves()
    assert data["moves"] == serve.game.moves
    assert data["current_player"] == serve.game.current_player()
    assert data["round"] == serve.game.round()
    assert data["ended"] is False
    assert data["history"] == []
    assert data["can_undo"] is False
    assert data["can_redo"] is False


def test_games_endpoint_and_selected_game_state():
    client = TestClient(serve.app)
    response = client.get("/games")
    assert response.status_code == 200
    data = response.json()
    assert "century" in data["games"]
    assert "tictactoe" in data["games"]

    response = client.get("/state", params={"game": "tictactoe"})
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "tictactoe"
    assert data["moves"] == [str(i) for i in range(9)]


def test_game_options_work_through_http_and_keep_sessions_separate():
    client = TestClient(serve.app)
    spec = "nim,num_stones=5,max_pick=2"

    response = client.get("/state", params={"game": spec})
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "nim"
    assert data["spec"] == spec
    assert data["moves"] == ["1", "2"]

    response = client.post("/do-one", params={"game": spec}, json={"action": "1"})
    assert response.status_code == 200
    assert response.json()["state"]["spec"] == spec

    default_nim = client.get("/state", params={"game": "nim"})
    assert default_nim.status_code == 200
    assert default_nim.json()["spec"] == "nim"
    assert default_nim.json()["moves"] == ["1", "2", "3"]

    response = client.post("/set-game", json={"game": spec})
    assert response.status_code == 200
    assert response.json()["spec"] == spec
    assert client.get("/games").json()["current_spec"] == spec

    assert client.get("/state", params={"game": "nim,unknown=1"}).status_code == 400
    assert client.get("/state", params={"game": "unknown"}).status_code == 404

    client.post("/set-game", json={"game": "century"})


def test_set_game_changes_legacy_default():
    client = TestClient(serve.app)
    response = client.post("/set-game", json={"game": "tictactoe"})
    assert response.status_code == 200
    assert response.json()["name"] == "tictactoe"
    assert serve.game_name == "tictactoe"

    response = client.get("/state")
    assert response.status_code == 200
    assert response.json()["name"] == "tictactoe"

    client.post("/set-game", json={"game": "century"})


def test_strategies_are_game_specific():
    client = TestClient(serve.app)
    response = client.get("/strategies", params={"game": "tictactoe"})
    assert response.status_code == 200
    data = response.json()
    assert "random" in data
    assert "random_buy" not in data
    assert not any("connectfour-ckpt" in strategy for strategy in data)
    assert not any("thegame-messages" in strategy for strategy in data)


def test_play_one_and_reset_endpoint():
    client = TestClient(serve.app)
    client.get("/reset")
    initial_game = serve.game
    resp = client.post("/play-one", json={"strategy": "random"})
    assert resp.status_code == 200
    assert isinstance(resp.json()["continue"], bool)
    resp = client.get("/reset")
    assert resp.status_code == 200
    assert resp.json() is True
    assert serve.game is not initial_game


def test_history_records_moves_and_reset_clears_it():
    client = TestClient(serve.app)
    client.post("/set-game", json={"game": "tictactoe"})
    client.get("/reset", params={"game": "tictactoe"})

    response = client.post("/do-one?game=tictactoe", json={"action": "0"})
    assert response.status_code == 200
    history = response.json()["state"]["history"]
    assert history == [
        {
            "turn": 1,
            "actor": "human",
            "action": "0",
            "strategy": None,
            "player": 1,
            "round": 0,
            "ended": False,
        }
    ]

    client.get("/reset", params={"game": "tictactoe"})
    response = client.get("/state", params={"game": "tictactoe"})
    assert response.json()["history"] == []

    client.post("/set-game", json={"game": "century"})


def test_undo_restores_previous_game_state_and_history():
    client = TestClient(serve.app)
    client.post("/set-game", json={"game": "tictactoe"})
    client.get("/reset", params={"game": "tictactoe"})
    before = client.get("/state", params={"game": "tictactoe"}).json()

    move_response = client.post("/do-one?game=tictactoe", json={"action": "0"})
    assert move_response.status_code == 200
    assert move_response.json()["state"]["history"]
    assert move_response.json()["state"]["can_undo"] is True
    assert move_response.json()["state"]["can_redo"] is False

    undo_response = client.post("/undo?game=tictactoe")
    assert undo_response.status_code == 200
    after = undo_response.json()
    assert after["board_with_moves"] == before["board_with_moves"]
    assert after["moves"] == before["moves"]
    assert after["history"] == []
    assert after["can_undo"] is False
    assert after["can_redo"] is True

    redo_response = client.post("/redo?game=tictactoe")
    assert redo_response.status_code == 200
    redone = redo_response.json()
    assert redone["history"][0]["action"] == "0"
    assert redone["can_undo"] is True
    assert redone["can_redo"] is False

    client.post("/undo?game=tictactoe")
    branch_response = client.post("/do-one?game=tictactoe", json={"action": "1"})
    assert branch_response.status_code == 200
    assert branch_response.json()["state"]["can_redo"] is False

    empty_redo = client.post("/redo?game=tictactoe")
    assert empty_redo.status_code == 400
    assert empty_redo.json()["detail"] == "No moves to redo."

    empty_undo = client.post("/undo?game=tictactoe")
    assert empty_undo.status_code == 200
    empty_undo = client.post("/undo?game=tictactoe")
    assert empty_undo.status_code == 400
    assert empty_undo.json()["detail"] == "No moves to undo."

    client.post("/set-game", json={"game": "century"})


def test_thegame_static_renderer_parses_current_display_format():
    run_static_renderer_check(
        r"""
const game = {
  name: 'thegame',
  moves: ['96->0', '96->1', '96->2', '96->3'],
  board_with_moves: [
    'Round: 0, Action: 0',
    'Piles: 1 1 100 100',
    'Cards: 84',
    'Hand: 96 39 29 32 71 19 20',
    '@96->0',
    '@96->1',
    '@96->2',
    '@96->3',
  ].join('\n'),
};
const piles = sandbox.parseTheGame(game).piles;
const html = sandbox.renderThegame(game);
if (piles.length !== 4) throw new Error(`expected 4 piles, got ${piles.length}`);
if (html.includes('undefined')) throw new Error('renderer leaked undefined');
if (!html.includes('Ascending') || !html.includes('Descending')) {
  throw new Error('renderer did not label pile directions');
}
"""
    )


def test_grid_static_renderers_parse_current_display_formats():
    run_static_renderer_check(
        r"""
const ttt = {
  name: 'tictactoe',
  moves: ['0','1','2','3','4','5','6','7','8'],
  board_with_moves: ['>O', '   ', '   ', '   ', 'Moves', '@0', '@1'].join('\n'),
};
const tttHtml = sandbox.renderTictactoe(ttt);
if ((tttHtml.match(/grid-cell/g) || []).length !== 9) {
  throw new Error('tictactoe did not render 9 cells');
}
if (!tttHtml.includes('data-action="0"')) {
  throw new Error('tictactoe did not expose clickable moves');
}

const connect = {
  name: 'connectfour',
  moves: ['0','1','2','3','4','5','6'],
  board_with_moves: ['>O', '|       |', '|       |', '|       |', '|       |', '|       |', '|       |', '---------', '@0', '@1'].join('\n'),
};
const connectHtml = sandbox.renderConnectfour(connect);
if ((connectHtml.match(/grid-cell/g) || []).length !== 42) {
  throw new Error('connectfour did not render 42 cells');
}
if ((connectHtml.match(/<button class="column-move/g) || []).length !== 7) {
  throw new Error('connectfour did not render 7 column controls');
}
"""
    )


def test_century_static_renderer_parses_current_display_format():
    run_static_renderer_check(
        r"""
const game = {
  name: 'century',
  moves: ['A0 >', 'A1 Y>', 'H0 >2Y', 'H1 Y>R'],
  board_with_moves: [
    '0    0',
    '_Board',
    'V0 2G2B>14',
    'V1 3Y2G>9',
    'A0 >YG >',
    'A1 YR>B X>',
    '_Him 1 0',
    'V 0',
    'S 4Y',
    '_Me 0',
    'V 0',
    'S 3Y',
    'H0 >2Y',
    'H1 XX',
    '_Moves',
    '@A0 >',
    '@A1 Y>',
    '@H0 >2Y',
    '@H1 Y>R',
  ].join('\n'),
};
const parsed = sandbox.parseCentury(game);
const html = sandbox.renderCentury(game);
if (parsed.victory.length !== 2) throw new Error('century victory market parse failed');
if (parsed.action.length !== 2) throw new Error('century action market parse failed');
if (parsed.hand.length !== 2) throw new Error('century hand parse failed');
if (html.includes('undefined')) throw new Error('century renderer leaked undefined');
if (!html.includes('Victory Market') || !html.includes('Action Market')) {
  throw new Error('century renderer missed expected sections');
}
"""
    )


def test_simple_static_renderers_parse_current_display_formats():
    run_static_renderer_check(
        r"""
const nim = {
  name: 'nim',
  moves: ['1', '2', '3'],
  board_with_moves: ['21', 'Moves', '@1', '@2', '@3'].join('\n'),
};
const nimHtml = sandbox.renderNim(nim);
if (!nimHtml.includes('21 stones')) throw new Error('nim stone count missing');
if ((nimHtml.match(/data-action="/g) || []).length !== 3) {
  throw new Error('nim did not expose 3 moves');
}

const rps = {
  name: 'rps',
  moves: ['rock', 'paper', 'scissors'],
  board_with_moves: ['>?|?', '@rock', '@paper', '@scissors'].join('\n'),
};
const rpsHtml = sandbox.renderRps(rps);
if (!rpsHtml.includes('Rock') || !rpsHtml.includes('Scissors')) {
  throw new Error('rps choices missing');
}

const guess = {
  name: 'guessnumber',
  moves: ['A', 'B'],
  board_with_moves: ['Secret: 1', 'History:', '@A', '@B'].join('\n'),
};
const guessHtml = sandbox.renderGuessnumber(guess);
if (!guessHtml.includes('Secret 1') || !guessHtml.includes('Signal A')) {
  throw new Error('guessnumber renderer missed secret or symbols');
}

const sum = {
  name: 'sum',
  moves: ['0','1','2','3','4','5','6','7','8','9'],
  board_with_moves: ['0 | 0', '5 2', '@0', '@1'].join('\n'),
};
const sumHtml = sandbox.renderSum(sum);
if (!sumHtml.includes('5') || !sumHtml.includes('/ 2')) {
  throw new Error('sum problem missing');
}
if ((sumHtml.match(/data-action="/g) || []).length !== 10) {
  throw new Error('sum did not expose 10 number moves');
}

const take5Put = {
  name: 'take5',
  moves: ['12', '47'],
  board_with_moves: ['Put', 'Hand: 12 47', 'S0: 5 19', 'S1: 23', '@12', '@47'].join('\n'),
};
const take5PutHtml = sandbox.renderTake5(take5Put);
if (!take5PutHtml.includes('Play a card') || !take5PutHtml.includes('data-action="12"')) {
  throw new Error('take5 hand renderer missed playable cards');
}

const take5Take = {
  name: 'take5',
  moves: ['S0', 'S1'],
  board_with_moves: ['Take 5', 'Hand:', 'S0: 5 19', 'S1: 23', '@S0', '@S1'].join('\n'),
};
const take5TakeHtml = sandbox.renderTake5(take5Take);
if (!take5TakeHtml.includes('Take a stack') || !take5TakeHtml.includes('data-action="S0"')) {
  throw new Error('take5 stack renderer missed take actions');
}
"""
    )


def test_analyze_endpoint():
    client = TestClient(serve.app)
    client.get("/reset")
    resp = client.get("/analyze", params={"strategy": "random"})
    data = resp.json()
    assert "moves" in data
    assert isinstance(data["moves"], dict)


def test_analyze_post_supports_generic_text_strategies():
    client = TestClient(serve.app)
    state = "Some board\n@short\n@much-longer\n"

    response = client.post(
        "/analyze",
        json={"strategy": "random", "state": state},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["probabilities"] == {"short": 0.5, "much-longer": 0.5}

    response = client.post(
        "/analyze",
        json={"strategy": "longest_move", "state": state},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["probabilities"] == {"short": 0.0, "much-longer": 1.0}


def test_analyze_post_rejects_text_without_moves():
    client = TestClient(serve.app)
    response = client.post(
        "/analyze",
        json={"strategy": "random", "state": "No moves here"},
    )
    assert response.status_code == 400
    assert "does not contain any @ legal moves" in response.json()["detail"]
