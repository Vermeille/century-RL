from dataclasses import asdict
import json
from pathlib import Path
import subprocess

import pytest
from fastapi.testclient import TestClient

from boardrl.games.game2048.game import Game2048
from boardrl.games.game2048.versus import Versus2048
from boardrl.serve import serve


def snapshot(game, name="2048"):
    return asdict(serve.GameSnapshot.from_game(name, name, game))


def renderer_check(snapshots, script):
    harness = r"""
const fs = require('fs');
const vm = require('vm');
const listeners = {};
const noopElement = () => ({
  addEventListener() {}, replaceChildren() {}, append() {},
  disabled: false, textContent: '', innerHTML: '', value: '2048',
});
const sandbox = {
  assert: require('assert/strict'),
  document: {
    addEventListener(type, handler) { (listeners[type] ||= []).push(handler); },
    querySelector: noopElement, querySelectorAll: () => [],
  },
  listeners, console, setTimeout,
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('boardrl/serve/static/app.js', 'utf8')
  .replace(/init\(\);\s*$/, ''), sandbox);
vm.runInContext(fs.readFileSync('boardrl/serve/static/game2048.js', 'utf8'), sandbox);
"""
    harness += f"vm.runInContext({json.dumps('const snapshots = ' + json.dumps(snapshots))}, sandbox);\n"
    harness += f"vm.runInContext({json.dumps(script)}, sandbox);\n"
    result = subprocess.run(
        ["node", "-e", harness], cwd=Path(__file__).parents[1],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_ui_loads_assets_and_renders_real_observations_and_canonical_actions():
    index = Path("boardrl/serve/static/index.html").read_text()
    assert '/static/game2048.css' in index and '/static/game2048.js' in index
    game = Game2048(seed=0)
    game.board = [2, 0, 0, 0] + [0] * 12
    game._refresh_moves()
    renderer_check([snapshot(game)], r"""
state.game = snapshots[0];
state.game.board_with_moves += '\n@left';
state.analysis = { probabilities: { right: 0.75, down: 0.25 } };
const data = parse2048(state.game);
assert.deepEqual(data.cells, [2, ...Array(15).fill(0)]);
const html = render2048(state.game);
assert.equal((html.match(/role="gridcell"/g) || []).length, 16);
assert.equal((html.match(/role="row"/g) || []).length, 4);
assert.deepEqual([...html.matchAll(/data-action="([^"]+)"/g)].map(match => match[1]).sort(), ['down', 'right']);
assert.match(html, /direction-left[^>]*disabled/);
assert.match(html, /75%/);
assert.match(html, /Score/);
assert.match(html, /Largest tile/);
assert.ok(!html.includes('undefined'));
assert.equal(renderers['2048'], render2048);
""")


def test_terminal_ui_displays_win_or_loss_without_legal_actions():
    won = Game2048(seed=0)
    won.board = [2048] + [0] * 15
    won.score = 20000
    won._refresh_moves()
    lost = Game2048(seed=0)
    lost.board = [2, 4, 2, 4, 4, 2, 4, 2] * 2
    lost._refresh_moves()
    renderer_check([snapshot(won), snapshot(lost)], r"""
assert.match(render2048(snapshots[0]), /2048 reached!/);
assert.match(render2048(snapshots[0]), /20000/);
assert.match(render2048(snapshots[1]), /Game over/);
for (const game of snapshots) {
  assert.ok(!render2048(game).includes('data-action='));
}
""")


def test_arrow_keys_respect_legality_busy_state_typing_and_other_games():
    game = Game2048(seed=0)
    game.board = [2] + [0] * 15
    game._refresh_moves()
    renderer_check([snapshot(game)], r"""
state.game = snapshots[0];
const played = [];
playMove = move => played.push(move);
let prevented = 0;
const send = (key, extra = {}) => listeners.keydown.forEach(handler => handler({
  key, target: { tagName: 'BODY' }, preventDefault() { prevented++; }, ...extra,
}));
send('ArrowRight');
send('ArrowUp');
assert.deepEqual(played, ['right']);
assert.equal(prevented, 2);
state.busy = true;
send('ArrowDown');
state.busy = false;
state.autoPlaying = true;
send('ArrowDown');
state.autoPlaying = false;
send('ArrowDown', { target: { tagName: 'TEXTAREA' } });
send('ArrowDown', { ctrlKey: true });
state.game.ended = true;
send('ArrowDown');
state.game.ended = false;
state.game.name = 'tictactoe';
send('ArrowDown');
assert.deepEqual(played, ['right']);
""")


def test_renderer_falls_back_to_escaped_text_for_malformed_board():
    renderer_check([snapshot(Game2048(seed=0))], r"""
const game = snapshots[0];
game.board_with_moves = '<img src=x onerror="alert(1)">';
assert.equal(parse2048(game), null);
assert.match(render2048(game), /fallback-board/);
assert.match(render2048(game), /&lt;img/);
assert.ok(!render2048(game).includes('<img'));
""")


def test_server_play_analyze_undo_redo_replay_and_reset(monkeypatch):
    monkeypatch.setattr(serve, "sessions", {})
    client = TestClient(serve.app)
    query = {"game": "2048,seed=0"}
    before = client.get("/state", params=query).json()
    assert before["name"] == "2048" and before["current_player"] == 0
    assert before["round"] == 0 and "Score: 0" in before["board"]
    assert "random" in client.get("/strategies", params=query).json()
    analysis = client.get("/analyze", params={**query, "strategy": "random"}).json()
    assert set(analysis["moves"]) == set(before["moves"])
    move = before["moves"][0]
    response = client.post("/do-one", params=query, json={"action": move})
    assert response.status_code == 200
    after = response.json()["state"]
    assert after["round"] == 1 and after["can_undo"]
    assert after["history"][0]["action"] == move
    undone = client.post("/undo", params=query).json()
    assert undone["board"] == before["board"] and undone["can_redo"]
    redone = client.post("/redo", params=query).json()
    assert redone["board"] == after["board"]
    client.post("/undo", params=query)
    replayed = client.post("/do-one", params=query, json={"action": move}).json()["state"]
    assert replayed["board"] == after["board"]
    assert client.post("/do-one", params=query, json={"action": "diagonal"}).status_code == 400
    assert client.post("/play-one", params=query, json={"strategy": "random"}).status_code == 200
    assert client.get("/reset", params=query).status_code == 200
    assert client.get("/state", params=query).json()["board"] == before["board"]


def test_server_terminal_result_uses_score_and_no_actions(monkeypatch):
    monkeypatch.setattr(serve, "sessions", {})
    query = {"game": "2048,seed=0"}
    session = serve.get_session(query["game"])
    session.game.board = [1024, 1024] + [0] * 14
    session.game._refresh_moves()
    client = TestClient(serve.app)
    response = client.post("/do-one", params=query, json={"action": "left"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["continue"] is False
    assert payload["state"]["ended"] and payload["state"]["moves"] == []
    assert payload["state"]["points"] == 2048
    assert "Status: won" in payload["state"]["board"]


def test_versus_renderer_switches_between_arrows_and_clickable_empty_cells():
    game = Versus2048(seed=0)
    before = snapshot(game, "2048versus")
    game.play_str(game.moves[0])
    after = snapshot(game, "2048versus")
    renderer_check([before, after], r"""
assert.equal(renderers['2048versus'], render2048);
assert.match(render2048(snapshots[0]), /Player 0 · Slide/);
state.game = snapshots[1];
state.analysis = { probabilities: { [state.game.moves[0]]: 0.5 } };
const parsed = parse2048(state.game);
assert.ok(parsed.versus && parsed.placing);
assert.ok([2, 4].includes(parsed.pending));
const html = render2048(state.game);
assert.match(html, /Player 1 · Place the next/);
assert.match(html, /50%/);
const actions = [...html.matchAll(/data-action="([^"]+)"/g)].map(match => match[1]);
assert.deepEqual(actions.sort(), [...state.game.moves].sort());
assert.equal(actions.length, parsed.cells.filter(value => value === 0).length);
assert.equal((html.match(/role="gridcell"/g) || []).length, 16);
assert.ok(!html.includes('game2048-controls'));
const played = [];
playMove = move => played.push(move);
listeners.keydown.forEach(handler => handler({
  key: 'ArrowRight', target: { tagName: 'BODY' }, preventDefault() {},
}));
assert.deepEqual(played, []);
// Rendered controls use the canonical list, including when it is narrowed.
state.game.moves = [state.game.moves[0]];
const narrowed = render2048(state.game);
assert.equal((narrowed.match(/data-action=/g) || []).length, 1);
""")


def test_versus_renderer_reports_the_actual_winning_role():
    winner = Versus2048(seed=0)
    winner.board = [2048] + [0] * 15
    winner._refresh_moves()
    loser = Versus2048(seed=0)
    loser.board = [2, 4, 2, 4, 4, 2, 4, 2] * 2
    loser._refresh_moves()
    renderer_check([snapshot(winner, "2048versus"), snapshot(loser, "2048versus")], r"""
assert.match(render2048(snapshots[0]), /Player 0 wins/);
assert.match(render2048(snapshots[1]), /Player 1 wins/);
assert.ok(!render2048(snapshots[0]).includes('data-action='));
assert.ok(!render2048(snapshots[1]).includes('data-action='));
""")


def test_versus_server_turns_placement_analysis_and_undo_restore_pending_tile(monkeypatch):
    monkeypatch.setattr(serve, "sessions", {})
    client = TestClient(serve.app)
    query = {"game": "2048versus,seed=0"}
    before = client.get("/state", params=query).json()
    assert before["current_player"] == 0
    slide = before["moves"][0]
    placing = client.post("/do-one", params=query, json={"action": slide}).json()["state"]
    assert placing["current_player"] == 1 and "Phase: spawn" in placing["board"]
    assert all(move.startswith("spawn ") for move in placing["moves"])
    policy = client.get("/analyze", params={**query, "strategy": "random"}).json()
    assert set(policy["moves"]) == set(placing["moves"])
    assert client.post("/do-one", params=query, json={"action": slide}).status_code == 400
    spawn = placing["moves"][0]
    after = client.post("/do-one", params=query, json={"action": spawn}).json()["state"]
    assert after["current_player"] == 0 and "Phase: slide" in after["board"]
    assert [entry["action"] for entry in after["history"]] == [slide, spawn]
    undone = client.post("/undo", params=query).json()
    assert undone["board"] == placing["board"] and undone["moves"] == placing["moves"]
    assert client.post("/redo", params=query).json()["board"] == after["board"]
    client.post("/undo", params=query)
    client.post("/undo", params=query)
    assert client.get("/state", params=query).json()["board"] == before["board"]
    replayed = client.post("/do-one", params=query, json={"action": slide}).json()["state"]
    assert replayed["board"] == placing["board"]
    assert client.post("/play-one", params=query, json={"strategy": "random"}).json()["state"]["current_player"] == 0


@pytest.mark.parametrize("name,game_class", [("2048", Game2048), ("2048versus", Versus2048)])
def test_nolimits_renderer_keeps_controls_above_2048_and_shows_actual_game_over(name, game_class):
    game = game_class(seed=0, nolimits=True)
    game.board = [2048, 2048] + [0] * 14
    game._refresh_moves()
    game.play_str("left")
    active = snapshot(game, name)
    if name == "2048versus":
        game.play_str("spawn d4")
    game.board = [4096, 2, 4, 8, 2, 4, 8, 16, 4, 8, 16, 32, 8, 16, 32, 64]
    game._refresh_moves()
    renderer_check([active, snapshot(game, name)], r"""
const active = snapshots[0];
assert.ok(parse2048(active).nolimits);
assert.equal(Math.max(...parse2048(active).cells), 4096);
const html = render2048(active);
assert.match(html, /No tile limit/);
assert.match(html, /tile-high/);
assert.ok(html.includes('data-action='));
assert.ok(!html.includes('wins') && !html.includes('Game over'));
if (active.name === '2048') assert.match(html, /Keep merging/);
else assert.match(html, /Player 1 · Place/);
const terminal = render2048(snapshots[1]);
assert.match(terminal, /No moves left/);
assert.ok(!terminal.includes('data-action='));
""")


@pytest.mark.parametrize("name", ["2048", "2048versus"])
def test_nolimits_server_keeps_playing_and_preserves_flag_across_undo_redo(monkeypatch, name):
    monkeypatch.setattr(serve, "sessions", {})
    query = {"game": f"{name},nolimits=True,seed=0"}
    session = serve.get_session(query["game"])
    session.game.board = [1024, 1024] + [0] * 14
    session.game._refresh_moves()
    client = TestClient(serve.app)
    response = client.post("/do-one", params=query, json={"action": "left"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["continue"] is True and not payload["state"]["ended"]
    assert "2048" in payload["state"]["board"] and "Nolimits: True" in payload["state"]["board"]
    if name == "2048versus":
        assert payload["state"]["current_player"] == 1
        assert payload["state"]["moves"] == [f"spawn {chr(97 + col)}{row + 1}" for row in range(4) for col in range(4) if row or col]
    client.post("/undo", params=query)
    redone = client.post("/redo", params=query).json()
    assert redone["board"] == payload["state"]["board"]
    assert session.game.nolimits
    if name == "2048versus":
        assert client.post("/do-one", params=query, json={"action": "spawn d4"}).json()["continue"]
