from dataclasses import asdict
import json
from pathlib import Path
import subprocess

import pytest

from boardrl.games.traversal.game import EXPLORER, SENTINEL, VANGUARD, Piece, Traversal
from boardrl.serve.serve import GameSnapshot


def snapshot(game, history=()):
    return asdict(GameSnapshot.from_game(
        "traversal", f"traversal,terrain={game.terrain},seed={game.seed}", game,
        history=history,
    ))


def renderer_check(snapshots, script):
    harness = r"""
const fs = require('fs');
const vm = require('vm');
const listeners = {};
const noopElement = () => ({
  addEventListener() {}, replaceChildren() {}, append() {}, focus() {},
  disabled: false, textContent: '', innerHTML: '', value: 'traversal',
});
const sandbox = {
  assert: require('assert/strict'),
  document: {
    addEventListener(type, handler) { (listeners[type] ||= []).push(handler); },
    querySelector: noopElement,
    querySelectorAll: () => [],
  },
  listeners,
  console,
  setTimeout,
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('boardrl/serve/static/app.js', 'utf8')
  .replace(/init\(\);\s*$/, ''), sandbox);
vm.runInContext(fs.readFileSync('boardrl/serve/static/traversal.js', 'utf8'), sandbox);
"""
    harness += f"vm.runInContext({json.dumps('const snapshots = ' + json.dumps(snapshots))}, sandbox);\n"
    harness += f"vm.runInContext({json.dumps(script)}, sandbox);\n"
    result = subprocess.run(
        ["node", "-e", harness], cwd=Path(__file__).parents[1],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_shared_ui_loads_traversal_assets():
    index = Path("boardrl/serve/static/index.html").read_text()
    assert '/static/traversal.css' in index
    assert '/static/traversal.js' in index


def test_real_observation_renders_board_heights_pieces_and_raw_scores():
    game = Traversal(terrain="flat")
    game.heights = tuple(tuple(row + col for col in range(10)) for row in range(10))
    game.scores = [3, 1]
    game.finished = [Piece(0, SENTINEL), Piece(1, EXPLORER)]
    game.captured = [Piece(1, VANGUARD)]
    for position in ((9, 2), (0, 4), (0, 3)):
        game.board.pop(position)
    game._finish("two_finished", 0)
    game._refresh_moves()
    renderer_check([snapshot(game)], r"""
const game = snapshots[0];
const parsed = parseTraversal(game);
assert.equal(parsed.cells.length, 100);
assert.equal(parsed.cells[99].coord, 'j10');
assert.equal(parsed.cells[99].height, 18);
assert.equal(parsed.cells.filter(cell => cell.type).length, 9);
assert.equal(parsed.cells.find(cell => cell.coord === 'h10').type, 'S');
assert.equal(parsed.cells.find(cell => cell.coord === 'h1').player, 1);
assert.deepEqual(parsed.scores, [3, 1]);
assert.equal(parsed.finished.length, 2);
assert.equal(parsed.captured[0].type, 'V');
const html = renderTraversal(game);
assert.equal((html.match(/data-traversal-coord=/g) || []).length, 100);
assert.match(html, /3 <small>pts/);
assert.match(html, /Player 0 finish/);
assert.match(html, /Player 1 finish/);
assert.match(html, /Explorer/);
assert.match(html, /Vanguard/);
assert.match(html, /Sentinel/);
assert.ok(!html.includes('undefined'));
assert.equal(renderers.traversal, renderTraversal);
""")


def test_active_observation_derives_scores_and_captures_from_material():
    game = Traversal(terrain="flat")
    game.board.pop((9, 3))
    game.board.pop((0, 4))
    game.finished = [Piece(0, VANGUARD)]
    game.captured = [Piece(1, EXPLORER)]
    game.scores = [2, 0]
    game._refresh_moves()
    renderer_check([snapshot(game)], r"""
const game = snapshots[0];
assert.ok(!game.board.includes('Scores:') && !game.board.includes('Captured:'));
const parsed = parseTraversal(game);
assert.deepEqual(parsed.scores, [2, 0]);
assert.deepEqual(parsed.captured, [{ player: 1, type: 'E' }]);
assert.match(renderTraversal(game), /2 <small>pts/);
""")


def test_terminal_elimination_uses_actual_scores_beyond_finished_pieces():
    game = Traversal(terrain="flat")
    game.board = {(5, 2): Piece(0, SENTINEL), (5, 3): Piece(1, EXPLORER)}
    game._refresh_moves()
    game.play_str("c6>d6")
    assert game.ended() and game.scores == [3, 0] and game.finished == []
    renderer_check([snapshot(game)], r"""
const parsed = parseTraversal(snapshots[0]);
assert.deepEqual(parsed.scores, [3, 0]);
assert.match(renderTraversal(snapshots[0]), /Player 0 wins · Army eliminated/);
""")


def test_source_selection_uses_canonical_actions_and_resets_after_move_or_undo():
    game = Traversal(terrain="flat")
    before = snapshot(game)
    move = game.moves[0]
    game.play_str(move)
    after = snapshot(game, [{"action": move}])
    renderer_check([before, after], r"""
state.game = snapshots[0];
state.game.board_with_moves += '\n@a1>off';
renderVisual();
assert.ok(!traversalTargets(state.game).sources.has('a1'));
assert.equal(traversalTargets(state.game).sources.size, 6);
const source = 'e10';
const expected = state.game.moves.filter(move => move.startsWith(source + '>'));
traversalView.select(source);
assert.equal(traversalView.source, source);
const html = renderTraversal(state.game);
const actions = [...html.matchAll(/data-action="([^"]+)"/g)]
  .map(match => match[1].replaceAll('&gt;', '>'));
assert.deepEqual(actions.sort(), expected.sort());
assert.match(html, /Cancel/);
state.analysis = { probabilities: { [expected[0]]: 0.25 } };
assert.match(renderTraversal(state.game), /25%/);
assert.equal(traversalView.source, source); // policy refresh keeps the choice
state.busy = true;
traversalView.select('d10');
assert.equal(traversalView.source, source);
state.busy = false;
traversalView.select('d10');
assert.equal(traversalView.source, 'd10');
traversalView.select('d10');
assert.equal(traversalView.source, null);
traversalView.select(source);
state.game = snapshots[1];
const moved = renderTraversal(state.game);
assert.equal(traversalView.source, null);
assert.match(moved, /traversal-last-move/);
traversalView.select([...traversalTargets(state.game).sources][0]);
state.game = snapshots[0];
renderTraversal(state.game);
assert.equal(traversalView.source, null);
""")


@pytest.mark.parametrize("player", [0, 1])
def test_capture_and_finish_controls_use_correct_player_edge(player):
    game = Traversal(terrain="flat")
    row = 0 if player == 0 else 9
    game.turn = player
    game.board = {
        (row, 0): Piece(player, SENTINEL),
        (row, 1): Piece(1 - player, EXPLORER),
        (4, 9): Piece(1 - player, VANGUARD),
    }
    game._refresh_moves()
    renderer_check([snapshot(game)], r"""
state.game = snapshots[0];
renderVisual();
const player = state.game.current_player;
const source = player === 0 ? 'a1' : 'a10';
traversalView.select(source);
const html = renderTraversal(state.game);
assert.match(html, new RegExp(`traversal-goal-${player}[\\s\\S]*?data-action="${source}&gt;off"`));
assert.equal((html.match(/class="traversal-finish"/g) || []).length, 1);
assert.match(html, /Win now/);
assert.match(html, /Finish Sentinel · \+3/);
assert.match(html, /traversal-capture/);
assert.match(html, new RegExp(`data-action="${source}&gt;b${player === 0 ? 1 : 10}"`));
assert.match(html, /capture here/);
""")


@pytest.mark.parametrize("winner", [0, 1, None])
def test_terminal_result_reads_winner_instead_of_tied_raw_scores(winner):
    game = Traversal(terrain="flat")
    game._finish("move_limit_advancement" if winner is not None else "repetition", winner)
    game._refresh_moves()
    renderer_check([snapshot(game)], r"""
state.game = snapshots[0];
const data = parseTraversal(state.game);
const html = renderTraversal(state.game);
const expected = data.winner === 'draw' ? 'Draw' : `Player ${data.winner} wins`;
assert.match(html, new RegExp(expected));
assert.match(html, /0 <small>pts/);
assert.ok(!html.includes('data-action='));
assert.equal(traversalTargets(state.game, 'c10').sources.size, 0);
traversalView.select('c10');
assert.equal(traversalView.source, null);
""")


def test_cancel_keyboard_metadata_escaping_and_text_fallback():
    renderer_check([snapshot(Traversal(terrain="flat"))], r"""
state.game = snapshots[0];
renderVisual();
traversalView.select('c10');
const event = { key: 'Escape', target: { tagName: 'BUTTON' }, preventDefault() {} };
listeners.keydown.forEach(handler => handler(event));
assert.equal(traversalView.source, null);
traversalView.select('c10');
listeners.click.forEach(handler => handler({ target: {
  closest(selector) { return selector === '[data-traversal-clear]' ? {} : null; },
} }));
assert.equal(traversalView.source, null);
const malicious = '<img src=x onerror="alert(1)">';
state.game.board_with_moves = state.game.board_with_moves.replace('Terrain: flat', 'Terrain: ' + malicious);
assert.ok(!renderTraversal(state.game).includes(malicious));
assert.match(renderTraversal(state.game), /&lt;img/);
state.game.board_with_moves = '<broken board>';
const fallback = renderTraversal(state.game);
assert.match(fallback, /fallback-board/);
assert.match(fallback, /&lt;broken board&gt;/);
assert.equal(parseTraversalMove('a0>b1'), null);
assert.equal(parseTraversalMove('a1>j11'), null);
assert.equal(parseTraversalMove('a1>off').destination, 'off');
""")
