from pathlib import Path
import subprocess


def run_santorini_renderer_check(script: str):
    harness = rf"""
const fs = require('fs');
const vm = require('vm');
const appCode = fs.readFileSync('boardrl/serve/static/app.js', 'utf8');
const santoriniCode = fs.readFileSync('boardrl/serve/static/santorini.js', 'utf8');
const noopElement = () => ({{
  addEventListener() {{}},
  replaceChildren() {{}},
  append() {{}},
  disabled: false,
  textContent: '',
  innerHTML: '',
  value: 'random',
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
vm.runInContext(appCode.replace(/init\(\);\s*$/, ''), sandbox);
vm.runInContext(santoriniCode, sandbox);
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


def test_santorini_assets_are_loaded_by_shared_ui():
    index = Path("boardrl/serve/static/index.html").read_text()
    assert '/static/santorini.css' in index
    assert '/static/santorini.js' in index


def test_santorini_renderer_parses_setup_and_board():
    run_santorini_renderer_check(
        r"""
const setup = {
  name: 'santorini',
  current_player: 0,
  history: [],
  moves: ['P:a1', 'P:b1'],
  board: [
    '>0 setup',
    '   a  b  c  d  e',
    '1 0. 0. 0. 0. 0.',
    '2 0. 0. 0. 0. 0.',
    '3 0. 0. 0. 0. 0.',
    '4 0. 0. 0. 0. 0.',
    '5 0. 0. 0. 0. 0.',
  ].join('\n') + '\n',
  board_with_moves: [
    '>0 setup',
    '   a  b  c  d  e',
    '1 0. 0. 0. 0. 0.',
    '2 0. 0. 0. 0. 0.',
    '3 0. 0. 0. 0. 0.',
    '4 0. 0. 0. 0. 0.',
    '5 0. 0. 0. 0. 0.',
    '@P:a1',
    '@P:b1',
  ].join('\n'),
};
const parsed = sandbox.parseSantorini(setup);
const html = sandbox.renderSantorini(setup);
if (parsed.phase !== 'setup') throw new Error(`wrong phase ${parsed.phase}`);
if (parsed.cells.length !== 25) throw new Error(`expected 25 cells, got ${parsed.cells.length}`);
if ((html.match(/data-santorini-coord=/g) || []).length !== 25) {
  throw new Error('renderer did not render 25 board cells');
}
if (!html.includes('target-setup') || !html.includes('Place a worker')) {
  throw new Error('setup cells are not visibly playable');
}
"""
    )


def test_santorini_click_targets_follow_relative_move_destination_build_sequence():
    run_santorini_renderer_check(
        r"""
const game = {
  name: 'santorini',
  current_player: 0,
  history: [{ action: 'P:a5' }],
  moves: [
    'a1>r+l',
    'a1>r+d',
    'a1>d+u',
    'e5>ul+ul',
  ],
  board: [
    '>0 play',
    '   a  b  c  d  e',
    '1 0O 0. 0. 0. 0X',
    '2 0. 0. 0. 0. 0.',
    '3 0. 0. 2. 1. 0.',
    '4 0. 0. 0. 0. 0.',
    '5 0X 0. 0. 0. 0O',
  ].join('\n') + '\n',
  board_with_moves: '',
};
const sources = sandbox.santoriniTargets(game, { source: null, destination: null });
if (sources.kind !== 'source' || !sources.targets.a1 || !sources.targets.e5) {
  throw new Error('source selection did not expose both movable workers');
}
const destinations = sandbox.santoriniTargets(game, { source: 'a1', destination: null });
if (destinations.kind !== 'destination' || !destinations.targets.b1 || !destinations.targets.a2) {
  throw new Error('relative move directions did not resolve to board destinations');
}
const builds = sandbox.santoriniTargets(game, { source: 'a1', destination: 'b1' });
if (builds.kind !== 'build' || !builds.targets.a1 || !builds.targets.b2) {
  throw new Error('relative build directions did not resolve to board targets');
}
const diagonal = sandbox.parseSantoriniMove('e5>ul+ul');
if (diagonal.destination !== 'd4' || diagonal.build !== 'c3') {
  throw new Error('diagonal relative action was parsed incorrectly');
}
"""
    )


def test_santorini_winning_move_has_no_build_step():
    run_santorini_renderer_check(
        r"""
const game = {
  name: 'santorini',
  current_player: 0,
  history: [],
  moves: ['b2>dr'],
  board: [
    '>0 play',
    '   a  b  c  d  e',
    '1 0. 0. 0. 0. 0.',
    '2 0. 2O 0. 0. 0.',
    '3 0. 0. 3. 0. 0.',
    '4 0. 0. 0. 0. 0.',
    '5 0X 0. 0. 0. 0O',
  ].join('\n') + '\n',
  board_with_moves: '',
};
const target = sandbox.santoriniTargets(game, { source: 'b2', destination: null });
if (target.kind !== 'destination' || target.targets.c3[0] !== 'b2>dr') {
  throw new Error('winning relative destination was not exposed directly');
}
const parsed = sandbox.parseSantoriniMove('b2>dr');
if (parsed.destination !== 'c3') throw new Error('winning direction resolved incorrectly');
if (parsed.build !== null) throw new Error('winning move unexpectedly requires a build');
"""
    )
