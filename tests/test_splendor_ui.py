from pathlib import Path
import subprocess


def run_splendor_renderer_check(script: str):
    harness = rf"""
const fs = require('fs');
const vm = require('vm');
const appCode = fs.readFileSync('boardrl/serve/static/app.js', 'utf8');
const splendorCode = fs.readFileSync('boardrl/serve/static/splendor.js', 'utf8');
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
vm.runInContext(splendorCode, sandbox);
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


def test_splendor_assets_are_loaded_by_shared_ui():
    index = Path("boardrl/serve/static/index.html").read_text()
    assert "/static/splendor.css" in index
    assert "/static/splendor.js" in index


def test_splendor_renderer_parses_market_players_and_hidden_reserve():
    run_splendor_renderer_check(
        r"""
const game = {
  name: 'splendor',
  current_player: 0,
  moves: ['T:WBG', 'T:WW', 'R:1.0', 'R:1.D', 'B:1.0', 'B:1.0~B'],
  board: [
    '>0 main v0',
    'Bank W4B4G4R4K4Y5',
    'Nobles N2[W4B4] N5[B3G3R3] N8[G3R3K3]',
    'T3(16) 0=K4[R7] 1=B3[W3G3R3K5] 2=W4[K7] 3=G5[B7G3]',
    'T2(26) 0=R2[B4G2W1] 1=K1[W3B2G2] 2=B3[R6] 3=G2[B5]',
    'T1(36) 0=K1[B4] 1=W0[B3] 2=G0[R3] 3=R0[K3]',
    'P0* S1 D2 TW2B1G0R0K0Y1 CW1B0G1R0K0 H0=K0[W1B1G1R1]',
    'P1 S0 D1 TW0B2G1R1K0Y0 CW0B1G0R0K0 H0=? 1=R0[B1G1K1]',
  ].join('\n') + '\n',
  board_with_moves: '',
  history: [],
};
const parsed = sandbox.parseSplendor(game);
if (parsed.market[1].cards.length !== 4) throw new Error('wrong tier-1 cards');
if (parsed.players.length !== 2) throw new Error('wrong player count');
if (!parsed.players[1].reserved[0].hidden) throw new Error('blind reserve leaked in parser');
if (parsed.bank.Y !== 5) throw new Error('gold bank not parsed');
const html = sandbox.renderSplendor(game);
if (!html.includes('data-action="B:1.0~B"')) throw new Error('gold payment buy action missing');
if (!html.includes('data-action="R:1.D"')) throw new Error('blind reserve action missing');
if (!html.includes('splendor-card-back')) throw new Error('hidden reserve not rendered facedown');
"""
    )


def test_splendor_renderer_exposes_compact_followup_phases():
    run_splendor_renderer_check(
        r"""
const base = [
  'Bank W1B1G1R1K1Y5',
  'Nobles N2[W4B4] N9[W3B3G3]',
  'T3(16) 0=K4[R7] 1=B4[W7] 2=W4[K7] 3=G4[B7]',
  'T2(26) 0=R2[B4G2W1] 1=K1[W3B2G2] 2=B3[R6] 3=G2[B5]',
  'T1(36) 0=K1[B4] 1=W0[B3] 2=G0[R3] 3=R0[K3]',
  'P0* S0 D0 TW3B3G2R2K1Y1 CW0B0G0R0K0 H-',
  'P1 S0 D0 TW0B0G0R0K0Y0 CW0B0G0R0K0 H-',
];
const discard = {
  name: 'splendor',
  current_player: 0,
  moves: ['D:WY', 'D:BB'],
  board: ['>0 discard v0', ...base].join('\n') + '\n',
  board_with_moves: '',
  history: [],
};
const discardHtml = sandbox.renderSplendor(discard);
if (!discardHtml.includes('Return WY') || !discardHtml.includes('data-action="D:BB"')) {
  throw new Error('discard phase controls missing');
}
const noble = {
  ...discard,
  moves: ['N:2', 'N:9'],
  board: ['>0 noble v0', ...base].join('\n') + '\n',
};
const nobleHtml = sandbox.renderSplendor(noble);
if (!nobleHtml.includes('data-action="N:2"') || !nobleHtml.includes('data-action="N:9"')) {
  throw new Error('noble choice controls missing');
}
"""
    )
