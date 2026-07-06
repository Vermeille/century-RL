import argparse
import os

import torch
from natsort import natsorted
from pathlib import Path
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse

import pyximport

pyximport.install(setup_args={"script_args": ["--cython-cplus"]})
from boardrl.games import games_library
from boardrl.cyutils import fast_sample
from boardrl.utils import ModelPool


parser = argparse.ArgumentParser()
parser.add_argument("--game", default="century")
args, _ = parser.parse_known_args()

game_name = args.game.split(",")[0]
game_desc = games_library(args.game)
game = game_desc.make_game()
game_dir = Path(__file__).parent.parent / "games" / game_name


class Strategies:
    def __init__(
        self,
        game_desc=game_desc,
        cache_len: int = 5,
        batch_size: int = 32,
        timeout: float = 0.01,
    ):
        self.game_desc = game_desc
        self.strategies = self.populate_strategies()
        self.cache = []
        self.cache_len = cache_len
        # ModelPool handles loading models and batching inference similar to
        # the training setup in ``main.py``.
        self.pool = ModelPool(None, batch_size, timeout)

    @staticmethod
    def populate_strategies():
        strategies = []
        # find all .pth files in all directories
        for root, dirs, files in os.walk("."):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for file in files:
                if file.endswith(".pth"):
                    # strategies.append(f"argmax:{os.path.join(root, file)}")
                    strategies.append(
                        f"policy_sampling,model={os.path.join(root, file)}"
                    )

        strategies = natsorted(strategies)
        strategies += [
            "random",
            "random_buy",
            "all_actions_then_random_buy",
            "no_actions_random_buy",
        ]
        return strategies

    def get_strategy(self, name):
        if name not in self.strategies:
            raise HTTPException(status_code=400, detail="Unknown strategy")
        for cache_name, strategy in self.cache:
            if cache_name == name:
                return strategy
        self.cache = self.cache[-self.cache_len :]
        # Use the shared ModelPool when instantiating strategies so model
        # arguments are resolved correctly.
        self.cache.append(
            (name, self.game_desc.strategy_from_string(name, model=self.pool))
        )
        return self.cache[-1][1]


strategies = Strategies(game_desc)

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def read_root():
    return (game_dir / "ui.html").read_text()


def end_response():
    return {
        "continue": False,
        "points": game.diff_points_for(0),
        "num_turns": game.round(),
    }


@app.get("/strategies")
def get_strategies():
    return strategies.strategies


@app.get("/board", response_class=PlainTextResponse)
def board():
    if game.ended():
        return game.display(force=0)
    else:
        return game.display_with_moves()


@app.get("/analyze")
async def analyze(strategy: str):
    pred = await strategies.get_strategy(strategy)(game)
    return pred[1]


@app.post("/do")
async def do(
    action: str = Body(..., embed=True), strategy: str = Body(..., embed=True)
):
    if game.ended():
        return end_response()

    player = game.current_player()
    game.play_str(action)
    if game.ended():
        return end_response()

    strategy_fn = strategies.get_strategy(strategy)
    while not game.ended() and game.current_player() != player:
        dist, _ = await strategy_fn(game)
        action_idx = fast_sample(torch.softmax(dist, dim=0))
        game.play_idx(action_idx)

    if game.ended():
        return end_response()

    return {"continue": True}


@app.post("/do-one")
async def do_one(action: str = Body(..., embed=True)):
    if game.ended():
        return end_response()

    game.play_str(action)
    if game.ended():
        return end_response()
    return {"continue": True}


@app.post("/play-one")
async def play_one(strategy: str = Body(..., embed=True)):
    if game.ended():
        return end_response()

    dist, _ = await strategies.get_strategy(strategy)(game)
    action_idx = fast_sample(torch.softmax(dist, dim=0))
    game.play_idx(action_idx)

    if game.ended():
        return end_response()

    return {"continue": True}


@app.get("/reset")
async def reset():
    global game
    game = game_desc.make_game()
    return True


# ----------------------- Game-agnostic visualization -----------------------


@app.get("/agnostic", response_class=HTMLResponse)
def agnostic_ui():
    # Minimal, game-agnostic UI. Shows the board text (before first '@') in an
    # editable area, lists clickable '@' moves, and visualizes action
    # distribution. Edits auto-refresh the viz; Reset restores original state.
    html = r"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>BoardRL – Game-Agnostic Viewer</title>
    <style>
      :root {
        --bg: #0f172a; /* slate-900 */
        --panel: #111827; /* gray-900 */
        --muted: #475569; /* slate-500 */
        --text: #e5e7eb; /* gray-200 */
        --accent: #22c55e; /* green-500 */
        --accent2: #38bdf8; /* sky-400 */
        --danger: #ef4444; /* red-500 */
      }
      html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; height: 100%; }
      .wrap { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; }
      .panel { background: var(--panel); border-radius: 8px; padding: 12px; border: 1px solid #1f2937; }
      .row { display: flex; gap: 8px; align-items: center; }
      .controls { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
      label { color: var(--muted); font-size: 12px; }
      select, button { background: #0b1220; color: var(--text); border: 1px solid #293140; border-radius: 6px; padding: 6px 10px; cursor: pointer; }
      button:hover { border-color: #3a4456; }
      button.reset { color: var(--danger); }
      textarea { width: 100%; height: 360px; resize: vertical; background: #0b1220; color: var(--text); border: 1px solid #293140; border-radius: 6px; padding: 8px; white-space: pre; }
      .moves { display: flex; flex-direction: column; gap: 8px; }
      .move-row { display: grid; grid-template-columns: auto 1fr auto; gap: 8px; align-items: center; }
      .move-btn { white-space: pre; }
      .bar { height: 14px; background: #111827; border: 1px solid #293140; border-radius: 7px; overflow: hidden; position: relative; }
      .bar > .fill { height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2)); width: 0%; }
      .prob { color: var(--muted); font-size: 12px; }
      .hint { color: var(--muted); font-size: 12px; margin-top: 6px; }
      .section-title { color: var(--muted); margin: 0 0 8px; font-size: 12px; letter-spacing: .04em; text-transform: uppercase; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="panel">
        <div class="controls">
          <div class="row">
            <label for="strategy">Strategy</label>
            <select id="strategy"></select>
          </div>
          <button id="resetBtn" class="reset">Reset</button>
        </div>
        <h4 class="section-title">State (editable, before first @)</h4>
        <textarea id="state" spellcheck="false"></textarea>
        <div class="hint">Editing updates the distribution below using the model. Newlines preserved.</div>
      </div>
      <div class="panel">
        <h4 class="section-title">Moves and Distribution</h4>
        <div id="moves" class="moves"></div>
      </div>
    </div>

    <script>
      const el = (sel) => document.querySelector(sel);
      const stateEl = el('#state');
      const movesEl = el('#moves');
      const strategyEl = el('#strategy');
      const resetBtn = el('#resetBtn');

      let originalState = '';
      let currentMoves = []; // array of {label:'@X', id:'X'} in order
      let movesBlock = '';
      let isEditing = false;

      function softmax(xs) {
        if (!xs.length) return [];
        const m = Math.max(...xs);
        const exps = xs.map(x => Math.exp(x - m));
        const s = exps.reduce((a,b)=>a+b, 0) || 1;
        return exps.map(e => e / s);
      }

      function toProbs(moveToVal) {
        const vals = currentMoves.map(m => moveToVal[m.id] ?? 0);
        // If looks like probs (sum ~1 and within [0,1]), use as-is, else softmax
        const sum = vals.reduce((a,b)=>a+b,0);
        const all01 = vals.every(v => v >= -1e-6 && v <= 1+1e-6);
        const near1 = Math.abs(sum - 1) < 1e-3;
        const probs = (near1 && all01) ? vals : softmax(vals);
        const out = {};
        currentMoves.forEach((m,i)=> out[m.id] = probs[i] ?? 0);
        return out;
      }

      function renderMoves() {
        movesEl.innerHTML = '';
        for (const m of currentMoves) {
          const row = document.createElement('div');
          row.className = 'move-row';
          const btn = document.createElement('button');
          btn.className = 'move-btn';
          btn.textContent = '@' + m.id;
          btn.addEventListener('click', () => playMove(m.id));
          const bar = document.createElement('div');
          bar.className = 'bar';
          const fill = document.createElement('div');
          fill.className = 'fill';
          bar.appendChild(fill);
          const prob = document.createElement('div');
          prob.className = 'prob';
          prob.textContent = '0.000';
          row.appendChild(btn);
          row.appendChild(bar);
          row.appendChild(prob);
          movesEl.appendChild(row);
        }
      }

      function updateBars(moveToProb) {
        const rows = movesEl.querySelectorAll('.move-row');
        rows.forEach((row, idx) => {
          const p = Math.max(0, Math.min(1, moveToProb[currentMoves[idx].id] ?? 0));
          row.querySelector('.fill').style.width = (p * 100).toFixed(1) + '%';
          row.querySelector('.prob').textContent = p.toFixed(3);
        });
      }

      function parseBoard(boardText) {
        // Split at the first line that starts with '@'
        const atIdx = boardText.indexOf('\n@');
        const at0 = boardText.startsWith('@') ? 0 : (atIdx >= 0 ? atIdx + 1 : -1);
        const state = at0 === -1 ? boardText : boardText.slice(0, at0);
        const movesLines = (boardText.match(/^@.*$/gm) || []).map(s => s.trim());
        const moves = movesLines.map(line => ({ label: line, id: line.slice(1).trim() }));
        const movesText = movesLines.join('\n');
        return { state, moves, movesText };
      }

      async function refreshBoard(andAnalyze = true) {
        const board = await fetch('/board').then(r => r.text());
        const { state, moves, movesText } = parseBoard(board);
        originalState = state;
        stateEl.value = state;
        currentMoves = moves;
        movesBlock = movesText;
        renderMoves();
        isEditing = false;
        if (andAnalyze) await analyzeServer();
      }

      async function analyzeServer() {
        // Default: use /analyze for the current server-side game state
        if (!strategyEl.value) return;
        const text = (stateEl.value || '').replace(/\n*$/, '') + (movesBlock ? '\n' + movesBlock : '');
        const data = await fetch('/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ strategy: strategyEl.value, state: text })
        }).then(r => r.json());
        const probs = toProbs(data.moves || {});
        updateBars(probs);
      }

      async function analyzeText() {
        // Analyze the edited text through the generic /analyze endpoint (POST)
        const text = (stateEl.value || '').replace(/\n*$/, '') + (movesBlock ? '\n' + movesBlock : '');
        const resp = await fetch('/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ strategy: strategyEl.value, state: text })
        });
        const data = await resp.json();
        const probs = toProbs(data.moves || {});
        updateBars(probs);
      }

      let typingTimer = null;
      stateEl.addEventListener('input', () => {
        isEditing = true;
        if (typingTimer) clearTimeout(typingTimer);
        typingTimer = setTimeout(analyzeText, 250);
      });

      resetBtn.addEventListener('click', () => {
        stateEl.value = originalState;
        isEditing = false;
        analyzeServer();
      });

      async function playMove(moveId) {
        if (!strategyEl.value) return;
        const r = await fetch('/do-one', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: moveId, strategy: strategyEl.value }) });
        const data = await r.json();
        await refreshBoard(true);
      }

      function pickDefaultStrategy(strats) {
        // Prefer first policy_sampling entry, else fallback to random
        for (const s of strats) if (s.startsWith('policy_sampling')) return s;
        return strats.find(s => s === 'random') || strats[0] || '';
      }

      async function init() {
        const strategies = await fetch('/strategies').then(r=>r.json());
        const def = pickDefaultStrategy(strategies);
        for (const s of strategies) {
          const opt = document.createElement('option');
          opt.value = s; opt.textContent = s; if (s === def) opt.selected = true;
          strategyEl.appendChild(opt);
        }
        strategyEl.addEventListener('change', () => {
          if (isEditing) analyzeText(); else analyzeServer();
        });
        await refreshBoard(true);
      }

      init();
    </script>
  </body>
 </html>
    """
    return HTMLResponse(content=html)


@app.post("/analyze")
async def analyze_post(
    strategy: str = Body(..., embed=True),
    state: str | None = Body(None, embed=True),
):
    moves = state.split("@")[1:]
    pred, value = await strategies.get_strategy(strategy).nn(state)
    print(value)
    return {"moves": {m.strip(): p.item() for m, p in zip(moves, pred[0])}}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("boardrl.serve.serve:app")
