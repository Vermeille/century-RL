const TRAVERSAL_PIECES = {
  E: { name: "Explorer", movement: 5, strength: 1, ability: "Diagonal step or two-square leap" },
  V: { name: "Vanguard", movement: 4, strength: 2, ability: "+1 attack on a full-movement charge" },
  S: { name: "Sentinel", movement: 3, strength: 3, ability: "Finish first to win immediately" },
};

function parseTraversalMove(move) {
  const match = move.match(/^([a-j](?:10|[1-9]))>([a-j](?:10|[1-9])|off)$/);
  return match ? { move, source: match[1], destination: match[2] } : null;
}

function parseTraversal(game) {
  const lines = linesWithoutMoves(game);
  const field = (name, fallback = "") => {
    const prefix = `${name}: `;
    return lines.find((line) => line.startsWith(prefix))?.slice(prefix.length) ?? fallback;
  };
  const heightStart = lines.indexOf("Terrain (height):") + 2;
  const pieceStart = lines.indexOf("Pieces:") + 2;
  if (heightStart < 2 || pieceStart < 2) return null;
  const cells = [];
  for (let row = 0; row < 10; row++) {
    const heights = lines[heightStart + row]?.trim().split(/\s+/) || [];
    const pieces = lines[pieceStart + row]?.trim().split(/\s+/) || [];
    if (heights.length !== 11 || pieces.length !== 11 ||
        Number(heights[0]) !== row + 1 || Number(pieces[0]) !== row + 1) return null;
    for (let col = 0; col < 10; col++) {
      const height = Number(heights[col + 1]);
      const token = pieces[col + 1];
      if (!Number.isInteger(height) || height < 0 || !/^(\.|[01][EVS])$/.test(token)) return null;
      cells.push({
        coord: `${String.fromCharCode(97 + col)}${row + 1}`, row, col, height,
        player: token === "." ? null : Number(token[0]),
        type: token === "." ? null : token[1],
      });
    }
  }
  const pieceList = (name) => field(name).split(/\s+/)
    .filter((token) => /^[01][EVS]$/.test(token))
    .map((token) => ({ player: Number(token[0]), type: token[1] }));
  const finished = pieceList("Finished");
  const scores = field("Scores").split(/\s+/).map(Number);
  const finishScores = [0, 1].map((player) => finished
    .filter((piece) => piece.player === player)
    .reduce((sum, piece) => sum + TRAVERSAL_PIECES[piece.type].strength, 0));
  const captured = [];
  for (const player of [0, 1]) {
    for (const type of Object.keys(TRAVERSAL_PIECES)) {
      const present = [...cells, ...finished].filter((piece) => piece.player === player && piece.type === type).length;
      for (let i = present; i < 2; i++) captured.push({ player, type });
    }
  }
  return {
    cells, scores: scores.length === 2 && scores.every(Number.isFinite) ? scores : finishScores,
    preset: field("Terrain", "Terrain"), status: field("Status", "ongoing"),
    winner: field("Winner"), finished, captured,
    turn: field("Turn").split(" ")[0],
  };
}

function traversalTargets(game, source = null) {
  const parts = game.ended ? [] : (game.moves || []).map(parseTraversalMove).filter(Boolean);
  const sources = new Set(parts.map((part) => part.source));
  const selected = sources.has(source) ? source : null;
  const targets = new Map();
  for (const part of parts) {
    if (selected && part.source !== selected) continue;
    const key = selected ? part.destination : part.source;
    if (!targets.has(key)) targets.set(key, []);
    targets.get(key).push(part.move);
  }
  return { selected, sources, targets };
}

function traversalPiece(piece, extraClass = "") {
  return `<span class="traversal-piece traversal-player-${piece.player} traversal-type-${piece.type} ${extraClass}" aria-hidden="true">${piece.type}</span>`;
}

function traversalResult(data) {
  const reasons = {
    sentinel: "First Sentinel finish", two_finished: "Two pieces finished",
    elimination: "Army eliminated", repetition: "Repeated position",
    stalemate: "No legal moves", move_limit_score: "Move limit · Score",
    move_limit_advancement: "Move limit · Advancement tiebreak",
    move_limit_draw: "Move limit",
  };
  const winner = data.winner === "draw" ? "Draw" :
    /^[01]$/.test(data.winner) ? `Player ${data.winner} wins` : "Game over";
  return `${winner} · ${reasons[data.status] || data.status}`;
}

function traversalScoreCard(data, game, player) {
  const remaining = data.cells.filter((cell) => cell.player === player).length;
  const tokens = (pieces) => pieces.filter((piece) => piece.player === player)
    .map((piece) => traversalPiece(piece, "traversal-piece-small")).join("") || "—";
  return `<section class="traversal-score traversal-score-${player} ${!game.ended && game.current_player === player ? "active" : ""}" aria-label="Player ${player}, ${data.scores[player]} points">
    <div class="traversal-score-heading"><span class="traversal-player-dot traversal-player-${player}"></span><strong>Player ${player}</strong><span class="traversal-points">${data.scores[player]} <small>pts</small></span></div>
    <div class="traversal-army"><span>${remaining} on board</span><span>Finished <span class="traversal-tokens">${tokens(data.finished)}</span></span><span>Captured <span class="traversal-tokens">${tokens(data.captured)}</span></span></div>
  </section>`;
}

function traversalCell(cell, targetData, selected, lastMove, minHeight, maxHeight) {
  const moves = targetData.targets.get(cell.coord) || [];
  const selectable = targetData.sources.has(cell.coord);
  const action = selected && moves.length ? moves[0] : null;
  const capture = Boolean(action && cell.type);
  const mass = moves.reduce((sum, move) => sum + probabilityFor(move), 0);
  const fraction = (cell.height - minHeight) / Math.max(1, maxHeight - minHeight);
  const background = `hsl(${Math.round(108 - fraction * 70)} 32% ${94 - fraction * 12}%)`;
  const classes = [
    "traversal-cell", selectable ? "traversal-source" : "",
    cell.coord === selected ? "selected" : "",
    action ? (capture ? "traversal-capture" : "traversal-destination") : "",
    cell.coord === lastMove?.source || cell.coord === lastMove?.destination ? "traversal-last-move" : "",
  ].filter(Boolean).join(" ");
  const occupant = cell.type ? `player ${cell.player} ${TRAVERSAL_PIECES[cell.type].name}` : "empty";
  const interaction = action ? (capture ? ", capture here" : ", move here") :
    selectable ? (cell.coord === selected ? ", selected; click to cancel" : ", select piece") : "";
  const label = `${cell.coord}, height ${cell.height}, ${occupant}${interaction}`;
  return `<button type="button" class="${classes}" data-traversal-coord="${cell.coord}"
    ${action ? `data-action="${escapeHtml(action)}"` : ""}
    ${!selectable && !action ? "disabled" : ""} ${selectable ? `aria-pressed="${cell.coord === selected}"` : ""}
    style="--terrain-color:${background};--terrain-relief:${1 + Math.round(fraction * 4)}px;--policy-mass:${Math.min(0.3, mass * 0.6).toFixed(3)}"
    title="${escapeHtml(label)}" aria-label="${escapeHtml(label)}">
    <span class="traversal-heat" aria-hidden="true"></span>
    <span class="traversal-height" aria-hidden="true">${cell.height}</span>
    ${cell.type ? traversalPiece(cell) : action ? '<span class="traversal-target-dot" aria-hidden="true"></span>' : ""}
    ${capture ? '<span class="traversal-capture-mark" aria-hidden="true">×</span>' : ""}
    ${mass > 0 ? `<span class="traversal-prob" title="Policy probability across all legal moves">${Math.round(mass * 100)}%</span>` : ""}
  </button>`;
}

function traversalGoal(data, selectedCell, finishMove, player) {
  const canFinish = finishMove && selectedCell?.player === player;
  const kind = selectedCell ? TRAVERSAL_PIECES[selectedCell.type] : null;
  const instant = canFinish && selectedCell.type === "S" && data.finished.length === 0;
  return `<div class="traversal-goal traversal-goal-${player}">
    <span>${player === 0 ? "↑ " : "↓ "}Player ${player} finish</span>
    ${canFinish ? `<button type="button" class="traversal-finish" data-action="${escapeHtml(finishMove)}">Finish ${escapeHtml(kind.name)} · +${kind.strength}${instant ? " · Win now" : ""} ${moveBadge(finishMove)}</button>` : ""}
  </div>`;
}

class TraversalView {
  constructor() {
    this.source = null;
    this.revision = null;
  }

  render(game) {
    const data = parseTraversal(game);
    if (!data) return renderGeneric(game);
    const revision = `${game.spec || game.name}\n${game.board || game.board_with_moves}\n${(game.moves || []).join("\n")}`;
    if (revision !== this.revision) {
      this.source = null;
      this.revision = revision;
    }
    const targetData = traversalTargets(game, this.source);
    this.source = targetData.selected;
    const selectedCell = data.cells.find((cell) => cell.coord === this.source);
    const finishMove = targetData.selected ? targetData.targets.get("off")?.[0] : null;
    const lastMove = parseTraversalMove(game.history?.at(-1)?.action || "");
    const minHeight = Math.min(...data.cells.map((cell) => cell.height));
    const maxHeight = Math.max(...data.cells.map((cell) => cell.height));
    const instruction = game.ended ? traversalResult(data) : selectedCell ?
      `${this.source} · ${TRAVERSAL_PIECES[selectedCell.type].name} — choose a destination${finishMove ? " or the finish edge" : ""}` : "Choose one of the highlighted pieces";
    return `<div class="traversal-board">
      <header class="traversal-heading"><span class="traversal-preset">${escapeHtml(data.preset)}</span><span>Turn ${escapeHtml(data.turn)}</span></header>
      <div class="traversal-scores">${[0, 1].map((player) => traversalScoreCard(data, game, player)).join("")}</div>
      <div class="traversal-instruction ${game.ended ? "traversal-result" : ""}" aria-live="polite"><span>${escapeHtml(instruction)}</span>${this.source ? '<button type="button" class="traversal-clear" data-traversal-clear>Cancel <kbd>Esc</kbd></button>' : ""}</div>
      <div class="traversal-map-scroll">
        <div class="traversal-map">
          ${traversalGoal(data, selectedCell, finishMove, 0)}
          <div class="traversal-grid" role="group" aria-label="Traversal terrain board, row 1 at the top">
            <span aria-hidden="true"></span>${Array.from("abcdefghij", (letter) => `<span class="traversal-axis" aria-hidden="true">${letter}</span>`).join("")}
            ${Array.from({ length: 10 }, (_, row) => `<span class="traversal-axis" aria-hidden="true">${row + 1}</span>${data.cells.slice(row * 10, row * 10 + 10).map((cell) => traversalCell(cell, targetData, this.source, lastMove, minHeight, maxHeight)).join("")}`).join("")}
          </div>
          ${traversalGoal(data, selectedCell, finishMove, 1)}
        </div>
      </div>
      <div class="traversal-terrain-key"><span>Low ${minHeight}</span><span class="traversal-terrain-gradient" aria-hidden="true"></span><span>High ${maxHeight}</span><span>Cell numbers show height</span></div>
      <div class="traversal-legend">${Object.entries(TRAVERSAL_PIECES).map(([type, kind]) => `<div>${traversalPiece({ player: game.current_player, type })}<span><strong>${kind.name}</strong><small>${kind.movement} movement · ${kind.strength} strength</small><small>${kind.ability}</small></span></div>`).join("")}</div>
      <p class="traversal-help">Move cost: 1 + height difference. Capture an enemy by matching or beating its strength. A piece scores its strength when it finishes.</p>
    </div>`;
  }

  select(coord) {
    const game = state.game;
    if (state.busy || !game || game.name !== "traversal" || game.ended) return;
    if (coord !== null && !traversalTargets(game).sources.has(coord)) return;
    this.source = coord === this.source ? null : coord;
    renderVisual();
    const focus = this.source ?
      "#visualBoard .traversal-cell[data-action], #visualBoard .traversal-finish" :
      "#visualBoard .traversal-source";
    document.querySelector(focus)?.focus?.();
  }
}

const traversalView = new TraversalView();
function renderTraversal(game) { return traversalView.render(game); }
renderers.traversal = renderTraversal;

document.addEventListener("click", (event) => {
  const clear = event.target.closest?.("[data-traversal-clear]");
  if (clear) { traversalView.select(null); return; }
  const cell = event.target.closest?.("[data-traversal-coord]");
  if (cell && !cell.dataset.action) traversalView.select(cell.dataset.traversalCoord);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !isTypingTarget(event.target) &&
      state.game?.name === "traversal" && traversalView.source && !state.busy) {
    event.preventDefault();
    traversalView.select(null);
  }
});
if (state.game?.name === "traversal") renderVisual();
