const directions2048 = {
  up: { arrow: "↑", key: "ArrowUp" },
  left: { arrow: "←", key: "ArrowLeft" },
  down: { arrow: "↓", key: "ArrowDown" },
  right: { arrow: "→", key: "ArrowRight" },
};

function parse2048(game) {
  const lines = linesWithoutMoves(game);
  const start = lines.indexOf("Board:");
  if (start < 0) return null;
  const rows = lines.slice(start + 1, start + 5).map((line) => line.trim().split(/\s+/));
  if (rows.length !== 4 || rows.some((row) => row.length !== 4)) return null;
  const cells = rows.flat().map((cell) => cell === "." ? 0 : Number(cell));
  if (cells.some((value) => !Number.isSafeInteger(value) || value < 0 ||
      (value !== 0 && (value < 2 || !Number.isInteger(Math.log2(value)))))) return null;
  const field = (name, fallback) => Number(
    lines.find((line) => line.startsWith(`${name}: `))?.split(": ")[1] ?? fallback,
  );
  const versus = game.name === "2048versus";
  return {
    cells,
    score: field("Score", 0),
    moves: field("Moves", game.round),
    target: field("Target", 2048),
    won: cells.some((value) => value >= field("Target", 2048)),
    nolimits: lines.includes("Nolimits: True"),
    versus,
    placing: versus && lines.includes("Phase: spawn"),
    pending: field("Next tile", 0),
  };
}

function status2048(game, data) {
  if (data.versus) {
    if (game.ended && data.nolimits) return `Player ${data.won ? 0 : 1} wins · No moves left`;
    if (game.ended) return data.won ? "Player 0 wins · 2048 reached!" : "Player 1 wins · No moves left";
    return data.placing ? `Player 1 · Place the next ${data.pending} on an empty cell` : "Player 0 · Slide the tiles";
  }
  if (data.nolimits) {
    if (game.ended) return "Game over · No moves left";
    return data.won ? `${data.target} reached · Keep merging` : "Merge tiles · No tile limit";
  }
  return data.won ? `${data.target} reached!` : game.ended ? "Game over · No moves left" : `Merge tiles to reach ${data.target}`;
}

function tile2048(game, data, row, col) {
  const value = data.cells[row * 4 + col];
  const coord = `${String.fromCharCode(97 + col)}${row + 1}`;
  const move = `spawn ${coord}`;
  const legal = data.placing && !game.ended && value === 0 && game.moves.includes(move);
  const tag = legal ? "button" : "div";
  const action = legal ? `type="button" data-action="${move}"` : "";
  const label = legal ? `Place ${data.pending} at ${coord}` : `Row ${row + 1}, column ${col + 1}: ${value || "empty"}`;
  const preview = legal ? `<span class="game2048-preview">+${escapeHtml(data.pending)}</span>${moveBadge(move)}` : "";
  return `<${tag} ${action} class="game2048-tile tile-${value <= 2048 ? value : "high"}${legal ? " game2048-spawn playable" : ""}" role="gridcell" aria-label="${escapeHtml(label)}">${value || preview}</${tag}>`;
}

function render2048(game) {
  const data = parse2048(game);
  if (!data) return renderGeneric(game);
  const status = status2048(game, data);
  return `<div class="game2048-board">
    <section class="game2048-stats" aria-label="Game statistics">
      <div><span>Score</span><strong>${escapeHtml(data.score)}</strong></div>
      <div><span>Largest tile</span><strong>${Math.max(...data.cells)}</strong></div>
      <div><span>Moves</span><strong>${escapeHtml(data.moves)}</strong></div>
    </section>
    <p class="game2048-status${data.won ? " game2048-won" : ""}" role="status">${escapeHtml(status)}</p>
    <div class="game2048-grid" role="grid" aria-label="2048 board">
      ${Array.from({ length: 4 }, (_, row) => `<div class="game2048-row" role="row">
        ${Array.from({ length: 4 }, (_, col) => tile2048(game, data, row, col)).join("")}
      </div>`).join("")}
    </div>
    ${data.placing ? "" : `<section class="game2048-controls" aria-label="Slide tiles">
      ${Object.entries(directions2048).map(([move, direction]) => {
        const legal = !game.ended && game.moves.includes(move);
        return `<button type="button" class="game2048-direction direction-${move}${legal ? " playable" : ""}" ${legal ? `data-action="${move}"` : "disabled"} aria-label="Slide ${move}" title="${direction.key}">
          <span class="game2048-arrow" aria-hidden="true">${direction.arrow}</span>
          <span>${move}</span>${legal ? moveBadge(move) : ""}
        </button>`;
      }).join("")}
    </section>`}
    <p class="game2048-help">${data.nolimits ? "No tile limit: play continues until no slide remains. " : ""}${data.versus
      ? "Player 0 slides with arrow keys or direction buttons. Player 1 clicks an empty cell to place the next tile. Its value is drawn at random: 2 (90%) or 4 (10%)."
      : "Use the arrow keys or direction buttons. Equal tiles merge once per move. A new 2 (90%) or 4 (10%) appears after each slide."}</p>
  </div>`;
}

renderers["2048"] = render2048;
renderers["2048versus"] = render2048;

document.addEventListener("keydown", (event) => {
  const game = state.game;
  if (!game || !["2048", "2048versus"].includes(game.name) || game.ended || state.busy || state.autoPlaying ||
      isTypingTarget(event.target) || event.metaKey || event.ctrlKey || event.altKey) return;
  if (parse2048(game)?.placing) return;
  const move = Object.keys(directions2048).find((name) => directions2048[name].key === event.key);
  if (!move) return;
  event.preventDefault();
  if (game.moves.includes(move)) playMove(move);
});
