const state = {
  game: null,
  games: [],
  strategies: [],
  analysis: null,
  comparison: null,
  busy: false,
  autoPlaying: false,
  rawDirty: false,
  orderedMoves: [],
};

const els = {
  gameName: document.querySelector("#gameName"),
  game: document.querySelector("#game"),
  gameSpec: document.querySelector("#gameSpec"),
  applyGame: document.querySelector("#applyGame"),
  currentPlayer: document.querySelector("#currentPlayer"),
  round: document.querySelector("#round"),
  status: document.querySelector("#status"),
  strategy: document.querySelector("#strategy"),
  compareStrategy: document.querySelector("#compareStrategy"),
  replyWithAgent: document.querySelector("#replyWithAgent"),
  playTop: document.querySelector("#playTop"),
  agentMove: document.querySelector("#agentMove"),
  autoPlay: document.querySelector("#autoPlay"),
  undo: document.querySelector("#undo"),
  redo: document.querySelector("#redo"),
  reset: document.querySelector("#reset"),
  message: document.querySelector("#message"),
  visualBoard: document.querySelector("#visualBoard"),
  rawState: document.querySelector("#rawState"),
  syncRaw: document.querySelector("#syncRaw"),
  analyzeRaw: document.querySelector("#analyzeRaw"),
  moves: document.querySelector("#moves"),
  moveCount: document.querySelector("#moveCount"),
  inspector: document.querySelector("#inspector"),
  history: document.querySelector("#history"),
};

function gameQuery() {
  const spec = els.gameSpec.value.trim() || els.game.value || state.game?.spec || "";
  return `game=${encodeURIComponent(spec)}`;
}

function api(path) {
  const sep = path.includes("?") ? "&" : "?";
  return `${path}${sep}${gameQuery()}`;
}

function setBusy(value) {
  state.busy = value;
  els.playTop.disabled = value || !state.orderedMoves.length;
  els.agentMove.disabled = value || !state.game || state.game.ended;
  els.reset.disabled = value;
  els.undo.disabled = value || !state.game?.can_undo;
  els.redo.disabled = value || !state.game?.can_redo;
  els.strategy.disabled = value;
  els.compareStrategy.disabled = value;
  els.game.disabled = value || state.autoPlaying;
  els.gameSpec.disabled = value || state.autoPlaying;
  els.applyGame.disabled = value || state.autoPlaying;
  els.autoPlay.disabled = value && !state.autoPlaying;
  for (const button of document.querySelectorAll("button[data-action]")) {
    button.disabled = value;
  }
}

function showMessage(text) {
  els.message.hidden = !text;
  els.message.textContent = text || "";
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  if (response.ok) {
    const contentType = response.headers.get("content-type") || "";
    return contentType.includes("application/json")
      ? response.json()
      : response.text();
  }
  let detail = response.statusText;
  try {
    const data = await response.json();
    detail = data.detail || detail;
  } catch (_error) {
    detail = await response.text();
  }
  throw new Error(detail || `Request failed: ${response.status}`);
}

function pickDefaultStrategy(strategies) {
  return (
    strategies.find((strategy) => strategy.startsWith("policy_sampling")) ||
    strategies.find((strategy) => strategy === "random") ||
    strategies[0] ||
    ""
  );
}

function renderGameSelect(current, currentSpec = current) {
  els.game.replaceChildren();
  for (const game of state.games) {
    const option = document.createElement("option");
    option.value = game;
    option.textContent = game;
    option.selected = game === current;
    els.game.append(option);
  }
  els.gameSpec.value = currentSpec;
}

function renderStrategies() {
  const selected = els.strategy.value || pickDefaultStrategy(state.strategies);
  els.strategy.replaceChildren();
  const compareSelected = els.compareStrategy.value || "";
  els.compareStrategy.replaceChildren();
  const none = document.createElement("option");
  none.value = "";
  none.textContent = "None";
  none.selected = compareSelected === "";
  els.compareStrategy.append(none);
  for (const strategy of state.strategies) {
    const option = document.createElement("option");
    option.value = strategy;
    option.textContent = strategy;
    option.selected = strategy === selected;
    els.strategy.append(option);

    const compareOption = document.createElement("option");
    compareOption.value = strategy;
    compareOption.textContent = strategy;
    compareOption.selected = strategy === compareSelected;
    els.compareStrategy.append(compareOption);
  }
}

function probabilityFor(move) {
  return Number(state.analysis?.probabilities?.[move] || 0);
}

function formatProbability(move) {
  const probability = probabilityFor(move);
  return probability ? probability.toFixed(3) : "-";
}

function comparisonProbabilityFor(move) {
  return Number(state.comparison?.probabilities?.[move] || 0);
}

function formatComparison(move) {
  if (!state.comparison) return "";
  const delta = probabilityFor(move) - comparisonProbabilityFor(move);
  const sign = delta > 0 ? "+" : "";
  return ` / ${sign}${delta.toFixed(3)}`;
}

function heatStyle(move) {
  const probability = probabilityFor(move);
  return `opacity: ${Math.min(0.68, probability * 0.9).toFixed(3)}`;
}

function moveBadge(move) {
  const probability = probabilityFor(move);
  if (!probability) return "";
  return `<span class="prob-badge">${Math.round(probability * 100)}%</span>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderGame() {
  const game = state.game;
  els.gameName.textContent = game?.name || "Game";
  els.currentPlayer.textContent = game ? String(game.current_player) : "-";
  els.round.textContent = game ? String(game.round) : "-";
  els.status.textContent = game?.ended ? `Ended (${game.points})` : "Playing";
  if (!state.rawDirty) {
    els.rawState.value = game?.board_with_moves || game?.board || "";
  }
  els.moveCount.textContent = `${game?.moves.length || 0} moves`;
  renderVisual();
  renderInspector();
  renderHistory();
  renderMoves();
}

function renderInspector() {
  const game = state.game;
  const topMove = game?.moves
    .map((move) => [move, probabilityFor(move)])
    .sort((a, b) => b[1] - a[1])[0];
  els.inspector.innerHTML = `
    <div class="metric-grid">
      <div class="metric"><span>Policy top</span><strong>${topMove ? escapeHtml(topMove[0]) : "-"}</strong></div>
      <div class="metric"><span>Confidence</span><strong>${topMove ? `${Math.round(topMove[1] * 100)}%` : "-"}</strong></div>
      <div class="metric"><span>Value</span><strong>${state.analysis?.reference_value?.toFixed?.(3) ?? "-"}</strong></div>
      <div class="metric"><span>Compare</span><strong>${state.comparison ? escapeHtml(els.compareStrategy.value) : "-"}</strong></div>
    </div>`;
}

function renderHistory() {
  els.history.replaceChildren();
  const history = state.game?.history || [];
  if (!history.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No moves yet.";
    els.history.append(empty);
    return;
  }

  for (const item of history.slice().reverse()) {
    const row = document.createElement("div");
    row.className = "history-item";
    row.innerHTML = `
      <span class="history-actor">${escapeHtml(item.actor)}</span>
      <code title="${escapeHtml(item.strategy || "")}">${escapeHtml(item.action)}</code>
      <span class="history-meta">P${item.player} R${item.round}</span>`;
    els.history.append(row);
  }
}

function renderVisual() {
  if (!state.game) {
    els.visualBoard.innerHTML = "";
    return;
  }
  const renderer = renderers[state.game.name] || renderGeneric;
  els.visualBoard.innerHTML = renderer(state.game);
  bindVisualActions();
}

function renderMoves() {
  els.moves.replaceChildren();
  const game = state.game;
  if (!game || game.moves.length === 0) {
    state.orderedMoves = [];
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = game?.ended ? "Game over." : "No legal moves.";
    els.moves.append(empty);
    return;
  }

  const orderedMoves = [...game.moves].sort(
    (a, b) => probabilityFor(b) - probabilityFor(a),
  );
  state.orderedMoves = orderedMoves;
  for (const move of orderedMoves) {
    const shortcut = orderedMoves.indexOf(move) + 1;
    const row = document.createElement("div");
    row.className = "move";

    const button = document.createElement("button");
    button.type = "button";
    button.dataset.action = move;
    button.textContent = `${shortcut <= 9 ? `${shortcut}. ` : ""}@${move}`;
    button.title = shortcut <= 9 ? `Press ${shortcut} to play ${move}` : `Play ${move}`;
    button.addEventListener("click", () => playMove(move));

    const meter = document.createElement("div");
    meter.className = "meter";
    const fill = document.createElement("div");
    fill.className = "meter-fill";
    const probability = probabilityFor(move);
    fill.style.width = `${Math.max(0, Math.min(100, probability * 100))}%`;
    meter.append(fill);

    const label = document.createElement("div");
    label.className = "probability";
    label.textContent = `${formatProbability(move)}${formatComparison(move)}`;

    row.append(button, meter, label);
    els.moves.append(row);
  }
}

function isTypingTarget(target) {
  return ["INPUT", "SELECT", "TEXTAREA"].includes(target?.tagName);
}

function handleShortcut(event) {
  if (isTypingTarget(event.target)) {
    return;
  }
  if (event.metaKey || event.ctrlKey) {
    if (event.key.toLowerCase() === "z" && event.shiftKey) {
      event.preventDefault();
      redoMove();
      return;
    }
    if (event.key.toLowerCase() === "z") {
      event.preventDefault();
      undoMove();
    }
    return;
  }
  if (event.key >= "1" && event.key <= "9") {
    const move = state.orderedMoves[Number(event.key) - 1];
    if (move) {
      event.preventDefault();
      playMove(move);
    }
    return;
  }
  if (event.key.toLowerCase() === "a") {
    event.preventDefault();
    playAgentMove();
    return;
  }
  if (event.key.toLowerCase() === "t") {
    event.preventDefault();
    playTopMove();
  }
}

function bindVisualActions() {
  for (const element of document.querySelectorAll("[data-action]")) {
    element.addEventListener("click", () => playMove(element.dataset.action));
  }
}

function linesWithoutMoves(game) {
  return (game.board_with_moves || game.board || "")
    .split("\n")
    .filter((line) => !line.startsWith("@") && line !== "Moves" && line !== "_Moves");
}

function renderGeneric(game) {
  return `<pre class="fallback-board">${escapeHtml(game.board_with_moves || game.board)}</pre>`;
}

function renderTictactoe(game) {
  const lines = linesWithoutMoves(game).filter((line) => line && !line.startsWith(">"));
  const rows = lines.slice(0, 3);
  return `<div class="grid-game"><div class="grid-board">${rows
    .map(
      (row, r) =>
        `<div class="grid-row">${[...row.padEnd(3, " ")]
          .slice(0, 3)
          .map((cell, c) => {
            const move = String(r * 3 + c);
            const playable = game.moves.includes(move);
            const markClass = cell === "X" ? "mark-x" : cell === "O" ? "mark-o" : "";
            return `<button class="grid-cell ${markClass} ${playable ? "playable" : ""}" ${playable ? `data-action="${move}"` : ""}>
              <span class="heat" style="${heatStyle(move)}"></span>
              <span class="cell-label">${escapeHtml(cell.trim() || move)}</span>
              ${moveBadge(move)}
            </button>`;
          })
          .join("")}</div>`,
    )
    .join("")}</div></div>`;
}

function renderConnectfour(game) {
  const boardRows = linesWithoutMoves(game).filter((line) => line.startsWith("|"));
  const columns = ["0", "1", "2", "3", "4", "5", "6"];
  return `<div class="grid-game">
    <div class="column-moves">${columns
      .map((move) => {
        const playable = game.moves.includes(move);
        return `<button class="column-move ${playable ? "playable" : ""}" ${playable ? `data-action="${move}"` : ""}>
          <span class="heat" style="${heatStyle(move)}"></span>
          <span class="move-label">${move}</span>
          ${moveBadge(move)}
        </button>`;
      })
      .join("")}</div>
    <div class="grid-board">${boardRows
      .map(
        (row) =>
          `<div class="grid-row">${[...row.slice(1, -1)]
            .map((cell) => {
              const markClass = cell === "X" ? "mark-x" : cell === "O" ? "mark-o" : "";
              return `<div class="grid-cell ${markClass}"><span class="cell-label">${escapeHtml(cell)}</span></div>`;
            })
            .join("")}</div>`,
      )
      .join("")}</div>
  </div>`;
}

function expandStock(stock) {
  let out = "";
  let digits = "";
  for (const char of stock || "") {
    if (/\d/.test(char)) {
      digits += char;
      continue;
    }
    const count = digits ? Number(digits) : 1;
    digits = "";
    out += char.repeat(Number.isFinite(count) ? count : 1);
  }
  return out;
}

function renderStock(stock) {
  return `<div class="token-row">${[...expandStock(stock)]
    .map((token) => `<span class="token token-${escapeHtml(token)}"></span>`)
    .join("")}</div>`;
}

function parseCentury(game) {
  const lines = (game.board_with_moves || "").split("\n").filter(Boolean);
  const parsed = {
    others: [],
    victory: [],
    action: [],
    hand: [],
    discard: [],
    moves: game.moves,
    stock: "",
    points: 0,
    victories: 0,
  };
  let i = 2;
  while (lines[i]?.startsWith("V")) {
    parsed.victory.push(lines[i].split(" ")[1].split(">"));
    i += 1;
  }
  while (lines[i]?.startsWith("A")) {
    const [, transfo, token] = lines[i].split(" ");
    parsed.action.push([transfo.split(">"), token.split(">")]);
    i += 1;
  }
  while (lines[i]?.startsWith("_Him")) {
    const [, num, points] = lines[i].split(" ");
    const [, victory] = lines[i + 1].split(" ");
    const [, stock] = lines[i + 2].split(" ");
    parsed.others.push({ num, points, victory, stock });
    i += 3;
  }
  if (lines[i]?.startsWith("_Me")) {
    const [, points] = lines[i].split(" ");
    const [, victory] = lines[i + 1].split(" ");
    const [, stock] = lines[i + 2].split(" ");
    parsed.points = Number(points);
    parsed.victories = Number(victory);
    parsed.stock = stock;
    i += 3;
  }
  while (lines[i]?.startsWith("H")) {
    parsed.hand.push(lines[i].split(" ")[1].split(">"));
    i += 1;
  }
  while (lines[i]?.startsWith("D")) {
    parsed.discard.push(lines[i].split(" ")[1]);
    i += 1;
  }
  return parsed;
}

function movesWithPrefix(moves, prefix) {
  return moves.filter((move) => move.startsWith(prefix));
}

function centuryCard({ title, top = "", bottom = "", moves = [], id = "" }) {
  const bestMove = moves.sort((a, b) => probabilityFor(b) - probabilityFor(a))[0];
  const actionAttr = bestMove ? `data-action="${escapeHtml(bestMove)}"` : "";
  return `<button class="century-card ${bestMove ? "playable" : ""}" ${actionAttr} data-card-id="${escapeHtml(id)}">
    ${moveBadge(bestMove)}
    <div class="title">${title}</div>
    <div>${top}</div>
    <div>${bottom}</div>
  </button>`;
}

function renderCentury(game) {
  const data = parseCentury(game);
  return `<div class="century-board">
    <section class="century-section">
      <div class="section-label">Opponents</div>
      <div class="century-row">${data.others
        .map(
          (other) =>
            `<div class="century-player">${renderStock(other.stock)}<span class="score-pill">P${other.num}: ${other.points} (${other.victory})</span></div>`,
        )
        .join("")}</div>
    </section>
    <section class="century-section">
      <div class="section-label">Victory Market</div>
      <div class="century-market">${data.victory
        .map((card, i) =>
          centuryCard({
            title: `${card[1]} points`,
            bottom: renderStock(card[0]),
            moves: movesWithPrefix(data.moves, `V${i}`),
            id: `V${i}`,
          }),
        )
        .join("")}</div>
    </section>
    <section class="century-section">
      <div class="section-label">Action Market</div>
      <div class="century-market">${data.action
        .map((card, i) =>
          centuryCard({
            title: "Action",
            top: renderStock(card[0][0]),
            bottom: `<strong>to</strong>${renderStock(card[0][1])}`,
            moves: movesWithPrefix(data.moves, `A${i}`),
            id: `A${i}`,
          }),
        )
        .join("")}</div>
    </section>
    <section class="century-section">
      <div class="section-label">You</div>
      <div class="century-player">${renderStock(data.stock)}<span class="score-pill">${data.points} (${data.victories})</span></div>
      <div class="century-hand">
        ${centuryCard({ title: "Reload", moves: movesWithPrefix(data.moves, "R"), id: "R" })}
        ${data.hand
          .map((card, i) =>
            centuryCard({
              title: "Hand",
              top: renderStock(card[0]),
              bottom: `<strong>to</strong>${renderStock(card[1])}`,
              moves: movesWithPrefix(data.moves, `H${i}`),
              id: `H${i}`,
            }),
          )
          .join("")}
      </div>
    </section>
  </div>`;
}

function parseTheGame(game) {
  const data = {
    round: 0,
    action: 0,
    piles: [],
    hand: [],
    cards: 0,
    messages: "",
    moves: game.moves,
  };
  for (const line of (game.board_with_moves || "").split("\n")) {
    if (line.startsWith("Round:")) {
      const match = line.match(/Round:\s*(\d+),\s*Action:\s*(\d+)/);
      if (match) {
        data.round = Number(match[1]);
        data.action = Number(match[2]);
      }
    } else if (line.startsWith("Piles:")) {
      data.piles = parseTheGamePiles(line.slice(6).trim());
    } else if (line.startsWith("Cards:")) {
      data.cards = Number(line.slice(6));
    } else if (line.startsWith("Hand:")) {
      data.hand = line.slice(5).trim().split(/\s+/).filter(Boolean);
    } else if (line.startsWith("Msgs:")) {
      data.messages = line.slice(5).trim();
    }
  }
  return data;
}

function parseTheGamePiles(text) {
  if (text.includes(":")) {
    return text.split(",").map((part, index) => {
      const [type, value] = part.trim().split(":");
      return normalizeTheGamePile(type, value, index);
    });
  }

  return text
    .split(/\s+/)
    .filter(Boolean)
    .map((value, index) => normalizeTheGamePile(index < 2 ? "asc" : "desc", value, index));
}

function normalizeTheGamePile(type, value, index) {
  const normalizedType = type === "asc" || type === "ascending" ? "asc" : "desc";
  return {
    index,
    type: normalizedType,
    value,
  };
}

function pileLabel(pile) {
  return pile.type === "asc" ? "Ascending" : "Descending";
}

function pileDirection(pile) {
  return pile.type === "asc" ? "up" : "down";
}

function renderThegame(game) {
  const data = parseTheGame(game);
  const moveByCard = {};
  const standaloneMoves = [];
  for (const move of data.moves) {
    if (move.includes("->")) {
      const [card] = move.split("->");
      moveByCard[card] ||= [];
      moveByCard[card].push(move);
    } else {
      standaloneMoves.push(move);
    }
  }
  return `<div class="thegame-board">
    <section class="thegame-status">
      <span class="score-pill">Round ${data.round}</span>
      <span class="score-pill">Played ${data.action}</span>
      <span class="score-pill">Deck ${data.cards}</span>
      ${data.messages ? `<span class="score-pill">Msgs ${escapeHtml(data.messages)}</span>` : ""}
    </section>
    <section class="thegame-section">
      <div class="section-label">Piles</div>
      <div class="piles">${data.piles
        .map(
          (pile, i) =>
            `<div class="pile-card pile-${pile.type}">
              <div class="title">${pileLabel(pile)}</div>
              <div class="pile-value">${escapeHtml(pile.value)}</div>
              <div>${pileDirection(pile)} pile ${i + 1}</div>
              <div class="pile-hint">${bestMoveForPile(data.moves, i)}</div>
            </div>`,
        )
        .join("")}</div>
    </section>
    ${standaloneMoves.length ? renderTheGameStandaloneMoves(standaloneMoves) : ""}
    <section class="thegame-section">
      <div class="section-label">Hand</div>
      <div class="hand">${data.hand
        .map((card) => {
          const moves = moveByCard[card] || [];
          return `<div class="game-card ${moves.length ? "playable" : ""}">
            <div class="card-value">${escapeHtml(card)}</div>
            <div class="pile-actions">${moves
              .map((move) => {
                const pile = Number(move.split("->")[1]);
                const target = data.piles[pile];
                const label = `${pile + 1}${target?.type === "asc" ? " up" : " down"}`;
                return `<button class="pile-action playable" data-action="${escapeHtml(move)}">${label} ${moveBadge(move)}</button>`;
              })
              .join("")}</div>
          </div>`;
        })
        .join("")}</div>
    </section>
  </div>`;
}

function bestMoveForPile(moves, pileIndex) {
  const targetMoves = moves
    .filter((move) => move.endsWith(`->${pileIndex}`))
    .sort((a, b) => probabilityFor(b) - probabilityFor(a));
  const bestMove = targetMoves[0];
  if (!bestMove || !probabilityFor(bestMove)) return "";
  return `best ${bestMove.split("->")[0]} (${Math.round(probabilityFor(bestMove) * 100)}%)`;
}

function renderTheGameStandaloneMoves(moves) {
  return `<section class="thegame-section">
    <div class="section-label">Turn Actions</div>
    <div class="turn-actions">${moves
      .map((move) => {
        const label = move === "x" ? "End turn" : `Message ${move}`;
        return `<button class="turn-action playable" data-action="${escapeHtml(move)}">${escapeHtml(label)} ${moveBadge(move)}</button>`;
      })
      .join("")}</div>
  </section>`;
}

function renderNim(game) {
  const lines = linesWithoutMoves(game).filter(Boolean);
  const status = lines[0] === "WIN" || lines[0] === "LOST" ? lines[0] : "";
  const stones = Number(status ? lines[1] : lines[0]) || 0;
  const visibleStones = Math.min(stones, 60);
  return `<div class="simple-board nim-board">
    <section class="simple-status">
      ${status ? `<span class="score-pill">${escapeHtml(status)}</span>` : ""}
      <span class="score-pill">${stones} stones</span>
    </section>
    <div class="stones" aria-label="${stones} stones">${Array.from({ length: visibleStones }, (_, i) => `<span class="stone" title="stone ${i + 1}"></span>`).join("")}${stones > visibleStones ? `<span class="stone-more">+${stones - visibleStones}</span>` : ""}</div>
    <section class="simple-actions">
      ${game.moves
        .map((move) => `<button class="choice-action playable" data-action="${escapeHtml(move)}">Take ${escapeHtml(move)} ${moveBadge(move)}</button>`)
        .join("")}
    </section>
  </div>`;
}

function renderRps(game) {
  const line = linesWithoutMoves(game).find((candidate) => candidate.startsWith(">")) || "?>?";
  const [mine = "?", opponent = "?"] = line.slice(1).split("|");
  const labels = { rock: "Rock", paper: "Paper", scissors: "Scissors" };
  return `<div class="simple-board rps-board">
    <section class="rps-showdown">
      <div class="rps-choice"><span>You</span><strong>${escapeHtml(mine || "?")}</strong></div>
      <div class="versus">vs</div>
      <div class="rps-choice"><span>Opponent</span><strong>${escapeHtml(opponent || "?")}</strong></div>
    </section>
    <section class="simple-actions rps-actions">
      ${game.moves
        .map((move) => `<button class="choice-action playable" data-action="${escapeHtml(move)}">${labels[move] || escapeHtml(move)} ${moveBadge(move)}</button>`)
        .join("")}
    </section>
  </div>`;
}

function parseGuessNumber(game) {
  const parsed = { secret: "", history: [], prompt: "", moves: game.moves };
  for (const line of linesWithoutMoves(game)) {
    if (line.startsWith("Secret:")) {
      parsed.secret = line.slice(7).trim();
    } else if (line === "History:") {
      continue;
    } else if (line.trim()) {
      if (line.endsWith("?")) {
        parsed.prompt = line;
      } else {
        parsed.history.push(line);
      }
    }
  }
  return parsed;
}

function renderGuessnumber(game) {
  const data = parseGuessNumber(game);
  const isGuessing = data.moves.every((move) => /^\d+$/.test(move));
  return `<div class="simple-board guess-board">
    <section class="simple-status">
      ${data.secret ? `<span class="score-pill">Secret ${escapeHtml(data.secret)}</span>` : ""}
      ${data.prompt ? `<span class="score-pill">${escapeHtml(data.prompt)}</span>` : ""}
      <span class="score-pill">${data.history.length} guesses</span>
    </section>
    <section class="guess-history">
      ${data.history.length ? data.history.map((item) => `<div class="history-card">${escapeHtml(item)}</div>`).join("") : `<div class="empty">No guesses yet.</div>`}
    </section>
    <section class="simple-actions ${isGuessing ? "number-pad" : ""}">
      ${data.moves
        .map((move) => `<button class="choice-action playable" data-action="${escapeHtml(move)}">${isGuessing ? "" : "Signal "}${escapeHtml(move)} ${moveBadge(move)}</button>`)
        .join("")}
    </section>
  </div>`;
}

function renderSum(game) {
  const lines = linesWithoutMoves(game).filter(Boolean);
  const [mine = "0", opponent = "0"] = (lines[0] || "0 | 0").split("|").map((part) => part.trim());
  const [a = "?", b = "?"] = (lines[1] || "? ?").split(/\s+/);
  return `<div class="simple-board sum-board">
    <section class="simple-status">
      <span class="score-pill">You ${escapeHtml(mine)}</span>
      <span class="score-pill">Opponent ${escapeHtml(opponent)}</span>
    </section>
    <section class="sum-problem">
      <span>${escapeHtml(a)}</span>
      <span>+</span>
      <span>${escapeHtml(b)}</span>
      <span class="sum-equals">/ 2</span>
    </section>
    <section class="simple-actions number-pad">
      ${game.moves
        .map((move) => `<button class="choice-action playable" data-action="${escapeHtml(move)}">${escapeHtml(move)} ${moveBadge(move)}</button>`)
        .join("")}
    </section>
  </div>`;
}

const renderers = {
  tictactoe: renderTictactoe,
  connectfour: renderConnectfour,
  century: renderCentury,
  thegame: renderThegame,
  nim: renderNim,
  rps: renderRps,
  guessnumber: renderGuessnumber,
  sum: renderSum,
};

async function refresh({ analyze = true } = {}) {
  state.game = await request(api("/state"));
  if (analyze && !state.game.ended && els.strategy.value) {
    state.analysis = await request(
      api(`/analyze?strategy=${encodeURIComponent(els.strategy.value)}`),
    );
    if (els.compareStrategy.value) {
      state.comparison = await request(
        api(`/analyze?strategy=${encodeURIComponent(els.compareStrategy.value)}`),
      );
    } else {
      state.comparison = null;
    }
  } else {
    state.analysis = null;
    state.comparison = null;
  }
  renderGame();
}

async function loadStrategies() {
  state.strategies = await request(api("/strategies"));
  renderStrategies();
}

function syncRawState() {
  state.rawDirty = false;
  els.rawState.value = state.game?.board_with_moves || state.game?.board || "";
}

async function analyzeRawState() {
  await runAction(
    async () => {
      state.analysis = await request(api("/analyze"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          strategy: els.strategy.value,
          state: els.rawState.value,
        }),
      });
      if (els.compareStrategy.value) {
        state.comparison = await request(api("/analyze"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            strategy: els.compareStrategy.value,
            state: els.rawState.value,
          }),
        });
      } else {
        state.comparison = null;
      }
      renderGame();
    },
    { refreshAfter: false },
  );
}

async function runAction(action, { refreshAfter = true } = {}) {
  if (state.busy) return;
  setBusy(true);
  showMessage("");
  try {
    await action();
    if (refreshAfter) await refresh();
  } catch (error) {
    showMessage(error.message);
  } finally {
    setBusy(false);
  }
}

async function playMove(move) {
  const path = els.replyWithAgent.checked ? "/do" : "/do-one";
  await runAction(() =>
    request(api(path), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: move, strategy: els.strategy.value }),
    }),
  );
}

async function playTopMove() {
  const move = state.orderedMoves[0];
  if (move) {
    await playMove(move);
  }
}

async function playAgentMove() {
  await runAction(() =>
    request(api("/play-one"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ strategy: els.strategy.value }),
    }),
  );
}

async function resetGame() {
  state.rawDirty = false;
  await runAction(() => request(api("/reset")));
}

async function undoMove() {
  state.rawDirty = false;
  await runAction(() =>
    request(api("/undo"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    }),
  );
}

async function redoMove() {
  state.rawDirty = false;
  await runAction(() =>
    request(api("/redo"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    }),
  );
}

async function changeGame() {
  els.gameSpec.value = els.game.value;
  await applyGameSpec();
}

async function applyGameSpec() {
  await runAction(async () => {
    state.rawDirty = false;
    const snapshot = await request("/set-game", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ game: els.gameSpec.value.trim() }),
    });
    els.game.value = snapshot.name;
    els.gameSpec.value = snapshot.spec;
    await loadStrategies();
  });
}

async function autoPlay() {
  state.autoPlaying = !state.autoPlaying;
  els.autoPlay.textContent = state.autoPlaying ? "Stop" : "Auto Play";
  while (state.autoPlaying && state.game && !state.game.ended) {
    await playAgentMove();
    await new Promise((resolve) => setTimeout(resolve, 180));
  }
  state.autoPlaying = false;
  els.autoPlay.textContent = "Auto Play";
  setBusy(false);
}

async function init() {
  setBusy(true);
  try {
    const gamesPayload = await request("/games");
    state.games = gamesPayload.games;
    renderGameSelect(gamesPayload.current, gamesPayload.current_spec);
    await loadStrategies();
    await refresh();
  } catch (error) {
    showMessage(error.message);
  } finally {
    setBusy(false);
  }
}

els.game.addEventListener("change", changeGame);
els.applyGame.addEventListener("click", applyGameSpec);
els.gameSpec.addEventListener("keydown", (event) => {
  if (event.key === "Enter") applyGameSpec();
});
els.strategy.addEventListener("change", () => runAction(async () => {}));
els.compareStrategy.addEventListener("change", () => runAction(async () => {}));
els.rawState.addEventListener("input", () => {
  state.rawDirty = true;
});
els.syncRaw.addEventListener("click", syncRawState);
els.analyzeRaw.addEventListener("click", analyzeRawState);
els.playTop.addEventListener("click", playTopMove);
els.agentMove.addEventListener("click", playAgentMove);
els.autoPlay.addEventListener("click", autoPlay);
els.undo.addEventListener("click", undoMove);
els.redo.addEventListener("click", redoMove);
els.reset.addEventListener("click", resetGame);
document.addEventListener("keydown", handleShortcut);

init();
