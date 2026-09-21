let santoriniSelection = { source: null, destination: null };
let santoriniRevision = null;

const santoriniDeltas = {
  u: [0, -1],
  d: [0, 1],
  l: [-1, 0],
  r: [1, 0],
  ul: [-1, -1],
  ur: [1, -1],
  dl: [-1, 1],
  dr: [1, 1],
};

function santoriniStep(coord, direction) {
  const [dx, dy] = santoriniDeltas[direction];
  const col = coord.charCodeAt(0) - 97 + dx;
  const row = Number(coord[1]) - 1 + dy;
  return `${String.fromCharCode(97 + col)}${row + 1}`;
}

function parseSantorini(game) {
  const data = {
    phase: "play",
    cells: [],
    moves: game.moves || [],
  };

  const lines = linesWithoutMoves(game);
  const status = lines.find((line) => line.startsWith(">")) || "";
  if (status.includes("setup")) {
    data.phase = "setup";
  }

  for (const line of lines) {
    const match = line.match(/^([1-5])\s+(.+)$/);
    if (!match) continue;
    const row = Number(match[1]) - 1;
    const tokens = match[2].trim().split(/\s+/);
    if (tokens.length !== 5) continue;

    tokens.forEach((token, col) => {
      const cell = token.match(/^([0-4])([.OX#])$/);
      if (!cell) return;
      data.cells.push({
        coord: `${String.fromCharCode(97 + col)}${row + 1}`,
        row,
        col,
        height: Number(cell[1]),
        occupant: cell[2],
      });
    });
  }

  return data;
}

function parseSantoriniMove(move) {
  if (move.startsWith("P:")) {
    return { setup: move.slice(2), source: null, destination: null, build: null };
  }

  const [movement, buildDirection = null] = move.split("+");
  const [source, moveDirection] = movement.split(">");
  const destination = santoriniStep(source, moveDirection);
  const build = buildDirection ? santoriniStep(destination, buildDirection) : null;
  return {
    setup: null,
    source,
    destination,
    build,
    moveDirection,
    buildDirection,
  };
}

function santoriniMoveMass(moves) {
  return moves.reduce((sum, move) => sum + probabilityFor(move), 0);
}

function santoriniTargets(game, selection = santoriniSelection) {
  const parsed = parseSantorini(game);
  const parts = game.moves.map((move) => ({ move, ...parseSantoriniMove(move) }));

  if (parsed.phase === "setup") {
    return {
      kind: "setup",
      selected: null,
      targets: Object.fromEntries(
        parts.map((part) => [part.setup, [part.move]]),
      ),
    };
  }

  let source = selection.source;
  let destination = selection.destination;
  const sourceParts = parts.filter((part) => part.source);

  if (source && !sourceParts.some((part) => part.source === source)) {
    source = null;
    destination = null;
  }
  if (
    source &&
    destination &&
    !sourceParts.some(
      (part) => part.source === source && part.destination === destination,
    )
  ) {
    destination = null;
  }

  if (!source) {
    const targets = {};
    for (const part of sourceParts) {
      (targets[part.source] ||= []).push(part.move);
    }
    return { kind: "source", selected: null, targets };
  }

  if (!destination) {
    const targets = {};
    for (const part of sourceParts.filter((part) => part.source === source)) {
      (targets[part.destination] ||= []).push(part.move);
    }
    return { kind: "destination", selected: source, targets };
  }

  const candidates = sourceParts.filter(
    (part) => part.source === source && part.destination === destination,
  );
  const winningMove = candidates.find((part) => part.build === null);
  if (winningMove) {
    return {
      kind: "win",
      selected: destination,
      targets: { [destination]: [winningMove.move] },
    };
  }

  const targets = {};
  for (const part of candidates) {
    (targets[part.build] ||= []).push(part.move);
  }
  return { kind: "build", selected: destination, targets };
}

function santoriniInstruction(kind) {
  return {
    setup: "Place a worker",
    source: "Choose a worker",
    destination: "Choose where to move",
    build: "Choose where to build",
    win: "Winning move",
  }[kind] || "Play";
}

function santoriniTower(cell) {
  const levels = Array.from(
    { length: Math.min(cell.height, 3) },
    (_, index) => `<span class="santorini-level santorini-level-${index + 1}"></span>`,
  ).join("");
  const dome = cell.height === 4 ? '<span class="santorini-dome"></span>' : "";
  const worker =
    cell.occupant === "O" || cell.occupant === "X"
      ? `<span class="santorini-worker worker-${cell.occupant}">${cell.occupant}</span>`
      : "";
  return `<span class="santorini-tower">${levels}${dome}${worker}</span>`;
}

function renderSantorini(game) {
  const revision = `${game.current_player}:${game.history?.length || 0}:${game.board || ""}`;
  if (revision !== santoriniRevision) {
    santoriniSelection = { source: null, destination: null };
    santoriniRevision = revision;
  }

  let targetData = santoriniTargets(game);
  if (targetData.kind === "source" && santoriniSelection.source) {
    santoriniSelection = { source: null, destination: null };
    targetData = santoriniTargets(game);
  } else if (
    targetData.kind === "destination" &&
    santoriniSelection.destination
  ) {
    santoriniSelection.destination = null;
    targetData = santoriniTargets(game);
  }

  const data = parseSantorini(game);
  const cells = new Map(data.cells.map((cell) => [cell.coord, cell]));
  const selected = new Set(
    [santoriniSelection.source, santoriniSelection.destination].filter(Boolean),
  );

  return `<div class="santorini-board">
    <section class="santorini-status">
      <span class="score-pill">${escapeHtml(santoriniInstruction(targetData.kind))}</span>
      <span class="score-pill">P${game.current_player}</span>
    </section>
    <div class="santorini-grid" role="grid" aria-label="Santorini board">
      ${Array.from({ length: 25 }, (_, index) => {
        const row = Math.floor(index / 5);
        const col = index % 5;
        const coord = `${String.fromCharCode(97 + col)}${row + 1}`;
        const cell = cells.get(coord) || {
          coord,
          height: 0,
          occupant: ".",
        };
        const actions = targetData.targets[coord] || [];
        const target = actions.length > 0;
        const mass = target ? santoriniMoveMass(actions) : 0;
        const classes = [
          "santorini-cell",
          `height-${cell.height}`,
          target ? `target-${targetData.kind}` : "",
          selected.has(coord) ? "selected" : "",
        ]
          .filter(Boolean)
          .join(" ");
        const probability =
          target && mass > 0
            ? `<span class="santorini-prob">${Math.round(mass * 100)}%</span>`
            : "";
        const heat = target
          ? `<span class="santorini-heat" style="opacity:${Math.min(0.55, mass * 0.9).toFixed(3)}"></span>`
          : "";
        return `<button type="button" class="${classes}" data-santorini-coord="${coord}" aria-label="${coord}, level ${cell.height}${target ? ", playable" : ""}">
          ${heat}
          <span class="santorini-coord">${coord}</span>
          <span class="santorini-height">${cell.height}</span>
          ${santoriniTower(cell)}
          ${probability}
        </button>`;
      }).join("")}
    </div>
  </div>`;
}

function selectSantoriniCell(coord) {
  const game = state.game;
  if (!game || game.name !== "santorini" || game.ended) return;

  const targetData = santoriniTargets(game);
  const actions = targetData.targets[coord] || [];

  if (targetData.kind === "setup") {
    if (actions[0]) {
      santoriniSelection = { source: null, destination: null };
      playMove(actions[0]);
    }
    return;
  }

  const allSources = new Set(
    game.moves.map(parseSantoriniMove).map((part) => part.source).filter(Boolean),
  );

  if (coord === santoriniSelection.source && targetData.kind === "destination") {
    santoriniSelection = { source: null, destination: null };
    renderVisual();
    return;
  }

  if (allSources.has(coord) && targetData.kind !== "source") {
    santoriniSelection = { source: coord, destination: null };
    renderVisual();
    return;
  }

  if (!actions.length) return;

  if (targetData.kind === "source") {
    santoriniSelection = { source: coord, destination: null };
    renderVisual();
    return;
  }

  if (targetData.kind === "destination") {
    const winningMove = actions.find((move) => !move.includes("+"));
    if (winningMove) {
      santoriniSelection = { source: null, destination: null };
      playMove(winningMove);
      return;
    }
    santoriniSelection.destination = coord;
    renderVisual();
    return;
  }

  if (targetData.kind === "build") {
    santoriniSelection = { source: null, destination: null };
    playMove(actions[0]);
  }
}

renderers.santorini = renderSantorini;

document.addEventListener("click", (event) => {
  const cell = event.target.closest?.("[data-santorini-coord]");
  if (!cell) return;
  selectSantoriniCell(cell.dataset.santoriniCoord);
});
