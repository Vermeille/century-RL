function parseSplendorCounts(text) {
  const counts = {};
  for (const match of text.matchAll(/([WBGRKY])(\d+)/g)) {
    counts[match[1]] = Number(match[2]);
  }
  return counts;
}

function parseSplendorCard(text) {
  if (!text || text === "-" || text === "?") {
    return { hidden: text === "?", empty: text === "-", raw: text };
  }
  const match = text.match(/^([WBGRK])(\d+)\[([^\]]+)\]$/);
  if (!match) return { raw: text };
  return {
    raw: text,
    bonus: match[1],
    points: Number(match[2]),
    cost: parseSplendorCounts(match[3]),
  };
}

function parseSplendor(game) {
  const data = {
    phase: "main",
    bank: {},
    nobles: [],
    market: {},
    players: [],
    moves: game.moves || [],
  };

  for (const line of linesWithoutMoves(game)) {
    if (line.startsWith(">")) {
      const match = line.match(/^>(\d+)\s+(\w+)/);
      if (match) data.phase = match[2];
      continue;
    }
    if (line.startsWith("Bank ")) {
      data.bank = parseSplendorCounts(line.slice(5));
      continue;
    }
    if (line.startsWith("Nobles ")) {
      const tokens = line.slice(7).trim().split(/\s+/).filter(Boolean);
      data.nobles = tokens.map((token) => {
        const match = token.match(/^N(\d+)\[([^\]]+)\]$/);
        return match
          ? { id: Number(match[1]), requirement: parseSplendorCounts(match[2]) }
          : { raw: token };
      });
      continue;
    }
    const tier = line.match(/^T([123])\((\d+)\)\s+(.*)$/);
    if (tier) {
      const tierNumber = Number(tier[1]);
      data.market[tierNumber] = {
        deck: Number(tier[2]),
        cards: tier[3]
          .trim()
          .split(/\s+/)
          .map((entry) => {
            const split = entry.indexOf("=");
            return {
              slot: Number(entry.slice(0, split)),
              ...parseSplendorCard(entry.slice(split + 1)),
            };
          }),
      };
      continue;
    }
    const player = line.match(
      /^P(\d+)(\*)?\s+S(\d+)\s+D(\d+)\s+T([^\s]+)\s+C([^\s]+)\s+H(.*)$/,
    );
    if (player) {
      const reservedText = player[7].trim();
      data.players.push({
        id: Number(player[1]),
        active: Boolean(player[2]),
        prestige: Number(player[3]),
        developments: Number(player[4]),
        tokens: parseSplendorCounts(player[5]),
        bonuses: parseSplendorCounts(player[6]),
        reserved:
          reservedText === "-"
            ? []
            : reservedText.split(/\s+/).map((entry) => {
                const split = entry.indexOf("=");
                return {
                  slot: Number(entry.slice(0, split)),
                  ...parseSplendorCard(entry.slice(split + 1)),
                };
              }),
      });
    }
  }
  return data;
}

const SPLENDOR_COLORS = ["W", "B", "G", "R", "K", "Y"];

function splendorGem(color, value, extraClass = "") {
  return `<span class="splendor-gem gem-${color} ${extraClass}"><b>${color}</b>${value}</span>`;
}

function splendorCounts(counts, colors = SPLENDOR_COLORS) {
  return colors
    .filter((color) => counts[color])
    .map((color) => splendorGem(color, counts[color]))
    .join("");
}

function splendorActionButton(move, label, className = "") {
  if (!move) return "";
  return `<button type="button" class="splendor-action ${className}" data-action="${escapeHtml(move)}">
    ${escapeHtml(label)} ${moveBadge(move)}
  </button>`;
}

function splendorBuyLabel(move) {
  const split = move.split("~");
  if (split.length === 1) return "Buy";
  const gold = split[1];
  return `Buy · gold→${gold || "-"}`;
}

function splendorCard(card, actions = [], label = "") {
  if (card.hidden) {
    return `<article class="splendor-card hidden"><span class="splendor-card-back">?</span><span>${escapeHtml(label)}</span></article>`;
  }
  if (card.empty) {
    return `<article class="splendor-card empty-card"><span>Empty</span></article>`;
  }
  if (!card.bonus) {
    return `<article class="splendor-card"><code>${escapeHtml(card.raw || "?")}</code></article>`;
  }
  return `<article class="splendor-card bonus-${card.bonus}">
    <header>
      <span class="splendor-bonus gem-${card.bonus}">${card.bonus}</span>
      <strong>${card.points} VP</strong>
    </header>
    <div class="splendor-cost">${splendorCounts(card.cost || {}, ["W", "B", "G", "R", "K"]) || '<span class="free-card">free</span>'}</div>
    ${label ? `<small>${escapeHtml(label)}</small>` : ""}
    <div class="splendor-card-actions">${actions.join("")}</div>
  </article>`;
}

function splendorMarketCard(game, tier, card) {
  if (card.empty) return splendorCard(card);
  const buyPrefix = `B:${tier}.${card.slot}`;
  const buys = game.moves.filter(
    (move) => move === buyPrefix || move.startsWith(`${buyPrefix}~`),
  );
  const reserve = `R:${tier}.${card.slot}`;
  const actions = buys.map((move) => splendorActionButton(move, splendorBuyLabel(move), "buy"));
  if (game.moves.includes(reserve)) {
    actions.push(splendorActionButton(reserve, "Reserve", "reserve"));
  }
  return splendorCard(card, actions, `T${tier} · ${card.slot}`);
}

function splendorReservedCard(game, card, mine) {
  if (!mine || card.hidden) return splendorCard(card, [], `Reserve ${card.slot}`);
  const prefix = `B:H${card.slot}`;
  const buys = game.moves.filter(
    (move) => move === prefix || move.startsWith(`${prefix}~`),
  );
  return splendorCard(
    card,
    buys.map((move) => splendorActionButton(move, splendorBuyLabel(move), "buy")),
    `Reserve ${card.slot}`,
  );
}

function splendorNoble(noble, game) {
  const move = `N:${noble.id}`;
  const playable = game.moves.includes(move);
  return `<article class="splendor-noble ${playable ? "playable" : ""}">
    <strong>N${noble.id} · 3 VP</strong>
    <div>${splendorCounts(noble.requirement || {}, ["W", "B", "G", "R", "K"])}</div>
    ${playable ? splendorActionButton(move, "Choose noble", "noble") : ""}
  </article>`;
}

function renderSplendor(game) {
  const data = parseSplendor(game);
  const takeMoves = game.moves.filter((move) => move.startsWith("T:"));
  const discardMoves = game.moves.filter((move) => move.startsWith("D:"));
  const current = data.players.find((player) => player.active) || data.players[0];

  const phaseActions =
    data.phase === "discard"
      ? `<section class="splendor-section"><div class="section-label">Return tokens to 10</div><div class="splendor-actions">${discardMoves
          .map((move) => splendorActionButton(move, `Return ${move.slice(2) || "nothing"}`, "discard"))
          .join("")}</div></section>`
      : `<section class="splendor-section"><div class="section-label">Take gems</div><div class="splendor-actions">${takeMoves
          .map((move) => splendorActionButton(move, `Take ${move.slice(2)}`, "take"))
          .join("")}</div></section>`;

  return `<div class="splendor-board">
    <section class="splendor-status">
      <span class="score-pill">${escapeHtml(data.phase)}</span>
      <span class="score-pill">P${game.current_player}</span>
      ${current ? `<span class="score-pill">${current.prestige} VP</span>` : ""}
    </section>

    <section class="splendor-section">
      <div class="section-label">Bank</div>
      <div class="splendor-bank">${SPLENDOR_COLORS.map((color) => splendorGem(color, data.bank[color] || 0)).join("")}</div>
    </section>

    ${phaseActions}

    <section class="splendor-section">
      <div class="section-label">Nobles</div>
      <div class="splendor-nobles">${data.nobles.map((noble) => splendorNoble(noble, game)).join("")}</div>
    </section>

    <section class="splendor-section splendor-market">
      <div class="section-label">Development market</div>
      ${[3, 2, 1]
        .map((tier) => {
          const row = data.market[tier] || { deck: 0, cards: [] };
          const deckMove = `R:${tier}.D`;
          return `<div class="splendor-tier">
            <div class="splendor-tier-head">
              <strong>Tier ${tier}</strong><span>${row.deck} in deck</span>
              ${game.moves.includes(deckMove) ? splendorActionButton(deckMove, "Reserve blind", "reserve blind") : ""}
            </div>
            <div class="splendor-cards">${row.cards.map((card) => splendorMarketCard(game, tier, card)).join("")}</div>
          </div>`;
        })
        .join("")}
    </section>

    <section class="splendor-section">
      <div class="section-label">Players</div>
      <div class="splendor-players">${data.players
        .map((player, index) => {
          const mine = index === 0;
          return `<article class="splendor-player ${player.active ? "active" : ""}">
            <header><strong>P${player.id}${mine ? " · view" : ""}</strong><span>${player.prestige} VP · ${player.developments} cards</span></header>
            <div><small>Tokens</small><div class="splendor-token-row">${SPLENDOR_COLORS.map((color) => splendorGem(color, player.tokens[color] || 0)).join("")}</div></div>
            <div><small>Bonuses</small><div class="splendor-token-row">${["W", "B", "G", "R", "K"].map((color) => splendorGem(color, player.bonuses[color] || 0, "bonus-count")).join("")}</div></div>
            <div><small>Reserved</small><div class="splendor-reserved">${player.reserved.length ? player.reserved.map((card) => splendorReservedCard(game, card, mine)).join("") : '<span class="splendor-none">none</span>'}</div></div>
          </article>`;
        })
        .join("")}</div>
    </section>
  </div>`;
}

renderers.splendor = renderSplendor;
