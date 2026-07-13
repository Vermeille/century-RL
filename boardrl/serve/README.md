# BoardRL Web UI

The server exposes one shared game workbench instead of one `ui.html` per game.
The shared shell owns game switching, strategy selection, agent play, policy
analysis, strategy comparison, move history, and raw-state inspection.
Game-specific code should be limited to visual renderers in `static/app.js`.

## Server Contract

Games are selected with the `game` query parameter on stateful endpoints:

- `GET /games`
- `GET /state?game=<name>`
- `GET /strategies?game=<name>`
- `GET /analyze?game=<name>&strategy=<strategy>`
- `POST /do-one?game=<name>`
- `POST /do?game=<name>`
- `POST /play-one?game=<name>`
- `POST /undo?game=<name>`
- `POST /redo?game=<name>`
- `GET /reset?game=<name>`

`<name>` may be a complete game specification using the same format as the
game registry, for example `nim,num_stones=5,max_pick=2`. The same value can
be supplied to `POST /set-game` as `{ "game": "nim,num_stones=5,max_pick=2" }`.
The old no-query behavior still maps to the current/default game for backwards
compatibility. `POST /set-game` changes that default for the current process.

The same specification can be selected at startup, for example:

```bash
python -m boardrl.serve.serve --game nim,num_stones=5,max_pick=2
```

`/state` returns the canonical UI payload:

- `name`
- `spec`
- `board`
- `board_with_moves`
- `moves`
- `current_player`
- `round`
- `ended`
- `points`
- `history`
- `can_undo`
- `can_redo`

`GET /games` continues to return registered base names in `games`, keeps the
selected base name in `current`, and also returns the complete selected
specification in `current_spec`.

The frontend should use `moves` for legal actions instead of scraping moves from
the raw text.

## Adding A Game UI

If the generic renderer is good enough, no UI work is needed. If the game needs
visuals:

1. Add a parser/helper in `static/app.js` that reads `game.board_with_moves`.
2. Add `render<YourGame>(game)` that returns HTML.
3. Put `data-action="<move>"` on clickable elements. The shared click handler
   sends the move to the server.
4. Register it in the `renderers` map:

   ```js
   const renderers = {
     yourgame: renderYourGame,
   };
   ```

Keep raw text support intact. The raw-state textarea is both a debugger and a
manual analysis scratchpad.

The optional Compare selector analyzes the same live or edited state with a
second strategy and shows probability deltas in the policy list.

## Keyboard Shortcuts

- `1` through `9`: play the corresponding move from the policy list.
- `T`: play the top move from the policy list.
- `A`: play one agent move.
- `Ctrl+Z` / `Cmd+Z`: undo.
- `Ctrl+Shift+Z` / `Cmd+Shift+Z`: redo.

Shortcuts are ignored while typing in inputs, selects, or the raw-state
textarea.

## Renderer Tests

When adding or changing a renderer, add a static renderer regression in
`tests/test_serve.py`. Use the actual `display_with_moves()` text format for the
game and assert that the renderer:

- creates the expected visual structure,
- exposes legal moves with `data-action`,
- does not render `undefined`,
- keeps important game-specific labels visible.

Run:

```bash
node --check boardrl/serve/static/app.js
uv run pytest tests/test_serve.py tests/test_serve_tictactoe.py
```
