# BoardRL Web UI

The canonical [workbench guide](../../docs/web-server.md) covers startup,
checkpoint discovery, play and policy comparison, shortcuts, the HTTP API,
renderer integration, tests, and hosting. Start the shared workbench from the
repository root:

```bash
uv run python -m boardrl.serve.serve --game tictactoe
```

Open <http://127.0.0.1:8000>. The server uses `static/index.html` and a shared
renderer registry in `static/app.js`; the old per-game `ui.html` files are not
the integration point. Registered games have a generic raw-text fallback.

Traversal uses `static/traversal.js` and `static/traversal.css` for its terrain
board, piece selection, capture and finish targets, scores, and game results.
Start it with `uv run python -m boardrl.serve.serve --game traversal`. The shared
policy list, raw state, and undo/redo controls remain available.

For renderer changes, use real game observations in regression tests and run:

```bash
node --check boardrl/serve/static/app.js
node --check boardrl/serve/static/traversal.js
uv run pytest tests/test_serve.py tests/test_serve_tictactoe.py tests/test_traversal_ui.py
```
