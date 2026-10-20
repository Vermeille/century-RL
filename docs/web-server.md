# Browser game workbench

The FastAPI server is a shared workbench for playing games and inspecting
strategies. It provides game switching, checkpoint selection, policy analysis,
comparison, raw observations, move history, and undo/redo.

## Start locally

From the repository root, with dependencies and the Cython compiler available:

```bash
uv run python -m boardrl.serve.serve --game tictactoe
```

Open <http://127.0.0.1:8000>. The module CLI accepts a full registry spec:

```bash
uv run python -m boardrl.serve.serve --game nim,num_stones=7,max_pick=2
```

For Uvicorn's host, port, or reload controls:

```bash
uv run uvicorn boardrl.serve.serve:app --host 127.0.0.1 --port 8000
```

The direct Uvicorn import uses the server's default game, Century. Select
another game in the UI or with `POST /set-game`. `GET /docs` exposes FastAPI's
interactive API schema. There is one mutable in-memory session per full spec
per process, shared by all browsers. A restart resets games and history. Use a
single worker so state does not diverge between processes.

## Play, analyze, and compare

Choose a game, then a built-in or discovered model strategy. Legal moves are
buttons in the rendered board or policy list. Analyze shows probabilities in
canonical move order; agent play samples from the chosen strategy. Compare
analyzes the same observation with a second strategy and shows probability
deltas. Undo/redo restores the game state as well as the move history.

Traversal has a graphical terrain board with height labels and distinct piece
shapes. Select a highlighted piece, then a marked destination or its finish
edge. Red destination outlines indicate captures; Esc or Cancel clears the
selection. Scores, finished and captured pieces, the last move, and the final
result appear on the board. Policy probabilities aggregate by source before
selection and show individual moves afterward. On small screens, scroll the
map horizontally. For a reproducible terrain, use a full spec such as
`traversal,terrain=mountains,seed=42` in Game Spec or the startup command.

The raw-state textarea is a model-analysis scratchpad. Edited text is analyzed
without mutating the live game; it is not validated as a reachable game state.
Preserve `@` legal-action lines. Model-backed strategies can analyze arbitrary
compatible text. `random` and `longest_move` have text fallbacks; other bots can
reject edited-state analysis.

| Key | Action |
| --- | --- |
| `1` through `9` | Corresponding move in the policy list |
| `T` | Top policy move |
| `A` | One agent action |
| Ctrl/Cmd+Z | Undo |
| Ctrl/Cmd+Shift+Z | Redo |

Shortcuts are ignored while typing in form controls and the raw-state area.

## Checkpoint discovery

At session creation the server recursively scans the working directory for
`*.pth`. Paths are matched to the base game name by directory/file components:
`tictactoe`, `tictactoe-*`, or `tictactoe,*` match Tic-Tac-Toe. A legacy path
containing a `*-ckpt` directory instead requires `<game>-ckpt`. Training layouts
such as `checkpoints/coop/thegame,mode=strict/cnn/run/step-100.pth` match.

The matching checks names, not checkpoint metadata or mode compatibility. A
strict and Omni model can both appear for The Game; choose the intended model.
Discovered strategies look like `policy_sampling,model=<encoded-path>`. Commas
and equals signs in paths are URL-encoded by discovery. Loaded strategies are
cached (five per session by default); discovery does not refresh on `/reset`.
Restart the server to pick up new files.

The server loads the first named model in each checkpoint. For a multi-model
file, export the desired model to a distinct file if the first is not the policy
you want to inspect:

```python
from boardrl import Checkpoints
from boardrl.rl.model import load_model

model = load_model("checkpoints/my-run/step-100.pth", name="environment", device="cpu")
Checkpoints("checkpoints/ui/connectfour-environment").save(100, {"current": model})
```

Point the input at a real file. The output directory includes `connectfour-`
so it is discovered for Connect Four. Files must use the named checkpoint
schema; loading untrusted `.pth` files is unsafe because they deserialize
Python objects. Keep the workbench and its checkpoint directory within your
trusted environment.

## HTTP API

State-bearing endpoints accept `?game=<full-spec>`. Without it, they use the
process-wide current/default spec. Use URL encoding for query values:

```bash
curl --get http://127.0.0.1:8000/state \
  --data-urlencode 'game=nim,num_stones=7,max_pick=2'
curl -X POST 'http://127.0.0.1:8000/do-one?game=nim,num_stones=7,max_pick=2' \
  -H 'Content-Type: application/json' -d '{"action":"1"}'
```

| Method and path | Input / response |
| --- | --- |
| `GET /games` | `{games: [base names], current: base name, current_spec: full spec}` |
| `POST /set-game` | JSON `{"game":"<spec>"}`; changes the default and returns a snapshot |
| `GET /state` | Canonical snapshot |
| `GET /board` | Plain text observation with moves; final forced-seat-0 text at game end |
| `GET /strategies` | JSON list of accepted strategy strings |
| `GET /analyze` | Query `strategy`; analyzes the live state |
| `POST /analyze` | JSON `{"strategy":"...", "state":"optional edited text"}` |
| `POST /do-one` | JSON `{"action":"<legal move>"}`; one human action |
| `POST /play-one` | JSON `{"strategy":"..."}`; one strategy action |
| `POST /do` | JSON `{"action":"...", "strategy":"..."}`; human action, then agent actions until that player's turn or game end |
| `POST /undo`, `POST /redo` | Restore a snapshot; no body required |
| `GET /reset` | Reset that session; returns `true` |

`/do-one`, `/do`, and `/play-one` return `{"continue": bool, "state": snapshot}`;
terminal responses also include `points` and `num_turns`. Undo/redo return the
snapshot directly. Invalid options or moves return 400; unknown game names
return 404. Strategies must come from the session's listed names.

The snapshot fields are:

| Field | Meaning |
| --- | --- |
| `name`, `spec` | Base name and full constructor specification |
| `board`, `board_with_moves` | Active-player observation and model input; at terminal both use forced seat 0 without moves |
| `moves` | Ordered legal strings, empty when ended; use these for controls |
| `current_player`, `round`, `ended` | Current game status |
| `points` | `null` while playing; final seat-0 score difference when ended |
| `history` | Entries with `turn`, `actor`, `action`, `strategy`, `player`, `round`, `ended` |
| `can_undo`, `can_redo` | Availability of history operations |

History status fields are captured after applying each move; `player` is the
resulting current player, not necessarily the mover. Analysis returns
`probabilities` (move-to-probability mapping), `move_order`, and any strategy
info, such as `reference_value` for a neural model. Use `probabilities` for
display; older `moves` info can be strategy-specific.

## Add a renderer

Registered games immediately work through the generic text view. Visual
renderers live in `boardrl/serve/static/app.js`, with separate helpers for
larger renderers such as Santorini, Splendor, and Traversal. To add visuals:

1. Parse the real `snapshot.board_with_moves` format; retain the raw view.
2. Implement `renderYourGame(game)` returning HTML. Escape game text before
   interpolating it into HTML or attributes, using the existing helpers.
3. Emit `data-action="<canonical move>"` on legal interactive controls. The
   shared handler sends that move to the server.
4. Register it under the base game name in the `renderers` map. Preserve the
   generic policy list for actions that have no board representation.
5. Add a regression based on a real game observation: visual structure, legal
   controls, important labels, and no `undefined` output.

```bash
node --check boardrl/serve/static/app.js
node --check boardrl/serve/static/traversal.js
uv run pytest tests/test_serve.py tests/test_serve_tictactoe.py tests/test_traversal_ui.py
```

The current server uses the shared static shell. Old `boardrl/games/*/ui.html`
files are historical and are not the integration point.

## Hosting

The documentation is static and can deploy to GitHub Pages; the game server
requires a running Python process, Torch, a compiler/cache, and model files.
For a private remote workbench, run one Uvicorn worker behind your existing
authenticated reverse proxy or SSH tunnel. The app itself has no authentication,
user-isolated sessions, or persistent game storage. Binding publicly would
expose shared mutable game and model-analysis endpoints.

See [documentation deployment](documentation.md) for the automated static site.
