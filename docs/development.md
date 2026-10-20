# Development and source map

Use a source checkout with Python 3.12+, `uv sync --locked`, and a compiler for
the Cython sampler/engine. Read repository `AGENTS.md` and the applicable
game contributor notes before implementation changes. Favor small collaborating
classes and polymorphism rather than type-dispatch conditionals.

## Find the implementation

| Area | Source / useful tests |
| --- | --- |
| Public experiment API | `boardrl/__init__.py` |
| Game descriptors and registration | `boardrl/games/__init__.py`, `tests/test_game_registration.py` |
| Game protocol and registry parsing | `boardrl/utils/__init__.py`, `boardrl/utils/registerbyname.py` |
| Scores, outcome, rewards | `boardrl/games/semantics.py`, `tests/test_returns.py` |
| Execution, identities, inference | `boardrl/rollouts.py`, `tests/test_rollouts.py` |
| Reward and target calculations | `boardrl/training/returns.py`, `postprocess.py`, `tests/test_returns.py` |
| Samples and optimization | `boardrl/training/sample.py`, `learner.py`, `tests/test_training_library.py` |
| Losses | `boardrl/rl/model/loss.py`, `tests/test_policy_gradient_loss.py`, `tests/test_loss_flags.py` |
| Model and architecture presets | `boardrl/rl/model/model.py`, `boardrl/models.py`, `tests/test_model_presets.py`, `tests/test_model_heads.py` |
| Checkpoints and run records | `boardrl/checkpoints.py`, `boardrl/run.py`, `tests/test_run_info.py` |
| Metrics | `boardrl/metrics.py`, `tests/test_metrics.py` |
| Evaluation and arena | `boardrl/evaluation.py`, `boardrl/arena/`, `tests/test_arena.py` |
| Shared browser and HTTP state | `boardrl/serve/serve.py`, `static/`, `tests/test_serve.py` |
| Complete experiments | `trainers/`, `examples/`, trainer-specific tests |

## Choose a reference game

- Tic-Tac-Toe and Connect Four: small alternating board games.
- Nim: configurable, bounded state machine and optimal heuristic.
- RPS: hidden decisions; GuessNumber: learned cooperative communication.
- Take5 and Skull King: private hands and multiple action phases.
- The Game: cooperative modes, messages, hidden/full information, augmentations.
- Hanabi and Regicide: cooperative card state and information constraints.
- Santorini: geometry, symmetries, rich web rendering.
- Splendor: private reservations, payment variants, tiebreak semantics.
- Century: Cython integration, card/resource mechanics, and custom descriptor.
- OpenSpiel: native chance resolution and exact evaluation adapters.

See [Add a game](adding-games.md) for the implementation contract and
[Write a training loop](training-library.md) for extension points in algorithms.

## Run focused checks

For general changes, the contributor guide's short suite is:

```bash
uv run pytest tests -k "not transformer and not model"
```

CI uses `uv run pytest tests -m "not slow"` and installs OpenSpiel as an
optional dependency. New long-running learning tests must use
`@pytest.mark.slow`. The full transformer/model learning suites are expensive;
run them when changing the corresponding modules. Use focused tests first:

```bash
uv run pytest tests/test_game_registration.py tests/test_games.py tests/test_strategies.py
uv run pytest tests/test_returns.py tests/test_rollouts.py tests/test_training_library.py
uv run pytest tests/test_action_augmentation.py tests/test_training_sample.py
uv run pytest tests/test_serve.py tests/test_serve_tictactoe.py
```

`test_fast_sample.py` exercises the compiler-backed sampler. Century source
changes need compiler-backed engine tests as well. Frontend renderer changes
also need `node --check boardrl/serve/static/app.js` and the game renderer's
tests. Slow model learning is not needed for a documentation edit; the docs
examples have focused smoke coverage in `tests/test_documentation.py`.

## Update the guides with the behavior

Keep action strings, observation formats, game specs, CLI flags, checkpoint
schemas, and HTTP payloads synchronized with their guides when changing those
interfaces. Explain score units and truncation behavior whenever adding an
algorithm or benchmark result. Run the [strict docs build](documentation.md)
to check navigation, internal links, and section anchors.
