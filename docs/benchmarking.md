# External board-game benchmarks

Century keeps benchmark dependencies outside the core training stack. The
training algorithms stay in `trainers/`; OpenSpiel is used as a game backend
and as an evaluator, while Othello engines are subprocess opponents.

## Install OpenSpiel

OpenSpiel is deliberately optional so benchmark tooling does not enlarge the
runtime dependency set for every Century experiment:

```bash
uv pip install open-spiel
```

## Train benchmark games with the normal Century trainers

`leduc` and `othello` are regular game descriptors backed by OpenSpiel. Chance
nodes are sampled inside the game adapter, so rollout code still sees only
player decisions. In imperfect-information games the model receives the acting
player's information state; in perfect-information games it receives the
observation/board state.

For example, any adversarial trainer that accepts a game spec can use:

```text
leduc
othello
openspiel,game=kuhn_poker
```

No training algorithm needs to be ported to OpenSpiel.

## Leduc: exact NashConv / exploitability

Evaluate a checkpoint with OpenSpiel's best-response implementation:

```bash
uv run python evaluate_nashconv.py checkpoints/step-100.pth \
  --model agent \
  --game leduc_poker
```

The JSON output reports NashConv, two-player exploitability (`NashConv / 2`),
per-player unilateral improvement, and the number of distinct information
states evaluated by the Century model. `--temperature 1` evaluates the policy
as trained; `--temperature 0` evaluates deterministic argmax play.

## Othello: fixed external engine strength

The evaluator speaks the standard subset of GTP used by Egaroucid and Edax. It
runs paired games from the same randomized opening, with the Century agent once
as black and once as white, to reduce first-player and opening bias.

Egaroucid example (flags may differ between releases):

```bash
uv run python evaluate_othello_engine.py checkpoints/step-100.pth \
  --model agent \
  --engine "./Egaroucid_for_Console -gtp -quiet -nobook -l 10 -t 4" \
  --pairs 64 \
  --opening-plies 8 \
  --seed 0
```

Edax can be used the same way by passing its GTP command. Sweep fixed engine
levels with the same paired openings to obtain an externally anchored strength
curve without moving the training code into an engine repository.
