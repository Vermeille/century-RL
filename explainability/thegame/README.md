# The Game policy explainability

These are the standalone analysis programs used to compare the final
`YOLO-yay2` Omni policy with the final `YOLO` Strict policy and to distill the
Strict policy into an interpretable move-ranking rule.

## Scripts

| Script | Purpose |
|---|---|
| `checkpoint_analysis.py` | Large fresh checkpoint evaluation, score distributions, behavior summaries, and input-scrambling sensitivity tests. |
| `memory_analysis.py` | Offline and online `Mem:` interventions, including zeroing, scrambling, and removing played-card memory. |
| `identity_memory_analysis.py` | Replaces `Mem:` with a different reachable memory vector matched on visible state identity and measures action/score changes. |
| `nonlowest_analysis.py` | Samples decisions where the policy rejects the cheapest card and replays the chosen branch against every minimum-cost alternative. |
| `strategy_ablation.py` | Factorial evaluation of lowest-cost tie-breaking, unrestricted card selection, and Omni stop/continue behavior. |
| `distill_strict.py` | Fits an interpretable linear move ranker to Strict, performs one DAgger iteration, and evaluates the distilled strategies in full games. |

All programs write detailed JSON suitable for follow-up analysis. They do not
modify checkpoints or training state.

## Checkpoints analyzed

```text
checkpoints/schedule-search/coop/thegame,mode=omni/patchformer-medium-p8/YOLO-yay2/step-900.pth
checkpoints/schedule-search/coop/thegame,mode=strict/patchformer-medium-p8/YOLO/step-900.pth
```

For exact historical reproduction, run a script with the BoardRL source that
created its checkpoint. The original investigation used source overlays for
the two runs because the active worktree had subsequently changed. When the
active checkout remains checkpoint-compatible, the `PYTHONPATH` prefix below
can be omitted.

```bash
STRICT_SOURCE=/tmp/thegame-policy-sources/strict
OMNI_SOURCE=/tmp/thegame-policy-sources/omni-yay2
PYTHON=.venv/bin/python
STRICT_CHECKPOINT='checkpoints/schedule-search/coop/thegame,mode=strict/patchformer-medium-p8/YOLO/step-900.pth'
OMNI_CHECKPOINT='checkpoints/schedule-search/coop/thegame,mode=omni/patchformer-medium-p8/YOLO-yay2/step-900.pth'
```

The temporary overlay paths above record the paths used during the original
analysis; they are not repository dependencies. Replace them with durable
source snapshots when reproducing elsewhere.

## Primary evaluation

Run both modes with the same game count and batching protocol:

```bash
PYTHONPATH="$OMNI_SOURCE" "$PYTHON" explainability/thegame/checkpoint_analysis.py \
  --mode omni --checkpoint "$OMNI_CHECKPOINT" --games 4096 \
  --ablation-games 1024 --output /tmp/thegame-omni-analysis.json

PYTHONPATH="$STRICT_SOURCE" "$PYTHON" explainability/thegame/checkpoint_analysis.py \
  --mode strict --checkpoint "$STRICT_CHECKPOINT" --games 4096 \
  --ablation-games 1024 --output /tmp/thegame-strict-analysis.json
```

## Memory interventions

`memory_analysis.py` produces its own matched baseline before running the
online interventions:

```bash
PYTHONPATH="$OMNI_SOURCE" "$PYTHON" explainability/thegame/memory_analysis.py \
  --mode omni --checkpoint "$OMNI_CHECKPOINT" \
  --output /tmp/thegame-omni-memory.json

PYTHONPATH="$STRICT_SOURCE" "$PYTHON" explainability/thegame/memory_analysis.py \
  --mode strict --checkpoint "$STRICT_CHECKPOINT" \
  --output /tmp/thegame-strict-memory.json
```

The identity-matched intervention is run similarly:

```bash
PYTHONPATH="$STRICT_SOURCE" "$PYTHON" explainability/thegame/identity_memory_analysis.py \
  --mode strict --checkpoint "$STRICT_CHECKPOINT" \
  --baseline-results /tmp/thegame-strict-analysis.json \
  --output /tmp/thegame-strict-identity-memory.json
```

Use `--mode omni`, the Omni source/checkpoint, and its primary JSON for the
corresponding Omni run.

## Non-lowest counterfactuals

```bash
PYTHONPATH="$OMNI_SOURCE" "$PYTHON" explainability/thegame/nonlowest_analysis.py \
  --game 'thegame,mode=omni' --checkpoint "$OMNI_CHECKPOINT" \
  --label omni-yay2 --games 256 --samples 2048 \
  --output /tmp/thegame-omni-nonlowest.json

PYTHONPATH="$STRICT_SOURCE" "$PYTHON" explainability/thegame/nonlowest_analysis.py \
  --game 'thegame,mode=strict' --checkpoint "$STRICT_CHECKPOINT" \
  --label strict-yolo --games 256 --samples 2048 \
  --output /tmp/thegame-strict-nonlowest.json
```

Each sampled state is replayed after the policy choice and after every tied
minimum-cost card choice. End-turn `x` is deliberately excluded from card-cost
comparisons.

## Strategy-factor ablation

```bash
PYTHONPATH="$OMNI_SOURCE" "$PYTHON" explainability/thegame/strategy_ablation.py \
  --mode omni --checkpoint "$OMNI_CHECKPOINT" --games 1024 \
  --output /tmp/thegame-omni-strategy-ablation.json

PYTHONPATH="$STRICT_SOURCE" "$PYTHON" explainability/thegame/strategy_ablation.py \
  --mode strict --checkpoint "$STRICT_CHECKPOINT" --games 1024 \
  --output /tmp/thegame-strict-strategy-ablation.json
```

The hybrids separate model tie-breaking, non-lowest card sequencing, and—in
Omni—learned stop/continue decisions. All variants reuse the same seed scheme
so their per-game scores are paired.

## Strict policy distillation

```bash
PYTHONPATH="$STRICT_SOURCE" "$PYTHON" explainability/thegame/distill_strict.py \
  --checkpoint "$STRICT_CHECKPOINT" \
  --teacher-games 512 --dagger-games 512 --eval-games 1024 \
  --output /tmp/thegame-strict-distillation.json
```

The JSON contains held-out action agreement, gameplay evaluation, the largest
coefficients, and complete weight maps for the initial and DAgger rankers with
and without memory features. The high-scoring interpretable result from the
original investigation was the initial no-memory ranker; the DAgger refit is a
diagnostic and should not automatically replace it merely because it was fit
on more states.

## Runtime notes

- These are CUDA-oriented analyses and intentionally do not silently fall back
  to CPU for checkpoint evaluation.
- Large evaluations can take several minutes.
- Keep game counts, seeds, checkpoint paths, and source revision matched when
  comparing outputs.
- JSON outputs can be large because some scripts retain per-game or
  per-decision records for paired statistics.
