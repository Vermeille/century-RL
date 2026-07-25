# Resume warning

`step-500.pth` was produced by resuming from `step-475.pth` before adaptive
training state was included in checkpoints. Scheduled-perplexity strength and
EMA, adaptive-KL state, and the learner learning-rate baseline were reset. The
entropy strength jumped from about 0.020 before interruption to near its 0.1
initial value, and policy performance regressed.

Use `step-475.pth` as the valid endpoint of this control. Do not use
`step-500.pth` for comparisons. Neither checkpoint can be resumed exactly,
because both predate the checkpoint-state fix.

Checkpoints created by the updated trainer save and restore the learner, whose
state includes all ordered loss state.
