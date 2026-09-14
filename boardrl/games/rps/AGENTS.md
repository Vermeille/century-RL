# Rock Paper Scissors

Single-round RPS with hidden choices. Each player picks `rock`, `paper` or `scissors`; choices reveal after both move. The winner scores one point, otherwise zero for a tie.

## Metrics
- `print_short_history()` prints each player's chosen move, predicted probabilities, and final points.
- `metrics()` aggregates average move probabilities across games and reports policy sensitivity.
