# Connect Four

Seven-column, six-row board. Players drop coloured discs into columns trying to connect four in a row horizontally, vertically or diagonally. Moves are column indices `0`-`6` as strings.

## Metrics
- `print_short_history()` prints move sequences for each player and shows the final coloured board and score.
- `metrics()` reports terminal rate, draw rate, and average actions per game.
- Policy metrics are grouped by stable strategy identity, independent of seat rotation.
- Per-strategy metrics include win rate, points, average actions, collapse, winning-game count, and the average probability assigned to the final winning move.
