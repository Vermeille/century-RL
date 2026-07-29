# Connect Four

Seven-column, six-row board. Players drop coloured discs into columns trying to connect four in a row horizontally, vertically or diagonally. Moves are column indices `0`-`6` as strings.

## Metrics
- `print_short_history()` prints move sequences for each player and shows the final coloured board and score.
- `metrics()` tracks the ratio of finished boards, average trace length, collapse, and the average probability of the chosen move in winning games.
