# TicTacToe

Simple 3x3 grid game where two players alternately mark cells with `O` and `X`. The first player to align three marks horizontally, vertically or diagonally wins. Moves are strings `0`-`8` representing board positions.

## Metrics
- `print_short_history()` displays the final board with colour for each symbol and prints the winner's points.
- `metrics()` reports the ratio of games completed without empty spaces and a collapse score.
