# Sum

Toy game where two digits `a` and `b` are shown each round. A player guesses a digit; if it matches the integer half of `a+b` they score a point and new digits are drawn, otherwise turn passes. First to 3 points wins. Moves are strings `0`-`9`.

## Metrics
- `print_short_history()` lists point totals accumulated by each player during play.
- `metrics_to_visdom()` records average trace length and a collapse value.
