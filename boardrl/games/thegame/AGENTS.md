# The Game

Simplified implementation of *The Game*. Four piles exist: two ascending starting at 1 and two descending starting at 100. Players play cards from hand in the form `card->pile`, obeying ascending/descending rules and optional "10 rule" jumps. The game ends when no moves remain or the deck empties. Score is higher with fewer cards left.

## Metrics
- `print_short_history()` lists the sequence of moves taken by each player.
- `metrics()` reports average points along with:
  - `avg_cost`: average cost of played moves;
  - `ratio_lowest_cost`: fraction of times the lowest-cost move was chosen;
  - `ten_rule_moves`: frequency of using the special 10-rule;
  - plus a collapse metric.
