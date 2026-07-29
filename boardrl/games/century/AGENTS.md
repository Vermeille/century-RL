# Century

Cython-based environment inspired by *Century: Spice Road*. Players acquire and upgrade coloured spices to purchase victory cards. Actions include playing action cards, harvesting resources, resting to refresh cards, and buying victory cards.

## Metrics
`Metrics.metrics()` provides statistics used by training sinks and `print_short_history()`:
- `prompt_size`: average/min/max serialized state length.
- `avg_len`: average number of turns per player.
- `avg_points`: average final point total per player.
- `causes`: frequency of game termination reasons (`proper` vs `toolong`).
- `avg_move_summary`: frequency of move types (`A`ction, `H`arvest, `R`est, `V`ictory).
- `buy_rank`: average ranking of purchased victory cards.
