"""External Othello engines as ordinary rollout strategies."""

from boardrl.games.strategies import one_hot


class GTPStrategy:
    def __init__(self, engine):
        self.engine = engine

    async def __call__(self, game):
        # Rollouts interleave games. Restore this position before each query;
        # replay and genmove are synchronous, so another game cannot intervene.
        self.engine.clear_board()
        replay = game._game.new_initial_state()
        for action in game._state.history():
            player = replay.current_player()
            self.engine.play("bw"[player], replay.action_to_string(player, action))
            replay.apply_action(action)
        move = self.engine.genmove("bw"[game.current_player()])
        if move not in game.moves:
            raise ValueError(
                f"engine returned illegal move {move!r}; legal moves: {game.moves}"
            )
        return one_hot(game.moves.index(move), len(game.moves)).log(), {}
