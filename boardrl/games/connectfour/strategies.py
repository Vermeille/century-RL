import random
import torch

from boardrl.games.connectfour.game import ConnectFour
from boardrl.utils import RegisterByName


def one_hot(i, n):
    x = torch.zeros(n, dtype=torch.float)
    x[i] = 1
    return x


strategy_from_string = RegisterByName()


@strategy_from_string.register("tactical_random")
class TacticalRandomStrategy:
    async def __call__(self, g: ConnectFour):
        me = g.current_player()
        opponent = 1 - me

        # 1. Winning move
        for idx, move in enumerate(g.moves):
            sim = g.copy()
            sim.play_str(move)

            if sim.winner() == me:
                probs = one_hot(idx, len(g.moves))
                return probs.log(), {
                    "moves": dict(zip(g.moves, probs.tolist())),
                }

        # 2. Block opponent's immediate winning move
        for idx, move in enumerate(g.moves):
            col = int(move)
            row = g.heights[col]

            sim = g.copy()
            sim.board[col][row] = opponent

            if sim._check_winner_at(col, row) == opponent:
                probs = one_hot(idx, len(g.moves))
                return probs.log(), {
                    "moves": dict(zip(g.moves, probs.tolist())),
                }

        # 3. Random otherwise
        idx = random.choice(range(len(g.moves)))
        probs = one_hot(idx, len(g.moves))
        return probs.log(), {
            "moves": dict(zip(g.moves, probs.tolist())),
        }
