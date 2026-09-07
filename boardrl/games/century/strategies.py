import torch

from boardrl.utils import RegisterByName, Game
from boardrl.rl.model import load_model


strategy_from_string = RegisterByName(arg_readers={"model": load_model})


_CUBE_VALUES = {"Y": 1, "R": 2, "G": 3, "B": 4}
_MIN_ACTION_POTENTIAL = 6.0
_ACTION_POTENTIAL_WEIGHT = 0.75
_CLAIM_BONUS = 2.0
_REJECT_SCORE = -1e9


def _stock_value(stock):
    return sum(_CUBE_VALUES[color] for color in stock)


def _card_potential(card):
    spent = card.takes()
    gained = card.gives()
    spent_size = sum(1 for _ in spent)
    gained_size = sum(1 for _ in gained)
    gain = _stock_value(gained) - _stock_value(spent)

    if spent_size == 0:
        return float(_stock_value(gained))

    repeats = max(1, 10 // max(spent_size, gained_size))
    return float(repeats * gain)


def _one_hot_policy(moves, index):
    policy = torch.zeros(len(moves), dtype=torch.float)
    policy[index] = 1
    return policy


@strategy_from_string.register("tempo_greedy")
class TempoGreedyStrategy:
    """Deterministic Century baseline built around weighted cube tempo."""

    def _score_harvest(self, g, move, me, stock_value):
        trial = g.copy(randomize=False)
        trial.play_str(move)
        return _stock_value(trial.get_player(me).stock) - stock_value

    def _score_rest(self, g, me):
        before = len(g.get_player(me).hand)
        trial = g.copy(randomize=False)
        trial.play_str("R")
        recovered = len(trial.get_player(me).hand) - before
        return 2.0 + 0.75 * recovered

    def _score_victory(self, g, move, me, stock_value, victory_points):
        trial = g.copy(randomize=False)
        trial.play_str(move)
        player = trial.get_player(me)

        if trial.ended():
            diff = trial.diff_points_for(me)
            if diff > 0:
                return 1000.0 + diff
            if diff == 0:
                return 500.0
            return -1000.0 + diff

        return (
            player.victory_points()
            - victory_points
            + _stock_value(player.stock)
            - stock_value
            + _CLAIM_BONUS
        )

    def _score_action_purchase(self, g, move, me, stock_value):
        action_index = int(move.split(" ", 1)[0][1:])
        potential = _card_potential(g.action.pile[action_index])
        if potential < _MIN_ACTION_POTENTIAL:
            return _REJECT_SCORE

        trial = g.copy(randomize=False)
        trial.play_str(move)
        stock_delta = _stock_value(trial.get_player(me).stock) - stock_value
        return _ACTION_POTENTIAL_WEIGHT * potential + stock_delta

    async def __call__(self, g: Game):
        moves = g.moves
        me = g.current_player()
        player = g.get_player(me)
        stock_value = _stock_value(player.stock)
        victory_points = player.victory_points()

        scores = []
        for move in moves:
            if move[0] == "H":
                score = self._score_harvest(g, move, me, stock_value)
            elif move == "R":
                score = self._score_rest(g, me)
            elif move[0] == "V":
                score = self._score_victory(
                    g, move, me, stock_value, victory_points
                )
            elif move[0] == "A":
                score = self._score_action_purchase(g, move, me, stock_value)
            else:
                score = _REJECT_SCORE
            scores.append(float(score))

        best = max(range(len(moves)), key=scores.__getitem__)
        policy = _one_hot_policy(moves, best)
        return policy.log(), {
            "moves": dict(zip(moves, policy.tolist())),
            "scores": dict(zip(moves, scores)),
        }


@strategy_from_string.register("random_buy")
class RandomBuyStrategy:
    async def __call__(self, g: Game):
        moves = g.moves
        for mov in moves:
            if mov[0] == "V":
                one_hot = torch.zeros(len(g.moves), dtype=torch.float)
                one_hot[g.moves.index(mov)] = 1
                return one_hot.log(), {"moves": dict(zip(g.moves, one_hot.tolist()))}

        uniform = torch.tensor([1 / len(g.moves)] * len(g.moves))
        return uniform.log(), {"moves": dict(zip(g.moves, uniform.tolist()))}


@strategy_from_string.register("all_actions_then_random_buy")
class AllActionsThenRandomBuyStrategy:
    async def __call__(self, g: Game):
        moves = g.moves
        for mov in moves:
            if mov.startswith("A0"):
                one_hot = torch.zeros(len(g.moves), dtype=torch.float)
                one_hot[g.moves.index(mov)] = 1
                return one_hot.log(), {"moves": dict(zip(g.moves, one_hot.tolist()))}

        for mov in moves:
            if mov[0] == "V":
                one_hot = torch.zeros(len(g.moves), dtype=torch.float)
                one_hot[g.moves.index(mov)] = 1
                return one_hot.log(), {"moves": dict(zip(g.moves, one_hot.tolist()))}

        uniform = torch.tensor([1 / len(g.moves)] * len(g.moves))
        return uniform.log(), {"moves": dict(zip(g.moves, uniform.tolist()))}


@strategy_from_string.register("no_actions_random_buy")
class NoActionsRandomBuyStrategy:
    async def __call__(self, g: Game):
        moves = g.moves
        for mov in moves:
            if mov[0] == "V":
                one_hot = torch.zeros(len(g.moves), dtype=torch.float)
                one_hot[g.moves.index(mov)] = 1
                return one_hot.log(), {"moves": dict(zip(g.moves, one_hot.tolist()))}

        num_no_action = sum(1 for mov in g.moves if mov[0] != "A")
        dist = torch.tensor(
            [(1 / num_no_action) if move[0] != "A" else 0 for move in g.moves]
        )
        return dist.log(), {"moves": dict(zip(g.moves, dist.tolist()))}


@strategy_from_string.register("never_buy")
class NeverBuyStrategy:
    async def __call__(self, g: Game):
        num_no_action = sum(1 for mov in g.moves if mov[0] != "V")
        dist = torch.tensor(
            [(1 / num_no_action) if move[0] != "V" else 0 for move in g.moves]
        )
        return dist.log(), {"moves": dict(zip(g.moves, dist.tolist()))}
