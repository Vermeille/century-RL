from collections import Counter

import torch

from boardrl.utils import RegisterByName, Game


strategy_from_string = RegisterByName()


_MIN_ACTION_POTENTIAL = 6.0
_MAX_ENGINE_CARDS = 5
_ACTION_POTENTIAL_WEIGHT = 0.75
_CLAIM_BONUS = 2.0
_VICTORY_POINT_WEIGHT = 1.0
_VICTORY_PROGRESS_WEIGHT = 2.0
_AFFORDABLE_VICTORY_BONUS = 100.0
_REJECT_SCORE = -1e9

_SPICE_VALUES = {"Y": 1, "R": 2, "G": 3, "B": 4}


def _card_potential(card):
    spent, gained = card.takes(), card.gives()
    spent_size = sum(1 for _ in spent)
    gained_size = sum(1 for _ in gained)
    gain = gained.weighted_value() - spent.weighted_value()
    if spent_size == 0:
        return float(gained.weighted_value())
    return float(max(1, 10 // max(spent_size, gained_size)) * gain)


def _victory_progress(stock, victory_cards):
    """Score how close ``stock`` is to its most attractive visible victory."""
    available = Counter(stock)
    return max(
        _VICTORY_POINT_WEIGHT * card.points
        - sum(
            _SPICE_VALUES[spice] * max(0, count - available[spice])
            for spice, count in Counter(card.cost).items()
        )
        for card in victory_cards
    )


@strategy_from_string.register("tempo_greedy")
class TempoGreedyStrategy:
    """One-ply Century baseline using only observable game objects."""

    def _score_harvest(self, g, move, stock_value):
        stock = g.get_player(g.current_player()).stock
        victory_cards = g.visible_victory()
        next_stock = g.preview_stock(move)
        stock_gain = next_stock.weighted_value() - stock_value
        progress = _victory_progress(next_stock, victory_cards) - _victory_progress(
            stock, victory_cards
        )
        return stock_gain + _VICTORY_PROGRESS_WEIGHT * progress

    def _score_rest(self, player):
        return 2.0 + 0.75 * player.discard_count()

    def _score_victory(self, g, move, me, player, stock_value, victory_points):
        card = g.visible_victory()[int(move[1:])]
        stock_delta = g.preview_stock(move).weighted_value() - stock_value
        vp_after = victory_points + card.points

        if player.victory_count() + 1 >= g.goal_card_count():
            if g.num_players == 1:
                diff = vp_after
            else:
                diff = vp_after - max(
                    g.points_for(i) for i in range(g.num_players) if i != me
                )
            return (1000.0 if diff > 0 else 500.0 if diff == 0 else -1000.0) + diff

        # An affordable victory is the payoff for all preceding engine work.
        # Do not let another small resource gain postpone it indefinitely.
        return _AFFORDABLE_VICTORY_BONUS + card.points + stock_delta + _CLAIM_BONUS

    def _score_action_purchase(self, g, move, player, stock_value):
        if len(player.hand) + player.discard_count() >= _MAX_ENGINE_CARDS:
            return _REJECT_SCORE

        action_index = int(move.split(" ", 1)[0][1:])
        card = g.action.visible()[action_index][0]
        potential = _card_potential(card)
        if potential < _MIN_ACTION_POTENTIAL:
            return _REJECT_SCORE
        stock_delta = g.preview_stock(move).weighted_value() - stock_value
        return _ACTION_POTENTIAL_WEIGHT * potential + stock_delta

    async def __call__(self, g: Game):
        moves = g.moves
        me = g.current_player()
        player = g.get_player(me)
        stock_value = player.stock.weighted_value()
        victory_points = player.victory_points()
        scores = []

        for move in moves:
            if move[0] == "H":
                score = self._score_harvest(g, move, stock_value)
            elif move == "R":
                score = self._score_rest(player)
            elif move[0] == "V":
                score = self._score_victory(
                    g, move, me, player, stock_value, victory_points
                )
            elif move[0] == "A":
                score = self._score_action_purchase(g, move, player, stock_value)
            else:
                score = _REJECT_SCORE
            scores.append(float(score))

        best = max(range(len(moves)), key=scores.__getitem__)
        policy = torch.zeros(len(moves), dtype=torch.float)
        policy[best] = 1
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
