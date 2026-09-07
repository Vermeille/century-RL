import torch

from boardrl.utils import RegisterByName, Game
from boardrl.rl.model import load_model


strategy_from_string = RegisterByName(arg_readers={"model": load_model})


_CUBE_VALUES = {"Y": 1, "R": 2, "G": 3, "B": 4}
_COLORS = tuple(_CUBE_VALUES)
_TARGET_WEIGHT = 0.25
_ACTION_POTENTIAL_WEIGHT = 0.75
_MIN_ACTION_POTENTIAL = 6.0
_MAX_ENGINE_CARDS = 5
_CLAIM_BONUS = 2.0
_REJECT_SCORE = -1e9


def _parse_stock(text: str):
    counts = {color: 0 for color in _COLORS}
    number = ""
    for char in text:
        if char.isdigit():
            number += char
            continue
        counts[char] += int(number) if number else 1
        number = ""
    return counts


def _stock_counts(stock):
    counts = {color: 0 for color in _COLORS}
    for color in stock:
        counts[color] += 1
    return counts


def _weighted_value(counts):
    return sum(_CUBE_VALUES[color] * counts[color] for color in _COLORS)


def _target_distance(counts, targets):
    if not targets:
        return 0
    return min(
        sum(
            _CUBE_VALUES[color] * max(target[color] - counts[color], 0)
            for color in _COLORS
        )
        for target in targets
    )


def _apply_transform(counts, action: str):
    spent_text, gained_text = action.split(">", 1)
    spent = _parse_stock(spent_text)
    gained = _parse_stock(gained_text)
    out = counts.copy()
    for color in _COLORS:
        out[color] += gained[color] - spent[color]

    # Match Stock.trim(): remove from the largest pile, breaking ties Y->R->G->B.
    while sum(out.values()) > 10:
        largest = max(_COLORS, key=lambda color: out[color])
        out[largest] -= 1
    return out


def _transform_score(counts, action: str, targets):
    after = _apply_transform(counts, action)
    economic_gain = _weighted_value(after) - _weighted_value(counts)
    target_gain = _target_distance(counts, targets) - _target_distance(after, targets)
    return economic_gain + _TARGET_WEIGHT * target_gain


def _card_potential_from_string(card: str):
    if ">" not in card:
        return 0.0
    spent_text, gained_text = card.split(">", 1)
    spent = _parse_stock(spent_text)
    gained = _parse_stock(gained_text)
    spent_size = sum(spent.values())
    gained_size = sum(gained.values())
    gain = _weighted_value(gained) - _weighted_value(spent)

    if spent_size == 0:
        return float(_weighted_value(gained))

    max_footprint = max(spent_size, gained_size)
    repeats = max(1, 10 // max_footprint)
    return float(repeats * gain)


def _visible_victory_targets(g):
    targets = []
    in_board = False
    for line in g.display().splitlines():
        if line == "_Board":
            in_board = True
            continue
        if not in_board:
            continue
        if line.startswith("A"):
            break
        if line.startswith("V"):
            _, card = line.split(" ", 1)
            cost, _ = card.split(">", 1)
            targets.append(_parse_stock(cost))
    return targets


def _owned_action_card_count(player):
    return sum(
        line.startswith("H") or line.startswith("D")
        for line in player.display(hidden=False).splitlines()
    )


def _one_hot_policy(moves, index):
    one_hot = torch.zeros(len(moves), dtype=torch.float)
    one_hot[index] = 1
    return one_hot


@strategy_from_string.register("tempo_greedy")
class TempoGreedyStrategy:
    """Simple deterministic Century baseline built around per-turn tempo.

    Yellow/red/green/blue cubes are valued 1/2/3/4. Merchant-card plays are
    scored by their actual post-trim economic gain plus a small bonus for moving
    toward any visible victory card. Rest is valued as half the best recovered
    play, victory cards cash out weighted cube value into VP, and only strong
    merchant cards are bought while the engine is still small.
    """

    def _score_harvest(self, move, stock, targets):
        _, action = move.split(" ", 1)
        return _transform_score(stock, action, targets)

    def _score_rest(self, g, me, targets):
        rested = g.copy(randomize=False)
        rested.play_str("R")
        player = rested.get_player(me)
        stock = _stock_counts(player.stock)
        best_recovered = _REJECT_SCORE
        for card in player.hand:
            for action in card.gen_move(player.stock):
                best_recovered = max(
                    best_recovered,
                    _transform_score(stock, action, targets),
                )
        if best_recovered == _REJECT_SCORE:
            return _REJECT_SCORE
        return 0.5 * best_recovered

    def _score_victory(self, g, move, me, stock_value, victory_points):
        trial = g.copy(randomize=False)
        trial.play_str(move)
        player = trial.get_player(me)

        if trial.ended():
            point_diff = trial.diff_points_for(me)
            if point_diff > 0:
                return 1000.0 + point_diff
            if point_diff == 0:
                return 500.0
            return -1000.0 + point_diff

        stock_delta = _weighted_value(_stock_counts(player.stock)) - stock_value
        vp_gain = player.victory_points() - victory_points
        return vp_gain + stock_delta + _CLAIM_BONUS

    def _score_action_purchase(self, g, move, me, stock_value, stock, targets):
        player = g.get_player(me)
        if _owned_action_card_count(player) >= _MAX_ENGINE_CARDS:
            return _REJECT_SCORE

        action_name = move.split(" ", 1)[0]
        action_index = int(action_name[1:])
        potential = _card_potential_from_string(str(g.action.pile[action_index]))
        if potential < _MIN_ACTION_POTENTIAL:
            return _REJECT_SCORE

        trial = g.copy(randomize=False)
        trial.play_str(move)
        after_stock = _stock_counts(trial.get_player(me).stock)
        immediate_gain = _weighted_value(after_stock) - stock_value
        target_gain = _target_distance(stock, targets) - _target_distance(
            after_stock, targets
        )
        return (
            _ACTION_POTENTIAL_WEIGHT * potential
            + immediate_gain
            + _TARGET_WEIGHT * target_gain
        )

    async def __call__(self, g: Game):
        moves = g.moves
        me = g.current_player()
        player = g.get_player(me)
        stock = _stock_counts(player.stock)
        stock_value = _weighted_value(stock)
        victory_points = player.victory_points()
        targets = _visible_victory_targets(g)

        scores = []
        for move in moves:
            if move[0] == "H":
                score = self._score_harvest(move, stock, targets)
            elif move == "R":
                score = self._score_rest(g, me, targets)
            elif move[0] == "V":
                score = self._score_victory(
                    g, move, me, stock_value, victory_points
                )
            elif move[0] == "A":
                score = self._score_action_purchase(
                    g, move, me, stock_value, stock, targets
                )
            else:
                score = _REJECT_SCORE
            scores.append(float(score))

        best = max(range(len(moves)), key=lambda index: scores[index])
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
