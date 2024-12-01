import torch
from random import choice as rndchoice

from centuryrl.rl.model import load_model
import pyximport

pyximport.install(setup_args={"script_args": ["--cython-cplus"]})
from centuryrl.century.engine import Game


class RandomStrategy:
    def __call__(self, g: Game):
        uniform = torch.tensor([1 / len(g.moves)] * len(g.moves))
        return uniform, {
            "moves": dict(zip(g.moves, uniform.tolist())),
        }


class RandomBuyStrategy:
    def __call__(self, g: Game):
        moves = g.moves
        for mov in moves:
            if mov[0] == "V":
                one_hot = torch.zeros(len(g.moves), dtype=torch.float)
                one_hot[g.moves.index(mov)] = 1
                return one_hot, {"moves": dict(zip(g.moves, one_hot.tolist()))}

        uniform = torch.tensor([1 / len(g.moves)] * len(g.moves))
        return uniform, {"moves": dict(zip(g.moves, uniform.tolist()))}


class AllActionsThenRandomBuyStrategy:
    def __call__(self, g: Game):
        moves = g.moves
        for mov in moves:
            if mov.startswith("A0"):
                one_hot = torch.zeros(len(g.moves), dtype=torch.float)
                one_hot[g.moves.index(mov)] = 1
                return one_hot, {"moves": dict(zip(g.moves, one_hot.tolist()))}

        for mov in moves:
            if mov[0] == "V":
                one_hot = torch.zeros(len(g.moves), dtype=torch.float)
                one_hot[g.moves.index(mov)] = 1
                return one_hot, {"moves": dict(zip(g.moves, one_hot.tolist()))}

        uniform = torch.tensor([1 / len(g.moves)] * len(g.moves))
        return uniform, {"moves": dict(zip(g.moves, uniform.tolist()))}


class NoActionsRandomBuyStrategy:
    def __call__(self, g: Game):
        moves = g.moves
        for mov in moves:
            if mov[0] == "V":
                one_hot = torch.zeros(len(g.moves), dtype=torch.float)
                one_hot[g.moves.index(mov)] = 1
                return one_hot, {"moves": dict(zip(g.moves, one_hot.tolist()))}

        num_no_action = sum(1 for mov in g.moves if mov[0] != "A")
        dist = torch.tensor(
            [(1 / num_no_action) if move[0] != "A" else 0 for move in g.moves]
        )
        return rndchoice([mov for mov in g.moves if mov[0] != "A"]), {
            "moves": dict(zip(g.moves, dist.tolist()))
        }


class ArgmaxStrategy:
    def __init__(self, nn):
        nn.eval()
        self.nn = nn

    def __call__(self, g: Game):
        policy = self.nn([g.display_with_moves()]).policy[0]
        one_hot = torch.zeros(len(g.moves), dtype=torch.float)
        one_hot[torch.argmax(policy).item()] = 1
        return one_hot, {
            "moves": dict(zip(g.moves, torch.softmax(policy, dim=0).tolist()))
        }


def mean(xs):
    return sum(xs) / len(xs)


class PickBestMCValueStrategy:
    def __init__(self, budget: int):
        self.budget = budget

    def __call__(self, g: Game):
        values = [[] for _ in g.moves]
        me = g.current_player()

        for _ in range(self.budget):
            for m_i, m in enumerate(g.moves):
                g2 = g.copy()
                g2.play_str(m)
                g2.simulate_to_end(RandomBuyStrategy())
                values[m_i].append(g2.diff_points_for(me))
        means = [mean(vs) for vs in values]
        return torch.softmax(torch.tensor(means) * 100, dim=0), {
            "moves": dict(zip(g.moves, means)),
        }


class LongestMoveStrategy:
    def __call__(self, g: Game):
        one_hot = torch.zeros(len(g.moves), dtype=torch.float)
        one_hot[max(range(len(g.moves)), key=lambda i: len(g.moves[i]))] = 1
        total = sum(len(move) for move in g.moves)
        return one_hot, {"moves": {move: len(g.moves) / total for move in g.moves}}


class PolicySamplingStrategy:
    def __init__(self, nn, temperature=1.0, epsilon=0):
        self.nn = nn
        nn.eval()
        self.temperature = temperature
        self.epsilon = epsilon

    @torch.no_grad()
    def __call__(self, g: Game):
        if len(g.moves) == 1:
            return torch.tensor([1.0]), {"moves": {g.moves[0]: 1.0}}

        policy = self.nn([g.display_with_moves()]).policy[0] / self.temperature
        policy = torch.softmax(policy, dim=0)
        policy = (1 - self.epsilon) * policy + self.epsilon / len(g.moves)
        return policy, {"moves": dict(zip(g.moves, policy.tolist()))}


def strategy_from_string(strategy_string):
    if strategy_string == "random":
        return RandomStrategy()
    elif strategy_string == "random_buy":
        return RandomBuyStrategy()
    elif strategy_string == "all_actions_then_random_buy":
        return AllActionsThenRandomBuyStrategy()
    elif strategy_string == "no_actions_random_buy":
        return NoActionsRandomBuyStrategy()
    elif strategy_string.startswith("argmax"):
        model_path = strategy_string.split(":")[1]
        return ArgmaxStrategy(load_model(model_path))
    elif strategy_string.startswith("policy_sampling"):
        model_path = strategy_string.split(":")[1]
        return PolicySamplingStrategy(load_model(model_path))
    elif strategy_string.startswith("pick_best_mc_value"):
        budget = int(strategy_string.split(":")[1])
        return PickBestMCValueStrategy(budget)
    else:
        raise ValueError(f"Unknown strategy: {strategy_string}")
