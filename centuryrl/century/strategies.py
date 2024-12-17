import torch
import inspect

from centuryrl.rl.model import load_model
import pyximport

pyximport.install()
from centuryrl.century.engine import Game

strategy_registry = {}


def register_strategy(cls):
    # Extract the argument names, types, and defaults from the __init__ method
    if "__init__" in cls.__dict__:
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        arg_info = {
            name: (
                param.annotation
                if param.annotation != inspect.Parameter.empty
                else lambda x: x,
                param.default,
            )
            for name, param in params.items()
            if name != "self"
        }
    else:
        arg_info = {}

    strategy_registry[cls.name] = (cls, arg_info)
    return cls


@register_strategy
class RandomStrategy:
    name = "random"

    def __call__(self, g: Game):
        uniform = torch.tensor([1 / len(g.moves)] * len(g.moves))
        return uniform.log(), {
            "moves": dict(zip(g.moves, uniform.tolist())),
        }


@register_strategy
class RandomBuyStrategy:
    name = "random_buy"

    def __call__(self, g: Game):
        moves = g.moves
        for mov in moves:
            if mov[0] == "V":
                one_hot = torch.zeros(len(g.moves), dtype=torch.float)
                one_hot[g.moves.index(mov)] = 1
                return one_hot.log(), {"moves": dict(zip(g.moves, one_hot.tolist()))}

        uniform = torch.tensor([1 / len(g.moves)] * len(g.moves))
        return uniform.log(), {"moves": dict(zip(g.moves, uniform.tolist()))}


@register_strategy
class AllActionsThenRandomBuyStrategy:
    name = "all_actions_then_random_buy"

    def __call__(self, g: Game):
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


@register_strategy
class NoActionsRandomBuyStrategy:
    name = "no_actions_random_buy"

    def __call__(self, g: Game):
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


@register_strategy
class ArgmaxStrategy:
    name = "argmax"

    def __init__(self, nn):
        nn.eval()
        self.nn = nn

    def __call__(self, g: Game):
        policy = self.nn([g.display_with_moves()]).policy[0]
        one_hot = torch.zeros(len(g.moves), dtype=torch.float)
        one_hot[torch.argmax(policy).item()] = 1
        return one_hot.log(), {
            "moves": dict(zip(g.moves, torch.softmax(policy, dim=0).tolist()))
        }


def mean(xs):
    return sum(xs) / len(xs)


@register_strategy
class PickBestMCValueStrategy:
    name = "pick_best_mc_value"

    def __init__(self, budget: int):
        self.budget = budget

    def __call__(self, g: Game):
        values = [[] for _ in g.moves]
        me = g.current_player()

        for _ in range(self.budget):
            for m_i, m in enumerate(g.moves):
                g2 = g.copy()
                g2.play_str(m)
                g2.simulate_to_end()
                values[m_i].append(g2.diff_points_for(me))
        means = [mean(vs) for vs in values]
        policy = torch.median(torch.tensor(values).float(), dim=1).values
        return policy, {
            "moves": dict(zip(g.moves, means)),
        }


@register_strategy
class PickBestValueStrategy:
    name = "pick_best_value"

    def __init__(self, budget: int, model):
        self.budget = budget
        self.model = model
        model.eval()

    def __call__(self, g: Game):
        values = [[] for _ in g.moves]
        me = g.current_player()

        for _ in range(self.budget):
            for m_i, m in enumerate(g.moves):
                g2 = g.copy()
                g2.play_str(m)
                g2.simulate_to_end()
                values[m_i].append(g2.diff_points_for(me))
        means = [mean(vs) for vs in values]
        policy = torch.median(torch.tensor(values).float(), dim=1).values
        return policy, {
            "moves": dict(zip(g.moves, means)),
        }


@register_strategy
class LongestMoveStrategy:
    name = "longest_move"

    def __call__(self, g: Game):
        one_hot = torch.zeros(len(g.moves), dtype=torch.float)
        one_hot[max(range(len(g.moves)), key=lambda i: len(g.moves[i]))] = 1
        total = sum(len(move) for move in g.moves)
        return one_hot, {"moves": {move: len(g.moves) / total for move in g.moves}}


@register_strategy
class PolicySamplingStrategy:
    name = "policy_sampling"

    def __init__(self, model, temperature: float = 1.0, epsilon: float = 0):
        self.nn = model
        model.eval()
        self.temperature = temperature
        self.epsilon = epsilon

    @torch.no_grad()
    def __call__(self, g: Game):
        if len(g.moves) == 1:
            return torch.tensor([1.0]), {"moves": {g.moves[0]: 1.0}}

        policy = self.nn([g.display_with_moves()]).policy[0] / self.temperature
        policy = torch.softmax(policy, dim=0)
        policy = (1 - self.epsilon) * policy + self.epsilon / len(g.moves)
        return policy.log(), {"moves": dict(zip(g.moves, policy.tolist()))}


def strategy_from_string(strategy_string, model=None):
    strategy_name, *arg_list = strategy_string.split(",")
    args = {arg.split("=")[0]: arg.split("=")[1] for arg in arg_list}

    if strategy_name not in strategy_registry:
        raise ValueError(f"Unknown strategy: {strategy_string}")

    strategy_class, arg_info = strategy_registry[strategy_name]
    init_args = {}

    for arg_name, (arg_type, default) in arg_info.items():
        if arg_name == "model":
            init_args[arg_name] = model or load_model(args.get(arg_name, "this"))
        elif arg_name in args:
            init_args[arg_name] = arg_type(args[arg_name])
        else:
            init_args[arg_name] = default

    return strategy_class(**init_args)
