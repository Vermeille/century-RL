import torch
from random import choice as rndchoice

from centuryrl.rl.model import load_model
import pyximport

pyximport.install(setup_args={"script_args": ["--cython-cplus"]})
from centuryrl.century.engine import Game


class RandomStrategy:
    def __call__(self, g: Game):
        return rndchoice(g.moves), {
            "moves": {move: 1 / len(g.moves) for move in g.moves}
        }


class RandomBuyStrategy:
    def __call__(self, g: Game):
        moves = g.moves
        for mov in moves:
            if mov[0] == "V":
                return mov, {"moves": {move: 1 if move == mov else 0 for move in moves}}
        return rndchoice(g.moves), {"moves": {move: 1 / len(moves) for move in moves}}


class AllActionsThenRandomBuyStrategy:
    def __call__(self, g: Game):
        moves = g.moves
        for mov in moves:
            if mov.startswith("A0"):
                return mov, {"moves": {move: 1 if move == mov else 0 for move in moves}}

        for mov in moves:
            if mov[0] == "V":
                return mov, {"moves": {move: 1 if move == mov else 0 for move in moves}}

        return rndchoice(g.moves), {"moves": {move: 1 / len(moves) for move in moves}}


class NoActionsRandomBuyStrategy:
    def __call__(self, g: Game):
        moves = g.moves
        for mov in moves:
            if mov[0] == "V":
                return mov, {"moves": {move: 1 if move == mov else 0 for move in moves}}

        num_no_action = sum(1 for mov in g.moves if mov[0] != "A")
        return rndchoice([mov for mov in g.moves if mov[0] != "A"]), {
            "moves": {
                move: (1 / num_no_action) if move[0] != "A" else 0 for move in g.moves
            }
        }


class ArgmaxStrategy:
    def __init__(self, nn):
        nn.eval()
        self.nn = nn

    def __call__(self, g: Game):
        policy = self.nn([g.display_with_moves()]).policy[0]
        idx = policy.argmax()
        return g.moves[idx], {
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
        return g.moves[max(range(len(values)), key=lambda i: mean(values[i]))], {
            "moves": dict(zip(g.moves, [mean(vs) for vs in values])),
        }


class LongestMoveStrategy:
    def __call__(self, g: Game):
        return max(g.moves, key=len), {
            "moves": {move: 1 / len(g.moves) for move in g.moves}
        }


class PolicyGuidedMCMCStrategy:
    def __init__(self, budget: int, nn):
        self.budget = budget
        nn.eval()
        self.nn = nn

    @torch.no_grad()
    def __call__(self, g: Game):
        assert g.num_players == 2
        model_out = self.nn([g.display_with_moves()])
        policy = torch.softmax(model_out.policy[0], dim=0)
        value = model_out.value[0].item()
        prompts = []
        for i in torch.multinomial(policy, self.budget, replacement=True):
            g2 = g.copy()
            g2.play_str(g.moves[i])
            if not g2.ended():
                prompts.append((i, g2.display_with_moves()))

        scores = [[value] for _ in range(len(g.moves))]
        if prompts:
            for i, val in zip(
                [p[0] for p in prompts], self.nn([p[1] for p in prompts]).value.tolist()
            ):
                scores[i].append(-val)
        return g.moves[max(range(len(scores)), key=lambda i: mean(scores[i]))], {
            "moves": dict(zip(g.moves, zip(policy.tolist(), scores))),
            "board": value,
        }


class PolicySamplingStrategy:
    def __init__(self, nn, temperature=1.0, epsilon=0):
        self.nn = nn
        nn.eval()
        self.temperature = temperature
        self.epsilon = epsilon

    @torch.no_grad()
    def __call__(self, g: Game):
        if len(g.moves) == 1:
            return g.moves[0], [(g.moves[0], 1.0)]

        policy = self.nn([g.display_with_moves()]).policy[0] / self.temperature
        policy = torch.softmax(policy, dim=0)
        policy = (1 - self.epsilon) * policy + self.epsilon / len(g.moves)
        idx = torch.multinomial(policy, 1)
        return g.moves[idx], {
            "moves": dict(zip(g.moves, torch.softmax(policy, dim=0).tolist()))
        }


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
