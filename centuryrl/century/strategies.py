import torch
from random import choice as rndchoice

import pyximport

pyximport.install(setup_args={"script_args": ["--cython-cplus"]})
from centuryrl.century.engine import Game


class RandomStrategy:
    def __call__(self, g: Game):
        return rndchoice(g.moves), g.moves


class RandomBuyStrategy:
    def __call__(self, g: Game):
        moves = g.moves
        for mov in moves:
            if mov[0] == "V":
                return mov, moves
        return rndchoice(g.moves), g.moves


class AllActionsThenRandomBuyStrategy:
    def __call__(self, g: Game):
        moves = g.moves
        for mov in moves:
            if mov.startswith("A0"):
                return mov, moves
        for mov in moves:
            if mov[0] == "V":
                return mov, moves
        return rndchoice(g.moves), g.moves


class NoActionsRandomBuyStrategy:
    def __call__(self, g: Game):
        moves = g.moves
        for mov in moves:
            if mov[0] == "V":
                return mov, moves
        return rndchoice([mov for mov in g.moves if mov[0] != "A"]), g.moves


class ArgmaxStrategy:
    def __init__(self, nn):
        nn.eval()
        self.nn = nn

    def __call__(self, g: Game):
        policy = self.nn([g.display_with_moves()])[0][0]
        idx = policy.argmax()
        return g.moves[idx], list(zip(policy.tolist(), g.moves))


class PolicyGuidedMCMCStrategy:
    def __init__(self, budget: int, nn):
        self.budget = budget
        nn.eval()
        self.nn = nn

    @torch.no_grad()
    def __call__(self, g: Game):
        policy = self.nn([g.display_with_moves()])[0][0]
        policy_sorted = policy.argsort(descending=True)
        scores = []
        for move in policy_sorted[: self.budget]:
            me = g.current_player()
            g2 = g.copy()
            g2.play_str(g.moves[move.item()])

            if not g2.ended():
                g2.simulate_to_end()
            scores.append(g2.diff_points_for(me))
        return g.moves[policy_sorted[scores.index(max(scores))]], list(
            zip(policy.tolist(), scores, g.moves)
        )


class PolicySamplingStrategy:
    def __init__(self, nn):
        self.nn = nn
        nn.eval()

    @torch.no_grad()
    def __call__(self, g: Game):
        if len(g.moves) == 1:
            return g.moves[0], [(g.moves[0], 1.0)]

        policy = self.nn([g.display_with_moves()])[0][0]
        idx = torch.multinomial(torch.softmax(policy, dim=0), 1)
        return g.moves[idx], list(zip(torch.softmax(policy, dim=0).tolist(), g.moves))
