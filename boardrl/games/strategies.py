import torch

from boardrl.utils import CachedBatchProcessor, RegisterByName, Game
from boardrl.rl.model import load_model
import pyximport

pyximport.install()
from boardrl.cyutils import fast_sample


strategy_from_string = RegisterByName(arg_readers={"model": load_model})


def one_hot(i, n, smooth=0.0):
    x = torch.ones(n, dtype=torch.float) * smooth / n
    x[i] += 1 - smooth
    return x


@strategy_from_string.register("random")
class RandomStrategy:
    def __call__(self, g: Game):
        uniform = one_hot(0, len(g.moves), smooth=1)  # uniform distribution
        return uniform.log(), {
            "moves": dict(zip(g.moves, uniform.tolist())),
        }


@strategy_from_string.register("argmax")
class ArgmaxStrategy:
    def __init__(self, nn):
        nn.eval()
        self.nn = nn

    def __call__(self, g: Game):
        policy = self.nn([g.display_with_moves()]).policy[0].cpu()
        distribution = one_hot(torch.argmax(policy).item(), len(g.moves))
        return distribution.log(), {
            "moves": dict(zip(g.moves, torch.softmax(policy, dim=0).tolist()))
        }


def mean(xs):
    return sum(xs) / len(xs)


@strategy_from_string.register("pick_best_mc_value")
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
                g2.simulate_to_end()
                values[m_i].append(g2.diff_points_for(me))
        means = [mean(vs) for vs in values]
        policy = torch.median(torch.tensor(values).float(), dim=1).values
        return policy, {
            "moves": dict(zip(g.moves, means)),
        }


@strategy_from_string.register("minimax_value")
class MinimaxValueStrategy:
    def __init__(self, budget: int, model, discount: float, temperature: float = 0.001):
        self.budget = budget
        self.model = model
        model.eval()
        self.discount = discount
        self.temperature = temperature

    @torch.no_grad()
    def __call__(self, g: Game):
        values = [[] for _ in g.moves]
        me = g.current_player()

        processor = CachedBatchProcessor(
            batch_size=64, process_fn=self.model, timeout=0.01, cache_size=1000
        )

        async def try_move(m_i):
            m = g.moves[m_i]
            g2 = g.copy()
            g2.play_str(m)
            for _ in range(g2.num_players - 1):
                if g2.ended():
                    break
                board = g2.display_with_moves()
                pred = await processor.send(board)
                policy = pred.policy[0]
                m_j = fast_sample(torch.softmax(policy, dim=0))
                g2.play_idx(m_j)
            if g2.ended():
                values[m_i].append(g2.diff_points_for(me))
            else:
                assert g2.current_player() == me
                board = g2.display_with_moves()
                values[m_i].append(
                    g2.diff_points_for(me)
                    + self.discount * (await processor.send(board)).value.mean[0].item()
                )

        processor.run_tasks(
            [try_move(m_i) for m_i in range(len(g.moves)) for _ in range(self.budget)]
        )
        policy = torch.min(torch.tensor(values).float(), dim=1).values
        print(policy)
        sm = one_hot(torch.argmax(policy).item(), len(g.moves), smooth=0.05)
        return sm.log(), {
            "moves": dict(zip(g.moves, policy)),
        }


@strategy_from_string.register("longest_move")
class LongestMoveStrategy:
    def __call__(self, g: Game):
        distribution = one_hot(
            max(range(len(g.moves)), key=lambda i: len(g.moves[i])), len(g.moves)
        )
        total = sum(len(move) for move in g.moves)
        return distribution, {"moves": {move: len(g.moves) / total for move in g.moves}}


@strategy_from_string.register("policy_sampling")
class PolicySamplingStrategy:
    def __init__(self, model, temperature: float = 1.0):
        self.nn = model
        model.eval()
        self.temperature = temperature

    @torch.no_grad()
    def __call__(self, g: Game):
        if len(g.moves) == 1:
            return torch.tensor([1.0]), {"moves": {g.moves[0]: 1.0}}

        policy = self.nn([g.display_with_moves()]).policy[0].cpu() / self.temperature
        # print(policy)
        return policy, {"moves": dict(zip(g.moves, policy.tolist()))}
