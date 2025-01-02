import torch

from centuryrl.century.utils import BatchProcessor, RegisterByName
from centuryrl.rl.model import load_model
import pyximport

pyximport.install()
from centuryrl.century.engine import fast_sample


class Game:
    moves: list[str]
    num_players: int

    def current_player(self) -> int: ...

    def display_with_moves(self) -> str: ...

    def copy(self) -> "Game": ...

    def play_str(self, move: str): ...

    def play_idx(self, move: int): ...

    def simulate_to_end(self): ...

    def diff_points_for(self, player: int) -> float: ...

    def ended(self) -> bool: ...


strategy_from_string = RegisterByName(arg_readers={"model": load_model})


@strategy_from_string.register("random")
class RandomStrategy:
    def __call__(self, g: Game):
        uniform = torch.tensor([1 / len(g.moves)] * len(g.moves))
        return uniform.log(), {
            "moves": dict(zip(g.moves, uniform.tolist())),
        }


@strategy_from_string.register("century_random_buy")
class CenturyRandomBuyStrategy:
    def __call__(self, g: Game):
        moves = g.moves
        for mov in moves:
            if mov[0] == "V":
                one_hot = torch.zeros(len(g.moves), dtype=torch.float)
                one_hot[g.moves.index(mov)] = 1
                return one_hot.log(), {"moves": dict(zip(g.moves, one_hot.tolist()))}

        uniform = torch.tensor([1 / len(g.moves)] * len(g.moves))
        return uniform.log(), {"moves": dict(zip(g.moves, uniform.tolist()))}


@strategy_from_string.register("century_all_actions_then_random_buy")
class CenturyAllActionsThenRandomBuyStrategy:
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


@strategy_from_string.register("century_no_actions_random_buy")
class CenturyNoActionsRandomBuyStrategy:
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


@strategy_from_string.register("argmax")
class ArgmaxStrategy:
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


@strategy_from_string.register("pick_best_value")
class PickBestValueStrategy:
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

        processor = BatchProcessor(batch_size=64, process_fn=self.model, timeout=0.1)

        async def try_move(m_i):
            m = g.moves[m_i]
            g2 = g.copy()
            g2.play_str(m)
            for _ in range(g2.num_players - 1):
                if g2.ended():
                    break
                board = g2.display_with_moves()
                policy = (await processor.send(board)).policy[0]
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
            print(m_i, m, values[m_i])

        processor.run_tasks(
            [try_move(m_i) for m_i in range(len(g.moves)) for _ in range(self.budget)]
        )
        means = [mean(vs) for vs in values]
        policy = torch.median(torch.tensor(values).float(), dim=1).values
        sm = torch.softmax(policy / self.temperature, dim=0)
        sm = 0.95 * sm + 0.05 / len(g.moves)
        return sm.log(), {
            "moves": dict(zip(g.moves, means)),
        }


@strategy_from_string.register("longest_move")
class LongestMoveStrategy:
    def __call__(self, g: Game):
        one_hot = torch.zeros(len(g.moves), dtype=torch.float)
        one_hot[max(range(len(g.moves)), key=lambda i: len(g.moves[i]))] = 1
        total = sum(len(move) for move in g.moves)
        return one_hot, {"moves": {move: len(g.moves) / total for move in g.moves}}


@strategy_from_string.register("policy_sampling")
class PolicySamplingStrategy:
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
