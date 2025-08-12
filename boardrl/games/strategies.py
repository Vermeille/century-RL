import torch

from boardrl.utils import Game, ModelPool, RegisterByName


def get_model(arg_str, default, provided_arg):
    """Resolve the model argument for strategy creation.

    ``arg_str`` is the model specification from the strategy string.  If a
    :class:`ModelPool` is supplied as ``provided_arg`` the resolution is
    delegated to it, allowing ``model=this``, paths, or ``recent-N`` specs to
    share caching and batching behaviour.  When no pool is supplied, fall back
    to loading a model directly from ``arg_str``.
    """

    assert isinstance(provided_arg, ModelPool)
    return provided_arg(arg_str)


strategy_from_string = RegisterByName(arg_readers={"model": get_model})


def one_hot(i, n, smooth=0.0):
    x = torch.ones(n, dtype=torch.float) * smooth / n
    x[i] += 1 - smooth
    return x


@strategy_from_string.register("random")
class RandomStrategy:
    async def __call__(self, g: Game):
        uniform = one_hot(0, len(g.moves), smooth=1)  # uniform distribution
        return uniform.log(), {
            "moves": dict(zip(g.moves, uniform.tolist())),
        }


@strategy_from_string.register("argmax")
class ArgmaxStrategy:
    def __init__(self, model, epsilon: float = 0.0):
        self.nn = model
        self.epsilon = epsilon

    async def __call__(self, g: Game):
        if torch.rand(1).item() < self.epsilon:
            distribution = one_hot(
                torch.randint(len(g.moves), (1,)).item(), len(g.moves)
            )
            return distribution.log(), {
                "moves": dict(zip(g.moves, distribution.tolist()))
            }
        else:
            policy = (await self.nn(g.display_with_moves())).policy[0].cpu()
            distribution = one_hot(torch.argmax(policy).item(), len(g.moves))
            return distribution.log(), {
                "moves": dict(zip(g.moves, torch.softmax(policy, dim=0).tolist()))
            }


def mean(xs):
    return sum(xs) / len(xs)


@strategy_from_string.register("longest_move")
class LongestMoveStrategy:
    async def __call__(self, g: Game):
        distribution = one_hot(
            max(range(len(g.moves)), key=lambda i: len(g.moves[i])), len(g.moves)
        )
        total = sum(len(move) for move in g.moves)
        return distribution, {"moves": {move: len(g.moves) / total for move in g.moves}}


@strategy_from_string.register("policy_sampling")
class PolicySamplingStrategy:
    def __init__(self, model, temperature: float = 1.0):
        self.nn = model
        self.temperature = temperature

    @torch.no_grad()
    async def __call__(self, g: Game):
        if len(g.moves) == 1:
            return torch.tensor([1.0]), {"moves": {g.moves[0]: 1.0}}

        policy = (await self.nn(g.display_with_moves())).policy[0].cpu()
        policy = policy / self.temperature
        return policy, {"moves": dict(zip(g.moves, policy.tolist()))}


