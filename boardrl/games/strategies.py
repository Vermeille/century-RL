import torch
from inspect import signature
from urllib.parse import unquote

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
    return provided_arg(unquote(arg_str) if arg_str is not None else None)


strategy_from_string = RegisterByName(arg_readers={"model": get_model})


def one_hot(i, n, *, smooth=0.0):
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


def randomized_copy(g: Game):
    if "randomize" in signature(g.copy).parameters:
        return g.copy(randomize=True)
    return g.copy()


@strategy_from_string.register("longest_move")
class LongestMoveStrategy:
    async def __call__(self, g: Game):
        distribution = one_hot(
            max(range(len(g.moves)), key=lambda i: len(g.moves[i])), len(g.moves)
        )
        total = sum(len(move) for move in g.moves)
        return distribution.log(), {
            "moves": {move: len(g.moves) / total for move in g.moves}
        }


@strategy_from_string.register("policy_sampling")
class PolicySamplingStrategy:
    def __init__(
        self,
        model,
        temperature: float = 1.0,
        epsilon: float = 0.0,
        dirichlet_alpha: float = 0.3,
        include_moves: bool = True,
    ):
        self.nn = model
        self.temperature = temperature
        self.epsilon = epsilon
        self.dirichlet_alpha = dirichlet_alpha
        self.include_moves = include_moves

    @torch.no_grad()
    async def __call__(self, g: Game):
        info: dict[str, object]
        if len(g.moves) == 1:
            info = {"moves": {g.moves[0]: 1.0}} if self.include_moves else {}
            return torch.tensor([1.0]), info

        state = g.display_with_moves()
        nn_out = await self.nn(state)
        raw_policy = nn_out.policy[0].cpu()
        raw_value = nn_out.value.mean.cpu()[0]
        policy = raw_policy / self.temperature
        if self.epsilon != 0.0:
            policy = torch.softmax(policy, dim=0)
            noise = torch.distributions.Dirichlet(
                policy.new_full((len(policy),), self.dirichlet_alpha)
            ).sample()
            policy = (1 - self.epsilon) * policy + self.epsilon * noise
            policy = policy.log()
        info = {
            "state": state,
            "reference_policy": raw_policy,
            "reference_value": raw_value.item(),
            "reference_max_q": (raw_value + (raw_policy - raw_policy.mean()).max()).item(),
        }
        if self.include_moves:
            info["moves"] = dict(zip(g.moves, policy.tolist()))
        return policy, info


@strategy_from_string.register("gumbel")
class Gumbel:
    def __init__(
        self, model, discount_factor: float, num_evals: int = 2, q_scale: float = 1.0
    ):
        self.model = model
        self.num_evals = num_evals
        self.gamma = discount_factor
        self.q_scale = q_scale

    async def __call__(self, g: Game):
        if len(g.moves) == 1:
            return torch.tensor([1.0]), {"moves": {g.moves[0]: 1.0}}

        player = g.current_player()
        base_points = g.diff_points_for(player)
        nn_out = await self.model(g.display_with_moves())
        base_val = nn_out.value.mean.cpu()
        logits = nn_out.policy[0].cpu()
        gumbels = torch.distributions.Gumbel(
            torch.tensor([0.0] * len(logits)), torch.tensor([1.0] * len(logits))
        ).sample((1,))[0]
        qs = torch.full_like(logits, fill_value=float("-inf"))
        num_evals = min(self.num_evals, len(logits))
        for mov in (logits + gumbels).topk(num_evals).indices:
            g2 = randomized_copy(g)
            g2.play_idx(mov.item())
            val = (await self.model(g2.display_with_moves())).value.mean.cpu()
            r = g2.diff_points_for(player) - base_points
            q = 0.1 * r + self.gamma * val - base_val
            qs[mov] = q
        final_move = torch.argmax(logits + gumbels + self.q_scale * qs)
        return one_hot(final_move, len(logits)).log(), {
            "moves": dict(
                zip(g.moves, zip(logits.tolist(), gumbels.tolist(), qs.tolist()))
            )
        }
