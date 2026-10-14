"""Rollout execution, traces, and batched model inference.

Algorithms provide concrete player strategies. This module owns the mechanics
of seating those strategies, executing games, and representing the resulting
trajectories. Opponent selection deliberately lives in experiment code.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Sequence
from contextlib import contextmanager

import torch
from tqdm import tqdm  # type: ignore[import-untyped]

from boardrl.games.semantics import CompetitiveOutcome, CooperativeOutcome
from boardrl.games.strategies import PolicySamplingStrategy
from boardrl.utils import BatchProcessor, Game, run_tasks

import pyximport  # type: ignore[import-untyped]

pyximport.install()
from boardrl.cyutils import fast_sample  # type: ignore[import-not-found]  # noqa: E402


Strategy = Callable[[Game], Awaitable[tuple[torch.Tensor, dict]]]
Lineup = Sequence[Strategy]
LineupFactory = Callable[[int], Sequence[Strategy]]


class Record:
    def __init__(self, game: Game, action_distribution, action: int, info: dict):
        self.state = info.get("state")
        if self.state is None:
            self.state = game.display_with_moves()
        self.metadata = {
            key: value for key, value in info.items() if key not in ("state", "moves")
        }
        self.moves = game.moves[:]
        self.action_distribution = action_distribution
        self.action_idx = action
        self.current_diff_points = game.diff_points()
        self.my_points = game.points()
        self.episodic_utility = 0.0
        self.terminal = False
        self.truncated = False
        self.player = game.current_player()
        self.round = game.round()


class EndState:
    def __init__(self, game: Game, player: int, *, outcome=None):
        outcome = outcome or CompetitiveOutcome()
        self.state = game.display(force=player)
        self.terminal = game.ended()
        self.truncated = not self.terminal
        self.cause = "proper" if self.terminal else "toolong"
        self.my_points = game.points_for(player)
        self.current_diff_points = game.diff_points_for(player)
        self.won = outcome.won(game, self.terminal)
        self.episodic_utility = outcome.utility(game, player, self.terminal)
        self.player = player
        self.round = game.round()


class PlayerTrace(list):
    def __init__(self, seat_id: int, strategy_id: int):
        super().__init__()
        self.seat_id = seat_id
        self.strategy_id = strategy_id


class GameTrace(list):
    """History of one game, indexed by physical seat."""

    def __init__(self, seat_traces: list[PlayerTrace]):
        self.by_seat = seat_traces
        self.by_strategy = sorted(seat_traces, key=lambda trace: trace.strategy_id)
        super().__init__(self.by_seat)

    def num_players(self):
        return len(self)

    def won(self):
        return self.by_seat[0][-1].won


class TraceGroup(list[PlayerTrace]):
    """The traces belonging to one seat or strategy across many games."""

    def __init__(self, identity: int, traces: list[PlayerTrace]):
        self.identity = identity
        super().__init__(traces)

    def points(self) -> list[float]:
        return [trace[-1].current_diff_points for trace in self]

    def outcomes(self) -> list[float]:
        return [
            getattr(
                trace[-1],
                "episodic_utility",
                float(trace[-1].current_diff_points > 0)
                - float(trace[-1].current_diff_points < 0),
            )
            for trace in self
        ]

    def wins(self) -> list[float]:
        return [
            1.0 if outcome > 0 else (0.5 if outcome == 0 else 0.0)
            for outcome in self.outcomes()
        ]

    def win_rate(self) -> float:
        wins = self.wins()
        return sum(wins) / len(wins)

    def avg_points(self) -> float:
        points = self.points()
        return sum(points) / len(points)

    def avg_reward(self) -> float:
        records = [record for trace in self for record in trace]
        return sum(record.reward for record in records) / len(records)

    def num_actions(self) -> int:
        return sum(max(len(trace) - 1, 0) for trace in self)

    def avg_actions(self) -> float:
        return self.num_actions() / len(self)

    def sensitivity(self) -> float:
        """How strongly the policy distribution depends on the observed state."""
        records = [record for trace in self for record in trace[:-1]]
        if not records:
            return 0.0

        marginal: dict[str, float] = {}
        conditional_entropy = 0.0
        for record in records:
            if len(record.moves) != len(record.action_distribution):
                raise ValueError(
                    "action distribution length does not match the move list"
                )
            probabilities = torch.softmax(
                torch.as_tensor(record.action_distribution).detach().float(),
                dim=0,
            ).cpu()
            positive = probabilities[probabilities > 0]
            conditional_entropy -= float((positive * positive.log()).sum())
            for move, probability in zip(record.moves, probabilities.tolist()):
                marginal[move] = marginal.get(move, 0.0) + probability

        vocabulary_size = len(marginal)
        if vocabulary_size <= 1:
            return 0.0

        count = len(records)
        marginal_entropy = -sum(
            probability / count * math.log(probability / count)
            for probability in marginal.values()
            if probability > 0
        )
        information = max(
            0.0,
            marginal_entropy - conditional_entropy / count,
        )
        return min(1.0, information / math.log(vocabulary_size))


class TraceGroups(list[TraceGroup]):
    """Rollout traces grouped explicitly by seat or strategy identity."""

    def __init__(
        self,
        games: "Rollouts",
        identity: Callable[[PlayerTrace], int],
    ):
        traces = [trace for game in games for trace in game.by_seat]
        super().__init__(
            TraceGroup(key, [trace for trace in traces if identity(trace) == key])
            for key in sorted({identity(trace) for trace in traces})
        )

    def group(self, identity: int) -> TraceGroup:
        try:
            return next(group for group in self if group.identity == identity)
        except StopIteration as exc:
            raise KeyError(f"unknown trace identity {identity}") from exc

    def sensitivity(self) -> list[float]:
        return [group.sensitivity() for group in self]


class Rollouts(list):
    """A batch of completed or truncated game trajectories."""

    def __init__(self, games: list[GameTrace]):
        super().__init__(games)

    def only_player(self, players: list[int]):
        return Rollouts(
            [
                GameTrace(
                    sorted(
                        [game.by_seat[player] for player in players],
                        key=lambda trace: trace.seat_id,
                    )
                )
                for game in self
            ]
        )

    def only_strategy(self, strategies: list[int]):
        return Rollouts(
            [
                GameTrace(
                    sorted(
                        [game.by_strategy[strategy] for strategy in strategies],
                        key=lambda trace: trace.seat_id,
                    )
                )
                for game in self
            ]
        )

    def num_players(self):
        return self[0].num_players()

    @property
    def by_seat(self) -> TraceGroups:
        return TraceGroups(self, lambda trace: trace.seat_id)

    @property
    def by_strategy(self) -> TraceGroups:
        return TraceGroups(self, lambda trace: trace.strategy_id)

    def grouped(self, by: str) -> TraceGroups:
        try:
            return {"strategy": self.by_strategy, "seat": self.by_seat}[by]
        except KeyError as exc:
            raise ValueError("by must be 'strategy' or 'seat'") from exc

    def all_traces(self):
        for game in self:
            for player in game:
                yield player

    def num_samples(self):
        return sum(len(player) for player in self.all_traces())

    def num_traces(self):
        return len(self) * self.num_players()

    def sensitivity(self) -> list[float]:
        return self.by_seat.sensitivity()

    def my_games(self, num: int, *, by: str = "strategy"):
        return list(self.grouped(by).group(num))

    def my_points(self, num: int, *, by: str = "strategy"):
        return self.grouped(by).group(num).points()

    def my_wins(self, num: int, *, by: str = "strategy"):
        return self.grouped(by).group(num).wins()

    def win_rate(self, num: int, *, by: str = "strategy"):
        return self.grouped(by).group(num).win_rate()

    def objective_win_rate(self):
        return sum(game.won() for game in self) / len(self)

    def my_avg_points(self, num: int, *, by: str = "strategy"):
        return self.grouped(by).group(num).avg_points()

    def my_avg_reward(self, num: int, *, by: str = "strategy"):
        return self.grouped(by).group(num).avg_reward()


async def _play_game(
    game: Game,
    strategies: Sequence[Strategy],
    max_steps: int,
    outcome,
):
    for _ in range(max_steps):
        if game.ended():
            break
        player = game.current_player()
        distribution, info = await strategies[player](game)
        if distribution.ndim != 1:
            raise ValueError("strategy distribution must be a 1D tensor")
        action = fast_sample(torch.softmax(distribution, dim=0))
        yield Record(game, distribution, action, info)
        game.play_idx(action)

    for player in range(len(strategies)):
        yield EndState(game, player, outcome=outcome)


@torch.no_grad()
def play_games(
    make_game,
    lineups: Sequence[Sequence[Strategy]],
    *,
    max_steps: int,
    rotate: bool = True,
    description: str | None = "playing games",
    outcome=None,
) -> Rollouts:
    """Execute concrete player lineups and return their trajectories."""
    outcome = outcome or CompetitiveOutcome()
    data: list[GameTrace | None] = [None] * len(lineups)

    with tqdm(
        total=len(lineups),
        desc=description,
        disable=description is None,
    ) as progress:
        async def run_game(index: int):
            lineup = lineups[index]
            num_players = len(lineup)
            offset = index if rotate else 0
            seated = [
                lineup[(seat + offset) % num_players]
                for seat in range(num_players)
            ]
            traces = [
                PlayerTrace(
                    seat_id=seat,
                    strategy_id=(seat + offset) % num_players,
                )
                for seat in range(num_players)
            ]
            game = make_game(num_players=num_players)
            async for record in _play_game(
                game,
                seated,
                max_steps,
                outcome,
            ):
                traces[record.player].append(record)
            data[index] = GameTrace(traces)
            progress.update(1)

        run_tasks([run_game(index) for index in range(len(lineups))])

    return Rollouts(data)  # type: ignore[arg-type]


def _materialize_lineups(
    lineup: Lineup | LineupFactory,
    games: int,
) -> list[list[Strategy]]:
    if isinstance(lineup, Sequence):
        return [list(lineup) for _ in range(games)]
    return [list(lineup(game)) for game in range(games)]


class Inference:
    """Turn a model into batched player strategies."""

    def __init__(self, model, *, batch_size: int, timeout: float = 0.001, name="model"):
        self.model = model
        self.processor = BatchProcessor(
            batch_size, model, timeout=timeout, model_name=name
        )

    def policy(
        self,
        *,
        temperature: float = 1.0,
        epsilon: float = 0.0,
        dirichlet_alpha: float = 0.3,
        record_moves: bool = False,
    ) -> PolicySamplingStrategy:
        return PolicySamplingStrategy(
            self.processor,
            temperature=temperature,
            epsilon=epsilon,
            dirichlet_alpha=dirichlet_alpha,
            include_moves=record_moves,
        )

    @contextmanager
    def evaluating(self):
        was_training = self.model.training
        self.model.eval()
        try:
            yield self
        finally:
            self.model.train(was_training)


class RolloutRunner:
    """Configured façade over the canonical :func:`play_games` primitive."""

    def __init__(
        self,
        make_game,
        *,
        progress: bool = True,
        outcome=None,
        coop: bool = False,
    ):
        self.make_game = make_game
        self.progress = progress
        self.coop = coop
        self.outcome = outcome or (
            CooperativeOutcome() if coop else CompetitiveOutcome()
        )

    def play(
        self,
        lineup: Lineup | LineupFactory,
        *,
        games: int,
        max_steps: int,
        rotate: bool = True,
        description: str = "rollouts",
    ) -> Rollouts:
        concrete = _materialize_lineups(lineup, games)
        if not concrete:
            return Rollouts([])
        return play_games(
            self.make_game,
            concrete,
            max_steps=max_steps,
            rotate=rotate,
            description=description if self.progress else None,
            outcome=self.outcome,
        )
