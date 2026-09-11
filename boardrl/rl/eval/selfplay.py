import torch
from tqdm import tqdm  # type: ignore[import-untyped]
from collections.abc import Callable, Iterator, Sequence
from typing import Awaitable, Tuple

from boardrl.utils import Game, run_tasks
import pyximport  # type: ignore[import-untyped]

pyximport.install()
from boardrl.cyutils import fast_sample  # type: ignore[import-not-found]  # noqa: E402


class Record:
    def __init__(self, game: Game, action_distribution, action: int, info: dict):
        self.state = info.get("state")
        if self.state is None:
            self.state = game.display_with_moves()
        self.moves = game.moves[:]
        self.action_distribution = action_distribution
        self.action_idx = action
        self.current_diff_points = game.diff_points()
        self.my_points = game.points()
        self.terminal = False
        self.truncated = False
        self.player = game.current_player()
        self.round = game.round()
        self._training_info = {
            key: info[key]
            for key in ("reference_policy", "reference_value", "reference_max_q")
            if key in info
        }

    def training_sample(self):
        """Convert this rollout record without exposing recorder internals."""
        from boardrl.training.sample import TrainingSample

        return TrainingSample(
            state=self.state,
            action_idx=self.action_idx,
            action_distribution=self.action_distribution,
            score=float(self.score),
            reward=float(self.reward),
            returns=self.returns,
            next=None,
            terminal=False,
            truncated=False,
            **self._training_info,
        )


class EndState:
    def __init__(self, game: Game, player: int, *, coop: bool = False):
        self.state = game.display(force=player)
        self.terminal = game.ended()
        self.truncated = not self.terminal
        self.cause = "proper" if self.terminal else "toolong"
        self.my_points = game.points_for(player)
        self.current_diff_points = game.diff_points_for(player)
        if coop:
            self.won = game.won() if self.terminal else False
        else:
            self.won = None
        self.player = player
        self.round = game.round()


class PlayerTrace(list):
    def __init__(self, seat_id: int, strategy_id: int):
        super().__init__()
        self.seat_id = seat_id
        self.strategy_id = strategy_id


class GameTrace(list):
    """History of a single self-play game.

    Indexing the instance directly returns traces ordered by *seat* id. The
    ``by_strategy`` attribute provides the same traces ordered by strategy id.
    """

    def __init__(self, seat_traces: list[PlayerTrace]):
        self.by_seat = seat_traces
        self.by_strategy = sorted(seat_traces, key=lambda t: t.strategy_id)
        super().__init__(self.by_seat)

    def num_players(self):
        return len(self)

    def won(self):
        return self.by_seat[0][-1].won


class TraceGrouping:
    """Select one stable trace identity from every game in a rollout."""

    def identity(self, trace: PlayerTrace) -> int:
        raise NotImplementedError


class SeatGrouping(TraceGrouping):
    def identity(self, trace: PlayerTrace) -> int:
        return trace.seat_id


class StrategyGrouping(TraceGrouping):
    def identity(self, trace: PlayerTrace) -> int:
        return trace.strategy_id


class TraceGroup(Sequence[PlayerTrace]):
    """The traces belonging to one seat or strategy across many games."""

    def __init__(self, identity: int, traces: list[PlayerTrace]):
        self.identity = identity
        self.traces = traces

    def __getitem__(self, index):
        return self.traces[index]

    def __len__(self) -> int:
        return len(self.traces)

    def __iter__(self) -> Iterator[PlayerTrace]:
        return iter(self.traces)

    def points(self) -> list[float]:
        return [trace[-1].current_diff_points for trace in self]

    def wins(self) -> list[float]:
        return [
            1.0 if points > 0 else (0.5 if points == 0 else 0.0)
            for points in self.points()
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

    def collapse(self) -> float:
        """Average pairwise action-sequence similarity for this identity."""
        sequences = [
            [record.moves[record.action_idx] for record in trace[:-1]]
            for trace in self
        ]
        n = len(sequences)
        if n <= 1:
            return 1.0

        max_len = max(len(sequence) for sequence in sequences)
        if max_len == 0:
            return 1.0

        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                first, second = sequences[i], sequences[j]
                matches = sum(int(a == b) for a, b in zip(first, second))
                matches += max_len - max(len(first), len(second))
                total += matches / max_len
                count += 1
        return total / count


class TraceGroups:
    """Rollout traces grouped explicitly by seat or strategy identity."""

    def __init__(self, games: "SelfPlayResults", grouping: TraceGrouping):
        self.games = games
        self.grouping = grouping
        self.identities = tuple(
            sorted(
                {
                    grouping.identity(trace)
                    for game in games
                    for trace in game.by_seat
                }
            )
        )

    def __getitem__(self, index):
        return self.group(self.identities[index])

    def __len__(self) -> int:
        return len(self.identities)

    def __iter__(self) -> Iterator[TraceGroup]:
        return (self.group(identity) for identity in self.identities)

    def group(self, identity: int) -> TraceGroup:
        if identity not in self.identities:
            raise KeyError(f"unknown trace identity {identity}")
        return TraceGroup(
            identity,
            [
                trace
                for game in self.games
                for trace in game.by_seat
                if self.grouping.identity(trace) == identity
            ],
        )

    def map(self, metric: Callable[[TraceGroup], object]) -> dict[str, object]:
        return {str(group.identity): metric(group) for group in self}

    def collapse(self) -> list[float]:
        return [group.collapse() for group in self]


BY_SEAT = SeatGrouping()
BY_STRATEGY = StrategyGrouping()


class SelfPlayResults(list):
    """Container for a batch of self-play games.

    Provides convenience helpers for analysing the outcome of self-play. Games
    are stored as :class:`GameTrace` objects.
    """

    def __init__(self, games: list[GameTrace]):
        super().__init__(games)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def only_player(self, players: list[int]):
        return SelfPlayResults(
            [
                GameTrace(
                    list(
                        sorted([g.by_seat[p] for p in players], key=lambda t: t.seat_id)
                    )
                )
                for g in self
            ]
        )

    def only_strategy(self, strategies: list[int]):
        return SelfPlayResults(
            [
                GameTrace(
                    list(
                        sorted(
                            [g.by_strategy[s] for s in strategies],
                            key=lambda t: t.seat_id,
                        )
                    )
                )
                for g in self
            ]
        )

    def num_players(self):
        return self[0].num_players()

    @property
    def by_seat(self) -> TraceGroups:
        return TraceGroups(self, BY_SEAT)

    @property
    def by_strategy(self) -> TraceGroups:
        return TraceGroups(self, BY_STRATEGY)

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

    def collapse(self) -> list[float]:
        """Return per-seat similarity of action sequences.

        For each seat, compare that seat's move sequences across games
        using the string representation of each action. Sequences are
        padded to the longest trace before computing the fraction of
        matching actions, and scores are averaged over all pairs. Values
        lie in ``[0, 1]`` with ``1.0`` for identical play traces and
        ``0.0`` when no pair shared an action at the same step.
        """

        return self.by_seat.collapse()

    #
    # ------------------------------------------------------------------
    # Metrics previously provided by ``PitResults``
    # ------------------------------------------------------------------
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


Strategy = Callable[[Game], Awaitable[Tuple[torch.Tensor, dict]]]


async def call_strategy(strategy, game: Game):
    try:
        result = strategy(game)
    except TypeError:
        result = strategy()(game)
    return await result


async def play_game(
    game: Game,
    strategies: list[Strategy],
    max_len: int,
    coop: bool = False,
):
    for _ in range(max_len):
        if game.ended():
            break
        p = game.current_player()
        dist, info = await call_strategy(strategies[p], game)
        assert dist.ndim == 1, "Distribution must be a 1D tensor"
        action = fast_sample(torch.softmax(dist, dim=0))
        rec = Record(game, dist, action, info)
        yield rec
        game.play_idx(action)
    for p in range(len(strategies)):
        yield EndState(game, p, coop=coop)


@torch.no_grad()
def self_play2(
    make_game,
    strategies: list[list[Strategy]],
    max_len: int,
    rotate: bool = True,
    desc: str | None = "playing games",
    coop: bool = False,
):
    n_games = len(strategies)
    data: list[GameTrace | None] = [None] * n_games

    with tqdm(total=n_games, desc=desc, disable=desc is None) as pbar:
        pbar.update(0)

        async def run_game(idx):
            n_players = len(strategies[idx])
            offset = idx if rotate else 0
            mixed_strategies = [
                strategies[idx][(i + offset) % n_players] for i in range(n_players)
            ]
            traces = [
                PlayerTrace(seat_id=i, strategy_id=(i + offset) % n_players)
                for i in range(n_players)
            ]
            game = make_game(num_players=n_players)
            async for record in play_game(
                game, mixed_strategies, max_len, coop=coop
            ):
                traces[record.player].append(record)
            data[idx] = GameTrace(traces)
            pbar.update(1)

        run_tasks([run_game(i) for i in range(n_games)])

    return SelfPlayResults(data)  # type: ignore[arg-type]


@torch.no_grad()
def self_play(
    make_game,
    strategies: list[Strategy],
    n_games: int,
    max_len: int,
    rotate: bool = True,
    desc: str = "playing games",
    coop: bool = False,
):
    return self_play2(
        make_game,
        [strategies] * n_games,
        max_len,
        rotate,
        desc=desc,
        coop=coop,
    )


@torch.no_grad()
def pit(
    make_game,
    strategies,
    n_games,
    max_len,
    *,
    rotate: bool = True,
    coop: bool = False,
):
    return self_play(
        make_game,
        strategies,
        n_games,
        max_len,
        rotate=rotate,
        desc="pit",
        coop=coop,
    )
