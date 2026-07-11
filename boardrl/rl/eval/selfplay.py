import torch
from tqdm import tqdm
from typing import Tuple, Callable, Awaitable

from boardrl.utils import Game, run_tasks
import pyximport

pyximport.install()
from boardrl.cyutils import fast_sample  # noqa: E402


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
        for key in ("reference_policy", "reference_value", "reference_max_q"):
            if key in info:
                setattr(self, key, info[key])


class EndState:
    def __init__(self, game: Game, player: int):
        self.state = game.display(force=player)
        self.terminal = game.ended()
        self.truncated = not self.terminal
        self.cause = "proper" if self.terminal else "toolong"
        self.my_points = game.points_for(player)
        self.current_diff_points = game.diff_points_for(player)
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
    @property
    def games(self):  # backward compatibility
        return self

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

    def all_traces(self):
        for game in self:
            for player in game:
                yield player

    def num_samples(self):
        return sum(len(player) for player in self.all_traces())

    def num_traces(self):
        return len(self) * self.num_players()

    def collapse(self) -> list[float]:
        """Return per-player similarity of action sequences.

        For each seat, compare that player's move sequences across games
        using the string representation of each action. Sequences are
        padded to the longest trace before computing the fraction of
        matching actions, and scores are averaged over all pairs. Values
        lie in ``[0, 1]`` with ``1.0`` for identical play traces and
        ``0.0`` when no pair shared an action at the same step.
        """

        def score(sequences: list[list[str]]) -> float:
            n = len(sequences)
            if n <= 1:
                return 1.0

            max_len = max(len(s) for s in sequences)
            if max_len == 0:
                return 1.0

            total = 0.0
            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = sequences[i], sequences[j]
                    matches = sum(int(x == y) for x, y in zip(a, b))
                    matches += max_len - max(len(a), len(b))
                    total += matches / max_len
                    count += 1
            return total / count

        return [
            score([[r.moves[r.action_idx] for r in game[seat][:-1]] for game in self])
            for seat in range(self.num_players())
        ]

    #
    # ------------------------------------------------------------------
    # Metrics previously provided by ``PitResults``
    # ------------------------------------------------------------------
    def my_games(self, num: int, *, by: str = "strategy"):
        if by == "strategy":
            return [game.by_strategy[num] for game in self]
        if by == "seat":
            return [game.by_seat[num] for game in self]
        raise ValueError("by must be 'strategy' or 'seat'")

    def my_points(self, num: int, *, by: str = "strategy"):
        return [hist[-1].current_diff_points for hist in self.my_games(num, by=by)]

    def my_wins(self, num: int, *, by: str = "strategy"):
        return [
            1 if p > 0 else (0.5 if p == 0 else 0) for p in self.my_points(num, by=by)
        ]

    def win_rate(self, num: int, *, by: str = "strategy"):
        wins = self.my_wins(num, by=by)
        return sum(wins) / len(wins)

    def my_avg_points(self, num: int, *, by: str = "strategy"):
        pts = self.my_points(num, by=by)
        return sum(pts) / len(pts)

    def my_avg_reward(self, num: int, *, by: str = "strategy"):
        my_games = self.my_games(num, by=by)
        return sum(g.reward for game in my_games for g in game) / sum(
            len(game) for game in my_games
        )


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
        yield EndState(game, p)


@torch.no_grad()
def self_play2(
    make_game,
    strategies: list[list[Strategy]],
    max_len: int,
    rotate: bool = True,
    desc: str = "playing games",
):
    n_games = len(strategies)
    data: list[GameTrace | None] = [None] * n_games

    with tqdm(total=n_games, desc=desc) as pbar:
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
            async for record in play_game(game, mixed_strategies, max_len):
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
):
    return self_play2(make_game, [strategies] * n_games, max_len, rotate, desc=desc)


@torch.no_grad()
def pit(make_game, strategies, n_games, max_len, *, rotate: bool = True):
    return self_play(make_game, strategies, n_games, max_len, rotate=rotate, desc="pit")
