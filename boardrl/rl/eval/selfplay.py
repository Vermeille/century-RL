import torch
from tqdm import tqdm

from boardrl.utils import Game, run_tasks
import pyximport

pyximport.install()
from boardrl.cyutils import fast_sample


class Record:
    def __init__(self, game: Game, action_distribution, action: int):
        self.state = game.display_with_moves()
        self.moves = game.moves[:]
        self.action_distribution = action_distribution
        self.action_idx = action
        self.current_diff_points = game.diff_points()
        self.my_points = game.points()
        self.final = False
        self.player = game.current_player()
        self.round = game.round()


class EndState:
    def __init__(self, game: Game, player: int):
        self.state = game.display(force=player)
        self.cause = "proper" if game.ended() else "toolong"
        self.my_points = game.points_for(player)
        self.current_diff_points = game.diff_points_for(player)
        self.player = player
        self.round = game.round()
        self.final = True


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
                GameTrace(sorted([g.by_seat[p] for p in players], key=lambda t: t.seat_id))
                for g in self
            ]
        )

    def only_strategy(self, strategies: list[int]):
        return SelfPlayResults(
            [
                GameTrace(
                    sorted(
                        [g.by_strategy[s] for s in strategies],
                        key=lambda t: t.seat_id,
                    )
                )
                for g in self
            ]
        )

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
        return [1 if p > 0 else 0.5 if p == 0 else 0 for p in self.my_points(num, by=by)]

    def win_rate(self, num: int, *, by: str = "strategy"):
        wins = self.my_wins(num, by=by)
        return sum(wins) / len(wins)

    def my_avg_points(self, num: int, *, by: str = "strategy"):
        pts = self.my_points(num, by=by)
        return sum(pts) / len(pts)


async def play_game(game, strategies, max_len):
    for _ in range(max_len):
        if game.ended():
            break
        p = game.current_player()
        dist, _ = await strategies[p](game)
        action = fast_sample(torch.softmax(dist, dim=0))
        rec = Record(game, dist, action)
        yield rec
        game.play_idx(action)
    for p in range(len(strategies)):
        yield EndState(game, p)


@torch.no_grad()
def self_play(make_game, strategies, n_games, max_len, desc="playing games"):
    n_players = len(strategies)
    data: list[GameTrace | None] = [None] * n_games

    with tqdm(total=n_games, desc=desc) as pbar:

        async def run_game(idx):
            mixed_strategies = [
                strategies[(i + idx) % n_players] for i in range(n_players)
            ]
            game = make_game(num_players=n_players)
            traces = [
                PlayerTrace(seat_id=i, strategy_id=(i + idx) % n_players)
                for i in range(n_players)
            ]
            async for record in play_game(game, mixed_strategies, max_len):
                traces[record.player].append(record)
            data[idx] = GameTrace(traces)
            pbar.update(1)

        run_tasks([run_game(i) for i in range(n_games)])

    return SelfPlayResults(data)  # type: ignore[arg-type]


@torch.no_grad()
def pit(make_game, strategies, n_games, max_len):
    return self_play(make_game, strategies, n_games, max_len)

