import torch
from tqdm import tqdm
import itertools
from boardrl.utils import Game
from boardrl.utils import run_tasks
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
    data = [[[] for _ in range(n_players)] for _ in range(n_games)]

    with tqdm(total=n_games, desc=desc) as pbar:

        async def run_game(idx):
            game = make_game(num_players=n_players)
            async for record in play_game(game, strategies, max_len):
                data[idx][record.player].append(record)
            pbar.update(1)

        run_tasks([run_game(i) for i in range(n_games)])

    return data


class PitResults:
    def __init__(self, games):
        self.games = games
        self.num_players = len(games[0])

    def my_points(self, player_num):
        return [
            history[-1].current_diff_points for history in self.my_games(player_num)
        ]

    def my_wins(self, player_num):
        return [
            (1 if p > 0 else 0.5 if p == 0 else 0) for p in self.my_points(player_num)
        ]

    def win_rate(self, player_num):
        my_wins = self.my_wins(player_num)
        return sum(my_wins) / len(my_wins)

    def my_games(self, player_num):
        return [game[player_num] for game in self.games]

    def my_avg_points(self, player_num):
        my_points = self.my_points(player_num)
        return sum(my_points) / len(my_points)


flatten = itertools.chain.from_iterable


@torch.no_grad()
def pit(make_game, strategies, n_games, max_len):
    dat = self_play(make_game, strategies, n_games, max_len)
    return PitResults(dat)
