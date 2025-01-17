import torch
from tqdm import tqdm
import itertools
import pyximport

pyximport.install()
from boardrl.cyutils import fast_sample
from boardrl.utils import Game


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


@torch.no_grad()
def self_play(make_game, strategies, n_games, max_len, desc="playing games"):
    n_players = len(strategies)
    data = [[[] for _ in range(n_players)] for _ in range(n_games)]

    for i_game in tqdm(range(n_games), desc=desc):
        g = make_game(num_players=n_players)

        for i_mov in range(max_len):
            if g.ended():
                break

            p = g.current_player()
            dist, debug = strategies[p](g)
            action = fast_sample(torch.softmax(dist, dim=0))

            rec = Record(g, dist, action)
            data[i_game][p].append(rec)

            g.play_idx(action)

        for p in range(n_players):
            data[i_game][p].append(EndState(g, p))

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
