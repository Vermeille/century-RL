import torch
from tqdm import tqdm
import itertools
import pyximport

pyximport.install()
from boardrl.century.engine import Century, fast_sample


class Game: ...


class Record:
    def __init__(self, game: Game, action_distribution, action: int):
        self.state = game.display_with_moves()
        self.moves = game.moves[:]
        self.action_distribution = action_distribution
        self.action_idx = action
        self.current_diff_points = game.diff_points()
        self.my_points = game.points()
        self.notes = []
        self.final = False


class EndState:
    def __init__(self, game: Game, player: int):
        self.cause = "proper" if game.ended() else "toolong"
        self.state = game.display(force=player)
        self.my_points = game.points_for(player)
        self.current_diff_points = game.diff_points_for(player)
        self.notes = []
        self.final = True


@torch.no_grad()
def self_play(strategies, n_games, max_len, desc="playing games"):
    n_players = len(strategies)
    data = [[[] for _ in range(n_players)] for _ in range(n_games)]

    for i_game in tqdm(range(n_games), desc=desc):
        g = Century(num_players=n_players)

        for i_mov in range(max_len):
            if g.ended():
                break

            dist, debug = strategies[g.current_player()](g)
            action = fast_sample(torch.softmax(dist, dim=0))

            rec = Record(g, dist, action)
            rec.notes += [str(debug)]
            data[i_game][g.current_player()].append(rec)

            g.play_idx(action)

        for p in range(n_players):
            data[i_game][p].append(EndState(g, p))

    return data


class PitResults:
    def __init__(self, games, num_players):
        self.games = games
        self.num_players = num_players

    def my_points(self, player_num):
        return [
            history[-1].current_diff_points for history in self.my_games(player_num)
        ]

    def my_wins(self, player_num):
        return [p >= 0 for p in self.my_points(player_num)]

    def win_rate(self, player_num):
        my_wins = self.my_wins(player_num)
        return sum(my_wins) / len(my_wins)

    def my_games(self, player_num):
        return self.games[player_num :: self.num_players]

    def my_avg_points(self, player_num):
        my_points = self.my_points(player_num)
        return sum(my_points) / len(my_points)


flatten = itertools.chain.from_iterable


@torch.no_grad()
def pit(strategies, n_games, max_len):
    dat = self_play(strategies, n_games, max_len)
    return PitResults(list(flatten(dat)), len(strategies))
