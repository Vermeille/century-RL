import os
import torch
from natsort import natsorted
from pathlib import Path
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, PlainTextResponse

import pyximport

pyximport.install(setup_args={"script_args": ["--cython-cplus"]})
from boardrl.games import games_library


class Strategies:
    def __init__(self, cache_len=5):
        self.strategies = self.populate_strategies()
        self.cache = []
        self.cache_len = cache_len

    @staticmethod
    def populate_strategies():
        strategies = []
        # find all .pth files in all directories
        for root, dirs, files in os.walk("."):
            for d in dirs:
                if d.startswith("."):
                    dirs.remove(d)
            for file in files:
                if file.endswith(".pth"):
                    # strategies.append(f"argmax:{os.path.join(root, file)}")
                    strategies.append(
                        f"policy_sampling,model={os.path.join(root, file)}"
                    )

        strategies = natsorted(strategies)
        strategies += [
            "random",
            "random_buy",
            "all_actions_then_random_buy",
            "no_actions_random_buy",
            "pick_best_mc_value,budget=5",
            "pick_best_mc_value,budget=10",
            "pick_best_mc_value,budget=100",
        ]
        return strategies

    def get_strategy(self, name):
        for cache_name, strategy in self.cache:
            if cache_name == name:
                return strategy
        self.cache = self.cache[-self.cache_len :]
        self.cache.append((name, century.strategy_from_string(name)))
        return self.cache[-1][1]


strategies = Strategies()

app = FastAPI()

game_name = "century"
century = games_library(game_name)
game = century.make_game()
game_dir = Path(__file__).parent.parent / "games" / game_name


@app.get("/", response_class=HTMLResponse)
def read_root():
    return open(game_dir / "ui.html").read()


@app.get("/strategies")
def get_strategies():
    return strategies.strategies


@app.get("/board", response_class=PlainTextResponse)
def board():
    if game.ended():
        return game.display(force=0)
    else:
        return game.display_with_moves()


@app.get("/analyze")
def analyze(strategy: str):
    return strategies.get_strategy(strategy)(game)[1]


@app.post("/do")
def do(action: str = Body(..., embed=True), strategy: str = Body(..., embed=True)):
    if game.ended():
        return {
            "continue": False,
            "points": game.diff_points_for(0),
            "num_turns": game.round(),
        }

    game.play_str(action)
    if game.ended():
        return {
            "continue": False,
            "points": game.diff_points_for(0),
            "num_turns": game.round(),
        }

    dist, _ = strategies.get_strategy(strategy)(game)
    dist = torch.softmax(dist, dim=0)
    game.play_distribution(dist)

    if game.ended():
        return {
            "continue": False,
            "points": game.diff_points_for(0),
            "num_turns": game.round(),
        }

    return {"continue": True}


@app.get("/reset")
def reset():
    global game
    game = century.make_game()
    return True
