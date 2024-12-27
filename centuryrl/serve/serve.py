import os
import torch
from natsort import natsorted
from pathlib import Path
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, PlainTextResponse

import pyximport

pyximport.install(setup_args={"script_args": ["--cython-cplus"]})
from centuryrl.century.engine import Game
from centuryrl.century.strategies import RandomBuyStrategy, strategy_from_string


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
        self.cache.append((name, strategy_from_string(name)))
        return self.cache[-1][1]


strategies = Strategies()

app = FastAPI()

game = Game()
strategy = RandomBuyStrategy()
current_dir = Path(__file__).parent


@app.get("/", response_class=HTMLResponse)
def read_root():
    return open(current_dir / "century.html").read()


@app.get("/strategies")
def get_strategies():
    return strategies.strategies


@app.get("/board", response_class=PlainTextResponse)
def board():
    return game.display_with_moves(force=0)


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
    game = Game()
    return True
