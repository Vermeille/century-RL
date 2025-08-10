import argparse
import os

import torch
from natsort import natsorted
from pathlib import Path
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse

import pyximport

pyximport.install(setup_args={"script_args": ["--cython-cplus"]})
from boardrl.games import games_library
from boardrl.cyutils import fast_sample
from boardrl.games.strategies import ModelPool


parser = argparse.ArgumentParser()
parser.add_argument("--game", default="century")
args, _ = parser.parse_known_args()

game_name = args.game
game_desc = games_library(game_name)
game = game_desc.make_game()
game_dir = Path(__file__).parent.parent / "games" / game_name


class Strategies:
    def __init__(
        self,
        game_desc=game_desc,
        cache_len: int = 5,
        batch_size: int = 32,
        timeout: float = 0.01,
    ):
        self.game_desc = game_desc
        self.strategies = self.populate_strategies()
        self.cache = []
        self.cache_len = cache_len
        # ModelPool handles loading models and batching inference similar to
        # the training setup in ``main.py``.
        self.pool = ModelPool(None, batch_size, timeout)

    @staticmethod
    def populate_strategies():
        strategies = []
        # find all .pth files in all directories
        for root, dirs, files in os.walk("."):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
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
            "mcts,iterations=50,max_unroll=30,discount_factor=0.9",
            "mcts,iterations=100,max_unroll=30,discount_factor=0.9",
            "mcts,iterations=1000,max_unroll=30,discount_factor=0.9",
        ]
        return strategies

    def get_strategy(self, name):
        if name not in self.strategies:
            raise HTTPException(status_code=400, detail="Unknown strategy")
        for cache_name, strategy in self.cache:
            if cache_name == name:
                return strategy
        self.cache = self.cache[-self.cache_len :]
        # Use the shared ModelPool when instantiating strategies so model
        # arguments are resolved correctly.
        self.cache.append(
            (name, self.game_desc.strategy_from_string(name, model=self.pool))
        )
        return self.cache[-1][1]


strategies = Strategies(game_desc)

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def read_root():
    return (game_dir / "ui.html").read_text()


def end_response():
    return {
        "continue": False,
        "points": game.diff_points_for(0),
        "num_turns": game.round(),
    }


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
async def analyze(strategy: str):
    pred = await strategies.get_strategy(strategy)(game)
    return pred[1]


@app.post("/do")
async def do(
    action: str = Body(..., embed=True), strategy: str = Body(..., embed=True)
):
    if game.ended():
        return end_response()

    player = game.current_player()
    game.play_str(action)
    if game.ended():
        return end_response()

    strategy_fn = strategies.get_strategy(strategy)
    while not game.ended() and game.current_player() != player:
        dist, _ = await strategy_fn(game)
        action_idx = fast_sample(torch.softmax(dist, dim=0))
        game.play_idx(action_idx)

    if game.ended():
        return end_response()

    return {"continue": True}


@app.post("/play-one")
async def play_one(strategy: str = Body(..., embed=True)):
    if game.ended():
        return end_response()

    dist, _ = await strategies.get_strategy(strategy)(game)
    action_idx = fast_sample(torch.softmax(dist, dim=0))
    game.play_idx(action_idx)

    if game.ended():
        return end_response()

    return {"continue": True}


@app.get("/reset")
async def reset():
    global game
    game = game_desc.make_game()
    return True


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("boardrl.serve.serve:app")
