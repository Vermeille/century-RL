from pathlib import Path
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, PlainTextResponse

import pyximport

pyximport.install(setup_args={"script_args": ["--cython-cplus"]})
from centuryrl.century.engine import Game
from centuryrl.century.strategies import RandomBuyStrategy

app = FastAPI()

game = Game()
num_turns = 0
strategy = RandomBuyStrategy()
current_dir = Path(__file__).parent


@app.get("/", response_class=HTMLResponse)
def read_root():
    return open(current_dir / "century.html").read()


@app.get("/board", response_class=PlainTextResponse)
def board():
    return game.display_with_moves()


@app.get("/analyze")
def analyze():
    return strategy(game)[1]


@app.post("/do")
def do(action: str = Body(..., embed=True)):
    global num_turns
    if game.ended():
        return {"continue": False, "points": game.points_for(0), "num_turns": num_turns}

    num_turns += 1
    game.play_str(action)
    if game.ended():
        return {"continue": False, "points": game.points_for(0), "num_turns": num_turns}

    move, _ = strategy(game)

    game.play_str(move)
    if game.ended():
        return {"continue": False, "points": game.points_for(0), "num_turns": num_turns}

    return {"continue": True}


@app.get("/reset")
def reset():
    global game
    game = Game()
    return True
