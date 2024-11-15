from pathlib import Path
import pyximport

pyximport.install(setup_args={"script_args": ["--cython-cplus"]})
from centuryrl.century.engine import Game
from centuryrl.century.strategies import RandomBuyStrategy

from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, PlainTextResponse

app = FastAPI()

game = Game()
strategy = RandomBuyStrategy()
current_dir = Path(__file__).parent


@app.get("/", response_class=HTMLResponse)
def read_root():
    return open(current_dir / "century.html").read()


@app.get("/board", response_class=PlainTextResponse)
def board():
    return game.display_with_moves()


@app.post("/do")
def do(action: str = Body(..., embed=True)):
    game.play_str(action)
    move, _ = strategy(game)
    game.play_str(move)
    return True
