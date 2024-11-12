import pyximport

pyximport.install(setup_args={"script_args": ["--cython-cplus"]})
from engine import Game, RandomBuyStrategy

# from main import Model
import torch
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, PlainTextResponse

# m = Model()
# m.load_state_dict(torch.load("model.pth"))
# m.to("cuda")

app = FastAPI()

game = Game()
strategy = RandomBuyStrategy()


@app.get("/", response_class=HTMLResponse)
def read_root():
    return open("century.html").read()


@app.get("/board", response_class=PlainTextResponse)
def board():
    return game.display_with_moves()


@app.post("/do")
def do(action: str = Body(..., embed=True)):
    game.play_str(action)
    move, _ = strategy(game)
    game.play_str(move)
    return True
