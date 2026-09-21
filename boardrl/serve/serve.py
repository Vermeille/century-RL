import argparse
import copy
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import torch
from natsort import natsorted
from pathlib import Path
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

import pyximport  # type: ignore[import-untyped]

pyximport.install(setup_args={"script_args": ["--cython-cplus"]})
from boardrl.games import games_library
from boardrl.cyutils import fast_sample  # type: ignore[import-not-found]
from boardrl.rl.model import load_model
from boardrl.utils import BatchProcessor


parser = argparse.ArgumentParser()
parser.add_argument("--game", default="century")
args, _ = parser.parse_known_args()

initial_game_spec = args.game
initial_game_spec = initial_game_spec.strip()
game_name = initial_game_spec.split(",", 1)[0].strip()
serve_dir = Path(__file__).parent
static_dir = serve_dir / "static"


@dataclass
class GameSnapshot:
    name: str
    spec: str
    board: str
    board_with_moves: str
    moves: list[str]
    current_player: int
    round: int
    ended: bool
    points: Any
    history: list[dict[str, Any]]
    can_undo: bool
    can_redo: bool

    @classmethod
    def from_game(
        cls, name, spec, game, history=None, can_undo=False, can_redo=False
    ):
        ended = bool(game.ended())
        return cls(
            name=name,
            spec=spec,
            board=game.display(force=0) if ended else game.display(),
            board_with_moves=(
                game.display(force=0) if ended else game.display_with_moves()
            ),
            moves=[] if ended else list(game.moves),
            current_player=game.current_player(),
            round=game.round(),
            ended=ended,
            points=game.diff_points_for(0) if ended else None,
            history=list(history or []),
            can_undo=can_undo,
            can_redo=can_redo,
        )


@dataclass
class GameSession:
    name: str
    spec: str
    game_desc: Any
    game: Any
    strategies: Any
    history: list[dict[str, Any]]
    undo_stack: list[tuple[Any, list[dict[str, Any]]]]
    redo_stack: list[tuple[Any, list[dict[str, Any]]]]

    @classmethod
    def create(cls, name: str, spec: str):
        game_desc = games_library(spec)
        return cls(
            name=name,
            spec=spec,
            game_desc=game_desc,
            game=game_desc.make_game(),
            strategies=Strategies(game_desc, game_name=name),
            history=[],
            undo_stack=[],
            redo_stack=[],
        )

    def reset(self):
        self.game = self.game_desc.make_game()
        self.history = []
        self.undo_stack = []
        self.redo_stack = []

    def snapshot(self):
        return GameSnapshot.from_game(
            self.name,
            self.spec,
            self.game,
            self.history,
            can_undo=bool(self.undo_stack),
            can_redo=bool(self.redo_stack),
        ).__dict__

    def save_undo(self):
        self.undo_stack.append((clone_game(self.game), list(self.history)))
        self.redo_stack = []

    def undo(self):
        if not self.undo_stack:
            raise HTTPException(status_code=400, detail="No moves to undo.")
        self.redo_stack.append((clone_game(self.game), list(self.history)))
        self.game, self.history = self.undo_stack.pop()

    def redo(self):
        if not self.redo_stack:
            raise HTTPException(status_code=400, detail="No moves to redo.")
        self.undo_stack.append((clone_game(self.game), list(self.history)))
        self.game, self.history = self.redo_stack.pop()

    def record_move(self, actor: str, action: str, strategy: str | None = None):
        self.history.append(
            {
                "turn": len(self.history) + 1,
                "actor": actor,
                "action": action,
                "strategy": strategy,
                "player": self.game.current_player(),
                "round": self.game.round(),
                "ended": bool(self.game.ended()),
            }
        )


class Strategies:
    def __init__(
        self,
        game_desc=None,
        game_name: str | None = None,
        cache_len: int = 5,
        batch_size: int = 32,
        timeout: float = 0.01,
    ):
        self.game_desc = game_desc or get_session().game_desc
        self.game_name = game_name or current_game_name
        self.cache: list[tuple[str, Any]] = []
        self.cache_len = cache_len
        self.batch_size = batch_size
        self.timeout = timeout
        self.model_paths: dict[str, str] = {}
        self.strategies = self.populate_strategies()

    def populate_strategies(self):
        strategies = []
        # Checkpoint runs store their weights several directories below the
        # repository's ``checkpoints`` directory, so search recursively.
        for model_path in Path(".").rglob("*.pth"):
            if not self.model_matches_game(model_path):
                continue
            # Strategy descriptions use commas as argument separators, while
            # checkpoint run directories may themselves contain commas.
            name = f"policy_sampling,model={quote(str(model_path), safe='/')}"
            strategies.append(name)
            self.model_paths[name] = str(model_path)

        strategies = natsorted(strategies)
        strategies += [
            name
            for name, (_, arg_info) in sorted(
                self.game_desc.strategy_from_string.registry.items()
            )
            if all(default is not None for _, default in arg_info.values())
        ]
        return strategies

    def model_matches_game(self, model_path: str):
        path = Path(model_path)
        parts = path.parts
        checkpoint_dirs = [part for part in parts if part.endswith("-ckpt")]
        if not checkpoint_dirs:
            return any(
                part == self.game_name
                or part.startswith(f"{self.game_name}-")
                or part.startswith(f"{self.game_name},")
                for part in parts
            )
        return f"{self.game_name}-ckpt" in checkpoint_dirs

    def get_strategy(self, name):
        if name not in self.strategies:
            raise HTTPException(status_code=400, detail="Unknown strategy")
        for cache_name, strategy in self.cache:
            if cache_name == name:
                return strategy

        kwargs = {}
        model_path = self.model_paths.get(name)
        if model_path is not None:
            model = load_model(model_path)
            model.eval()
            kwargs["model"] = BatchProcessor(
                self.batch_size,
                model,
                timeout=self.timeout,
                model_name=model_path,
            )

        strategy = self.game_desc.strategy_from_string(name, **kwargs)
        self.cache.append((name, strategy))
        if len(self.cache) > self.cache_len:
            self.cache.pop(0)
        return strategy


sessions: dict[str, GameSession] = {}
current_game_name = game_name
current_game_spec = initial_game_spec
game_desc: Any
game: Any
strategies: Strategies


def available_games():
    return sorted(games_library.registry.keys())


def base_game_name(spec: str) -> str:
    return spec.split(",", 1)[0].strip()


def get_session(spec: str | None = None):
    spec = (spec or current_game_spec).strip()
    name = base_game_name(spec)
    if name not in games_library.registry:
        raise HTTPException(status_code=404, detail=f"Unknown game: {name}")
    if spec not in sessions:
        try:
            sessions[spec] = GameSession.create(name, spec)
        except (AssertionError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid options for game '{name}': {exc}",
            ) from exc
    return sessions[spec]


def set_current_session(spec: str):
    global current_game_name, current_game_spec, game_name, game_desc, game, strategies
    session = get_session(spec)
    current_game_name = session.name
    current_game_spec = session.spec
    game_name = session.name
    game_desc = session.game_desc
    game = session.game
    strategies = session.strategies
    return session


set_current_session(initial_game_spec)

app = FastAPI()
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
def read_root():
    return (static_dir / "index.html").read_text()


def end_response():
    session = get_session()
    return {
        "continue": False,
        "points": session.game.diff_points_for(0),
        "num_turns": session.game.round(),
        "state": session.snapshot(),
    }


@app.get("/strategies")
def get_strategies(game: str | None = None):
    return get_session(game).strategies.strategies


@app.get("/games")
def get_games():
    return {
        "games": available_games(),
        "current": current_game_name,
        "current_spec": current_game_spec,
    }


@app.post("/set-game")
def set_game(game: str = Body(..., embed=True)):
    set_current_session(game)
    return get_session(game).snapshot()


@app.get("/state")
def state(game: str | None = None):
    return get_session(game).snapshot()


@app.get("/board", response_class=PlainTextResponse)
def board(game: str | None = None):
    session = get_session(game)
    if session.game.ended():
        return session.game.display(force=0)
    else:
        return session.game.display_with_moves()


@app.get("/analyze")
async def analyze(strategy: str, game: str | None = None):
    session = get_session(game)
    distribution, info = await session.strategies.get_strategy(strategy)(session.game)
    return analysis_payload(session, distribution, info)


@app.post("/do")
async def do(
    action: str = Body(..., embed=True),
    strategy: str = Body(..., embed=True),
    game: str | None = None,
):
    session = get_session(game)
    if session.game.ended():
        return end_response_for(session)

    player = session.game.current_player()
    play_action(session, action, actor="human")
    if session.game.ended():
        return end_response_for(session)

    strategy_fn = session.strategies.get_strategy(strategy)
    while not session.game.ended() and session.game.current_player() != player:
        dist, _ = await strategy_fn(session.game)
        action_idx = fast_sample(torch.softmax(dist, dim=0))
        play_action_idx(session, action_idx, actor="agent", strategy=strategy)

    if session.game.ended():
        return end_response_for(session)

    return {"continue": True, "state": session.snapshot()}


@app.post("/do-one")
async def do_one(action: str = Body(..., embed=True), game: str | None = None):
    session = get_session(game)
    if session.game.ended():
        return end_response_for(session)

    play_action(session, action, actor="human")
    if session.game.ended():
        return end_response_for(session)
    return {"continue": True, "state": session.snapshot()}


@app.post("/play-one")
async def play_one(strategy: str = Body(..., embed=True), game: str | None = None):
    session = get_session(game)
    if session.game.ended():
        return end_response_for(session)

    dist, _ = await session.strategies.get_strategy(strategy)(session.game)
    action_idx = fast_sample(torch.softmax(dist, dim=0))
    play_action_idx(session, action_idx, actor="agent", strategy=strategy)

    if session.game.ended():
        return end_response_for(session)

    return {"continue": True, "state": session.snapshot()}


@app.get("/reset")
async def reset(game: str | None = None):
    session = get_session(game)
    session.reset()
    if session.spec == current_game_spec:
        set_current_session(session.spec)
    return True


@app.post("/undo")
async def undo(game: str | None = None):
    session = get_session(game)
    session.undo()
    if session.spec == current_game_spec:
        set_current_session(session.spec)
    return session.snapshot()


@app.post("/redo")
async def redo(game: str | None = None):
    session = get_session(game)
    session.redo()
    if session.spec == current_game_spec:
        set_current_session(session.spec)
    return session.snapshot()


@app.post("/analyze")
async def analyze_post(
    strategy: str = Body(..., embed=True),
    state: str | None = Body(None, embed=True),
    game: str | None = None,
):
    if state is None:
        return await analyze(strategy, game=game)

    session = get_session(game)
    strategy_fn = session.strategies.get_strategy(strategy)
    moves = moves_from_state(state)
    if not moves:
        raise HTTPException(
            status_code=400,
            detail="Edited state does not contain any @ legal moves.",
        )

    if not hasattr(strategy_fn, "nn"):
        return text_strategy_analysis(strategy, moves)

    nn_output = await strategy_fn.nn(state)
    distribution = nn_output.policy[0].cpu()
    info = {
        "moves": {move: value.item() for move, value in zip(moves, distribution)},
        "reference_value": nn_output.value.mean.cpu()[0].item(),
    }
    return analysis_payload(session, distribution, info, moves=moves)


@app.get("/agnostic", response_class=HTMLResponse)
def agnostic_ui():
    return read_root()


def end_response_for(session: GameSession):
    return {
        "continue": False,
        "points": session.game.diff_points_for(0),
        "num_turns": session.game.round(),
        "state": session.snapshot(),
    }


def play_action(
    session: GameSession,
    action: str,
    actor: str,
    strategy: str | None = None,
):
    try:
        session.save_undo()
        session.game.play_str(action)
        session.record_move(actor=actor, action=action, strategy=strategy)
    except (AssertionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def play_action_idx(
    session: GameSession,
    action_idx: int,
    actor: str,
    strategy: str | None = None,
):
    try:
        action = session.game.moves[action_idx]
        session.save_undo()
        session.game.play_idx(action_idx)
        session.record_move(actor=actor, action=action, strategy=strategy)
    except (AssertionError, ValueError, IndexError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def analysis_payload(session: GameSession, distribution, info, moves=None):
    move_names = moves if moves is not None else list(session.game.moves)
    probabilities = torch.softmax(distribution, dim=0).tolist()
    payload = dict(info)
    payload["probabilities"] = dict(zip(move_names, probabilities))
    payload["move_order"] = move_names
    return payload


def moves_from_state(state: str):
    return [
        line[1:].strip()
        for line in state.splitlines()
        if line.startswith("@") and line[1:].strip()
    ]


def text_strategy_analysis(strategy: str, moves: list[str]):
    if strategy == "random":
        probabilities = {move: 1 / len(moves) for move in moves}
        return {
            "moves": probabilities,
            "probabilities": probabilities,
            "move_order": moves,
        }

    if strategy == "longest_move":
        best_move = max(moves, key=len)
        probabilities = {move: 1.0 if move == best_move else 0.0 for move in moves}
        return {
            "moves": probabilities,
            "probabilities": probabilities,
            "move_order": moves,
        }

    raise HTTPException(
        status_code=400,
        detail="This strategy cannot analyze arbitrary text state.",
    )


def clone_game(game):
    if hasattr(game, "copy"):
        return game.copy()
    return copy.deepcopy(game)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("boardrl.serve.serve:app")