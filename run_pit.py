"""Small CLI around the library evaluation API."""

import argparse
import json

from boardrl import Evaluator, Inference
from boardrl.games import games_library
from boardrl.games.strategies import RandomStrategy
from boardrl.rl.model import load_model


def checkpoint_player(path, batch_size, temperature):
    model = load_model(path)
    return Inference(model, batch_size=batch_size, name=path).policy(
        temperature=temperature
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="tictactoe")
    parser.add_argument("--player1")
    parser.add_argument("--player2")
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--no-rotate", action="store_true")
    args = parser.parse_args()

    make_player = lambda path: (
        RandomStrategy()
        if path is None
        else checkpoint_player(path, args.batch_size, args.temperature)
    )
    game = games_library(args.game)
    result = Evaluator(game.make_game, outcome=game.outcome).compare(
        [make_player(args.player1), make_player(args.player2)],
        names=[args.player1 or "random", args.player2 or "random"],
        games=args.games,
        max_steps=args.max_steps,
        rotate=not args.no_rotate,
    )
    print(
        json.dumps(
            {
                "players": result.names,
                "games": result.games,
                "win_rate": result.win_rate(),
                "avg_points": [result.avg_points(0), result.avg_points(1)],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
