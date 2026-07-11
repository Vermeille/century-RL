import argparse
import json

from boardrl.games import games_library
from boardrl.games.strategies import strategy_from_string
from boardrl.rl.eval.selfplay import pit
from boardrl.rl.model import load_model
from boardrl.utils import BatchProcessor, ModelPool


def summarize(points):
    return {
        "avg": sum(points) / len(points),
        "min": min(points),
        "max": max(points),
        "points": points,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--game", default=None)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--opponent", choices=["self", "random"], default="self")
    parser.add_argument(
        "--strategy",
        choices=["policy_sampling", "argmax", "gumbel"],
        default="policy_sampling",
    )
    parser.add_argument("--num-evals", type=int, default=4)
    parser.add_argument("--q-scale", type=float, default=1.0)
    parser.add_argument("--rotate", action="store_true")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    game = args.game
    if game is None:
        import torch

        ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
        game = ckpt["config"].get("game", "thegame")

    bp = BatchProcessor(args.batch_size, model, timeout=0.001, model_name=args.checkpoint)
    pool = ModelPool(bp, args.batch_size, 0.001)

    if args.strategy == "policy_sampling":
        strategy_spec = f"policy_sampling,model=this,temperature={args.temperature}"
    elif args.strategy == "argmax":
        strategy_spec = "argmax,model=this"
    elif args.strategy == "gumbel":
        strategy_spec = (
            f"gumbel,model=this,num_evals={args.num_evals},q_scale={args.q_scale}"
        )
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    policy = strategy_from_string(strategy_spec, model=pool, discount_factor=1.0)
    opponent = policy if args.opponent == "self" else strategy_from_string("random")

    desc = games_library(game)
    results = pit(
        desc.make_game,
        [policy, opponent],
        args.games,
        args.max_len,
        rotate=args.rotate,
    )
    out = {
        "checkpoint": args.checkpoint,
        "game": game,
        "games": args.games,
        "max_len": args.max_len,
        "temperature": args.temperature,
        "strategy": args.strategy,
        "num_evals": args.num_evals,
        "q_scale": args.q_scale,
        "opponent": args.opponent,
        "seat0": summarize(results.my_points(0, by="seat")),
        "seat1": summarize(results.my_points(1, by="seat")),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
