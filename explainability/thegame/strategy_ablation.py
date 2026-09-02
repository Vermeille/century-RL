import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from boardrl.cyutils import init_seed
from boardrl.games import games_library
from boardrl.games.strategies import one_hot
from boardrl.games.thegame.strategies import LowestCostStrategy
from boardrl.models import make
from boardrl.rollouts import Inference, RolloutRunner


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    init_seed(seed)


def cost(game, move):
    card, pile = map(int, move.split("->"))
    return card - game.piles[pile] if pile < 2 else game.piles[pile] - card


class ModelDecision:
    def __init__(self, processor, *, restrict_cards, always_stop):
        self.processor = processor
        self.restrict_cards = restrict_cards
        self.always_stop = always_stop

    async def __call__(self, game):
        output = await self.processor(game.display_with_moves())
        logits = output.policy[0].cpu()
        if self.always_stop and "x" in game.moves:
            action = game.moves.index("x")
        else:
            action = int(logits.argmax())
            if game.moves[action] != "x" and self.restrict_cards:
                card_indices = [i for i, move in enumerate(game.moves) if move != "x"]
                minimum = min(cost(game, game.moves[i]) for i in card_indices)
                eligible = [i for i in card_indices if cost(game, game.moves[i]) == minimum]
                action = max(eligible, key=lambda i: float(logits[i]))
        return one_hot(action, len(game.moves)).log(), {"state": game.display_with_moves()}


def behavior(results):
    card_choices = nonlowest = tie_choices = nonfirst_ties = optional_cards = x_moves = 0
    total_cost = 0
    for game in results:
        for player in game:
            for record in player[:-1]:
                chosen = record.moves[record.action_idx]
                if chosen == "x":
                    x_moves += 1
                    continue
                if "->" not in chosen:
                    continue
                piles_line = next(line for line in record.state.splitlines() if line.startswith("Piles:"))
                piles = list(map(int, piles_line.removeprefix("Piles:").split()))
                card_indices = [i for i, move in enumerate(record.moves) if "->" in move]
                costs = []
                for i in card_indices:
                    card, pile = map(int, record.moves[i].split("->"))
                    costs.append(card - piles[pile] if pile < 2 else piles[pile] - card)
                chosen_position = card_indices.index(record.action_idx)
                chosen_cost = costs[chosen_position]
                minimum = min(costs)
                minimum_indices = [card_indices[j] for j, value in enumerate(costs) if value == minimum]
                card_choices += 1
                total_cost += chosen_cost
                nonlowest += chosen_cost > minimum
                if len(minimum_indices) > 1 and chosen_cost == minimum:
                    tie_choices += 1
                    nonfirst_ties += record.action_idx != minimum_indices[0]
                optional_cards += "x" in record.moves
    return {
        "card_choices": card_choices,
        "nonlowest_pct": 100 * nonlowest / card_choices,
        "avg_card_cost": total_cost / card_choices,
        "tied_minimum_decisions": tie_choices,
        "nonfirst_tie_pct": 100 * nonfirst_ties / tie_choices if tie_choices else 0,
        "optional_cards": optional_cards,
        "optional_cards_per_game": optional_cards / len(results),
        "x_moves_per_game": x_moves / len(results),
    }


def paired_summary(scores, baseline):
    x = np.asarray(scores, dtype=float)
    b = np.asarray(baseline, dtype=float)
    delta = x - b
    se = delta.std(ddof=1) / math.sqrt(len(delta))
    return {
        "n": len(x),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)),
        "win_rate": float(np.mean(x == 100)),
        "delta_vs_lowest": float(delta.mean()),
        "delta_ci95": [float(delta.mean() - 1.96 * se), float(delta.mean() + 1.96 * se)],
        "better_tie_worse_pct": [
            float(100 * np.mean(delta > 0)),
            float(100 * np.mean(delta == 0)),
            float(100 * np.mean(delta < 0)),
        ],
    }


def run_policy(runner, strategy, games, batch_games, seed):
    scores = []
    collected = []
    for batch, start in enumerate(range(0, games, batch_games)):
        count = min(batch_games, games - start)
        seed_all(seed + batch)
        result = runner.play([strategy, strategy], games=count, max_steps=800, rotate=False)
        collected.extend(result)
        scores.extend(float(game.by_seat[0][-1].my_points) for game in result)
    return scores, behavior(collected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["omni", "strict"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=4096)
    parser.add_argument("--batch-games", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    model = make(checkpoint["metadata"]["architecture"])
    model.load_state_dict(checkpoint["models"]["current"])
    model.cuda().eval()
    inference = Inference(model, batch_size=1024)
    game_desc = games_library(f"thegame,mode={args.mode}")
    runner = RolloutRunner(game_desc.make_game, progress=False, coop=True)
    strategies = {
        "lowest_first_tie": LowestCostStrategy(),
        "model_tie_always_stop": ModelDecision(inference.processor, restrict_cards=True, always_stop=True),
        "model_cards_always_stop": ModelDecision(inference.processor, restrict_cards=False, always_stop=True),
    }
    if args.mode == "omni":
        strategies.update({
            "minimum_cards_model_stop": ModelDecision(inference.processor, restrict_cards=True, always_stop=False),
            "full_model": ModelDecision(inference.processor, restrict_cards=False, always_stop=False),
        })
    raw = {}
    for name, strategy in strategies.items():
        scores, stats = run_policy(runner, strategy, args.games, args.batch_games, 912300)
        raw[name] = {"scores": scores, "behavior": stats}
        print(name, np.mean(scores), stats, flush=True)
    baseline = raw["lowest_first_tie"]["scores"]
    result = {
        "mode": args.mode,
        "checkpoint": str(args.checkpoint),
        "games": args.games,
        "strategies": {
            name: {"scores": paired_summary(data["scores"], baseline), "behavior": data["behavior"]}
            for name, data in raw.items()
        },
        "per_game_scores": {name: data["scores"] for name, data in raw.items()},
    }
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
