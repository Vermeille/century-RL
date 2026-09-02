import argparse
import copy
import json
import math
import random
from collections import defaultdict

import numpy as np
import torch

from boardrl.games import games_library
from boardrl.games.strategies import one_hot
from boardrl.models import make
from boardrl.rollouts import Inference, RolloutRunner


def move_cost(game, move):
    if move == "x":
        return None
    card, pile = map(int, move.split("->"))
    top = game.piles[pile]
    return card - top if pile < 2 else top - card


def card_moves(game):
    return [(m, move_cost(game, m)) for m in game.moves if m != "x"]


def after_features(game, move, selected_pile):
    branch = copy.deepcopy(game)
    branch.play_str(move)
    available = card_moves(branch)
    return {
        "mobility": len(available),
        "any_ten": any(cost == -10 for _, cost in available),
        "same_pile_ten": any(
            cost == -10 and int(candidate.split("->")[1]) == selected_pile
            for candidate, cost in available
        ),
    }


class ArgmaxPolicy:
    def __init__(self, model):
        self.inference = Inference(model, batch_size=1024)

    async def __call__(self, game):
        output = await self.inference.processor(game.display_with_moves())
        action = int(output.policy[0].argmax().item())
        return one_hot(action, len(game.moves)).log(), {
            "state": game.display_with_moves()
        }


class RecordingArgmaxPolicy(ArgmaxPolicy):
    def __init__(self, model, target, seed):
        super().__init__(model)
        self.target = target
        self.rng = random.Random(seed)
        self.card_decisions = 0
        self.nonlowest_seen = 0
        self.records = []
        self.episode_ids = {}
        self.next_episode_id = 0

    async def __call__(self, game):
        output = await self.inference.processor(game.display_with_moves())
        action = int(output.policy[0].argmax().item())
        move = game.moves[action]
        available = card_moves(game)
        if move != "x" and available:
            self.card_decisions += 1
            costs = dict(available)
            minimum = min(costs.values())
            chosen_cost = costs[move]
            if chosen_cost > minimum:
                self.nonlowest_seen += 1
                object_id = id(game)
                if object_id not in self.episode_ids:
                    self.episode_ids[object_id] = self.next_episode_id
                    self.next_episode_id += 1
                minimum_moves = [candidate for candidate, cost in available if cost == minimum]
                chosen_card, chosen_pile = map(int, move.split("->"))
                chosen_after = after_features(game, move, chosen_pile)
                first_min_card, first_min_pile = map(int, minimum_moves[0].split("->"))
                first_after = after_features(game, minimum_moves[0], first_min_pile)
                record = {
                    "game": copy.deepcopy(game),
                    "episode_id": self.episode_ids[object_id],
                    "chosen": move,
                    "minimum_moves": minimum_moves,
                    "chosen_cost": chosen_cost,
                    "minimum_cost": minimum,
                    "regret": chosen_cost - minimum,
                    "choice_type": (
                        "same_card_different_pile"
                        if any(int(candidate.split("->")[0]) == chosen_card for candidate in minimum_moves)
                        else "different_card"
                    ),
                    "optional": not game.needs_more_cards_this_turn(),
                    "deck_count": len(game.deck),
                    "hand_count": len(game.hands[game.curplay]),
                    "chosen_card": chosen_card,
                    "chosen_pile": chosen_pile,
                    "first_min_card": first_min_card,
                    "first_min_pile": first_min_pile,
                    "chosen_after": chosen_after,
                    "first_after": first_after,
                    "logit_margin": float(
                        output.policy[0][action].item()
                        - torch.topk(output.policy[0], k=min(2, len(output.policy[0]))).values[-1].item()
                    ),
                }
                if len(self.records) < self.target:
                    self.records.append(record)
                else:
                    replacement = self.rng.randrange(self.nonlowest_seen)
                    if replacement < self.target:
                        self.records[replacement] = record
        return one_hot(action, len(game.moves)).log(), {
            "state": game.display_with_moves()
        }


def mean_ci(values):
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return {"n": 0}
    mean = float(a.mean())
    se = float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0
    return {"n": len(a), "mean": mean, "ci95": [mean - 1.96 * se, mean + 1.96 * se]}


def delta_summary(rows):
    delta = np.asarray([row["delta_first"] for row in rows], dtype=float)
    if not len(delta):
        return {"n": 0}
    return {
        "n": len(delta),
        "delta_first": mean_ci(delta),
        "delta_mean_min": mean_ci([row["delta_mean_min"] for row in rows]),
        "delta_best_min": mean_ci([row["delta_best_min"] for row in rows]),
        "better_first_pct": float(100 * np.mean(delta > 0)),
        "tie_first_pct": float(100 * np.mean(delta == 0)),
        "worse_first_pct": float(100 * np.mean(delta < 0)),
    }


def group_summary(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {key: delta_summary(value) for key, value in sorted(grouped.items())}


def stage(deck_count):
    if deck_count >= 48:
        return "early"
    if deck_count >= 20:
        return "middle"
    return "late"


def regret_bin(regret):
    if regret == 1:
        return "1"
    if regret == 2:
        return "2"
    if regret <= 5:
        return "3-5"
    if regret <= 10:
        return "6-10"
    return ">10"


def evaluate_states(policy, states, chunk_size):
    all_scores = []
    for offset in range(0, len(states), chunk_size):
        chunk = states[offset : offset + chunk_size]
        cursor = 0

        def make_game(num_players):
            nonlocal cursor
            result = copy.deepcopy(chunk[cursor])
            cursor += 1
            return result

        runner = RolloutRunner(make_game, progress=False, coop=True)
        games = runner.play([policy, policy], games=len(chunk), max_steps=800, rotate=False)
        all_scores.extend(float(game.by_seat[0][-1].my_points) for game in games)
    return all_scores


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    model = make(checkpoint["metadata"]["architecture"])
    model.load_state_dict(checkpoint["models"]["current"])
    model.to(args.device).eval()

    recorder = RecordingArgmaxPolicy(model, args.samples, args.seed + 91)
    game_desc = games_library(args.game)
    runner = RolloutRunner(game_desc.make_game, progress=True, coop=True)
    baseline_games = runner.play(
        [recorder, recorder], games=args.games, max_steps=800, rotate=False
    )
    records = recorder.records
    if not records:
        raise RuntimeError("No non-lowest-cost decisions found")

    branch_states = []
    branch_keys = []
    for index, record in enumerate(records):
        for kind, move in [("chosen", record["chosen"])] + [
            (f"minimum_{alt}", move) for alt, move in enumerate(record["minimum_moves"])
        ]:
            branch = copy.deepcopy(record["game"])
            branch.play_str(move)
            branch_states.append(branch)
            branch_keys.append((index, kind))

    continuation = ArgmaxPolicy(model)
    scores = evaluate_states(continuation, branch_states, args.chunk_size)
    score_map = defaultdict(dict)
    for key, score in zip(branch_keys, scores):
        score_map[key[0]][key[1]] = score

    rows = []
    for index, record in enumerate(records):
        chosen_score = score_map[index]["chosen"]
        minimum_scores = [score_map[index][f"minimum_{i}"] for i in range(len(record["minimum_moves"]))]
        row = {key: value for key, value in record.items() if key != "game"}
        row.update(
            {
                "stage": stage(record["deck_count"]),
                "regret_bin": regret_bin(record["regret"]),
                "sets_up_ten": record["chosen_after"]["same_pile_ten"],
                "mobility_delta": record["chosen_after"]["mobility"] - record["first_after"]["mobility"],
                "chosen_score": chosen_score,
                "minimum_scores": minimum_scores,
                "delta_first": chosen_score - minimum_scores[0],
                "delta_mean_min": chosen_score - float(np.mean(minimum_scores)),
                "delta_best_min": chosen_score - max(minimum_scores),
            }
        )
        rows.append(row)

    simple = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "baseline_games": args.games,
        "baseline_score": {
            "mean": float(np.mean([game.by_seat[0][-1].my_points for game in baseline_games])),
            "std": float(np.std([game.by_seat[0][-1].my_points for game in baseline_games], ddof=1)),
        },
        "card_decisions": recorder.card_decisions,
        "nonlowest_seen": recorder.nonlowest_seen,
        "nonlowest_pct": 100 * recorder.nonlowest_seen / recorder.card_decisions,
        "sampled_decisions": len(rows),
        "chosen_cost": mean_ci([row["chosen_cost"] for row in rows]),
        "minimum_cost": mean_ci([row["minimum_cost"] for row in rows]),
        "regret": mean_ci([row["regret"] for row in rows]),
        "regret_quantiles": {
            str(q): float(np.quantile([row["regret"] for row in rows], q))
            for q in [0.25, 0.5, 0.75, 0.9, 0.95]
        },
        "choice_type_pct": {
            key: 100 * sum(row["choice_type"] == key for row in rows) / len(rows)
            for key in ["same_card_different_pile", "different_card"]
        },
        "optional_pct": 100 * sum(row["optional"] for row in rows) / len(rows),
        "sets_up_ten_pct": 100 * sum(row["sets_up_ten"] for row in rows) / len(rows),
        "chosen_any_ten_pct": 100 * sum(row["chosen_after"]["any_ten"] for row in rows) / len(rows),
        "first_min_any_ten_pct": 100 * sum(row["first_after"]["any_ten"] for row in rows) / len(rows),
        "mobility_delta": mean_ci([row["mobility_delta"] for row in rows]),
        "overall": delta_summary(rows),
        "by_choice_type": group_summary(rows, "choice_type"),
        "by_optional": group_summary(rows, "optional"),
        "by_stage": group_summary(rows, "stage"),
        "by_regret_bin": group_summary(rows, "regret_bin"),
        "by_sets_up_ten": group_summary(rows, "sets_up_ten"),
    }
    with open(args.output, "w") as handle:
        json.dump({"summary": simple, "rows": rows}, handle, indent=2)
    print(json.dumps(simple, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--game", default="thegame,mode=omni")
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=73191)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())
