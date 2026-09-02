import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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


def move_cost(piles, move):
    card, pile = map(int, move.split("->"))
    return card - piles[pile] if pile < 2 else piles[pile] - card


def legal_costs(card, piles):
    values = []
    for pile, top in enumerate(piles):
        value = card - top if pile < 2 else top - card
        if value >= 0 or value == -10:
            values.append((pile, value))
    return values


FEATURE_NAMES = [
    "cost", "cost_sq", "is_reverse_10", "cost_0_2", "cost_3_5", "cost_6_10", "cost_gt_10",
    "regret", "regret_1_2", "regret_3_5", "regret_6_10", "regret_gt_10",
    "card_best_cost", "card_legal_piles", "card_single_pile", "card_value",
    "chosen_pile_advance", "chosen_pile_open_after", "direction_gap_after", "direction_gap_change",
    "remaining_playable_cards", "remaining_dead_cards", "remaining_legal_moves",
    "next_min_cost", "next_mean_best_cost", "next_max_best_cost", "next_reverse_moves",
    "sets_up_reverse_10", "chosen_vs_cheapest_flexibility",
    "cost_x_progress", "regret_x_progress", "next_max_x_progress",
    "memory_selected_pile_options", "memory_all_pile_options",
]


def features(game, *, use_memory=True):
    piles = list(game.piles)
    hand = list(game.hands[game.curplay])
    moves = [move for move in game.moves if "->" in move]
    assert len(moves) == len(game.moves), "strict card phase expected"
    all_costs = [move_cost(piles, move) for move in moves]
    global_min = min(all_costs)
    options = {card: legal_costs(card, piles) for card in hand}
    cheapest_cards = {
        int(move.split("->")[0]) for move, value in zip(moves, all_costs) if value == global_min
    }
    cheapest_flex = np.mean([len(options[card]) for card in cheapest_cards])
    progress = 1.0 - len(game.deck) / 84.0
    rows = []
    for move, immediate in zip(moves, all_costs):
        card, pile = map(int, move.split("->"))
        new_piles = piles[:]
        new_piles[pile] = card
        remaining = hand[:]
        remaining.remove(card)
        remaining_options = [legal_costs(other, new_piles) for other in remaining]
        playable = [values for values in remaining_options if values]
        best = [min(value for _, value in values) for values in playable]
        reverse_moves = sum(value == -10 for values in remaining_options for _, value in values)
        direction_other = 1 - pile if pile < 2 else 5 - pile
        old_gap = abs(piles[pile] - piles[direction_other])
        new_gap = abs(new_piles[pile] - new_piles[direction_other])
        open_after = 100 - card if pile < 2 else card - 1
        normal_advance = max(immediate, 0)
        regret = immediate - global_min
        selected_memory_options = 0
        all_memory_options = 0
        if use_memory:
            own = set(hand)
            for unseen in range(2, game.max_value):
                if game._played_cards & (1 << unseen) or unseen in own:
                    continue
                legal = legal_costs(unseen, new_piles)
                all_memory_options += len(legal)
                selected_memory_options += sum(candidate_pile == pile for candidate_pile, _ in legal)
        next_min = min(best) if best else 40
        next_mean = float(np.mean(best)) if best else 40
        next_max = max(best) if best else 40
        row = [
            immediate / 10, immediate * immediate / 100, int(immediate == -10),
            int(0 <= immediate <= 2), int(3 <= immediate <= 5), int(6 <= immediate <= 10), int(immediate > 10),
            regret / 10, int(1 <= regret <= 2), int(3 <= regret <= 5), int(6 <= regret <= 10), int(regret > 10),
            min(value for _, value in options[card]) / 10, len(options[card]), int(len(options[card]) == 1), card / 100,
            normal_advance / 10, open_after / 100, new_gap / 100, (new_gap - old_gap) / 100,
            len(playable), len(remaining) - len(playable), sum(len(values) for values in remaining_options),
            next_min / 10, next_mean / 10, next_max / 10, reverse_moves,
            int(any(candidate_pile == pile and value == -10 for values in remaining_options for candidate_pile, value in values)),
            len(options[card]) - cheapest_flex,
            immediate / 10 * progress, regret / 10 * progress, next_max / 10 * progress,
            selected_memory_options / 100, all_memory_options / 400,
        ]
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


@dataclass
class Decision:
    x: np.ndarray
    chosen: int
    episode: int


class LinearRanker:
    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=np.float32)

    def action(self, game):
        use_memory = bool(np.any(self.weights[-2:] != 0))
        return int(np.argmax(features(game, use_memory=use_memory) @ self.weights))

    async def __call__(self, game):
        action = self.action(game)
        return one_hot(action, len(game.moves)).log(), {}


class TeacherCollector:
    def __init__(self, processor, *, behavior=None, episode_offset=0):
        self.processor = processor
        self.behavior = behavior
        self.decisions = []
        self.episode_ids = {}
        self.next_episode = episode_offset

    async def __call__(self, game):
        output = await self.processor(game.display_with_moves())
        chosen = int(output.policy[0].argmax())
        object_id = id(game)
        if object_id not in self.episode_ids:
            self.episode_ids[object_id] = self.next_episode
            self.next_episode += 1
        self.decisions.append(Decision(features(game), chosen, self.episode_ids[object_id]))
        action = chosen if self.behavior is None else self.behavior.action(game)
        return one_hot(action, len(game.moves)).log(), {}


def collect(runner, collector, games, seed):
    seed_all(seed)
    return runner.play([collector, collector], games=games, max_steps=800, rotate=False)


def make_pairs(decisions, indices, rng, max_pairs=500_000):
    pairs = []
    for index in indices:
        decision = decisions[index]
        chosen = decision.x[decision.chosen]
        for alternative in range(len(decision.x)):
            if alternative != decision.chosen:
                pairs.append(chosen - decision.x[alternative])
    if len(pairs) > max_pairs:
        selected = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[i] for i in selected]
    positive = np.asarray(pairs, dtype=np.float32)
    x = np.concatenate([positive, -positive])
    y = np.concatenate([np.ones(len(positive), np.float32), -np.ones(len(positive), np.float32)])
    order = rng.permutation(len(x))
    return x[order], y[order]


def fit_ranker(decisions, train_episodes, *, include_memory, seed):
    indices = [i for i, d in enumerate(decisions) if d.episode in train_episodes]
    x, y = make_pairs(decisions, indices, np.random.default_rng(seed))
    included = np.ones(len(FEATURE_NAMES), dtype=bool)
    if not include_memory:
        included = np.array([not name.startswith("memory_") for name in FEATURE_NAMES])
    scale = x[:, included].std(axis=0)
    scale[scale < 1e-5] = 1
    xt = torch.from_numpy(x[:, included] / scale).cuda()
    yt = torch.from_numpy(y).cuda()
    w = torch.zeros(xt.shape[1], device="cuda", requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=0.08)
    for step in range(240):
        optimizer.zero_grad()
        loss = F.softplus(-yt * (xt @ w)).mean() + 2e-4 * w.abs().mean()
        loss.backward()
        optimizer.step()
    raw = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    raw[included] = w.detach().cpu().numpy() / scale
    return LinearRanker(raw), float(loss.detach().cpu())


def agreement(ranker, decisions, episodes):
    sample = [d for d in decisions if d.episode in episodes]
    exact = np.mean([np.argmax(d.x @ ranker.weights) == d.chosen for d in sample])
    lowest_match = []
    regret = []
    for d in sample:
        action = int(np.argmax(d.x @ ranker.weights))
        costs = d.x[:, FEATURE_NAMES.index("cost")] * 10
        lowest_match.append(costs[action] == costs.min())
        regret.append(costs[action] - costs.min())
    return {"states": len(sample), "top1": float(exact), "lowest_pct": float(np.mean(lowest_match)), "mean_regret": float(np.mean(regret))}


def evaluate(runner, strategy, games, seed):
    seed_all(seed)
    result = runner.play([strategy, strategy], games=games, max_steps=800, rotate=False)
    scores = np.asarray([game.by_seat[0][-1].my_points for game in result], dtype=float)
    return {
        "n": len(scores), "mean": float(scores.mean()), "std": float(scores.std(ddof=1)),
        "q05": float(np.quantile(scores, .05)), "win_rate": float(np.mean(scores == 100)),
        "scores": scores.tolist(),
    }


def top_weights(ranker, n=18):
    order = np.argsort(np.abs(ranker.weights))[::-1]
    return [{"feature": FEATURE_NAMES[i], "weight": float(ranker.weights[i])} for i in order[:n]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-games", type=int, default=512)
    parser.add_argument("--dagger-games", type=int, default=512)
    parser.add_argument("--eval-games", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    model = make(checkpoint["metadata"]["architecture"])
    model.load_state_dict(checkpoint["models"]["current"])
    model.cuda().eval()
    inference = Inference(model, batch_size=1024)
    runner = RolloutRunner(games_library("thegame,mode=strict").make_game, progress=False, coop=True)

    teacher = TeacherCollector(inference.processor)
    teacher_results = collect(runner, teacher, args.teacher_games, 661000)
    episodes = sorted(set(d.episode for d in teacher.decisions))
    split = int(.8 * len(episodes))
    train_episodes, test_episodes = set(episodes[:split]), set(episodes[split:])
    initial, initial_loss = fit_ranker(teacher.decisions, train_episodes, include_memory=False, seed=1)
    initial_memory, memory_loss = fit_ranker(teacher.decisions, train_episodes, include_memory=True, seed=2)
    print("teacher decisions", len(teacher.decisions), "initial agreement", agreement(initial, teacher.decisions, test_episodes), flush=True)

    dagger = TeacherCollector(inference.processor, behavior=initial, episode_offset=max(episodes) + 1)
    collect(runner, dagger, args.dagger_games, 662000)
    combined = teacher.decisions + dagger.decisions
    all_train_episodes = train_episodes | set(d.episode for d in dagger.decisions)
    final, final_loss = fit_ranker(combined, all_train_episodes, include_memory=False, seed=3)
    final_memory, final_memory_loss = fit_ranker(combined, all_train_episodes, include_memory=True, seed=4)
    print("dagger decisions", len(dagger.decisions), "final agreement", agreement(final, teacher.decisions, test_episodes), flush=True)

    evaluations = {
        "lowest": evaluate(runner, LowestCostStrategy(), args.eval_games, 663000),
        "initial_linear": evaluate(runner, initial, args.eval_games, 663000),
        "final_linear": evaluate(runner, final, args.eval_games, 663000),
        "final_linear_memory": evaluate(runner, final_memory, args.eval_games, 663000),
    }
    baseline = np.asarray(evaluations["lowest"]["scores"])
    for value in evaluations.values():
        delta = np.asarray(value["scores"]) - baseline
        se = delta.std(ddof=1) / math.sqrt(len(delta))
        value["delta_vs_lowest"] = float(delta.mean())
        value["delta_ci95"] = [float(delta.mean() - 1.96 * se), float(delta.mean() + 1.96 * se)]
        del value["scores"]
    output = {
        "teacher_checkpoint": str(args.checkpoint),
        "teacher_corpus": {"games": args.teacher_games, "decisions": len(teacher.decisions)},
        "dagger_corpus": {"games": args.dagger_games, "decisions": len(dagger.decisions)},
        "heldout_agreement": {
            "initial_no_memory": agreement(initial, teacher.decisions, test_episodes),
            "initial_memory": agreement(initial_memory, teacher.decisions, test_episodes),
            "final_no_memory": agreement(final, teacher.decisions, test_episodes),
            "final_memory": agreement(final_memory, teacher.decisions, test_episodes),
            "final_on_dagger_states": agreement(final, dagger.decisions, set(d.episode for d in dagger.decisions)),
        },
        "losses": {"initial": initial_loss, "initial_memory": memory_loss, "final": final_loss, "final_memory": final_memory_loss},
        "evaluations": evaluations,
        "top_weights": {"initial": top_weights(initial), "final": top_weights(final), "final_memory": top_weights(final_memory)},
        "initial_weights": dict(zip(FEATURE_NAMES, map(float, initial.weights))),
        "initial_memory_weights": dict(zip(FEATURE_NAMES, map(float, initial_memory.weights))),
        "final_weights": dict(zip(FEATURE_NAMES, map(float, final.weights))),
        "final_memory_weights": dict(zip(FEATURE_NAMES, map(float, final_memory.weights))),
    }
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps({"agreement": output["heldout_agreement"], "evaluations": evaluations, "top_weights": output["top_weights"]}, indent=2))


if __name__ == "__main__":
    main()
