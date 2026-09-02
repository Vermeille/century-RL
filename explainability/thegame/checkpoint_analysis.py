from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from boardrl.cyutils import init_seed
from boardrl.games import games_library
from boardrl.models import make
from boardrl.rollouts import Inference, RolloutRunner


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    init_seed(seed)


def parse_line_int(state: str, prefix: str) -> int:
    line = next(line for line in state.splitlines() if line.startswith(prefix))
    return int(line.removeprefix(prefix).strip())


def parse_action(state: str) -> int:
    line = next(line for line in state.splitlines() if line.startswith("Round:"))
    return int(line.rsplit("Action:", 1)[1].strip())


def parse_piles(state: str) -> list[int]:
    line = next(line for line in state.splitlines() if line.startswith("Piles:"))
    return [int(x) for x in line.removeprefix("Piles:").split()]


def move_cost(move: str, piles: list[int]) -> tuple[int, bool]:
    card_s, pile_s = move.split("->")
    card, pile = int(card_s), int(pile_s)
    cost = card - piles[pile] if pile < 2 else piles[pile] - card
    return cost, cost == -10


def stage(deck: int) -> str:
    if deck >= 56:
        return "early_deck_56_plus"
    if deck >= 20:
        return "mid_deck_20_55"
    if deck > 0:
        return "late_deck_1_19"
    return "post_deck"


def scramble_privileged(state: str) -> str:
    """Scramble other-hand/deck card identities without moving characters."""
    lines = state.splitlines()
    hand_seen = 0
    selected = []
    for i, line in enumerate(lines):
        if line.startswith("Hand:"):
            hand_seen += 1
            if hand_seen > 1:
                selected.append(i)
        elif line.startswith("Deck:"):
            selected.append(i)
    if not selected:
        return state
    tokens = []
    for i in selected:
        tokens.extend(re.findall(r"\d+", lines[i]))
    rng = random.Random(int.from_bytes(hashlib.sha256(state.encode()).digest()[:8], "big"))
    by_width: dict[int, list[str]] = defaultdict(list)
    for token in tokens:
        by_width[len(token)].append(token)
    for values in by_width.values():
        rng.shuffle(values)
    positions = {width: 0 for width in by_width}
    for i in selected:
        def replacement(match):
            width = len(match.group())
            value = by_width[width][positions[width]]
            positions[width] += 1
            return value
        lines[i] = re.sub(r"\d+", replacement, lines[i])
    return "\n".join(lines)


def scramble_memory(state: str) -> str:
    """Permute fixed-width played-card decade summaries, preserving alignment."""
    lines = state.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("Mem:"):
            continue
        chunks = line.removeprefix("Mem:").strip().split()
        rng = random.Random(int.from_bytes(hashlib.sha256((state + "mem").encode()).digest()[:8], "big"))
        rng.shuffle(chunks)
        lines[i] = "Mem: " + " ".join(chunks)
        break
    return "\n".join(lines)


class ScrambledPolicy:
    def __init__(self, processor, transform, temperature: float):
        self.processor = processor
        self.transform = transform
        self.temperature = temperature

    @torch.no_grad()
    async def __call__(self, game):
        if len(game.moves) == 1:
            return torch.tensor([1.0]), {}
        state = self.transform(game.display_with_moves())
        output = await self.processor(state)
        return output.policy[0].cpu() / self.temperature, {"state": state}


def game_summary(game, reservoir, reservoir_rng, reservoir_limit: int):
    end = game.by_seat[0][-1]
    records = [record for trace in game for record in trace[:-1]]
    by_round = defaultdict(list)
    costs, regrets, lowest, ten, ten_opportunity, legal_counts = ([] for _ in range(6))
    pile_counts = [0, 0, 0, 0]
    per_stage = defaultdict(lambda: defaultdict(list))

    for record in records:
        chosen = record.moves[record.action_idx]
        by_round[record.round].append(record)
        if len(record.moves) > 1:
            seen = getattr(game_summary, "seen", 0) + 1
            game_summary.seen = seen
            if len(reservoir) < reservoir_limit:
                reservoir.append(record.state)
            else:
                j = reservoir_rng.randrange(seen)
                if j < reservoir_limit:
                    reservoir[j] = record.state
        if "->" not in chosen:
            continue
        piles = parse_piles(record.state)
        deck = parse_line_int(record.state, "Cards:")
        card_costs = [move_cost(move, piles) for move in record.moves if "->" in move]
        chosen_cost, chosen_ten = move_cost(chosen, piles)
        minimum = min(cost for cost, _ in card_costs)
        opportunity = any(flag for _, flag in card_costs)
        pile = int(chosen.split("->")[1])
        values = {
            "cost": chosen_cost,
            "regret": chosen_cost - minimum,
            "lowest": int(chosen_cost == minimum),
            "ten": int(chosen_ten),
            "ten_given_opportunity": int(chosen_ten) if opportunity else None,
            "legal_actions": len(record.moves),
        }
        costs.append(chosen_cost)
        regrets.append(chosen_cost - minimum)
        lowest.append(int(chosen_cost == minimum))
        ten.append(int(chosen_ten))
        if opportunity:
            ten_opportunity.append(int(chosen_ten))
        pile_counts[pile] += 1
        legal_counts.append(len(record.moves))
        bucket = per_stage[stage(deck)]
        for key, value in values.items():
            if value is not None:
                bucket[key].append(value)

    turn_extras = []
    turns_ended_with_x = 0
    for turn_records in by_round.values():
        turn_records.sort(key=lambda record: parse_action(record.state))
        deck = parse_line_int(turn_records[0].state, "Cards:")
        required = 2 if deck else 1
        cards = sum("->" in record.moves[record.action_idx] for record in turn_records)
        turn_extras.append(max(cards - required, 0))
        turns_ended_with_x += any(record.moves[record.action_idx] == "x" for record in turn_records)

    def mean(values):
        return float(np.mean(values)) if values else None

    return {
        "score": end.my_points,
        "win": int(game.won()),
        "rounds": end.round,
        "card_plays": len(costs),
        "decisions": len(records),
        "avg_cost": mean(costs),
        "avg_regret": mean(regrets),
        "lowest_rate": mean(lowest),
        "ten_rate": mean(ten),
        "ten_capture_rate": mean(ten_opportunity),
        "ten_opportunities": len(ten_opportunity),
        "avg_legal_actions": mean(legal_counts),
        "pile_counts": pile_counts,
        "turns": len(by_round),
        "turn_extra_cards": int(sum(turn_extras)),
        "turns_with_extra": int(sum(extra > 0 for extra in turn_extras)),
        "turns_ended_with_x": turns_ended_with_x,
        "max_extra_in_turn": max(turn_extras, default=0),
        "stage": {
            name: {key: {"sum": float(sum(values)), "n": len(values)} for key, values in metrics.items()}
            for name, metrics in per_stage.items()
        },
    }


@torch.no_grad()
def sensitivity(model, states: list[str], transform, batch_size=256):
    original, changed = [], []
    actually_changed = 0
    for start in range(0, len(states), batch_size):
        source = states[start : start + batch_size]
        altered = [transform(state) for state in source]
        actually_changed += sum(a != b for a, b in zip(source, altered))
        original.extend(model(source).policy)
        changed.extend(model(altered).policy)
    flips, tvs, jsds, confidence_drops = [], [], [], []
    for first_logits, second_logits in zip(original, changed):
        p = torch.softmax(first_logits.float(), 0)
        q = torch.softmax(second_logits.float(), 0)
        midpoint = (p + q) / 2
        jsd = 0.5 * (
            torch.sum(p * (p.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log()))
            + torch.sum(q * (q.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log()))
        )
        top = int(torch.argmax(p))
        flips.append(int(top != int(torch.argmax(q))))
        tvs.append(float(0.5 * torch.abs(p - q).sum()))
        jsds.append(float(jsd))
        confidence_drops.append(float(p[top] - q[top]))
    return {
        "n": len(states),
        "changed_fraction": actually_changed / len(states),
        "top_action_flip_rate": float(np.mean(flips)),
        "mean_total_variation": float(np.mean(tvs)),
        "mean_jsd_nats": float(np.mean(jsds)),
        "mean_original_top_probability_drop": float(np.mean(confidence_drops)),
    }


def evaluate(model, mode, games, batch_games, seed_base, reservoir_limit):
    game_desc = games_library(f"thegame,mode={mode}")
    inference = Inference(model, batch_size=1024)
    runner = RolloutRunner(game_desc.make_game, progress=False, coop=True)
    summaries, reservoir = [], []
    reservoir_rng = random.Random(991827)
    started = time.time()
    for batch_index, start in enumerate(range(0, games, batch_games)):
        count = min(batch_games, games - start)
        seed_all(seed_base + batch_index)
        with inference.evaluating():
            policy = inference.policy(temperature=0.02)
            results = runner.play([policy, policy], games=count, max_steps=800, rotate=True)
        summaries.extend(game_summary(game, reservoir, reservoir_rng, reservoir_limit) for game in results)
        print(f"{mode} primary {len(summaries)}/{games} elapsed={time.time()-started:.1f}s", file=sys.stderr, flush=True)
    return summaries, reservoir, inference, runner


def ablation_scores(inference, runner, transform, games, batch_games, seed_base, label):
    scores, wins = [], []
    started = time.time()
    for batch_index, start in enumerate(range(0, games, batch_games)):
        count = min(batch_games, games - start)
        seed_all(seed_base + batch_index)
        with inference.evaluating():
            policy = ScrambledPolicy(inference.processor, transform, 0.02)
            results = runner.play([policy, policy], games=count, max_steps=800, rotate=True)
        scores.extend(game.by_seat[0][-1].my_points for game in results)
        wins.extend(int(game.won()) for game in results)
        print(f"{label} {len(scores)}/{games} elapsed={time.time()-started:.1f}s", file=sys.stderr, flush=True)
    return {"scores": scores, "wins": wins}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("omni", "strict"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=4096)
    parser.add_argument("--batch-games", type=int, default=256)
    parser.add_argument("--ablation-games", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    model = make(checkpoint["metadata"]["architecture"])
    model.load_state_dict(checkpoint["models"]["current"])
    model.cuda().eval()
    summaries, reservoir, inference, runner = evaluate(
        model, args.mode, args.games, args.batch_games, 781000, 4096
    )
    result = {
        "mode": args.mode,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint["step"],
        "games": args.games,
        "temperature": 0.02,
        "seed_scheme": "batch seeds 781000+i, identical across modes",
        "per_game": summaries,
        "sensitivity": {"memory_scrambled": sensitivity(model, reservoir, scramble_memory)},
        "ablations": {},
    }
    if args.mode == "omni":
        result["sensitivity"]["privileged_scrambled"] = sensitivity(model, reservoir, scramble_privileged)
        result["ablations"]["privileged_scrambled"] = ablation_scores(
            inference, runner, scramble_privileged, args.ablation_games,
            args.batch_games, 781000, "omni privileged ablation"
        )
    result["ablations"]["memory_scrambled"] = ablation_scores(
        inference, runner, scramble_memory, args.ablation_games,
        args.batch_games, 781000, f"{args.mode} memory ablation"
    )
    args.output.write_text(json.dumps(result))
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
