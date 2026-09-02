from __future__ import annotations

import argparse
import hashlib
import json
import math
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


EMPTY_MEMORY = "9aa 8aa 7aa 6aa 5aa 4aa 3aa 2aa 1aa 0aa"
FULL_MEMORY = "9FF 8FF 7FF 6FF 5FF 4FF 3FF 2FF 1FF 0FF"


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    init_seed(seed)


def line(state: str, prefix: str) -> str:
    return next(value for value in state.splitlines() if value.startswith(prefix))


def line_int(state: str, prefix: str) -> int:
    return int(line(state, prefix).removeprefix(prefix).strip())


def round_action(state: str) -> tuple[int, int]:
    value = line(state, "Round:")
    left, right = value.split(", Action:")
    return int(left.removeprefix("Round:").strip()), int(right.strip())


def memory(state: str) -> str:
    return line(state, "Mem:").removeprefix("Mem:").strip()


def replace_memory(state: str, replacement: str) -> str:
    return re.sub(r"(?m)^Mem: .*$", "Mem: " + replacement, state)


def stage(state: str) -> str:
    cards = line_int(state, "Cards:")
    if cards >= 56:
        return "early"
    if cards >= 20:
        return "middle"
    if cards > 0:
        return "late"
    return "post_deck"


def zero_memory(state: str) -> str:
    return replace_memory(state, EMPTY_MEMORY)


def full_memory(state: str) -> str:
    return replace_memory(state, FULL_MEMORY)


def zero_decades(state: str, decades: set[int]) -> str:
    tokens = memory(state).split()
    transformed = [token[0] + "aa" if int(token[0]) in decades else token for token in tokens]
    return replace_memory(state, " ".join(transformed))


def shuffle_symbols(state: str) -> str:
    tokens = memory(state).split()
    symbols = [symbol for token in tokens for symbol in token[1:]]
    rng = random.Random(int.from_bytes(hashlib.sha256((state + "symbols").encode()).digest()[:8], "big"))
    rng.shuffle(symbols)
    transformed = []
    for token in tokens:
        transformed.append(token[0] + symbols.pop(0) + symbols.pop(0))
    return replace_memory(state, " ".join(transformed))


class DonorMemory:
    def __init__(self, states: list[str]):
        self.exact = defaultdict(list)
        self.coarse = defaultdict(list)
        self.cards_only = defaultdict(list)
        for state in states:
            cards = line_int(state, "Cards:")
            game_round, action = round_action(state)
            value = memory(state)
            self.exact[(cards, action, game_round // 2)].append(value)
            self.coarse[(cards, action)].append(value)
            self.cards_only[cards].append(value)
        self.calls = 0
        self.changed = 0

    def __call__(self, state: str) -> str:
        cards = line_int(state, "Cards:")
        game_round, action = round_action(state)
        current = memory(state)
        candidates = self.exact.get((cards, action, game_round // 2))
        if not candidates:
            candidates = self.coarse.get((cards, action))
        if not candidates:
            candidates = self.cards_only.get(cards)
        self.calls += 1
        if not candidates:
            return state
        index = int.from_bytes(hashlib.sha256((state + "donor").encode()).digest()[:8], "big") % len(candidates)
        replacement = candidates[index]
        if replacement == current and len(candidates) > 1:
            replacement = candidates[(index + 1) % len(candidates)]
        self.changed += replacement != current
        return replace_memory(state, replacement)

    @property
    def changed_fraction(self):
        return self.changed / self.calls if self.calls else 0.0


class TransformedPolicy:
    def __init__(self, processor, transform, temperature=0.02):
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


def reservoir_add(reservoir, state, seen, limit, rng):
    seen += 1
    if len(reservoir) < limit:
        reservoir.append(state)
    else:
        index = rng.randrange(seen)
        if index < limit:
            reservoir[index] = state
    return seen


def play_baseline(model, mode, games, batch_games, reservoir_limit):
    game_desc = games_library(f"thegame,mode={mode}")
    inference = Inference(model, batch_size=1024)
    runner = RolloutRunner(game_desc.make_game, progress=False, coop=True)
    scores, wins, reservoir = [], [], []
    reservoir_rng = random.Random(47621)
    seen = 0
    started = time.time()
    for batch_index, start in enumerate(range(0, games, batch_games)):
        count = min(batch_games, games - start)
        seed_all(781000 + batch_index)
        with inference.evaluating():
            policy = inference.policy(temperature=0.02)
            results = runner.play([policy, policy], games=count, max_steps=800, rotate=True)
        for game in results:
            scores.append(game.by_seat[0][-1].my_points)
            wins.append(int(game.won()))
            for trace in game:
                for record in trace[:-1]:
                    if len(record.moves) > 1:
                        seen = reservoir_add(
                            reservoir, record.state, seen, reservoir_limit, reservoir_rng
                        )
        print(f"{mode} baseline {len(scores)}/{games} elapsed={time.time()-started:.1f}s", file=sys.stderr, flush=True)
    return inference, runner, scores, wins, reservoir


@torch.no_grad()
def offline_effect(model, states, transform, *, only_changed=True, batch_size=256):
    source_states = []
    changed_states = []
    stage_names = []
    for state in states:
        changed = transform(state)
        if only_changed and changed == state:
            continue
        source_states.append(state)
        changed_states.append(changed)
        stage_names.append(stage(state))
    rows = []
    for start in range(0, len(source_states), batch_size):
        source = source_states[start : start + batch_size]
        changed = changed_states[start : start + batch_size]
        first = model(source)
        second = model(changed)
        for p_logits, q_logits, p_value, q_value in zip(
            first.policy, second.policy, first.value.mean, second.value.mean
        ):
            p = torch.softmax(p_logits.float(), 0)
            q = torch.softmax(q_logits.float(), 0)
            midpoint = (p + q) / 2
            jsd = 0.5 * (
                torch.sum(p * (p.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log()))
                + torch.sum(q * (q.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log()))
            )
            top = int(torch.argmax(p))
            rows.append(
                (
                    int(top != int(torch.argmax(q))),
                    float(0.5 * torch.abs(p - q).sum()),
                    float(jsd),
                    float(p[top] - q[top]),
                    float(q_value - p_value),
                    float(abs(q_value - p_value)),
                )
            )
    values = np.asarray(rows, dtype=float)

    def summarize(indices):
        subset = values[indices]
        return {
            "n": len(subset),
            "top_flip_rate": float(subset[:, 0].mean()),
            "mean_total_variation": float(subset[:, 1].mean()),
            "mean_jsd_nats": float(subset[:, 2].mean()),
            "mean_original_top_probability_drop": float(subset[:, 3].mean()),
            "mean_value_delta_points": float(subset[:, 4].mean() * 100),
            "mean_abs_value_delta_points": float(subset[:, 5].mean() * 100),
        }

    result = summarize(np.arange(len(values)))
    result["fraction_of_corpus_changed"] = len(values) / len(states)
    result["by_stage"] = {
        name: summarize(np.asarray([i for i, value in enumerate(stage_names) if value == name]))
        for name in ("early", "middle", "late", "post_deck")
        if any(value == name for value in stage_names)
    }
    return result


def run_ablation(inference, runner, transform, games, batch_games, label):
    scores, wins = [], []
    started = time.time()
    for batch_index, start in enumerate(range(0, games, batch_games)):
        count = min(batch_games, games - start)
        seed_all(781000 + batch_index)
        with inference.evaluating():
            policy = TransformedPolicy(inference.processor, transform)
            results = runner.play([policy, policy], games=count, max_steps=800, rotate=True)
        scores.extend(game.by_seat[0][-1].my_points for game in results)
        wins.extend(int(game.won()) for game in results)
        print(f"{label} {len(scores)}/{games} elapsed={time.time()-started:.1f}s", file=sys.stderr, flush=True)
    return {"scores": scores, "wins": wins}


def paired_summary(baseline_scores, baseline_wins, ablation):
    first = np.asarray(baseline_scores[: len(ablation["scores"])], dtype=float)
    second = np.asarray(ablation["scores"], dtype=float)
    difference = second - first
    half_width = 1.96 * difference.std(ddof=1) / math.sqrt(len(difference))
    return {
        "n": len(difference),
        "baseline_mean": float(first.mean()),
        "ablation_mean": float(second.mean()),
        "paired_score_delta": float(difference.mean()),
        "paired_score_delta_ci95": [
            float(difference.mean() - half_width),
            float(difference.mean() + half_width),
        ],
        "baseline_win_rate": float(np.mean(baseline_wins[: len(difference)])),
        "ablation_win_rate": float(np.mean(ablation["wins"])),
        "ablation_score_sd": float(second.std(ddof=1)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("omni", "strict"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=2048)
    parser.add_argument("--batch-games", type=int, default=256)
    parser.add_argument("--reservoir", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    model = make(checkpoint["metadata"]["architecture"])
    model.load_state_dict(checkpoint["models"]["current"])
    model.cuda().eval()
    inference, runner, scores, wins, states = play_baseline(
        model, args.mode, args.games, args.batch_games, args.reservoir
    )
    donor = DonorMemory(states)
    transforms = {
        "zero_all": zero_memory,
        "full_all": full_memory,
        "shuffle_symbols": shuffle_symbols,
        "matched_donor": donor,
        "zero_low_decades_0_4": lambda state: zero_decades(state, set(range(0, 5))),
        "zero_high_decades_5_9": lambda state: zero_decades(state, set(range(5, 10))),
    }
    offline = {
        name: offline_effect(model, states, transform)
        for name, transform in transforms.items()
    }
    for decade in range(10):
        offline[f"zero_decade_{decade}"] = offline_effect(
            model, states, lambda state, decade=decade: zero_decades(state, {decade})
        )
    online_zero = run_ablation(
        inference, runner, zero_memory, args.games, args.batch_games,
        f"{args.mode} zero-memory"
    )
    online_donor_transform = DonorMemory(states)
    online_donor = run_ablation(
        inference, runner, online_donor_transform, args.games, args.batch_games,
        f"{args.mode} donor-memory"
    )
    result = {
        "mode": args.mode,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint["step"],
        "games": args.games,
        "temperature": 0.02,
        "reservoir_states": len(states),
        "baseline": {
            "mean": float(np.mean(scores)),
            "sd": float(np.std(scores, ddof=1)),
            "win_rate": float(np.mean(wins)),
        },
        "offline": offline,
        "online": {
            "zero_all": paired_summary(scores, wins, online_zero),
            "matched_donor": paired_summary(scores, wins, online_donor),
        },
        "donor_changed_fraction_offline": donor.changed_fraction,
        "donor_changed_fraction_online": online_donor_transform.changed_fraction,
    }
    args.output.write_text(json.dumps(result))
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
