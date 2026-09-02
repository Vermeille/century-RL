from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
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
from memory_analysis import memory, line_int, offline_effect, replace_memory, round_action


SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEF"


def played_count(value: str) -> int:
    return sum(SYMBOLS.index(symbol).bit_count() for token in value.split() for symbol in token[1:])


def state_key(state: str):
    game_round, action = round_action(state)
    value = memory(state)
    return line_int(state, "Cards:"), game_round, action, played_count(value)


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    init_seed(seed)


class IdentityDonor:
    def __init__(self, states):
        self.pool = defaultdict(list)
        for state in states:
            value = memory(state)
            values = self.pool[state_key(state)]
            if value not in values:
                values.append(value)
        self.calls = 0
        self.changed = 0

    def __call__(self, state):
        self.calls += 1
        current = memory(state)
        candidates = self.pool.get(state_key(state), ())
        if len(candidates) < 2:
            return state
        index = int.from_bytes(hashlib.sha256((state + "identity").encode()).digest()[:8], "big") % len(candidates)
        replacement = candidates[index]
        if replacement == current:
            replacement = candidates[(index + 1) % len(candidates)]
        self.changed += replacement != current
        return replace_memory(state, replacement)

    @property
    def changed_fraction(self):
        return self.changed / self.calls if self.calls else 0.0


class Policy:
    def __init__(self, processor, transform):
        self.processor = processor
        self.transform = transform

    @torch.no_grad()
    async def __call__(self, game):
        if len(game.moves) == 1:
            return torch.tensor([1.0]), {}
        state = self.transform(game.display_with_moves())
        output = await self.processor(state)
        return output.policy[0].cpu() / 0.02, {"state": state}


def collect_states(inference, runner, games, batch_games):
    states = []
    started = time.time()
    for batch_index, start in enumerate(range(0, games, batch_games)):
        count = min(batch_games, games - start)
        seed_all(991000 + batch_index)
        with inference.evaluating():
            policy = inference.policy(temperature=0.02)
            results = runner.play([policy, policy], games=count, max_steps=800, rotate=True)
        for game in results:
            for trace in game:
                states.extend(record.state for record in trace[:-1] if len(record.moves) > 1)
        print(f"donor corpus {start+count}/{games} states={len(states)} elapsed={time.time()-started:.1f}s", file=sys.stderr, flush=True)
    return states


def evaluate(inference, runner, transform, games, batch_games):
    scores, wins = [], []
    started = time.time()
    for batch_index, start in enumerate(range(0, games, batch_games)):
        count = min(batch_games, games - start)
        seed_all(781000 + batch_index)
        with inference.evaluating():
            policy = Policy(inference.processor, transform)
            results = runner.play([policy, policy], games=count, max_steps=800, rotate=True)
        scores.extend(game.by_seat[0][-1].my_points for game in results)
        wins.extend(int(game.won()) for game in results)
        print(f"identity donor {len(scores)}/{games} elapsed={time.time()-started:.1f}s", file=sys.stderr, flush=True)
    return scores, wins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("omni", "strict"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--games", type=int, default=2048)
    parser.add_argument("--corpus-games", type=int, default=1024)
    parser.add_argument("--batch-games", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    model = make(checkpoint["metadata"]["architecture"])
    model.load_state_dict(checkpoint["models"]["current"])
    model.cuda().eval()
    game_desc = games_library(f"thegame,mode={args.mode}")
    inference = Inference(model, batch_size=1024)
    runner = RolloutRunner(game_desc.make_game, progress=False, coop=True)
    states = collect_states(inference, runner, args.corpus_games, args.batch_games)
    donor = IdentityDonor(states)
    rng = random.Random(7781)
    sample = rng.sample(states, min(8192, len(states)))
    offline = offline_effect(model, sample, donor)
    donor_online = IdentityDonor(states)
    scores, wins = evaluate(inference, runner, donor_online, args.games, args.batch_games)

    baseline_data = json.loads(args.baseline_results.read_text())
    baseline_scores = np.asarray([game["score"] for game in baseline_data["per_game"][: args.games]], dtype=float)
    baseline_wins = np.asarray([game["win"] for game in baseline_data["per_game"][: args.games]], dtype=float)
    scores_array = np.asarray(scores, dtype=float)
    difference = scores_array - baseline_scores
    half_width = 1.96 * difference.std(ddof=1) / math.sqrt(len(difference))
    result = {
        "mode": args.mode,
        "corpus_games": args.corpus_games,
        "corpus_states": len(states),
        "unique_keys": len(donor.pool),
        "offline": offline,
        "offline_changed_fraction": donor.changed_fraction,
        "online_changed_fraction": donor_online.changed_fraction,
        "online": {
            "n": args.games,
            "baseline_mean": float(baseline_scores.mean()),
            "identity_donor_mean": float(scores_array.mean()),
            "paired_delta": float(difference.mean()),
            "paired_delta_ci95": [
                float(difference.mean() - half_width),
                float(difference.mean() + half_width),
            ],
            "baseline_win_rate": float(baseline_wins.mean()),
            "identity_donor_win_rate": float(np.mean(wins)),
        },
    }
    args.output.write_text(json.dumps(result))
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
