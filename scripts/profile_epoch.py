import argparse
import cProfile
import io
import os
import pstats
import random
import sys
import time

import pyximport
import torch
import yaml

pyximport.install()

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from boardrl.config import Config
from boardrl.cyutils import init_seed
from boardrl.training.returns import annotate_with_model
from main import Trainer, fix_dict, to_trainset


def load_config(config_file: str, fixes: list[str]) -> Config:
    with open(config_file) as f:
        raw_config = yaml.safe_load(f)

    for config_fix in fixes:
        fix_dict(raw_config, *config_fix.split("=", 1))
    raw_config.setdefault("visdom_url", "offline")
    raw_config.setdefault("visdom_port", 8097)

    if "model" not in raw_config:
        raise ValueError("config must specify 'model'")

    net_overrides = raw_config.get("net", {})
    with open(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "model-configs",
            f"{raw_config['model']}.yaml",
        )
    ) as f:
        raw_config["net"] = yaml.safe_load(f)
    raw_config["net"].update(net_overrides)

    config = Config.from_dict(raw_config)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        config.device = "cpu"
    return config


class Timer:
    def __init__(self):
        self.start = time.perf_counter()
        self.last = self.start
        self.rows: list[tuple[str, float]] = []

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.rows.append((name, now - self.last))
        self.last = now

    def total(self) -> float:
        return time.perf_counter() - self.start


def run_profile(args: argparse.Namespace) -> None:
    config = load_config(args.config_file, args.x)
    init_seed(config.seed)
    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)

    trainer = Trainer(config, args.ckpt)
    trainer.epoch = args.epoch

    profiler = cProfile.Profile()
    timer = Timer()
    profiler.enable()

    pit_results = None
    if args.include_pit and trainer.epoch % config.pit.every == 0:
        pit_results = trainer._log_pit()
        timer.mark("pit")

    if args.include_save and trainer.epoch % config.train.save_every == 0:
        saved = trainer._save_model()
        trainer.match_maker.ensure_strategy_registered(f"policy_sampling,model={saved}")
        timer.mark("save")

    data = trainer._run_episode()
    timer.mark("self_play_returns_metrics")

    reference_updated = trainer.reference_handler.update(
        trainer.model,
        epoch=trainer.epoch,
        pit_results=pit_results,
        episode_results=data,
    )
    timer.mark("reference_update")

    trainset = to_trainset(
        data,
        only_players=config.train.only_players,
        only_strategies=config.train.only_strategies,
    )
    random.shuffle(trainset)
    timer.mark("to_trainset_shuffle")

    needs_reference = any(
        loss_fn.needs_reference_policy_value for loss_fn in trainer.losses
    )
    if needs_reference:
        annotate_with_model(
            trainer.reference_handler.model,
            trainset,
            bs=max(config.self_play.batch_size, config.train.batch_size),
            gamma=config.train.discount_factor,
            lmbda=config.train.gae_lambda,
            use_cached_rollout=reference_updated,
        )
    timer.mark("annotate_reference")

    trainer._train_epoch_on_policy(trainset)
    timer.mark("train")
    profiler.disable()

    total = sum(seconds for _, seconds in timer.rows)
    print("PROFILE_EPOCH_SUMMARY")
    print(f"epoch={trainer.epoch}")
    print(f"games={config.self_play.num_games}")
    print(f"samples={len(trainset)}")
    print(f"trace_samples={data.num_samples()}")
    print(f"total_seconds={total:.6f}")
    print(f"samples_per_second={len(trainset) / total:.3f}")
    for name, seconds in timer.rows:
        print(f"stage {name} seconds={seconds:.6f} pct={seconds / total * 100:.2f}")

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs().sort_stats(args.sort).print_stats(args.limit)
    print("PROFILE_TOP")
    print(stream.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--include-pit", action="store_true")
    parser.add_argument("--include-save", action="store_true")
    parser.add_argument("--sort", default="tottime")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("-x", action="append", default=[])
    args = parser.parse_args()
    run_profile(args)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
