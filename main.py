from collections import defaultdict
import torch
import time
import os
import yaml
import random
from tqdm import tqdm
import math
import pyximport
from heavyball import ForeachMuon

from boardrl.config import Config
from boardrl.rl.model import Model
from boardrl.rl.model.loss import loss_from_string
from boardrl.rl.utils import explained_variance, pearson_corr, ReferenceModelHandler
from boardrl.rl.eval.matchmaker import MatchMaker
from boardrl.rl.eval.selfplay import SelfPlayResults
from boardrl.games import games_library

pyximport.install()
from boardrl.cyutils import init_seed
from boardrl.utils import BatchProcessor, Visualizer, ModelPool, chunk
from boardrl.training.returns import compute_returns, annotate_with_model
from boardrl.training import TrainingSample


def make_optimizer(params, train_cfg):
    betas = tuple(train_cfg.betas)
    if train_cfg.optimizer == "AdamW":
        return torch.optim.AdamW(
            params,
            lr=train_cfg.lr,
            betas=betas,
            weight_decay=train_cfg.weight_decay,
            fused=True,
        )
    elif train_cfg.optimizer == "Muon":
        return ForeachMuon(
            params,
            lr=train_cfg.lr,
            betas=betas,
            weight_decay=train_cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {train_cfg.optimizer}")


def measure_gradient_norm(parameters):
    norms = [
        parameter.grad.detach().norm()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(norms))


class Optimizer:
    def __init__(self, params, train_cfg):
        self.opt = make_optimizer(params=params, train_cfg=train_cfg)
        self._initial_lr = float(train_cfg.lr)
        self._total_iterations = float(train_cfg.iterations)
        self._warmup_epochs = train_cfg.warmup_epochs
        self._min_lr_scale = float(train_cfg.lr_min_scale)
        self.current_lr = train_cfg.lr

    def epoch_start(self, epoch: int) -> None:
        self._update_lr(epoch)

    def batch_start(self) -> None:
        self.opt.zero_grad(set_to_none=True)

    def batch_end(self) -> None:
        self.opt.step()

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, d):
        return self.opt.load_state_dict(d)

    def _update_lr(self, epoch: int) -> None:
        """Apply piecewise LR schedule with optional warmup and linear decay.

        - Warmup: linearly ramps 0 -> initial_lr over ``warmup_epochs``.
        - Decay: then linearly decays to ``lr_min_scale * initial_lr``.

        If ``iterations`` is not finite, only the warmup phase is applied.
        """
        total = self._total_iterations

        if not math.isfinite(total) or total < 0:
            return

        warmup = (
            float(self._warmup_epochs)
            if self._warmup_epochs is not None
            else min(100, total * 0.05)
        )
        warmup = max(warmup, 0.0)
        min_scale = max(self._min_lr_scale, 0.0)

        if warmup > 0 and epoch < warmup:
            scale = epoch / warmup
        else:
            decay_steps = max(total - warmup, 1)
            progress = (epoch - warmup) / decay_steps
            scale = min_scale + (1 - min_scale) * (1 - progress)

        new_lr = self._initial_lr * max(scale, 0.0)
        for pg in self.opt.param_groups:
            pg["lr"] = new_lr
        # Always log LR for traceability
        self.current_lr = new_lr


def to_trainset(
    games_data: SelfPlayResults,
    only_players: list[int] | None = None,
    only_strategies: list[int] | None = None,
):
    if only_players is not None:
        games_data = games_data.only_player(only_players)
    if only_strategies is not None:
        games_data = games_data.only_strategy(only_strategies)

    out = []
    for game in games_data:
        for hist in game:
            if len(hist) == 0:
                continue
            end = hist[-1]
            hist = hist[:-1]
            if len(hist) == 0:
                continue
            for i, log in list(enumerate(hist)):
                out.append(
                    TrainingSample(
                        state=log.state,
                        action_idx=log.action_idx,
                        action_distribution=log.action_distribution,
                        score=float(log.score),
                        reward=float(log.reward),
                        returns=log.returns,
                        next=None,
                        terminal=False,
                        truncated=False,
                    )
                )
                for key in ("reference_policy", "reference_value", "reference_max_q"):
                    if hasattr(log, key):
                        setattr(out[-1], key, getattr(log, key))
                if i != 0:
                    out[-2].next = out[-1]
            out[-1].next = end
    # While this looks like a good idea, this prevents
    # learning the value of those non terminal states
    # out = [o for o in out if len(o.moves) > 1]
    return out


class Trainer:
    def __init__(self, config, checkpoint_path=None):
        self.config = config
        self.model = Model(**self.config.net.__dict__)
        self.model.to(config.device)

        self.losses = [
            loss_from_string(
                loss_str, model=self.model, discount_factor=config.train.discount_factor
            )
            for loss_str in config.train.losses
        ]
        self.opt = Optimizer(
            self.model.parameters(),
            config.train,
        )
        # Keep LR scheduling inputs handy
        self.epoch = 0

        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path)
            self.model.load_state_dict(ckpt["model"])
            try:
                self.opt.load_state_dict(ckpt["opt"])
            except ValueError as e:
                print(f"Could not load optimizer state from {checkpoint_path}: {e}")
            self.epoch = ckpt["epoch"]
            ckpt_train = ckpt.get("config", {}).get("train", {})
            if ckpt_train.get("iterations") != config.train.iterations:
                print(
                    "Checkpoint was trained with train.iterations="
                    f"{ckpt_train.get('iterations')}; current run uses "
                    f"{config.train.iterations}. This changes the LR schedule."
                )

        self.reference_handler = ReferenceModelHandler(
            self.model, self.config.train.reference_model_update
        )
        print(self.losses)
        self.viz = Visualizer(
            f"{config.game}_{config.tag}-lr={config.train.lr}",
            url=config.visdom_url,
            port=config.visdom_port,
        )
        self.viz.html(
            "config",
            "<pre>\n" + yaml.dump(config.model_dump()) + "</pre>",
        )
        self.game_desc = games_library(config.game)
        self.game_name = config.game.split(",")[0]

        batch_size = self.config.pit.batch_size or self.config.train.batch_size
        timeout = 0.001
        bp = BatchProcessor(batch_size, self.model, timeout=timeout, model_name="this")
        reference_bp = BatchProcessor(
            batch_size,
            self.reference_handler.model,
            timeout=timeout,
            model_name="reference",
        )
        self.pool = ModelPool(bp, batch_size, timeout, reference_bp)
        self.match_maker = MatchMaker(
            self.game_desc,
            self.pool,
            self.config.train.discount_factor,
        )

    def _training_progress(self) -> float:
        total = float(self.config.train.iterations)
        if not math.isfinite(total) or total <= 0:
            return 0.0
        return max(0.0, min(1.0, self.epoch / total))

    def _log_pit(self):
        self.model.eval()
        print("PIT: ", " VS ".join(str(s) for s in self.config.pit.strategies))
        pit_results = self.match_maker.run_pit(
            self.config.pit.strategies,
            self.config.pit.num_games,
            self.config.pit.max_len,
            rotate=self.config.pit.rotate,
        )
        compute_returns(pit_results.games, self.config.train.discount_factor)
        if (
            self.config.train.print_histories
            and self.epoch % self.config.train.show_every == 0
        ):
            self.game_desc.make_metrics(pit_results.games).print_short_history()
        self.viz.push("pit.win_rate (strategy)", pit_results.win_rate(0), self.epoch)
        # If supported, also display min/max band around mean for pit points.
        self.viz.push_range("pit.points", pit_results.my_points(0), self.epoch)
        self.model.train()
        return pit_results

    def _train_epoch_on_policy(self, data):
        self.model.train()
        now = time.time()
        total_losses = defaultdict(float)
        num_batches = 0
        total_samples = 0
        if len(data) == 0:
            print("No training samples; skipping optimizer step")
            return
        self.opt.epoch_start(self.epoch)
        can_reuse_rollout = all(loss.supports_off_policy for loss in self.losses)
        gradient_epochs = self.config.train.gradient_epochs if can_reuse_rollout else 1
        if not can_reuse_rollout and self.config.train.gradient_epochs != 1:
            print("strict on-policy losses: ignoring gradient_epochs > 1")
        batch_total = math.ceil(len(data) / self.config.train.batch_size)
        for grad_epoch in range(gradient_epochs):
            epoch_data = data
            for augmentation in self.game_desc.augmentations:
                epoch_data = augmentation(epoch_data)
            random.shuffle(epoch_data)
            if not can_reuse_rollout:
                self.opt.batch_start()
            for batch in tqdm(
                chunk(epoch_data, self.config.train.batch_size),
                desc=f"epoch {self.epoch}.{grad_epoch}",
                total=batch_total,
            ):
                if can_reuse_rollout:
                    self.opt.batch_start()
                with torch.no_grad():
                    samples = TrainingSample.collate(batch).to(
                        self.config.device, non_blocking=True
                    )
                policy, value = self.model(samples.state)
                training_state = {"progress": self._training_progress()}
                loss_dict = {
                    loss_fn._registry_name: loss_fn(
                        policy,
                        value,
                        samples,
                        training_state,
                    )
                    for loss_fn in self.losses
                }
                loss = sum(loss_dict.values())
                if not can_reuse_rollout:
                    loss = loss * (len(batch) / len(data))
                loss.backward()
                if can_reuse_rollout:
                    if self.config.train.gradient_clip_norm is None:
                        grad_mag = measure_gradient_norm(self.model.parameters())
                    else:
                        grad_mag = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.config.train.gradient_clip_norm,
                        )
                    self.opt.batch_end()
                else:
                    grad_mag = None
                total_samples += len(batch)
                if self.epoch % self.config.train.show_every == 0:
                    with torch.no_grad():
                        for loss_name, loss_val in loss_dict.items():
                            total_losses[loss_name] += loss_val.item()
                        total_losses["normalized_perplexity"] += sum(
                            (
                                torch.exp(
                                    torch.sum(
                                        -torch.softmax(p, 0) * torch.log_softmax(p, 0)
                                    )
                                )
                                - 1
                            )
                            / (len(p) - 1 + 1e-8)
                            for p in policy
                        ).item() / len(policy)
                        total_losses["perplexity"] += sum(
                            (
                                torch.exp(
                                    torch.sum(
                                        -torch.softmax(p, 0) * torch.log_softmax(p, 0)
                                    )
                                )
                            )
                            for p in policy
                        ).item() / len(policy)
                        pearson = pearson_corr(value.mean, samples.returns)
                        total_losses["pearson"] += pearson.item()
                        total_losses["explained_variance"] += explained_variance(
                            value.mean, samples.returns
                        ).item()
                        total_losses["MAE"] += torch.nn.functional.l1_loss(
                            value.mean, samples.returns
                        ).item()
                        if grad_mag is not None:
                            total_losses["grad_mag"] += grad_mag.item()
                num_batches += 1
            if not can_reuse_rollout:
                if self.config.train.gradient_clip_norm is None:
                    grad_mag = measure_gradient_norm(self.model.parameters())
                else:
                    grad_mag = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.train.gradient_clip_norm,
                    )
                self.opt.batch_end()
                if self.epoch % self.config.train.show_every == 0:
                    total_losses["grad_mag"] += grad_mag.item()

        if num_batches == 0:
            print("No full training batches; skipping optimizer step")
            return

        # self.viz.push( "reference_model.version", self.reference_model.version, self.epoch)
        if self.epoch % self.config.train.show_every == 0:
            for k, v in total_losses.items():
                self.viz.push(
                    k,
                    v / num_batches,
                    self.epoch,
                )
            self.viz.push("lr", self.opt.current_lr, self.epoch)
        print(
            "throughput",
            total_samples / (time.time() - now),
        )
        print()

    def _save_model(self):
        import os

        os.makedirs(f"{self.game_name}-ckpt", exist_ok=True)
        path = f"{self.game_name}-ckpt/rl-{self.epoch}.pth"
        torch.save(
            {
                "model": self.model.state_dict(),
                "opt": self.opt.state_dict(),
                "epoch": self.epoch,
                "config": self.config.model_dump(),
            },
            path,
        )
        return path

    def _run_episode(self):
        print(
            "SELF-PLAY: ", " VS ".join(str(s) for s in self.config.self_play.strategies)
        )
        self.model.eval()
        data = self.match_maker.run_self_play(
            self.config.self_play.strategies,
            self.config.self_play.num_games,
            self.config.self_play.max_len,
            rotate=self.config.self_play.rotate,
        )
        compute_returns(
            data,
            self.config.train.discount_factor,
            # 1 - 1 / (1 + self.epoch * 0.1),
            entropy_reward_scale=self.config.train.entropy_reward_scale,
            reward_rescale=self.config.train.reward_rescale,
        )
        metrics = self.game_desc.make_metrics(data)
        if self.epoch % self.config.train.show_every == 0:
            if self.config.train.print_histories:
                metrics.print_short_history()
            metrics.metrics_to_visdom(self.viz, self.epoch)
            avg_reward_strat = [
                data.my_avg_reward(p, by="strategy") for p in range(data.num_players())
            ]
            self.viz.push("avg_reward (strategy)", avg_reward_strat, self.epoch)
            avg_reward_seat = [
                data.my_avg_reward(p, by="seat") for p in range(data.num_players())
            ]
            self.viz.push("avg_reward (seat)", avg_reward_seat, self.epoch)
            avg_winrate_strat = [
                data.win_rate(p, by="strategy") for p in range(data.num_players())
            ]
            self.viz.push("avg_winrate (strat)", avg_winrate_strat, self.epoch)
        self.model.train()
        return data

    def train(self):
        print("#parameters", sum(p.numel() for p in self.model.parameters()) / 1e6, "M")

        pit_results = None
        while self.epoch < self.config.train.iterations + 1:
            # Update learning-rate schedule (linear decay) and log it
            print("EPOCH", self.epoch)
            if self.epoch % self.config.pit.every == 0:
                pit_results = self._log_pit()

            if self.epoch % self.config.train.save_every == 0:
                saved = self._save_model()
                self.match_maker.ensure_strategy_registered(
                    f"policy_sampling,model={saved}"
                )

            data = self._run_episode()
            reference_updated = self.reference_handler.update(
                self.model,
                epoch=self.epoch,
                pit_results=pit_results,
                episode_results=data,
            )
            trainset = to_trainset(
                data,
                only_players=self.config.train.only_players,
                only_strategies=self.config.train.only_strategies,
            )

            random.shuffle(trainset)
            needs_reference = any(
                loss_fn.needs_reference_policy_value for loss_fn in self.losses
            )
            if needs_reference:
                annotate_with_model(
                    self.reference_handler.model,
                    trainset,
                    bs=max(
                        self.config.self_play.batch_size, self.config.train.batch_size
                    ),
                    gamma=self.config.train.discount_factor,
                    lmbda=self.config.train.gae_lambda,
                    use_cached_rollout=reference_updated,
                )
            print(len(trainset), "samples")
            self._train_epoch_on_policy(trainset)
            # torch.cuda.empty_cache()
            self.epoch += 1
        self._save_model()


def fix_dict(config, key, new_value):
    split = key.split(".", 1)

    if isinstance(config, list):
        split[0] = int(split[0])

    if len(split) == 1:
        if isinstance(config, dict):
            config[split[0]] = yaml.safe_load(new_value)
        elif isinstance(config, list):
            assert split[0] < len(config)
            config[split[0]] = yaml.safe_load(new_value)
    else:
        if isinstance(config, dict):
            config = config.setdefault(split[0], {})
        elif isinstance(config, list):
            assert split[0] < len(config)
            config = config[split[0]]
        fix_dict(config, split[1], new_value)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("-x", action="append", default=[])
    parser.add_argument("--visdom-url", default="http://localhost")
    parser.add_argument("--visdom-port", default=8097)
    opts = parser.parse_args()

    with open(opts.config_file) as f:
        raw_config = yaml.safe_load(f)

    for config_fix in opts.x:
        fix_dict(raw_config, *config_fix.split("=", 1))
    raw_config.setdefault("visdom_url", opts.visdom_url)
    raw_config.setdefault("visdom_port", opts.visdom_port)

    if "model" in raw_config:
        net_overrides = raw_config.get("net", {})
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "model-configs",
                f"{raw_config['model']}.yaml",
            )
        ) as f:
            raw_config["net"] = yaml.safe_load(f)
        raw_config["net"].update(net_overrides)
    else:
        raise ValueError("config must specify 'model'")

    config = Config.from_dict(raw_config)
    init_seed(config.seed)
    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        print(r"* - . /!\ /!\ CUDA not available, using CPU /!\ /!\ . - *")
        config.device = "cpu"

    ckpt = opts.ckpt if opts.ckpt != "None" else None

    trainer = Trainer(config, ckpt)
    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer._save_model()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
