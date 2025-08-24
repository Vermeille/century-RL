from collections import defaultdict
import torch
import time
import copy
import os
import yaml
import random
from tqdm import tqdm
from heavyball import ForeachMuon

from boardrl.config import Config
from boardrl.rl.model import Model
from boardrl.rl.model.loss import loss_from_string
from boardrl.rl.utils import pearson_corr
from boardrl.rl.eval.selfplay import self_play, pit, SelfPlayResults
from boardrl.games import games_library
from boardrl.cyutils import init_seed
from boardrl.utils import BatchProcessor, Visualizer, ModelPool, PythonExec
from boardrl.training.returns import compute_returns
from boardrl.training import TrainingSample


def make_optimizer(params, train_cfg):
    betas = tuple(train_cfg.betas)
    if train_cfg.optimizer == "AdamW":
        return torch.optim.AdamW(
            params,
            lr=train_cfg.lr,
            betas=betas,
            weight_decay=train_cfg.weight_decay,
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
            for i, log in list(enumerate(hist)):
                out.append(
                    TrainingSample(
                        round=log.round,
                        state=log.state,
                        moves=log.moves,
                        action_idx=log.action_idx,
                        action_distribution=log.action_distribution,
                        score=float(log.current_diff_points),
                        reward=float(log.reward),
                        returns=log.returns,
                        current_diff_points=float(log.current_diff_points),
                        next=None,
                        final=False,
                    )
                )
                if i != 0:
                    out[-2].next = out[-1]
            out[-1].next = end
    # While this looks like a good idea, this prevents
    # learning the value of those non terminal states
    # out = [o for o in out if len(o.moves) > 1]
    return out


def chunk(data, size):
    i = 0
    while i < len(data):
        yield data[i : i + size]
        i += size


def flatten(list_of_lists):
    out = []
    for l in list_of_lists:
        out += l
    return out


def autobatch(model, input, bs=None):
    if bs is None:
        bs = len(input)
    assert bs > 0

    try:
        with torch.no_grad():
            return flatten([model(x) for x in chunk(input, bs)])
    except Exception as e:
        print(e)
        return autobatch(model, input, bs // 2)


class Trainer:
    def __init__(self, config, checkpoint_path=None):
        self.config = config
        self.model = Model(**self.config.net.__dict__)
        self.model.to(config.device)
        self.opt = make_optimizer(self.model.parameters(), config.train)

        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path)
            self.model.load_state_dict(ckpt["model"])
            self.opt.load_state_dict(ckpt["opt"])
            # change lr
            for param_group in self.opt.param_groups:
                param_group["lr"] = config.train.lr

        self.prev_model = copy.deepcopy(self.model)
        self.prev_model.version = 0
        self.prev_model.eval()
        self.policy_loss = loss_from_string(
            config.train.loss.policy,
            model=self.model,
            discount_factor=config.train.discount_factor,
        )
        self.value_loss = loss_from_string(
            config.train.loss.value,
            model=self.model,
            discount_factor=config.train.discount_factor,
        )
        print(self.policy_loss, self.value_loss)
        self.viz = Visualizer(
            f"{config.game}_{config.tag}-lr={config.train.lr}",
            url=config.visdom_url,
            port=config.visdom_port,
        )
        self.viz.html(
            "config",
            "<pre>\n" + yaml.dump(config.model_dump()) + "</pre>",
        )
        self.epoch = 0
        self.game_desc = games_library(config.game)
        self.game_name = config.game.split(",")[0]
        self.pit_results = None
        self.episode_results = None
        self.prev_update_exec = PythonExec(self.config.train.prev_model_update)

    def _annotate_reference_model(self, trainset):
        with torch.no_grad():

            def eval_states(states):
                out = []
                for batch in chunk(states, self.config.train.batch_size):
                    out.extend(self.prev_model(batch).unbatched())
                return out

            preds = eval_states([s.state for s in trainset])
            for sample, pv in zip(trainset, preds):
                sample.reference_policy = pv.policy
                sample.reference_value = pv.value.mean.item()
                sample.reference_max_q = pv.q_value()[0].max().item()

                if getattr(sample.next, "final", False):
                    sample.next.reference_value = 0
                    sample.next.reference_max_q = 0

            for sample in trainset:
                if sample.next:
                    sample.next_reference_value = sample.next.reference_value
                    sample.next_reference_max_q = sample.next.reference_max_q

    def _log_pit(self):
        self.model.eval()
        self.prev_model.eval()
        batch_size = self.config.pit.batch_size or self.config.train.batch_size
        timeout = 0.01
        bp = BatchProcessor(batch_size, self.model, timeout=timeout)
        prev_bp = BatchProcessor(batch_size, self.prev_model, timeout=timeout)
        pool = ModelPool(bp, batch_size, timeout, prev_bp)
        print("PIT: ", " VS ".join(self.config.pit.strategies))
        pit_results = pit(
            self.game_desc.make_game,
            [
                self.game_desc.strategy_from_string(
                    s, model=pool, discount_factor=self.config.train.discount_factor
                )
                for s in self.config.pit.strategies
            ],
            self.config.pit.num_games,
            self.config.pit.max_len,
            rotate=self.config.pit.rotate,
        )
        self.pit_results = pit_results
        compute_returns(pit_results.games, self.config.train.discount_factor)
        self.game_desc.make_metrics(pit_results.games).print_short_history()
        self.viz.push("pit.win_rate (strategy)", pit_results.win_rate(0), self.epoch)
        self.viz.push("pit.avg_points", pit_results.my_avg_points(0), self.epoch)
        self.model.train()

    def _maybe_update_prev_model(self):
        env = {
            "epoch": self.epoch,
            "pit": self.pit_results,
            "episode": self.episode_results,
            "True": True,
            "False": False,
            "version": self.prev_model.version,
            "__builtins__": {
                "print": print,
            },
        }
        env["__builtins__"]["exists"] = lambda s: s in self.prev_update_exec.ctx

        update = self.prev_update_exec(env)
        if update:
            with torch.no_grad():
                for prev_param, param in zip(
                    self.prev_model.state_dict().values(),
                    self.model.state_dict().values(),
                ):
                    prev_param.data.copy_(param.data)
            self.prev_model.eval()
            self.prev_model.version = self.epoch

    def _train_epoch_off_policy(self, data):
        self.model.train()
        now = time.time()
        grad_pct = 1 / self.config.train.gradient_epochs
        for grad_ep in range(self.config.train.gradient_epochs):
            indices = torch.randperm(len(data))
            batch_pct = 1 / (len(indices) / self.config.train.batch_size + 1)
            for b_i, batch in enumerate(
                tqdm(
                    chunk(indices, self.config.train.batch_size),
                    desc=f"epoch {self.epoch}",
                    total=len(indices) // self.config.train.batch_size,
                )
            ):
                with torch.no_grad():
                    samples = TrainingSample.collate([data[bi] for bi in batch]).to(
                        self.config.device
                    )
                self.opt.zero_grad()
                total_losses = defaultdict(float)
                policy, value = self.model(samples.state)
                policy_loss = self.policy_loss(policy, value, samples)
                value_loss = self.value_loss(policy, value, samples)
                loss = policy_loss + value_loss
                loss = loss * len(samples.state) / self.config.train.batch_size
                loss.backward()
                total_losses["policy"] += policy_loss.item()
                total_losses["value"] += value_loss.item()

                grad_mag = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=5.0
                )
                self.opt.step()

                step = self.epoch + grad_ep * grad_pct + b_i * batch_pct * grad_pct

                if False:
                    for k, v in total_losses.items():
                        self.viz.push(
                            f"loss.{k}",
                            v,
                            step,
                        )
                    self.viz.push(
                        "MAE",
                        torch.nn.functional.l1_loss(value.mean, samples.returns).item(),
                        step,
                    )
                    self.viz.push(
                        "grad_mag",
                        grad_mag.item(),
                        step,
                    )
                    self.viz.push(
                        "perplexity",
                        sum(
                            torch.exp(
                                torch.sum(
                                    -torch.softmax(p, 0) * torch.log_softmax(p, 0)
                                )
                            )
                            / len(p)
                            for p in policy
                        ).item()
                        / len(policy),
                        step,
                    )
                    pearson = pearson_corr(value.mean, samples.returns)
                    self.viz.push(
                        "pearson",
                        pearson.item(),
                        step,
                    )
        print(
            "throughput",
            len(data) * self.config.train.gradient_epochs / (time.time() - now),
        )
        print()

    def _train_epoch_on_policy(self, data):
        self.model.train()
        now = time.time()
        total_losses = defaultdict(float)
        self.opt.zero_grad()
        num_batches = 1 + len(data) // self.config.train.batch_size
        for batch in tqdm(
            chunk(data, self.config.train.batch_size),
            desc=f"epoch {self.epoch}",
            total=num_batches,
        ):
            with torch.no_grad():
                samples = TrainingSample.collate(batch).to(
                    self.config.device, non_blocking=True
                )
            policy, value = self.model(samples.state)
            policy_loss = self.policy_loss(policy, value, samples)
            value_loss = self.value_loss(policy, value, samples)
            loss = policy_loss + value_loss
            loss = loss * len(samples.state)
            loss.backward()
            with torch.no_grad():
                total_losses["loss_policy"] += policy_loss.item()
                total_losses["loss_value"] += value_loss.item()

                total_losses["normalized_perplexity"] += sum(
                    torch.exp(torch.sum(-torch.softmax(p, 0) * torch.log_softmax(p, 0)))
                    / len(p)
                    for p in policy
                ).item() / len(policy)
                pearson = pearson_corr(value.mean, samples.returns)
                total_losses["pearson"] += pearson.item()
                total_losses["MAE"] += torch.nn.functional.l1_loss(
                    value.mean, samples.returns
                ).item()

        with torch.no_grad():
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad /= len(data)
        grad_mag = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=50.0
        )
        total_losses["grad_mag"] += grad_mag.item()
        self.opt.step()

        self.viz.push("prev_model.version", self.prev_model.version, self.epoch)
        if self.epoch % self.config.train.show_every == 0:
            for k, v in total_losses.items():
                self.viz.push(
                    k,
                    v / num_batches,
                    self.epoch,
                )
        print(
            "throughput",
            len(data) * self.config.train.gradient_epochs / (time.time() - now),
        )
        print()

    def _save_model(self):
        import os

        os.makedirs(f"{self.game_name}-ckpt", exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "opt": self.opt.state_dict(),
                "epoch": self.epoch,
                "config": self.config.net.model_dump(),
            },
            f"{self.game_name}-ckpt/rl-{self.epoch}.pth",
        )

    def _run_episode(self):
        print("SELF PLAY: ", " VS ".join(self.config.self_play.strategies))

        self.model.eval()
        self.prev_model.eval()
        batch_size = self.config.self_play.batch_size or self.config.train.batch_size
        timeout = 0.02
        bp = BatchProcessor(batch_size, self.model, timeout=timeout)
        prev_bp = BatchProcessor(batch_size, self.prev_model, timeout=timeout)
        pool = ModelPool(bp, batch_size, timeout, prev_bp)
        data = self_play(
            self.game_desc.make_game,
            [
                self.game_desc.strategy_from_string(
                    s, model=pool, discount_factor=self.config.train.discount_factor
                )
                for s in self.config.self_play.strategies
            ],
            self.config.self_play.num_games,
            self.config.self_play.max_len,
            rotate=self.config.self_play.rotate,
        )
        self.episode_results = data
        compute_returns(
            data,
            self.config.train.discount_factor,
            # 1 - 1 / (1 + self.epoch * 0.1),
            entropy_reward_scale=self.config.train.entropy_reward_scale,
            reward_rescale=self.config.train.reward_rescale,
        )
        metrics = self.game_desc.make_metrics(data)
        metrics.print_short_history()
        if self.epoch % self.config.train.show_every == 0:
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

        epoch = 0
        while epoch < self.config.train.iterations:
            self.epoch = epoch

            print("EPOCH", epoch)
            if epoch % self.config.pit.every == 0:
                self._log_pit()

            if epoch % self.config.train.save_every == 0:
                self._save_model()

            data = self._run_episode()
            self._maybe_update_prev_model()
            trainset = to_trainset(
                data,
                only_players=self.config.train.only_players,
                only_strategies=self.config.train.only_strategies,
            )

            random.shuffle(trainset)
            needs_reference = (
                self.policy_loss.needs_reference_policy_value
                or self.value_loss.needs_reference_policy_value
            )
            if needs_reference:
                self._annotate_reference_model(trainset)
            print(len(trainset), "samples")
            if (
                False
                and self.policy_loss.supports_off_policy
                and self.value_loss.supports_off_policy
            ):
                self._train_epoch_off_policy(trainset)
            else:
                self._train_epoch_on_policy(trainset)
            torch.cuda.empty_cache()
            epoch += 1


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

    init_seed()
    with open(opts.config_file) as f:
        raw_config = yaml.safe_load(f)

    for config_fix in opts.x:
        fix_dict(raw_config, *config_fix.split("=", 1))
    raw_config.setdefault("visdom_url", opts.visdom_url)
    raw_config.setdefault("visdom_port", opts.visdom_port)

    if "model" in raw_config:
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "model-configs",
                f"{raw_config['model']}.yaml",
            )
        ) as f:
            raw_config["net"] = yaml.safe_load(f)
    else:
        raise ValueError("config must specify 'model'")

    config = Config.from_dict(raw_config)

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        print(r"* - . /!\ /!\ CUDA not available, using CPU /!\ /!\ . - *")
        config.device = "cpu"

    ckpt = opts.ckpt if opts.ckpt != "None" else None

    Trainer(config, ckpt).train()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
