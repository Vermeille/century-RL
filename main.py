from collections import defaultdict
import torch
import time
import copy
import yaml
from visdom import Visdom
from tqdm import tqdm

from boardrl.rl.model import Model
from boardrl.rl.model.loss import loss_from_string
from boardrl.rl.utils import pearson_corr
from boardrl.rl.eval.selfplay import self_play, pit
from boardrl.games import games_library
from boardrl.cyutils import init_seed


class TrainingSample:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @staticmethod
    def collate(samples):
        return TrainingSample(
            **{
                k: collate([getattr(s, k) for s in samples])
                for k in samples[0].__dict__
            }
        )

    def to(self, *args, **kwargs):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(*args, **kwargs)
            elif isinstance(v[0], torch.Tensor):
                self.__dict__[k] = [x.to(*args, **kwargs) for x in v]
        return self

    def __repr__(self):
        out = ["TrainingSample:"]
        for k, v in self.__dict__.items():
            if k == "next":
                continue
            out.append(f"{k}: {v}")
        return "\n".join(out)


def collate(xs):
    if isinstance(xs[0], (int, float)):
        return torch.tensor(xs)
    if isinstance(xs[0], torch.Tensor):
        try:
            return torch.stack(xs, dim=0)
        except RuntimeError:
            return xs
    return xs


def discount(rews, discount_factor):
    return sum(discount_factor**i * r for i, r in enumerate(rews))


def to_trainset(games_data, discount_factor):
    out = []
    print("to trainset", len(games_data.data))
    for hist in games_data.data:
        end = hist[-1]
        rewards = [0] * (len(hist) - 1)
        for i in range(len(hist) - 1):
            rewards[i] = hist[i + 1].current_diff_points - hist[i].current_diff_points

        for i, log in reversed(list(enumerate(hist[:-1]))):
            out.append(
                TrainingSample(
                    round=float(i),
                    state=log.state,
                    moves=log.moves,
                    action_idx=log.action_idx,
                    action_distribution=log.action_distribution,
                    score=float(end.current_diff_points),
                    reward=float(rewards[i]),
                    returns=discount(rewards[i:], float(discount_factor)),
                    current_diff_points=float(log.current_diff_points),
                    next=end if i == len(hist) - 2 else out[-1],
                    final=False,
                )
            )
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


class Visualizer:
    def __init__(self, tag):
        self.viz = Visdom(
            env=tag,
            server="https://visdom.vermeille.fr",
            port=443,
        )
        self.viz.close()

    def push(self, name, value, epoch):
        self.viz.line(
            torch.tensor([value]),
            torch.tensor([epoch]),
            win=name,
            update="append",
            opts=dict(title=name),
        )


class Trainer:
    def __init__(self, config, checkpoint_path=None):
        self.config = config
        self.model = Model(**self.config.net)
        self.model.to(config.device)
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.train.lr,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )

        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path)
            self.model.load_state_dict(ckpt["model"])
            self.opt.load_state_dict(ckpt["opt"])

        self.prev_model = copy.deepcopy(self.model)
        self.policy_loss = loss_from_string(
            config.train.loss.policy, model=self.model, prev_model=self.prev_model
        )
        self.value_loss = loss_from_string(
            config.train.loss.value, model=self.model, prev_model=self.prev_model
        )
        self.viz = Visualizer(f"{config.game}_{config.tag}-lr={config.train.lr}")
        self.epoch = 0
        self.game_desc = games_library(config.game)
        self.game_name = config.game.split(",")[0]

    def _log_pit(self):
        print("PIT: ", " VS ".join(self.config.pit.strategies))
        pit_results = pit(
            self.game_desc.make_game,
            [
                self.game_desc.strategy_from_string(s, model=self.model)
                for s in self.config.pit.strategies
            ],
            self.config.pit.num_games,
            self.config.pit.max_len,
        )
        self.game_desc.make_metrics(pit_results.games).print_short_history()
        self.viz.push("pit.win_rate", pit_results.win_rate(0), self.epoch)
        self.viz.push("pit.avg_points", pit_results.my_avg_points(0), self.epoch)

    def _train_epoch(self, data):
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
                    self.model.parameters(), max_norm=50000.0
                )
                self.opt.step()

                step = self.epoch + grad_ep * grad_pct + b_i * batch_pct * grad_pct

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
                            torch.sum(-torch.softmax(p, 0) * torch.log_softmax(p, 0))
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

        with torch.no_grad():
            for prev_param, param in zip(
                self.prev_model.state_dict().values(), self.model.state_dict().values()
            ):
                prev_param.data.copy_(param.data)

    def _save_model(self):
        import os

        os.makedirs(self.game_name, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "opt": self.opt.state_dict(),
                "epoch": self.epoch,
                "config": self.config.net,
            },
            f"{self.game_name}/rl-{self.epoch}.pth",
        )

    def _run_episode(self):
        print("SELF PLAY: ", " VS ".join(self.config.self_play.strategies))
        data = self_play(
            self.game_desc.make_game,
            [
                self.game_desc.strategy_from_string(s, model=self.model)
                for s in self.config.self_play.strategies
            ],
            self.config.self_play.num_games,
            self.config.self_play.max_len,
        )
        data = self.game_desc.make_metrics(flatten(data))
        data.print_short_history()
        data.metrics_to_visdom(self.viz, self.epoch)
        return data

    def train(self):
        print("#parameters", sum(p.numel() for p in self.model.parameters()) / 1e6, "M")
        import random

        trainset_limit = 1000
        full_trainset = []
        for epoch in range(3000):
            self.epoch = epoch

            print("EPOCH", epoch)
            if epoch % self.config.pit.every == 0:
                self._log_pit()

            if epoch % self.config.train.save_every == 0:
                self._save_model()

            data = self._run_episode()
            trainset = to_trainset(data, self.config.train.discount_factor)

            full_trainset = trainset
            # full_trainset = full_trainset[-trainset_limit:]
            copy_trainset = full_trainset.copy()
            random.shuffle(copy_trainset)
            print(len(copy_trainset), "samples")
            self._train_epoch(copy_trainset)
            torch.cuda.empty_cache()


import torch.nn as nn
from boardrl.rl.model import RotarySingle


class StatePredictor(nn.Module):
    def __init__(self, dim, head_size) -> None:
        super().__init__()
        self.norm_hidden = nn.LayerNorm(dim)
        self.emb = nn.Embedding(256, dim, padding_idx=0)
        self.norm_in = nn.Linear(dim, dim)
        self.rotary = RotarySingle(dim, 512)
        self.body = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    dim,
                    dim // head_size,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(1)
            ]
        )
        self.proj = nn.Linear(dim, 256)

    def forward(self, hidden, target):
        hidden = self.norm_hidden(hidden)
        target = self.emb(target)
        target = self.norm_in(target)
        target = self.rotary(target)
        for layer in self.body:
            target = layer(
                target,
                hidden,
                tgt_is_causal=True,
                tgt_mask=nn.Transformer.generate_square_subsequent_mask(
                    target.size(1), device=target.device
                ),
            )
        return self.proj(target)


class PreTrainer:
    def __init__(self, config):
        self.config = config
        self.model = torch.nn.ModuleList(
            [
                Model(**self.config.net),
                StatePredictor(self.config.net.dim, self.config.net.head_size),
            ]
        )
        self.model.to(config.device)
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.train.lr,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )

        self.viz = Visualizer(f"{config.game}_{config.tag}-lr={config.train.lr}")
        self.epoch = 0
        self.game_desc = games_library(config.game)

    def _train_epoch(self, data):
        self.model.train()
        grad_pct = 1 / self.config.train.gradient_epochs
        for grad_ep in range(self.config.train.gradient_epochs):
            indices = torch.randperm(len(data))
            batch_pct = 1 / (len(indices) / self.config.train.batch_size)
            for b_i, batch in enumerate(
                tqdm(
                    chunk(indices, self.config.train.batch_size),
                    desc=f"epoch {self.epoch}",
                )
            ):
                with torch.no_grad():
                    samples = TrainingSample.collate([data[bi] for bi in batch]).to(
                        self.config.device
                    )
                self.opt.zero_grad()
                target = self.model[0].text_encode(
                    [chr(1) + n.state for n in samples.next],
                    2048,
                )
                board_moves = [
                    f"{samples.state[i]}\n{samples.moves[i][samples.action_idx[i]]}"
                    for i in range(len(samples.state))
                ]
                (policy, value), hidden = self.model[0](
                    board_moves,
                    return_hidden=True,
                )
                pred = self.model[1](hidden, target[:, :-1])
                pretrain_loss = nn.functional.cross_entropy(
                    pred.transpose(1, 2), target[:, 1:], ignore_index=0
                )
                value_loss = nn.functional.mse_loss(value.mean, samples.returns)
                loss = pretrain_loss + value_loss
                loss.backward()
                self.opt.step()

                self.viz.viz.text(
                    samples.state[0].replace("\n", "<br>")
                    + "<hr>"
                    + "".join(
                        f'<span style="color:{"green" if correct else "red"}">{chr(int(c)).replace(" ", "_")}</span>'
                        for c, correct in zip(
                            target[0, 1:], pred[0].argmax(-1) == target[0, 1:]
                        )
                    ).replace("\n", "<br>"),
                    win="display",
                )
                self.viz.push(
                    "pretrain loss",
                    pretrain_loss.item(),
                    self.epoch + grad_ep * grad_pct + b_i * batch_pct * grad_pct,
                )
                self.viz.push(
                    "pretrain acc",
                    ((pred.argmax(-1) == target[:, 1:]) & (target[:, 1:] != 0))
                    .float()
                    .sum()
                    .item()
                    / (target[:, 1:] != 0).sum().item(),
                    self.epoch + grad_ep * grad_pct + b_i * batch_pct * grad_pct,
                )
                self.viz.push(
                    "value loss",
                    value_loss.item(),
                    self.epoch + grad_ep * grad_pct + b_i * batch_pct * grad_pct,
                )

    def _run_episode(self):
        print("SELF PLAY: ", " VS ".join(self.config.self_play.strategies))
        data = self_play(
            self.game_desc.make_game,
            [
                self.game_desc.strategy_from_string("random")
                for s in self.config.self_play.strategies
            ],
            self.config.self_play.num_games,
            self.config.self_play.max_len,
        )
        data = self.game_desc.make_metrics(flatten(data))
        return data

    def pretrain(self):
        for epoch in range(1):
            print("EPOCH", epoch)
            self.epoch = epoch

            data = self._run_episode()
            trainset = to_trainset(data, self.config.train.discount_factor)

            print(len(trainset), "samples")
            self._train_epoch(trainset)
        return self.model[0]


def fix_dict(config, key, new_value):
    split = key.split(".", 1)

    if isinstance(config, list):
        split[0] = int(split[0])

    if len(split) == 1:
        assert split[0] in config
        config[split[0]] = yaml.safe_load(new_value)
    else:
        fix_dict(config[split[0]], split[1], new_value)


def main():
    from easydict import EasyDict
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("-x", action="append", default=[])
    opts = parser.parse_args()

    init_seed()
    with open(opts.config_file) as f:
        config = EasyDict(yaml.safe_load(f))

    for config_fix in opts.x:
        fix_dict(config, *config_fix.split("="))

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        print("* - . /!\\ /!\\ CUDA not available, using CPU /!\\ /!\\ . - *")
        config.device = "cpu"

    ckpt = opts.ckpt if opts.ckpt != "None" else None
    if ckpt is None:
        model = PreTrainer(config).pretrain()
        trainer = Trainer(config, ckpt)
        for trainer_model, pretrain_model in zip(
            trainer.model.parameters(), model.parameters()
        ):
            trainer_model.data.copy_(pretrain_model.data)
        trainer.train()
    else:
        Trainer(config, ckpt).train()


if __name__ == "__main__":
    main()
