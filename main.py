from collections import defaultdict
import torch
import time
import copy
import os
import yaml
from visdom import Visdom
from tqdm import tqdm
from heavyball import ForeachMuon

from boardrl.rl.model import Model
from boardrl.rl.model.loss import loss_from_string
from boardrl.rl.utils import pearson_corr
from boardrl.rl.eval.selfplay import self_play, pit
from boardrl.games import games_library
from boardrl.cyutils import init_seed
from boardrl.utils import BatchProcessor, easydict_to_dict, Visualizer


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
    return sum(discount_factor**i * r.reward for i, r in enumerate(rews))


def compute_returns(
    games,
    discount_factor,
    *,
    entropy_reward_scale: float | None = None,
    reward_rescale: float | None = None,
):
    def rescale(history, scale):
        for log in history:
            log.current_diff_points *= scale

    def set_next(history):
        for i, log in enumerate(history[:-1]):
            log.next = history[i + 1]

    def set_rewards(history):
        history[-1].reward = 0
        for i in range(len(history) - 1):
            history[i].reward = (
                history[i + 1].current_diff_points - history[i].current_diff_points
            )

    def entropy_reward(history, strength):
        for log in history[:-1]:
            log.reward += (
                strength
                * -torch.log_softmax(log.action_distribution, dim=0)[log.action_idx]
            )

    def set_returns(history):
        if history[-1].final:
            for i in range(len(history) - 1):
                history[i].returns = discount(history[i:], discount_factor)
        else:
            for i in range(len(history) - 1):
                history[i].returns = float("nan")

    def set_score(history):
        if history[-1].final:
            for i in range(len(history)):
                history[i].score = history[-1].current_diff_points
        else:
            for i in range(len(history)):
                history[i].score = float("nan")

    for game in games:
        for history in game:
            if len(history) == 0:
                continue
            if reward_rescale is not None:
                rescale(history, reward_rescale)
            set_next(history)
            set_rewards(history)
            if entropy_reward_scale is not None:
                entropy_reward(history, entropy_reward_scale)
            set_returns(history)
            set_score(history)


def to_trainset(games_data, only_players: list[int] | None = None):
    out = []
    for game in games_data:
        for player_id, hist in enumerate(game):
            if only_players is not None and player_id not in only_players:
                continue

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
        self.model = Model(**self.config.net)
        self.model.to(config.device)
        if self.config.train.optimizer == "AdamW":
            self.opt = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.train.lr,
                betas=(0.9, 0.99),
                weight_decay=0.01,
            )
        elif self.config.train.optimizer == "Muon":
            self.opt = ForeachMuon(
                self.model.parameters(),
                lr=config.train.lr,
                betas=(0.9, 0.99),
                weight_decay=0.01,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.train.optimizer}")

        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path)
            self.model.load_state_dict(ckpt["model"])
            self.opt.load_state_dict(ckpt["opt"])

        self.prev_model = copy.deepcopy(self.model)
        self.policy_loss = loss_from_string(
            config.train.loss.policy,
            model=self.model,
            prev_model=self.prev_model,
            discount_factor=config.train.discount_factor,
        )
        self.value_loss = loss_from_string(
            config.train.loss.value,
            model=self.model,
            prev_model=self.prev_model,
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
            "<pre>\n" + yaml.dump(easydict_to_dict(config)) + "</pre>",
        )
        self.epoch = 0
        self.game_desc = games_library(config.game)
        self.game_name = config.game.split(",")[0]

    def _log_pit(self):
        self.model.eval()
        bp = BatchProcessor(
            self.config.pit.get("batch_size", self.config.train.batch_size),
            self.model,
            timeout=0.01,
        )
        print("PIT: ", " VS ".join(self.config.pit.strategies))
        pit_results = pit(
            self.game_desc.make_game,
            [
                self.game_desc.strategy_from_string(
                    s, model=bp, discount_factor=self.config.train.discount_factor
                )
                for s in self.config.pit.strategies
            ],
            self.config.pit.num_games,
            self.config.pit.max_len,
        )
        compute_returns(pit_results.games, self.config.train.discount_factor)
        self.game_desc.make_metrics(pit_results.games).print_short_history()
        self.viz.push("pit.win_rate", pit_results.win_rate(0), self.epoch)
        self.viz.push("pit.avg_points", pit_results.my_avg_points(0), self.epoch)
        self.model.train()

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
                    self.model.parameters(), max_norm=50000.0
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

        with torch.no_grad():
            for prev_param, param in zip(
                self.prev_model.state_dict().values(), self.model.state_dict().values()
            ):
                prev_param.data.copy_(param.data)

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
                samples = TrainingSample.collate(batch).to(self.config.device)
            policy, value = self.model(samples.state)
            policy_loss = self.policy_loss(policy, value, samples)
            value_loss = self.value_loss(policy, value, samples)
            loss = policy_loss + value_loss
            loss = loss * len(samples.state)
            loss.backward()
            total_losses["loss_policy"] += policy_loss.item()
            total_losses["loss_value"] += value_loss.item()

            print(policy[0].shape)
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
        grad_mag = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        total_losses["grad_mag"] += grad_mag.item()
        self.opt.step()

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

        with torch.no_grad():
            for prev_param, param in zip(
                self.prev_model.state_dict().values(), self.model.state_dict().values()
            ):
                prev_param.data.copy_(param.data)

    def _save_model(self):
        import os

        os.makedirs(f"{self.game_name}-ckpt", exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "opt": self.opt.state_dict(),
                "epoch": self.epoch,
                "config": self.config.net,
            },
            f"{self.game_name}-ckpt/rl-{self.epoch}.pth",
        )

    def _run_episode(self):
        print("SELF PLAY: ", " VS ".join(self.config.self_play.strategies))

        self.model.eval()
        bp = BatchProcessor(
            self.config.self_play.get("batch_size", self.config.train.batch_size),
            self.model,
            timeout=0.02,
        )
        data = self_play(
            self.game_desc.make_game,
            [
                self.game_desc.strategy_from_string(
                    s, model=bp, discount_factor=self.config.train.discount_factor
                )
                for s in self.config.self_play.strategies
            ],
            self.config.self_play.num_games,
            self.config.self_play.max_len,
        )
        compute_returns(
            data,
            self.config.train.discount_factor,
            # 1 - 1 / (1 + self.epoch * 0.1),
            entropy_reward_scale=self.config.train.get("entropy_reward_scale"),
            reward_rescale=self.config.train.get("reward_rescale"),
        )
        metrics = self.game_desc.make_metrics(data)
        metrics.print_short_history()
        metrics.metrics_to_visdom(self.viz, self.epoch)
        avg_reward = [
            sum(h.reward for players in data for h in players[p])
            / sum(len(players[0]) for players in data)
            for p in range(len(data[0]))
        ]
        self.viz.push("avg_reward", avg_reward, self.epoch)
        self.model.train()
        return data

    def train(self):
        print("#parameters", sum(p.numel() for p in self.model.parameters()) / 1e6, "M")

        epoch = 0
        while True:
            self.epoch = epoch

            print("EPOCH", epoch)
            if epoch % self.config.pit.every == 0:
                self._log_pit()

            if epoch % self.config.train.save_every == 0:
                self._save_model()

            data = self._run_episode()
            trainset = to_trainset(
                data, only_players=self.config.train.get("only_players")
            )

            print(len(trainset), "samples")
            if (
                self.policy_loss.supports_off_policy
                and self.value_loss.supports_off_policy
            ):
                self._train_epoch_off_policy(trainset)
            else:
                self._train_epoch_on_policy(trainset)
            torch.cuda.empty_cache()
            epoch += 1


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
        if self.config.train.optimizer == "AdamW":
            self.opt = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.train.lr,
                betas=(0.9, 0.95),
                weight_decay=0.01,
            )
        elif self.config.train.optimizer == "Muon":
            self.opt = ForeachMuon(
                self.model.parameters(),
                lr=config.train.lr,
                betas=(0.9, 0.95),
                weight_decay=0.01,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.train.optimizer}")

        self.viz = Visualizer(
            f"{config.game}_{config.tag}-lr={config.train.lr}",
            url=config.visdom_url,
            port=config.visdom_port,
        )
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
                board_moves = self.model[0].text_encode(
                    [
                        f"{chr(1)}{samples.moves[i][samples.action_idx[i]]}\n{samples.next[i].state}"
                        for i in range(len(samples.state))
                    ],
                    2048,
                )
                (policy, value), hidden = self.model[0](
                    samples.state,
                    return_hidden=True,
                )
                pred = self.model[1](hidden, board_moves[:, :-1])
                pretrain_loss = nn.functional.cross_entropy(
                    pred.transpose(1, 2), board_moves[:, 1:], ignore_index=0
                )
                value_loss = nn.functional.mse_loss(value.mean, samples.returns)
                loss = pretrain_loss + value_loss
                loss.backward()
                self.opt.step()

                self.viz.html(
                    "display",
                    samples.state[0].replace("\n", "<br>")
                    + "<hr>"
                    + "".join(
                        f'<span style="color:{"green" if correct else "red"}">{chr(int(c)).replace(" ", "_")}</span>'
                        for c, correct in zip(
                            board_moves[0, 1:], pred[0].argmax(-1) == board_moves[0, 1:]
                        )
                    ).replace("\n", "<br>"),
                    win="display",
                )
                grad_mag = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=50000.0
                )
                self.viz.push(
                    "grad_mag",
                    grad_mag.item(),
                    self.epoch + grad_ep * grad_pct + b_i * batch_pct * grad_pct,
                )
                self.viz.push(
                    "pretrain loss",
                    pretrain_loss.item(),
                    self.epoch + grad_ep * grad_pct + b_i * batch_pct * grad_pct,
                )
                self.viz.push(
                    "pretrain acc",
                    (
                        (pred.argmax(-1) == board_moves[:, 1:])
                        & (board_moves[:, 1:] != 0)
                    )
                    .float()
                    .sum()
                    .item()
                    / (board_moves[:, 1:] != 0).sum().item(),
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
        return data

    def pretrain(self):
        for epoch in range(5):
            print("EPOCH", epoch)
            self.epoch = epoch

            data = self._run_episode()
            compute_returns(data, self.config.train.discount_factor)
            trainset = to_trainset(data)

            print(len(trainset), "samples")
            self._train_epoch(trainset)
        return self.model[0]


def fix_dict(config, key, new_value):
    split = key.split(".", 1)

    if isinstance(config, list):
        split[0] = int(split[0])

    if len(split) == 1:
        if isinstance(config, dict):
            assert split[0] in config
        elif isinstance(config, list):
            assert split[0] < len(config)
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
    parser.add_argument("--visdom-url", default="http://localhost")
    parser.add_argument("--visdom-port", default=8097)
    opts = parser.parse_args()

    init_seed()
    with open(opts.config_file) as f:
        config = EasyDict(yaml.safe_load(f))

    for config_fix in opts.x:
        fix_dict(config, *config_fix.split("=", 1))
    config.visdom_url = config.get("visdom_url", opts.visdom_url)
    config.visdom_port = config.get("visdom_port", opts.visdom_port)

    if "model" in config:
        with open(
            os.path.join(
                os.path.dirname(__file__), "model-configs", f"{config.model}.yaml"
            )
        ) as f:
            config.net = EasyDict(yaml.safe_load(f))
    else:
        raise ValueError("config must specify 'model'")

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
    torch.set_float32_matmul_precision("medium")
    main()
