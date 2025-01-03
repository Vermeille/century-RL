from collections import defaultdict
import torch
import time
from visdom import Visdom
from tqdm import tqdm

from boardrl.rl.model import Model
from boardrl.rl.model.loss import loss_from_string
from boardrl.rl.eval.selfplay import self_play, pit
from boardrl.games import games_library


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


def collate(xs):
    if isinstance(xs[0], (int, float)):
        return torch.tensor(xs)
    if isinstance(xs[0], torch.Tensor):
        try:
            return torch.stack(xs, dim=0)
        except RuntimeError:
            return xs
    return xs


def discount(rews):
    d = 0.98
    return sum(d**i * r for i, r in enumerate(rews))


def to_trainset(games_data):
    out = []
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
                    returns=discount(rewards[i:]),
                    current_diff_points=float(log.current_diff_points),
                    next=end if i == len(hist) - 2 else out[-1],
                    final=False,
                )
            )
    return out


def chunk(data, size):
    i = 0
    while i + size < len(data):
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


import copy


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
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=config.train.lr, betas=(0., 0.99))
        self.policy_loss = loss_from_string(config.train.loss.policy, model=self.model)
        self.value_loss = loss_from_string(config.train.loss.value, model=self.model)
        self.viz = Visualizer(f"{config.tag}-lr={config.train.lr}")
        self.epoch = 0
        self.game_desc = games_library("tictactoe")()

        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path)
            self.model.load_state_dict(ckpt["model"])
            self.opt.load_state_dict(ckpt["opt"])

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
            batch_pct = 1 / (len(indices) // self.config.train.batch_size)
            for b_i, batch in enumerate(
                tqdm(
                    chunk(indices, self.config.train.batch_size),
                    desc=f"epoch {self.epoch}",
                )
            ):
                with torch.no_grad():
                    samples = TrainingSample.collate(
                        [copy.deepcopy(data[bi]) for bi in batch]
                    ).to(self.config.device)
                self.opt.zero_grad()
                total_losses = defaultdict(float)
                policy, value = self.model(samples.state, samples)
                policy_loss = self.policy_loss(policy, value, samples)
                value_loss = self.value_loss(policy, value, samples)
                loss = policy_loss + value_loss
                loss.backward()
                total_losses["policy"] += policy_loss.item() / len(data) * len(batch)
                total_losses["value"] += value_loss.item() / len(data) * len(batch)

                grad_mag = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=5.0
                )
                self.opt.step()

                print(total_losses)
                for k, v in total_losses.items():
                    self.viz.push(
                        f"loss.{k}",
                        v,
                        self.epoch + grad_ep * grad_pct + b_i * batch_pct * grad_pct,
                    )
                self.viz.push(
                    "grad_mag",
                    grad_mag.item(),
                    self.epoch + grad_ep * grad_pct + b_i * batch_pct * grad_pct,
                )
        print(
            "throughput",
            len(data) * self.config.train.gradient_epochs / (time.time() - now),
        )
        print()

    def _save_model(self):
        torch.save(
            {
                "model": self.model.state_dict(),
                "opt": self.opt.state_dict(),
                "epoch": self.epoch,
                "config": self.config.net,
            },
            f"rl-{self.epoch}.pth",
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
        for epoch in range(3000):
            self.epoch = epoch

            print("EPOCH", epoch)
            if epoch % self.config.pit.every == 0:
                self._log_pit()

            if epoch % self.config.train.save_every == 0:
                self._save_model()

            data = self._run_episode()
            trainset = to_trainset(data)

            print(len(trainset), "samples")
            self._train_epoch(trainset)


def main():
    import sys
    import yaml
    from easydict import EasyDict

    with open(sys.argv[1]) as f:
        config = EasyDict(yaml.safe_load(f))
    Trainer(config, sys.argv[2] if len(sys.argv) > 2 else None).train()


if __name__ == "__main__":
    main()
