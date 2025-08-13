import random
from tqdm import tqdm
import torch
import torch.nn as nn
from heavyball import ForeachMuon

from boardrl.rl.model import Model, RotarySingle
from boardrl.rl.eval.selfplay import self_play, SelfPlayResults
from boardrl.games import games_library
from boardrl.training.returns import compute_returns
from boardrl.training import TrainingSample
from boardrl.utils import Visualizer


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
    return out


def chunk(data, size):
    i = 0
    while i < len(data):
        yield data[i : i + size]
        i += size


class DynamicTanh(nn.Module):
    """Normalization layer replacing LayerNorm."""

    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        return self.weight * torch.tanh(self.alpha * x) + self.bias


class StatePredictor(nn.Module):
    def __init__(self, dim, head_size) -> None:
        super().__init__()
        self.norm_hidden = DynamicTanh(dim)
        self.emb = nn.Embedding(256, dim, padding_idx=0)
        self.norm_in = nn.Linear(dim, dim)
        self.rotary = RotarySingle(dim)
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
                Model(**self.config.net.__dict__),
                StatePredictor(self.config.net.dim, self.config.net.head_size),
            ]
        )
        self.model.to(config.device)
        self.opt = make_optimizer(self.model.parameters(), config.train)

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
                )
                grad_mag = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=5.0
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
            rotate=self.config.self_play.rotate,
        )
        return data

    def pretrain(self):
        for epoch in range(0):
            print("EPOCH", epoch)
            self.epoch = epoch

            data = self._run_episode()
            compute_returns(data, self.config.train.discount_factor)
            trainset = to_trainset(data)

            print(len(trainset), "samples")
            self._train_epoch(trainset)
        return self.model[0]
