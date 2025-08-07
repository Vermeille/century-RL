"""Toy transformer training tasks used in tests and experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from boardrl.rl.model.transformer import Transformer


class ToyMLM(nn.Module):
    """Minimal transformer-based masked language model."""

    def __init__(self, dim: int = 16, max_len: int = 26, rotary: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.rotary = rotary
        self.embed = nn.Embedding(27, dim)
        if not rotary:
            self.pos = nn.Embedding(max_len, dim)
        self.tr = Transformer(dim, 2, 2, 8, rotary=rotary)
        self.out = nn.Linear(dim, 27)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: BL
        pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        y = self.embed(x)
        if hasattr(self, "pos"):
            y += self.pos(pos_ids)
        y = self.tr(y, torch.ones_like(x, dtype=torch.bool))
        return self.out(y)  # BLD

    def __str__(self) -> str:
        return f"dim={self.dim}-rotary={self.rotary}"


def transformer_learns_alphabet_mlm(model: ToyMLM) -> list[float]:
    """Train on the alphabet and ensure every masked letter is recovered."""

    torch.manual_seed(0)
    letters = torch.arange(1, 27)
    seq = letters.unsqueeze(0)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    history: list[float] = []

    for i in range(1000):
        x = seq.expand(3, -1).clone()  # Repeat for batch size of 3
        mask_idx = torch.randint(0, 26, (3,))
        target = x[torch.arange(3), mask_idx].clone()
        x[torch.arange(3), mask_idx] = 0
        logits = model(x)[torch.arange(x.shape[0]), mask_idx]
        loss = F.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 10 == 0:
            history.append(loss.item())

    with torch.no_grad():
        for i in range(26):
            x = seq.clone()
            x[0, i] = 0
            pred = model(x)[0, i].argmax().item()
            assert pred == letters[i].item()

    return history


def transformer_needs_context_mlm(model: ToyMLM) -> list[float]:
    """Verify that predictions depend on neighboring context, not only position."""

    torch.manual_seed(0)
    sequences = torch.stack([(torch.arange(1, 27) + s) % 26 + 1 for s in range(26)])
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    history: list[float] = []
    for i in range(1500):
        x = sequences.clone()  # 26 x 26
        mask_idx = torch.randint(0, 26, (26,))
        target = x[torch.arange(26), mask_idx].clone()
        x[torch.arange(26), mask_idx] = 0
        logits = model(x)
        masked_logits = logits[torch.arange(26), mask_idx]
        loss = F.cross_entropy(masked_logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 10 == 0:
            history.append(loss.item())

    with torch.no_grad():
        for i in range(1, 26):
            x = sequences.clone()
            x[:, i] = 0
            pred = model(x)[:, i].argmax(dim=-1)
            assert torch.all(pred == sequences[:, i]), f"Failed at position {i}: {pred} != {sequences[0, i].item()}"

    return history


def transformer_positional(model: ToyMLM) -> list[float]:
    """Set a fixed input and classify the nth position as the nth class."""

    torch.manual_seed(0)
    seq = torch.ones((1, 25), dtype=torch.int)
    seq[0, 0] = 0  # RoPE is relative, so we need an absolute position marker
    target = torch.arange(25).unsqueeze(0)
    opt = torch.optim.Adam(model.parameters(), lr=0.015)

    history: list[float] = []
    for i in range(500):
        x = seq.expand(3, -1).clone()  # Repeat for batch size of 3
        logits = model(x)
        loss = F.cross_entropy(logits.transpose(1, 2), target.expand_as(x))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 10 == 0:
            history.append(loss.item())

    with torch.no_grad():
        x = seq.clone()
        pred = model(x).argmax(2)
        assert torch.equal(pred, target)

    return history
