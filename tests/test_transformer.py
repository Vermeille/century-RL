"""Quick sanity checks for the Transformer module using tiny MLM tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from boardrl.rl.model.transformer import Transformer


class ToyMLM(nn.Module):
    def __init__(self, dim=16, max_len=26, rotary=False):
        super().__init__()
        self.dim = dim
        self.rotary = rotary
        self.embed = nn.Embedding(27, dim)
        if not rotary:
            self.pos = nn.Embedding(max_len, dim)
        self.tr = Transformer(dim, 2, 2, 8, rotary=rotary)
        self.out = nn.Linear(dim, 27)

    def forward(self, x):
        # x: BL
        pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        y = self.embed(x)
        if hasattr(self, "pos"):
            y += self.pos(pos_ids)
        y = self.tr(y, torch.ones_like(x, dtype=torch.bool))
        return self.out(y)  # BLD

    def __str__(self) -> str:
        return f"dim={self.dim}-rotary={self.rotary}"


def transformer_learns_alphabet_mlm(model):
    """Train on the alphabet and ensure every masked letter is recovered.

    A minimal transformer is trained as a masked language model on the
    sequence "a b c ... z". After a few iterations it should predict any
    single masked letter correctly, demonstrating that the transformer
    can learn simple MLM tasks.
    """
    torch.manual_seed(0)
    letters = torch.arange(1, 27)
    seq = letters.unsqueeze(0)
    opt = torch.optim.Adam(model.parameters(), lr=0.015)
    history = []

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

    plt.figure()
    plt.plot(history)
    plt.savefig(f"test_alphabet_mlm_{model}.png")
    with torch.no_grad():
        for i in range(26):
            x = seq.clone()
            x[0, i] = 0
            pred = model(x)[0, i].argmax().item()
            assert pred == letters[i].item()


def test_transformer_learns_alphabet_mlm_lpe():
    transformer_learns_alphabet_mlm(ToyMLM())


def test_transformer_learns_alphabet_mlm_rotary():
    transformer_learns_alphabet_mlm(ToyMLM(dim=16, rotary=True))


def test_transformer_needs_context_mlm():
    """Verify that predictions depend on neighboring context, not only position.

    The model trains on many shifted alphabet sequences. We mask a random
    interior token in each example and allow the model to attend to the masked
    position. Successful learning requires using the surrounding tokens rather
    than relying solely on positional embeddings.
    """
    torch.manual_seed(0)
    sequences = torch.stack([(torch.arange(1, 27) + s) % 26 + 1 for s in range(26)])
    model = ToyMLM(dim=64)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    history = []
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

    plt.figure()
    plt.plot(history)
    plt.savefig(f"test_alphabet_shifted_{model}.png")
    with torch.no_grad():
        for i in range(1, 26):
            x = sequences.clone()
            x[:, i] = 0
            pred = model(x)[:, i].argmax(dim=-1)
            assert torch.all(
                pred == sequences[:, i]
            ), f"Failed at position {i}: {pred} != {sequences[0, i].item()}"


def transformer_positional(model):
    """Set a fixed input and classify the nth position as the nth class.
    This won't work if positional encoding fails.
    """
    torch.manual_seed(0)
    seq = torch.ones((1, 25), dtype=torch.int)
    seq[0, 0] = 0  # RoPE is relative, so we need an absolute position marker
    target = torch.arange(25).unsqueeze(0)
    opt = torch.optim.Adam(model.parameters(), lr=0.015)

    history = []
    for i in range(500):
        x = seq.expand(3, -1).clone()  # Repeat for batch size of 3
        logits = model(x)
        loss = F.cross_entropy(logits.transpose(1, 2), target.expand_as(x))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 10 == 0:
            history.append(loss.item())

    plt.figure()
    plt.plot(history)
    plt.savefig(f"test_positional_{model}.png")
    with torch.no_grad():
        x = seq.clone()
        pred = model(x).argmax(2)
        assert torch.equal(pred, target)


def test_transformer_positional_lpe():
    transformer_positional(ToyMLM())


def test_transformer_positional_rotary():
    transformer_positional(ToyMLM(rotary=True))
