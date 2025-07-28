"""Quick sanity checks for the Transformer module using tiny MLM tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from boardrl.rl.model.transformer import Transformer

class ToyMLM(nn.Module):
    def __init__(self, dim=16, max_len=26):
        super().__init__()
        self.embed = nn.Embedding(27, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.tr = Transformer(dim, 1, 2, 8)
        self.out = nn.Linear(dim, 27)

    def forward(self, x, attend_masked=False):
        if attend_masked:
            mask = torch.ones_like(x, dtype=torch.bool)
        else:
            mask = x != 0
        pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        y = self.embed(x) + self.pos(pos_ids)
        y = self.tr(y, mask)
        return self.out(y)

def test_transformer_learns_alphabet_mlm():
    """Train on the alphabet and ensure every masked letter is recovered.

    A minimal transformer is trained as a masked language model on the
    sequence "a b c ... z". After a few iterations it should predict any
    single masked letter correctly, demonstrating that the transformer
    can learn simple MLM tasks.
    """
    torch.manual_seed(0)
    letters = torch.arange(1, 27)
    seq = letters.unsqueeze(0)
    model = ToyMLM()
    opt = torch.optim.Adam(model.parameters(), lr=0.15)

    for _ in range(150):
        x = seq.clone()
        mask_idx = torch.randint(0, 26, (3,))
        target = x[0, mask_idx]
        x[0, mask_idx] = 0
        logits = model(x)
        loss = F.cross_entropy(logits[0, mask_idx], target)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        for i in range(26):
            x = seq.clone()
            x[0, i] = 0
            pred = model(x)[0, i].argmax().item()
            assert pred == letters[i].item()


def test_transformer_needs_context_mlm():
    """Verify that predictions depend on neighboring context, not only position.

    The model trains on many shifted alphabet sequences. We mask a random
    interior token in each example and allow the model to attend to the masked
    position. Successful learning requires using the surrounding tokens rather
    than relying solely on positional embeddings.
    """
    torch.manual_seed(0)
    seq_len = 5
    sequences = torch.stack(
        [(torch.arange(s, s + seq_len) % 26 + 1) for s in range(26)]
    )
    model = ToyMLM(max_len=seq_len)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)

    for _ in range(800):
        x = sequences.clone()
        mask_idx = torch.randint(1, seq_len - 1, (26,))
        target = x[torch.arange(26), mask_idx]
        x[torch.arange(26), mask_idx] = 0
        logits = model(x, attend_masked=True)
        loss = F.cross_entropy(logits[torch.arange(26), mask_idx], target)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        for start in range(26):
            base = torch.arange(start, start + seq_len) % 26 + 1
            seq = base.unsqueeze(0)
            for i in range(1, seq_len - 1):
                x = seq.clone()
                x[0, i] = 0
                pred = model(x, attend_masked=True)[0, i].argmax().item()
                assert pred == base[i].item()

