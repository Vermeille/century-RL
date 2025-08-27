"""Training routines for the full RL model used in tests and experiments."""

from __future__ import annotations

import random
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torch.distributions as dist

from boardrl.rl.model.model import Model


def model_learns_policy_and_value(model: Model) -> List[float]:
    """Overfit the entire model on a trivial set of text games."""

    dist.Distribution.set_default_validate_args(False)
    torch.manual_seed(0)

    games = ["a@0b@1", "b@0c@1", "c@0d@1", "d@0e@1"]
    targets = torch.tensor([0, 1, 0, 1])
    values = torch.tensor([1.0, -1.0, 0.5, -0.5])

    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    history: List[float] = []
    for i in range(300):
        out = model(games)
        logits = torch.stack(out.policy)
        loss_policy = F.cross_entropy(logits, targets)
        loss_value = F.mse_loss(out.value.mean, values)
        (loss_policy + loss_value).backward()
        opt.step()
        opt.zero_grad()
        if i % 10 == 0:
            history.append((loss_policy + loss_value).item())

    with torch.no_grad():
        out = model(games)
        preds = torch.stack(out.policy).argmax(dim=1)
        assert torch.equal(preds, targets)
        assert torch.allclose(out.value.mean, values, atol=0.1)

    return history


def model_learns_from_context(model: Model) -> List[float]:
    """Model must use the surrounding letters to choose the correct action."""

    dist.Distribution.set_default_validate_args(False)
    torch.manual_seed(0)

    alphabet = list("abcdefghijklmnopqrstuvwxyz")
    vowels = set("aeiou")
    games, targets, values = [], [], []
    for shift in range(6):
        seq = alphabet[shift : shift + 6]
        pos0 = shift % 5
        pos1 = (shift * 2 + 1) % 5
        letters = seq.copy()
        if pos0 < pos1:
            letters.insert(pos0 + 1, "@0")
            letters.insert(pos1 + 2, "@1")
        else:
            letters.insert(pos1 + 1, "@1")
            letters.insert(pos0 + 2, "@0")
        games.append("".join(letters))
        targets.append(0 if seq[pos0] in vowels else 1)
        values.append(1.0 if targets[-1] == 0 else -1.0)

    targets_t = torch.tensor(targets)
    values_t = torch.tensor(values)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)

    history: List[float] = []
    for i in range(500):
        out = model(games)
        logits = torch.stack(out.policy)
        loss_policy = F.cross_entropy(logits, targets_t)
        loss_value = F.mse_loss(out.value.mean, values_t)
        (loss_policy + loss_value).backward()
        opt.step()
        opt.zero_grad()
        if i % 10 == 0:
            history.append((loss_policy + loss_value).item())

    with torch.no_grad():
        out = model(games)
        preds = torch.stack(out.policy).argmax(dim=1)
        assert torch.equal(preds, targets_t)
        assert torch.allclose(out.value.mean, values_t, atol=0.1)

    return history


# Dyck utilities -----------------------------------------------------------

def _random_balanced_string(max_len: int, max_depth: int) -> str:
    if max_len % 2 == 1:
        max_len -= 1
    seq: List[str] = []
    depth = 0
    while len(seq) < max_len:
        open_possible = depth < max_depth and len(seq) < max_len - (depth + 1)
        close_possible = depth > 0
        if not open_possible and not close_possible:
            break
        if open_possible and close_possible:
            p_open = 0.5 * (max_depth - depth) / max_depth
            step = "(" if random.random() < p_open else ")"
        elif open_possible:
            step = "("
        else:
            step = ")"
        seq.append(step)
        depth += 1 if step == "(" else -1
    seq.extend([")"] * depth)
    return "".join(seq)


def generate_dyck_example(
    max_len: int = 128, max_depth: int = 12, pad_to_len: bool = True
) -> Tuple[str, int, int]:
    dyck = _random_balanced_string(max_len, max_depth)
    q_pos = random.randint(0, len(dyck))
    depth = 0
    for tok in dyck[:q_pos]:
        depth += 1 if tok == "(" else -1
    tokens = dyck[:q_pos] + "Q" + dyck[q_pos:]
    if pad_to_len:
        pad_len = (max_len + 1) - len(tokens)
        tokens += " " * max(0, pad_len)
    return tokens, q_pos, depth


def _generate_dyck_examples(bs: int) -> Tuple[List[str], torch.Tensor]:
    xy = [generate_dyck_example(32, max_depth=10, pad_to_len=False) for _ in range(bs)]
    x = [x + "\n" + "".join(f"@{i}\n" for i in range(11)) for x, _, _ in xy]
    y = [y for _, _, y in xy]
    return x, torch.tensor(y)


def model_dyck(model: Model) -> List[float]:
    """Run Dyck query test."""

    torch.manual_seed(0)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0001)

    history: List[float] = []
    for i in range(10000):
        x, y = _generate_dyck_examples(64)
        out = model(x)
        logits = torch.stack(out.policy)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 10 == 0:
            history.append(logits.argmax(1).eq(y).float().mean().item())

    with torch.no_grad():
        x, y = _generate_dyck_examples(4)
        out = model(x)
        pred = torch.stack(out.policy).argmax(1)
        assert torch.equal(pred, y)

    return history


# Arithmetic expression utilities -------------------------------------------

def _random_arith_expr(max_depth: int) -> Tuple[str, int]:
    if max_depth <= 0 or random.random() < 0.5:
        n = random.randint(0, 19)
        return str(n), n % 20
    left_s, left_v = _random_arith_expr(max_depth - 1)
    right_s, right_v = _random_arith_expr(max_depth - 1)
    op = random.choice(["+", "*"])
    if op == "+":
        val = (left_v + right_v) % 20
    else:
        val = (left_v * right_v) % 20
    return f"({left_s}{op}{right_s})", val


def generate_arith_example(max_depth: int = 3) -> Tuple[str, int]:
    expr, val = _random_arith_expr(max_depth)
    return expr, val


def _generate_arith_examples(bs: int, max_depth: int) -> Tuple[List[str], torch.Tensor]:
    xy = [generate_arith_example(max_depth) for _ in range(bs)]
    x = [expr + "\n" + "".join(f"@{i}\n" for i in range(20)) for expr, _ in xy]
    y = torch.tensor([val for _, val in xy])
    return x, y


def model_arith_mod20(
    model: Model, max_depth: int = 3, num_iters: int = 10000, bs: int = 64
) -> List[float]:
    """Run modular arithmetic evaluation experiment."""

    torch.manual_seed(0)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0001)

    history: List[float] = []
    for i in range(num_iters):
        x, y = _generate_arith_examples(bs, max_depth)
        out = model(x)
        logits = torch.stack(out.policy)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 10 == 0:
            history.append(logits.argmax(1).eq(y).float().mean().item())

    with torch.no_grad():
        x, y = _generate_arith_examples(4, max_depth)
        out = model(x)
        pred = torch.stack(out.policy).argmax(1)
        assert torch.equal(pred, y)

    return history
