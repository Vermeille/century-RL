import random
import time
import torch

def fast_sample(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.ndim == 2:
        x = x[0]
    total = float(x.sum())
    if total <= 0:
        raise AssertionError(
            "Should not reach here. Called fast_sample on an invalid distribution (all zeros or negative values)"
        )
    r = random.random() * total
    acc = 0.0
    for i, val in enumerate(x.tolist()):
        acc += float(val)
        if acc >= r:
            return i
    return x.numel() - 1

def init_seed():
    random.seed(int(time.time()))
