import sys
import importlib
import pyximport
import torch
import pytest

# Remove stubbed module from other tests
if 'boardrl.cyutils' in sys.modules:
    del sys.modules['boardrl.cyutils']

pyximport.install()
cyutils = importlib.import_module('boardrl.cyutils')
fast_sample = cyutils.fast_sample


def test_fast_sample_single_index():
    x = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    for _ in range(5):
        assert fast_sample(x) == 1


def test_fast_sample_accepts_2d():
    x = torch.tensor([[0.0, 0.0, 5.0]], dtype=torch.float32)
    assert fast_sample(x) == 2


def test_fast_sample_distribution():
    dist = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float32)
    counts = [0, 0, 0]
    n = 3000
    for _ in range(n):
        counts[fast_sample(dist)] += 1
    probs = [c / n for c in counts]
    assert pytest.approx(probs[0], rel=0.1) == 0.1
    assert pytest.approx(probs[1], rel=0.1) == 0.2
    assert pytest.approx(probs[2], rel=0.1) == 0.7
