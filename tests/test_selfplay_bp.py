import asyncio
import os
import sys
import types
import torch
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

if "pyximport" not in sys.modules:
    pyx = types.ModuleType("pyximport")
    pyx.install = lambda *a, **k: None
    sys.modules["pyximport"] = pyx
if "boardrl.cyutils" not in sys.modules:
    cyutils = types.ModuleType("boardrl.cyutils")
    cyutils.fast_sample = lambda x: int(torch.multinomial(x.float(), 1).item()) if x.dim()==1 else int(torch.multinomial(x[0].float(), 1).item())
    sys.modules["boardrl.cyutils"] = cyutils

from boardrl.games.sum.game import Sum
from boardrl.games.strategies import ArgmaxStrategy
from boardrl.rl.eval.selfplay import self_play
from boardrl.utils import BatchProcessor
from boardrl.rl.model.model import PolicyValue

class DummyOut:
    def __init__(self, outs):
        self._outs = outs
    def unbatched(self):
        return self._outs

def process_fn(batch):
    outs = []
    for disp in batch:
        moves = [line[1:] for line in disp.splitlines() if line.startswith("@")]
        logits = torch.zeros(len(moves))
        outs.append(PolicyValue([logits], torch.distributions.Normal(torch.zeros(1), torch.ones(1))))
    return DummyOut(outs)

@pytest.mark.timeout(10)
def test_self_play_two_batch_processors():
    bp1 = BatchProcessor(batch_size=2, process_fn=process_fn, timeout=0.1)
    bp2 = BatchProcessor(batch_size=2, process_fn=process_fn, timeout=0.1)
    strat1 = ArgmaxStrategy(bp1)
    strat2 = ArgmaxStrategy(bp2)
    games = self_play(Sum, [strat1, strat2], n_games=2, max_len=5)
    assert len(games) == 2
    for g in games:
        assert len(g[0]) > 0 and len(g[1]) > 0
