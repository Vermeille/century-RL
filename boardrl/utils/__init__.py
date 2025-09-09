import torch
from boardrl.utils.batchprocessor import BatchProcessor, run_tasks, CachedBatchProcessor
from boardrl.utils.modelpool import ModelPool
from boardrl.utils.registerbyname import RegisterByName
from boardrl.utils.visualizer import Visualizer
from boardrl.utils.pythonexec import PythonExec


__all__ = [
    "BatchProcessor",
    "run_tasks",
    "CachedBatchProcessor",
    "ModelPool",
    "RegisterByName",
    "entropy",
    "Game",
    "PythonExec",
    "Visualizer",
]


#@torch.jit.script
def entropy(logits, dim: int):
    log_probs = torch.log_softmax(logits, dim=dim)
    return -torch.sum(torch.exp(log_probs) * log_probs, dim=dim)


class Game:
    moves: list[str]
    num_players: int

    def current_player(self) -> int: ...

    def display_with_moves(self) -> str: ...

    def copy(self) -> "Game": ...

    def play_str(self, move: str): ...

    def play_idx(self, move: int): ...

    def simulate_to_end(self): ...

    def diff_points_for(self, player: int) -> float: ...

    def ended(self) -> bool: ...

    ...
