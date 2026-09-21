import torch
from typing import Protocol
from torch._dynamo.decorators import skip
from boardrl.utils.batchprocessor import BatchProcessor, run_tasks
from boardrl.utils.registerbyname import RegisterByName, parse_spec


__all__ = [
    "BatchProcessor",
    "run_tasks",
    "RegisterByName",
    "parse_spec",
    "entropy",
    "Game",
]


def chunk(data, size, skip_last=False):
    if size <= 0:
        raise ValueError("chunk size must be positive")

    for i in range(0, len(data), size):
        batch = data[i : i + size]
        if skip_last and len(batch) < size:
            continue
        yield batch


# @torch.jit.script
def entropy(logits, dim: int):
    log_probs = torch.log_softmax(logits, dim=dim)
    return -torch.sum(torch.exp(log_probs) * log_probs, dim=dim)


class Game(Protocol):
    moves: list[str]
    num_players: int

    def current_player(self) -> int: ...

    def display_with_moves(self) -> str: ...

    def display(self, force: int = -1) -> str: ...

    def copy(self, *, randomize: bool = False) -> "Game": ...

    def round(self) -> int: ...

    def play_str(self, move: str): ...

    def play_idx(self, move: int): ...

    def simulate_to_end(self): ...

    def diff_points_for(self, player: int) -> float: ...

    def diff_points(self) -> float: ...

    def points_for(self, player: int) -> float: ...

    def points(self) -> float: ...

    def ended(self) -> bool: ...
