"""Small cross-cutting helpers shared by executable training runs."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from boardrl.metrics import Trackio, make_trackio


class TextSink(Protocol):
    def text(self, name: str, value: str) -> None: ...


def _git(cwd: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.rstrip()


@dataclass(frozen=True)
class RunInfo:
    """Arguments and exact repository state needed to identify a run."""

    entrypoint: Path
    arguments: Mapping[str, object]
    git_commit: str | None
    git_status: str
    git_diff: str

    @classmethod
    def capture(
        cls,
        arguments: argparse.Namespace,
        entrypoint: str | Path | None = None,
    ) -> RunInfo:
        path = Path(entrypoint or sys.argv[0]).resolve()
        root_text = _git(path.parent, "rev-parse", "--show-toplevel")
        if root_text is None:
            return cls(path, dict(vars(arguments)), None, "", "")

        root = Path(root_text)
        return cls(
            path,
            dict(vars(arguments)),
            _git(root, "rev-parse", "HEAD"),
            _git(root, "status", "--short") or "",
            _git(root, "diff", "--binary", "HEAD") or "",
        )

    @property
    def text(self) -> str:
        arguments = json.dumps(self.arguments, indent=2, sort_keys=True, default=str)
        commit = self.git_commit or "(not a git checkout)"
        status = self.git_status or ("(clean)" if self.git_commit else "(unavailable)")
        diff = self.git_diff or ("(clean)" if self.git_commit else "(unavailable)")
        return (
            f"Entrypoint\n==========\n{self.entrypoint}\n\n"
            f"Arguments\n=========\n{arguments}\n\n"
            f"Git commit\n==========\n{commit}\n\n"
            f"Git status\n==========\n{status}\n\n"
            f"Dirty diff\n==========\n{diff}\n"
        )

    def publish(self, sink: TextSink, *, name: str = "run") -> None:
        sink.text(name, self.text)

    def save(self, directory: str | Path, *, name: str = "run.txt") -> Path:
        path = Path(directory) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.text)
        return path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    from boardrl.cyutils import init_seed

    init_seed(seed)


@contextmanager
def trackio_run(
    *,
    project: str | None,
    name: str | None = None,
    server_url: str | None = None,
    config: Mapping[str, object] | None = None,
    factory: Callable[..., Trackio | None] = make_trackio,
) -> Iterator[Trackio | None]:
    """Create and reliably finish an optional Trackio run."""

    sink = factory(
        project=project,
        name=name,
        server_url=server_url,
        config=config,
    )
    try:
        yield sink
    finally:
        if sink is not None:
            sink.finish()
