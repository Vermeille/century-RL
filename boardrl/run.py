"""Capture and publish the reproducible description of a run."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class TextSink(Protocol):
    def text(self, name: str, value: str) -> None: ...


@dataclass(frozen=True)
class RunInfo:
    executable: Path
    arguments: Mapping[str, object]
    code: str

    @classmethod
    def capture(
        cls,
        arguments: argparse.Namespace,
        executable: str | Path | None = None,
    ) -> RunInfo:
        path = Path(executable or sys.argv[0]).resolve()
        return cls(path, dict(vars(arguments)), path.read_text())

    @property
    def text(self) -> str:
        arguments = json.dumps(self.arguments, indent=2, sort_keys=True, default=str)
        return f"Arguments\n=========\n{arguments}\n\nSource: {self.executable}\n{'=' * 80}\n{self.code}"

    def publish(self, sink: TextSink, *, name: str = "run") -> None:
        sink.text(name, self.text)

    def save(self, directory: str | Path, *, name: str = "run.txt") -> Path:
        path = Path(directory) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.text)
        return path
