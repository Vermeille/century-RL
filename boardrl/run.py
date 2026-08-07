"""Capture and publish the reproducible description of a run."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol


class TextSink(Protocol):
    def text(self, name: str, value: str) -> None: ...


@dataclass(frozen=True)
class RunInfo:
    executable: Path
    arguments: Mapping[str, object]
    code: str
    additional_code: Mapping[Path, str] = field(default_factory=dict)

    @classmethod
    def capture(
        cls,
        arguments: argparse.Namespace,
        executable: str | Path | None = None,
        additional_sources: Iterable[str | Path] = (),
    ) -> RunInfo:
        path = Path(executable or sys.argv[0]).resolve()
        sources = [Path(source).resolve() for source in additional_sources]
        return cls(
            path,
            dict(vars(arguments)),
            path.read_text(),
            {source: source.read_text() for source in sources},
        )

    @property
    def text(self) -> str:
        arguments = json.dumps(self.arguments, indent=2, sort_keys=True, default=str)
        sources = [(self.executable, self.code), *self.additional_code.items()]
        rendered_sources = "\n\n".join(
            f"Source: {path}\n{'=' * 80}\n{code}" for path, code in sources
        )
        return f"Arguments\n=========\n{arguments}\n\n{rendered_sources}"

    def publish(self, sink: TextSink, *, name: str = "run") -> None:
        sink.text(name, self.text)

    def save(self, directory: str | Path, *, name: str = "run.txt") -> Path:
        path = Path(directory) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.text)
        return path
