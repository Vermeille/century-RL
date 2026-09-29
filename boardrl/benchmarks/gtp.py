from __future__ import annotations

import subprocess
from collections.abc import Sequence


class GTPError(RuntimeError):
    pass


class GTPEngine:
    """Tiny subprocess client for the subset of GTP used by Othello engines."""

    def __init__(self, command: Sequence[str]):
        if not command:
            raise ValueError("engine command cannot be empty")
        self.command_line = tuple(command)
        self.process = subprocess.Popen(
            self.command_line,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def _response(self, command: str) -> str:
        if self.process.poll() is not None:
            raise GTPError(
                f"engine exited with code {self.process.returncode}: {self.command_line}"
            )
        if self.process.stdin is None or self.process.stdout is None:
            raise GTPError("engine pipes are unavailable")
        self.process.stdin.write(command.rstrip("\n") + "\n")
        self.process.stdin.flush()

        while True:
            line = self.process.stdout.readline()
            if line == "":
                raise GTPError(f"engine closed stdout while handling {command!r}")
            line = line.strip()
            if line:
                break

        if line.startswith("?"):
            raise GTPError(f"engine rejected {command!r}: {line[1:].strip()}")
        if not line.startswith("="):
            raise GTPError(f"invalid GTP response to {command!r}: {line!r}")
        return line[1:].strip()

    def clear_board(self) -> None:
        self._response("clear_board")

    def play(self, color: str, move: str) -> None:
        self._response(f"play {color} {move}")

    def genmove(self, color: str) -> str:
        payload = self._response(f"genmove {color}")
        if not payload:
            raise GTPError("genmove returned an empty move")
        return payload.split()[-1].lower()

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            self._response("quit")
        except GTPError:
            pass
        try:
            self.process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
