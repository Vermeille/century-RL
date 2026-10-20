"""Collect/resume an offline pool: python -m boardrl.games.traversal.collect_terrains."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from boardrl.games.traversal.terrain import SIZE, SOURCE, TERRAIN_TYPES


def fetch_board(terrain: str):
    request = Request(
        SOURCE,
        data=json.dumps({"terrain": terrain}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(4):
        try:
            with urlopen(request, timeout=30) as response:
                data = json.load(response)
            break
        except HTTPError as exc:
            if attempt == 3 or (exc.code != 429 and exc.code < 500):
                raise
            delay = max(2 ** attempt, float(exc.headers.get("Retry-After", 0)))
            print(f"{terrain}: HTTP {exc.code}; retrying in {delay:g}s", flush=True)
            time.sleep(delay)
        except (URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    if data["terrain_selected"] != terrain:
        raise ValueError(f"Requested {terrain}, received {data['terrain_selected']}")
    board = data["board"]
    if len(board) != SIZE or any(len(row) != SIZE for row in board):
        raise ValueError(f"{terrain}: expected a {SIZE}x{SIZE} board")
    if any(height < 0 or height != int(height) for row in board for height in row):
        raise ValueError(f"{terrain}: expected nonnegative integer heights")
    for row in range(SIZE):
        for col in range(SIZE):
            for dr, dc in ((0, 1), (1, 0)):
                r, c = row + dr, col + dc
                if r < SIZE and c < SIZE and abs(board[row][col] - board[r][c]) > 2:
                    raise ValueError(f"{terrain}: adjacent heights differ by more than two")
    return board


def save_pool(path: Path, data):
    # Keep one whole board per line so the generated data stays compact.
    metadata = {key: value for key, value in data.items() if key != "boards"}
    lines = json.dumps(metadata, indent=2).splitlines()[:-1]
    lines[-1] += ","
    lines.append('  "boards": {')
    for name_index, (name, boards) in enumerate(data["boards"].items()):
        lines.append(f'    "{name}": [')
        for index, board in enumerate(boards):
            comma = "," if index + 1 < len(boards) else ""
            lines.append("      " + json.dumps(board, separators=(",", ":")) + comma)
        comma = "," if name_index + 1 < len(data["boards"]) else ""
        lines.append("    ]" + comma)
    lines.extend(("  }", "}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-type", type=int, default=20)
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds between requests")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).with_name("terrain_samples.json"),
    )
    args = parser.parse_args()
    if args.per_type <= 0 or args.delay < 0:
        parser.error("--per-type must be positive and --delay nonnegative")
    if args.output.exists():
        data = json.loads(args.output.read_text(encoding="utf-8"))
        if data["source"] != SOURCE or data["samples_per_type"] != args.per_type:
            parser.error("Existing pool has different source/count; choose another output")
    else:
        data = {
            "source": SOURCE,
            "collected_at_utc": datetime.now(timezone.utc).isoformat(),
            "samples_per_type": args.per_type,
            "boards": {name: [] for name in TERRAIN_TYPES},
        }
    for terrain in TERRAIN_TYPES:
        boards = data["boards"][terrain]
        while len(boards) < args.per_type:
            boards.append(fetch_board(terrain))
            save_pool(args.output, data)
            time.sleep(args.delay)
        unique = len({json.dumps(board) for board in boards})
        print(f"{terrain}: {len(boards)} samples, {unique} distinct boards", flush=True)
    print(f"Saved {sum(map(len, data['boards'].values()))} samples to {args.output}", flush=True)


if __name__ == "__main__":
    main()
