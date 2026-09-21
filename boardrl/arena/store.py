"""SQLite persistence for arena policies, matches, fits, and events."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .events import ArenaEvent
from .ratings import Match, RatingFit


@dataclass(frozen=True)
class Policy:
    id: str
    kind: str
    step: int | None
    path: Path | None
    active: bool
    placement_batches: int


class ArenaStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS policies (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                step INTEGER,
                path TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                placement_batches INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS matches (
                batch_id TEXT PRIMARY KEY,
                first_id TEXT NOT NULL,
                second_id TEXT NOT NULL,
                games INTEGER NOT NULL,
                score REAL NOT NULL,
                seed INTEGER NOT NULL,
                purpose TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS posterior_fits (
                fit_id INTEGER NOT NULL,
                policy_id TEXT NOT NULL,
                rating REAL NOT NULL,
                deviation REAL NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (fit_id, policy_id)
            );
            CREATE TABLE IF NOT EXISTS events (
                event_key TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                level TEXT NOT NULL,
                step INTEGER NOT NULL,
                created_at REAL NOT NULL,
                alerted INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        self.connection.commit()

    def close(self):
        self.connection.close()

    def add_policy(
        self,
        policy_id: str,
        *,
        kind: str,
        step: int | None = None,
        path: Path | None = None,
    ) -> bool:
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO policies(id, kind, step, path)
            VALUES (?, ?, ?, ?)
            """,
            (policy_id, kind, step, str(path) if path is not None else None),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def policies(self, *, checkpoints_only=False, playable_only=False) -> list[Policy]:
        clauses = []
        if checkpoints_only:
            clauses.append("kind = 'checkpoint'")
        if playable_only:
            clauses.append("active = 1 AND path IS NOT NULL")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.connection.execute(
            f"SELECT * FROM policies {where} ORDER BY step, id"
        ).fetchall()
        return [self._policy_from_row(row) for row in rows]

    def policy(self, policy_id: str) -> Policy:
        row = self.connection.execute(
            "SELECT * FROM policies WHERE id = ?", (policy_id,)
        ).fetchone()
        if row is None:
            raise KeyError(policy_id)
        return self._policy_from_row(row)

    @staticmethod
    def _policy_from_row(row) -> Policy:
        return Policy(
            row["id"],
            row["kind"],
            row["step"],
            Path(row["path"]) if row["path"] else None,
            bool(row["active"]),
            row["placement_batches"],
        )

    def matches(self) -> list[Match]:
        return [
            Match(row["first_id"], row["second_id"], row["games"], row["score"])
            for row in self.connection.execute(
                "SELECT first_id, second_id, games, score FROM matches ORDER BY created_at"
            )
        ]

    def games_between(self, first: str, second: str) -> int:
        row = self.connection.execute(
            """
            SELECT COALESCE(SUM(games), 0) AS games FROM matches
            WHERE (first_id = ? AND second_id = ?)
               OR (first_id = ? AND second_id = ?)
            """,
            (first, second, second, first),
        ).fetchone()
        return int(row["games"])

    def record_match(
        self,
        *,
        batch_id: str,
        first: str,
        second: str,
        games: int,
        score: float,
        seed: int,
        purpose: str,
    ) -> bool:
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT OR IGNORE INTO matches(
                    batch_id, first_id, second_id, games, score,
                    seed, purpose, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (batch_id, first, second, games, score, seed, purpose, time.time()),
            )
            if cursor.rowcount and purpose.startswith("placement:"):
                policy_id = purpose.split(":", 2)[1]
                self.connection.execute(
                    """
                    UPDATE policies SET placement_batches = placement_batches + 1
                    WHERE id = ?
                    """,
                    (policy_id,),
                )
        return cursor.rowcount > 0

    def set_pool(self, retained: set[str]):
        checkpoints = self.policies(checkpoints_only=True)
        with self.connection:
            for policy in checkpoints:
                if policy.path is None:
                    continue
                if policy.id in retained:
                    self.connection.execute(
                        "UPDATE policies SET active = 1 WHERE id = ?", (policy.id,)
                    )
                else:
                    policy.path.unlink(missing_ok=True)
                    self.connection.execute(
                        "UPDATE policies SET active = 0, path = NULL WHERE id = ?",
                        (policy.id,),
                    )

    def get_int(self, key: str, default=0) -> int:
        value = self.get_text(key)
        return int(value) if value is not None else default

    def get_text(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_int(self, key: str, value: int):
        self.set_text(key, str(value))

    def set_text(self, key: str, value: str):
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO metadata(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )

    def record_fit(self, fit: RatingFit):
        fit_id = self.get_int("next_fit")
        created_at = time.time()
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO posterior_fits(
                    fit_id, policy_id, rating, deviation, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (
                        fit_id,
                        policy_id,
                        rating,
                        fit.deviations[policy_id],
                        created_at,
                    )
                    for policy_id, rating in fit.ratings.items()
                ),
            )
            self.connection.execute(
                """
                INSERT INTO metadata(key, value) VALUES ('next_fit', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(fit_id + 1),),
            )

    def record_event(self, event: ArenaEvent) -> bool:
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT OR IGNORE INTO events(
                    event_key, kind, title, text, level, step, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.key,
                    event.kind,
                    event.title,
                    event.text,
                    event.level,
                    event.step,
                    event.created_at,
                ),
            )
        return cursor.rowcount > 0

    def events(self) -> list[ArenaEvent]:
        rows = self.connection.execute(
            "SELECT * FROM events ORDER BY created_at, event_key"
        ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def pending_alerts(self) -> list[ArenaEvent]:
        rows = self.connection.execute(
            "SELECT * FROM events WHERE alerted = 0 ORDER BY created_at, event_key"
        ).fetchall()
        return [self._event_from_row(row) for row in rows]

    @staticmethod
    def _event_from_row(row) -> ArenaEvent:
        return ArenaEvent(
            row["event_key"],
            row["kind"],
            row["title"],
            row["text"],
            row["level"],
            row["step"],
            row["created_at"],
        )

    def mark_alerted(self, event_key: str):
        with self.connection:
            self.connection.execute(
                "UPDATE events SET alerted = 1 WHERE event_key = ?",
                (event_key,),
            )
