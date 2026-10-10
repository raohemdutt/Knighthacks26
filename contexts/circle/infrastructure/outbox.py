"""circle_outbox: approved shares waiting on the box until Neon has them (N1).

One SQLite file (default data/circle.sqlite). A connection is opened per call,
so the outbox can be used from the event loop and from a worker thread.
"""

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from contexts.circle.application.sharing import OutboxRow
from contracts.circle import ShareQueued

TABLE = """
CREATE TABLE IF NOT EXISTS circle_outbox (
    share_id    TEXT PRIMARY KEY,
    circle_id   TEXT NOT NULL,
    member_id   TEXT NOT NULL,
    kind        TEXT NOT NULL,
    payload     TEXT NOT NULL,  -- JSON
    created_at  TEXT NOT NULL,  -- UTC ISO 8601
    synced_at   TEXT
)
"""


def _iso(t: datetime) -> str:
    return (t.astimezone(UTC) if t.tzinfo else t.replace(tzinfo=UTC)).isoformat(timespec="microseconds")


class CircleOutbox:
    def __init__(self, path: str | Path = "data/circle.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._run(TABLE)

    def _run(self, sql: str, params: tuple = ()) -> tuple[list[tuple], int]:
        with closing(sqlite3.connect(self.path)) as db, db:
            cur = db.execute(sql, params)
            return cur.fetchall(), cur.rowcount

    def enqueue(self, circle_id: str, member_id: str, kind: str, payload: dict, now: datetime,
                share_id: str | None = None) -> ShareQueued | None:
        """Write the share locally. Returns None when this share_id is already in the outbox."""
        share_id = share_id or uuid4().hex
        _, added = self._run(
            "INSERT OR IGNORE INTO circle_outbox (share_id, circle_id, member_id, kind, payload, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (share_id, circle_id, member_id, kind, json.dumps(payload), _iso(now)),
        )
        if not added:
            return None
        return ShareQueued(ts=now, share_id=share_id, circle_id=circle_id, kind=kind, payload=payload)

    def pending(self) -> list[OutboxRow]:
        rows, _ = self._run(
            "SELECT share_id, circle_id, member_id, kind, payload, created_at FROM circle_outbox"
            " WHERE synced_at IS NULL ORDER BY created_at, rowid"
        )
        return [OutboxRow(s, c, m, k, json.loads(p), datetime.fromisoformat(t)) for s, c, m, k, p, t in rows]

    def mark_synced(self, share_id: str, now: datetime) -> None:
        self._run("UPDATE circle_outbox SET synced_at = ? WHERE share_id = ?", (_iso(now), share_id))
