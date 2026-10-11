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
from contracts.circle import PlaceShared, ShareQueued

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

    def stats(self) -> list[dict]:
        """Per kind: how many shares wait here, how many Neon has, and when the last one went (UTC ISO or None)."""
        rows, _ = self._run(
            "SELECT kind, SUM(synced_at IS NULL), SUM(synced_at IS NOT NULL), MAX(synced_at),"
            " MIN(CASE WHEN synced_at IS NULL THEN created_at END) FROM circle_outbox GROUP BY kind ORDER BY kind"
        )
        return [{"kind": k, "queued": q, "sent": s, "last_synced_at": last, "oldest_queued_at": oldest}
                for k, q, s, last, oldest in rows]


PLACE_TABLE = """
CREATE TABLE IF NOT EXISTS presence_outbox (
    circle_id   TEXT NOT NULL,
    member_id   TEXT NOT NULL,
    event_id    TEXT NOT NULL,
    event       TEXT NOT NULL,  -- the PlaceShared, JSON
    PRIMARY KEY (circle_id, member_id)
)
"""


class PresenceQueue:
    """The one place share (or "stop sharing") this box has not delivered yet, in the same SQLite file.

    Presence is a single overwritten row in Neon, so only the latest unsent one is worth keeping: one row per
    member here, replaced by a newer share. It survives a restart, which the in-memory `pending` did not."""

    def __init__(self, path: str | Path = "data/circle.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._run(PLACE_TABLE)

    def _run(self, sql: str, params: tuple = ()) -> list[tuple]:
        with closing(sqlite3.connect(self.path)) as db, db:
            return db.execute(sql, params).fetchall()

    def save(self, e: PlaceShared) -> None:
        self._run("INSERT OR REPLACE INTO presence_outbox (circle_id, member_id, event_id, event) VALUES (?, ?, ?, ?)",
                  (e.circle_id, e.member_id, e.event_id, e.model_dump_json()))

    def load(self, circle_id: str, member_id: str) -> PlaceShared | None:
        rows = self._run("SELECT event FROM presence_outbox WHERE circle_id = ? AND member_id = ?", (circle_id, member_id))
        try:
            return PlaceShared.model_validate_json(rows[0][0]) if rows else None
        except ValueError:  # a row written in a shape this build cannot read: nothing worth sending
            return None

    def clear(self, e: PlaceShared) -> None:
        """Delivered, or expired unsent. Only this share: a newer one saved meanwhile stays queued."""
        self._run("DELETE FROM presence_outbox WHERE circle_id = ? AND member_id = ? AND event_id = ?",
                  (e.circle_id, e.member_id, e.event_id))
