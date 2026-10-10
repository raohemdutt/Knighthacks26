"""Memory's SQLite file (M1): episodes, sightings, connector items and one FTS5 index.

Every table is a projection of the event log, so the file can be deleted and
rebuilt with `python -m contexts.memory rebuild`. Timestamps are stored as UTC
ISO strings of fixed width, so text order is time order.
"""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from contracts.memory import Episode, Item, Sighting

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_episodes (
    id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    start_ts TEXT NOT NULL,
    end_ts TEXT,                       -- NULL while the episode is open
    last_activity TEXT NOT NULL,       -- event time of the latest activity; becomes end_ts on close
    labels TEXT NOT NULL DEFAULT '[]',
    captions TEXT NOT NULL DEFAULT '[]',
    summary TEXT,
    keyframe_refs TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS memory_episodes_start ON memory_episodes (start_ts);
CREATE TABLE IF NOT EXISTS memory_sightings (
    id TEXT PRIMARY KEY,               -- camera, time, label and position: the same sighting never lands twice
    label TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    zone_id TEXT,
    ts TEXT NOT NULL,
    frame_ref TEXT NOT NULL,
    position TEXT
);
CREATE INDEX IF NOT EXISTS memory_sightings_label ON memory_sightings (label, ts);
CREATE TABLE IF NOT EXISTS memory_items (
    connector TEXT NOT NULL,
    external_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    when_ts TEXT,
    url TEXT,
    summary TEXT,
    synced_at TEXT NOT NULL,           -- event time of the latest sync
    PRIMARY KEY (connector, external_id)
);
CREATE INDEX IF NOT EXISTS memory_items_when ON memory_items (when_ts);
-- source is 'caption', 'summary' or 'item'; ref is the episode id or item_ref(...).
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5 (text, source UNINDEXED, ref UNINDEXED, tokenize = 'porter unicode61');
-- Events already projected, so replaying the log into a live file changes nothing.
CREATE TABLE IF NOT EXISTS memory_applied (event_id TEXT PRIMARY KEY) WITHOUT ROWID;
"""

TABLES = ("memory_episodes", "memory_sightings", "memory_items", "memory_fts", "memory_applied")


def iso(ts: datetime) -> str:
    """UTC, fixed width. A naive datetime is taken as UTC."""
    ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)
    return ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse(ts: str | None) -> datetime | None:
    return datetime.fromisoformat(ts) if ts else None


def item_ref(connector: str, external_id: str) -> str:
    return f"item:{connector}:{external_id}"


def episode_from(row: sqlite3.Row) -> Episode:
    return Episode(
        id=row["id"], camera_id=row["camera_id"], start=parse(row["start_ts"]), end=parse(row["end_ts"]),
        labels=json.loads(row["labels"]), captions=json.loads(row["captions"]), summary=row["summary"],
        keyframe_refs=json.loads(row["keyframe_refs"]),
    )


def item_from(row: sqlite3.Row) -> Item:
    return Item(connector=row["connector"], kind=row["kind"], external_id=row["external_id"], title=row["title"],
                when=parse(row["when_ts"]), url=row["url"], summary=row["summary"])


class MemoryStore:
    def __init__(self, path: str | Path = "data/memory.sqlite") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # One connection, used from the event loop; the API's tests call in from another thread.
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.execute("PRAGMA synchronous = NORMAL")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def first_time(self, event_id: str) -> bool:
        """Mark an event as projected. False when it already was (a replay or a retry)."""
        return self.db.execute("INSERT OR IGNORE INTO memory_applied VALUES (?)", (event_id,)).rowcount == 1

    # --- episodes ---

    def put_episode(self, ep: Episode, last_activity: datetime) -> None:
        self.db.execute(
            """INSERT INTO memory_episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (id) DO UPDATE SET camera_id = excluded.camera_id, start_ts = excluded.start_ts,
                   end_ts = excluded.end_ts, last_activity = excluded.last_activity, labels = excluded.labels,
                   captions = excluded.captions, summary = excluded.summary, keyframe_refs = excluded.keyframe_refs""",
            (ep.id, ep.camera_id, iso(ep.start), iso(ep.end) if ep.end else None, iso(last_activity),
             json.dumps(ep.labels), json.dumps(ep.captions), ep.summary, json.dumps(ep.keyframe_refs)),
        )

    def episode(self, episode_id: str) -> Episode | None:
        row = self.db.execute("SELECT * FROM memory_episodes WHERE id = ?", (episode_id,)).fetchone()
        return episode_from(row) if row else None

    def open_episodes(self) -> list[tuple[Episode, datetime]]:
        """Episodes still open, each with the time of its latest activity."""
        rows = self.db.execute("SELECT * FROM memory_episodes WHERE end_ts IS NULL ORDER BY start_ts, id")
        return [(episode_from(r), parse(r["last_activity"])) for r in rows]

    # --- sightings and items ---

    def add_sighting(self, s: Sighting) -> None:
        sid = f"{s.camera_id}:{iso(s.ts)}:{s.label}:{s.position or ''}"
        self.db.execute(
            "INSERT OR REPLACE INTO memory_sightings VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sid, s.label, s.camera_id, s.zone_id, iso(s.ts), s.frame_ref, s.position),
        )

    def put_item(self, item: Item, synced_at: datetime) -> None:
        """Insert or update on (connector, external_id), and keep its search text current."""
        self.db.execute(
            """INSERT INTO memory_items VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (connector, external_id) DO UPDATE SET kind = excluded.kind, title = excluded.title,
                   when_ts = excluded.when_ts, url = excluded.url, summary = excluded.summary,
                   synced_at = excluded.synced_at""",
            (item.connector, item.external_id, item.kind, item.title, iso(item.when) if item.when else None,
             item.url, item.summary, iso(synced_at)),
        )
        ref = item_ref(item.connector, item.external_id)
        self.unindex(ref)
        self.index("item", ref, f"{item.title}. {item.summary}" if item.summary else item.title)

    # --- search index ---

    def index(self, source: str, ref: str, text: str) -> None:
        self.db.execute("INSERT INTO memory_fts (text, source, ref) VALUES (?, ?, ?)", (text, source, ref))

    def unindex(self, ref: str) -> None:
        self.db.execute("DELETE FROM memory_fts WHERE ref = ?", (ref,))

    def dump(self) -> dict[str, list[tuple]]:
        """Every row of every table in a stable order, for comparing two rebuilds."""
        return {t: sorted(tuple(r) for r in self.db.execute(f"SELECT * FROM {t}")) for t in TABLES}
