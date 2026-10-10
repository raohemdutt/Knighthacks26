"""Memory's read side (M4): the agent's MemoryReader, served from SQLite.

Same semantics as the agent's FakeMemory, so the two can be swapped. Search
goes through FTS5 and returns episodes and connector items together.
"""

import re
from datetime import datetime

from contexts.memory.domain.window import group_by_hour
from contexts.memory.infrastructure.store import MemoryStore, episode_from, iso, item_from, parse
from contracts.memory import Episode, EpisodeWindow, Item, MemoryStats, SearchHit, Sighting

# Question words that match every caption and only add noise to the ranking.
STOPWORDS = frozenset("the and did does was were are has have had where what when who how for with any there you our about".split())


def fts_query(q: str) -> str:
    """Plain words OR-ed together, quoted so nothing in a question reads as FTS5 syntax.

    Each word is a prefix, so "milestone" finds a Canvas title that says "Milestone1"."""
    words = [w for w in re.findall(r"\w+", q.lower()) if len(w) > 2 and w not in STOPWORDS]
    return " OR ".join(f'"{w}"*' for w in dict.fromkeys(words))


class SqliteMemory:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self.db = store.db

    def episodes(self, start: datetime, end: datetime, camera: str | None = None, label: str | None = None) -> list[Episode]:
        """Episodes that start in [start, end), oldest first."""
        rows = self.db.execute(
            "SELECT * FROM memory_episodes WHERE start_ts >= ? AND start_ts < ? AND (? IS NULL OR camera_id = ?) ORDER BY start_ts, id",
            (iso(start), iso(end), camera, camera),
        )
        eps = [episode_from(r) for r in rows]
        return [e for e in eps if label is None or label in e.labels]

    def window(self, start: datetime, end: datetime) -> EpisodeWindow:
        return group_by_hour(start, end, self.episodes(start, end))

    def last_seen(self, label: str) -> Sighting | None:
        row = self.db.execute("SELECT * FROM memory_sightings WHERE label = ? ORDER BY ts DESC, id LIMIT 1", (label,)).fetchone()
        if row is None:
            return None
        return Sighting(label=row["label"], camera_id=row["camera_id"], zone_id=row["zone_id"], ts=parse(row["ts"]),
                        frame_ref=row["frame_ref"], position=row["position"])

    def items(self, start: datetime, end: datetime, kind: str | None = None) -> list[Item]:
        """Connector items with `when` in [start, end), soonest first."""
        rows = self.db.execute(
            "SELECT * FROM memory_items WHERE when_ts >= ? AND when_ts < ? AND (? IS NULL OR kind = ?) ORDER BY when_ts, connector, external_id",
            (iso(start), iso(end), kind, kind),
        )
        return [item_from(r) for r in rows]

    def search(self, q: str, limit: int = 5) -> list[SearchHit]:
        """Best first. An item comes back with `episode_id` "item:<connector>:<external_id>" and its connector as the camera."""
        match = fts_query(q)
        if not match:
            return []
        rows = self.db.execute(
            "SELECT text, source, ref, bm25(memory_fts) AS rank FROM memory_fts WHERE memory_fts MATCH ? ORDER BY rank, rowid LIMIT ?",
            (match, limit),
        ).fetchall()
        return [hit for r in rows if (hit := self._hit(r))]

    def _hit(self, row) -> SearchHit | None:
        score = -row["rank"]  # bm25 is lower-is-better
        if row["source"] == "item":
            _, connector, external_id = row["ref"].split(":", 2)
            item = self.db.execute("SELECT * FROM memory_items WHERE connector = ? AND external_id = ?", (connector, external_id)).fetchone()
            if item is None:
                return None
            return SearchHit(episode_id=row["ref"], camera_id=connector, ts=parse(item["when_ts"] or item["synced_at"]),
                             text=row["text"], score=score, kind=item["kind"], url=item["url"], title=item["title"])
        ep = self.store.episode(row["ref"])
        if ep is None:
            return None
        return SearchHit(episode_id=ep.id, camera_id=ep.camera_id, ts=ep.start, text=row["text"], score=score,
                         frame_ref=ep.keyframe_refs[0] if ep.keyframe_refs else None)

    def stats(self) -> MemoryStats:
        """Hours from the first activity to the last, with the quiet stretches in between."""
        row = self.db.execute("SELECT COUNT(*) AS n, MIN(start_ts) AS first, MAX(last_activity) AS last, "
                              "COALESCE(SUM(json_array_length(captions)), 0) AS captions FROM memory_episodes").fetchone()
        first, last = parse(row["first"]), parse(row["last"])
        hours = (last - first).total_seconds() / 3600 if first else 0.0
        return MemoryStats(hours_recorded=round(hours, 2), episodes=row["n"], captions=row["captions"], first_event=first)
