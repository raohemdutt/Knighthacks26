"""Memory's read side (M4): the agent's MemoryReader, served from SQLite.

Same semantics as the agent's FakeMemory, so the two can be swapped. Search
goes through FTS5 and returns episodes (captions and summaries), connector
items and the agent's hourly notes together.

With an embedder and stored vectors (M5) search is hybrid: the FTS ranking and
a cosine ranking of the same texts are merged by reciprocal rank fusion, so
"something to drink" finds a caption that says "energy drink can". Without
either, it is the keyword search alone.
"""

import json
import math
import operator
import re
from datetime import datetime

from contexts.memory.application.embedder import TextEmbedder
from contexts.memory.domain.window import group_by_hour
from contexts.memory.infrastructure.store import MemoryStore, episode_from, iso, item_from, parse, unpack_vector
from contracts.memory import Episode, EpisodeWindow, Item, MemoryStats, SearchHit, Sighting

# Question words that match every caption and only add noise to the ranking.
STOPWORDS = frozenset("the and did does was were are has have had where what when who how for with any there you our about".split())
RRF_K = 60  # reciprocal rank fusion: a result scores 1 / (RRF_K + rank) in each ranking that has it


def fts_query(q: str) -> str:
    """Plain words OR-ed together, quoted so nothing in a question reads as FTS5 syntax.

    Each word is a prefix, so "milestone" finds a Canvas title that says "Milestone1"."""
    words = [w for w in re.findall(r"\w+", q.lower()) if len(w) > 2 and w not in STOPWORDS]
    return " OR ".join(f'"{w}"*' for w in dict.fromkeys(words))


def fuse(keyword: list, semantic: list) -> list[dict]:
    """Two best-first rankings of rows, one row per ref, merged into one. A ref in both keeps its keyword text.

    The fused score goes out as `rank` with bm25's sign (lower is better), so `_hit` reads either kind of row."""
    scores: dict[str, float] = {}
    rows: dict[str, object] = {}
    for ranking in (keyword, semantic):
        for n, r in enumerate(ranking, start=1):
            scores[r["ref"]] = scores.get(r["ref"], 0.0) + 1 / (RRF_K + n)
            rows.setdefault(r["ref"], r)
    # sorted() is stable: on a tie the keyword results stay ahead of the vector-only ones
    return [{"source": rows[ref]["source"], "ref": ref, "text": rows[ref]["text"], "rank": -scores[ref]}
            for ref in sorted(scores, key=lambda ref: -scores[ref])]


class SqliteMemory:
    def __init__(self, store: MemoryStore, embedder: TextEmbedder | None = None) -> None:
        self.store = store
        self.db = store.db
        self.embedder = embedder  # None: keyword search only

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
        """The latest object-list sighting (it has a text position), unless the detector tracked the label after it.

        A track gives the camera, the zone its zone events put it in, the time and the frame, and no position."""
        row = self.db.execute("SELECT * FROM memory_sightings WHERE label = ? ORDER BY ts DESC, id LIMIT 1", (label,)).fetchone()
        track = self.db.execute("SELECT * FROM memory_tracks WHERE label = ?", (label,)).fetchone()
        if track and (row is None or track["ts"] > row["ts"]):
            return Sighting(label=label, camera_id=track["camera_id"], zone_id=track["zone_id"], ts=parse(track["ts"]),
                            frame_ref=track["frame_ref"] or "")
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
        """Best first. An item comes back with `episode_id` "item:<connector>:<external_id>" and its connector as the camera.

        An hourly note comes back as kind "note" on the first episode it covers, so it cites a real keyframe.
        An episode is indexed once per caption and once for its summary: it comes back once, with the
        text that matched best, and so does every item and note. `limit` counts those distinct results.

        With an embedder and vectors, results that only mean the same as the question come back too, and
        `score` is the fused score instead of the bm25 one. This call is synchronous, so the query vector
        comes from the embedder's cache or a bounded wait; `search_hybrid` is the same search for async callers."""
        match = fts_query(q)
        if not match or limit <= 0:
            return []
        rows = self.db.execute(
            "SELECT text, source, ref, bm25(memory_fts) AS rank FROM memory_fts WHERE memory_fts MATCH ? ORDER BY rank, rowid",
            (match,),
        )
        ranked: list = []
        seen: set[str] = set()
        for r in rows:  # best first, so the first row of a ref is its best
            if r["ref"] not in seen:
                seen.add(r["ref"])
                ranked.append(r)
        if semantic := self._semantic(q):
            ranked = fuse(ranked, semantic)
        hits: list[SearchHit] = []
        for r in ranked:
            if hit := self._hit(r):
                hits.append(hit)
                if len(hits) == limit:
                    break
        return hits

    async def search_hybrid(self, q: str, limit: int = 5) -> list[SearchHit]:
        """`search` for async callers: the query is embedded off the event loop first, then read from the cache."""
        if self.embedder and fts_query(q) and limit > 0 and self.store.vector_count(self.embedder.model):
            await self.embedder.aquery(q)
        return self.search(q, limit)

    def _semantic(self, q: str) -> list[dict]:
        """Indexed rows whose text is close to the question in meaning, best first, one per ref.

        Empty without an embedder, without vectors from its model, or when the model does not answer.
        Texts under the similarity floor are left out, so a question about nothing we saw finds nothing."""
        if self.embedder is None or not self.store.vector_count(self.embedder.model):
            return []
        query = self.embedder.query(q)
        norm = math.sqrt(sum(x * x for x in query)) if query else 0.0
        if not norm:
            return []
        query = [x / norm for x in query]
        by_text: dict[str, float] = {}  # a caption that repeats across episodes is scored once
        best: dict[str, dict] = {}
        for r in self.store.vector_rows(self.embedder.model, len(query)):  # a vector of another size is another model's
            text = r["text"]
            if text not in by_text:
                by_text[text] = sum(map(operator.mul, query, unpack_vector(r["vector"])))  # both unit length: the cosine
            similarity = by_text[text]
            if similarity >= self.embedder.min_similarity and (r["ref"] not in best or similarity > best[r["ref"]]["similarity"]):
                best[r["ref"]] = {"source": r["source"], "ref": r["ref"], "text": text, "similarity": similarity}
        return sorted(best.values(), key=lambda r: -r["similarity"])  # stable: ties stay in index order

    def _hit(self, row) -> SearchHit | None:
        score = -row["rank"]  # bm25 is lower-is-better
        if row["source"] == "item":
            _, connector, external_id = row["ref"].split(":", 2)
            item = self.db.execute("SELECT * FROM memory_items WHERE connector = ? AND external_id = ?", (connector, external_id)).fetchone()
            if item is None:
                return None
            return SearchHit(episode_id=row["ref"], camera_id=connector, ts=parse(item["when_ts"] or item["synced_at"]),
                             text=row["text"], score=score, kind=item["kind"], url=item["url"], title=item["title"])
        if row["source"] == "note":
            return self._note_hit(row, score)
        ep = self.store.episode(row["ref"])
        if ep is None:
            return None
        return SearchHit(episode_id=ep.id, camera_id=ep.camera_id, ts=ep.start, text=row["text"], score=score,
                         frame_ref=ep.keyframe_refs[0] if ep.keyframe_refs else None)

    def _note_hit(self, row, score: float) -> SearchHit | None:
        note = self.db.execute("SELECT * FROM memory_notes WHERE id = ?", (row["ref"].split(":", 1)[1],)).fetchone()
        if note is None:
            return None
        ep = next((ep for i in json.loads(note["episode_ids"]) if (ep := self.store.episode(i))), None)
        return SearchHit(episode_id=ep.id if ep else row["ref"], camera_id=ep.camera_id if ep else "agent",
                         ts=parse(note["hour"]), text=row["text"], score=score, kind="note",
                         frame_ref=ep.keyframe_refs[0] if ep and ep.keyframe_refs else None)

    def stats(self) -> MemoryStats:
        """Hours from the first activity to the last, with the quiet stretches in between."""
        row = self.db.execute("SELECT COUNT(*) AS n, MIN(start_ts) AS first, MAX(last_activity) AS last, "
                              "COALESCE(SUM(json_array_length(captions)), 0) AS captions FROM memory_episodes").fetchone()
        first, last = parse(row["first"]), parse(row["last"])
        hours = (last - first).total_seconds() / 3600 if first else 0.0
        return MemoryStats(hours_recorded=round(hours, 2), episodes=row["n"], captions=row["captions"], first_event=first)
