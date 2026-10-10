"""What the Agent needs from Memory (A1). Shapes come from contracts.memory.

The real adapter calls Memory's read API (M4); FakeMemory serves canned
episodes so the agent can be built and tested before M4/M7 land.
"""

from collections import defaultdict
from datetime import datetime
from typing import Protocol

from contracts.memory import Episode, EpisodeWindow, HourBucket, Item, SearchHit, Sighting

MAX_CAPTIONS_PER_HOUR = 6


class MemoryReader(Protocol):
    def episodes(
        self, start: datetime, end: datetime, camera: str | None = None, label: str | None = None
    ) -> list[Episode]: ...

    def window(self, start: datetime, end: datetime) -> EpisodeWindow: ...

    def last_seen(self, label: str) -> Sighting | None: ...

    def search(self, q: str, limit: int = 5) -> list[SearchHit]: ...

    def items(self, start: datetime, end: datetime, kind: str | None = None) -> list[Item]:
        """Connector items (Canvas, calendar, campus events) with `when` in [start, end)."""
        ...


def group_by_hour(start: datetime, end: datetime, episodes: list[Episode]) -> EpisodeWindow:
    buckets: dict[datetime, HourBucket] = {}
    for ep in sorted(episodes, key=lambda e: e.start):
        hour = ep.start.replace(minute=0, second=0, microsecond=0)
        b = buckets.setdefault(hour, HourBucket(hour=hour, episode_ids=[], captions=[]))
        b.episode_ids.append(ep.id)
        for c in [ep.summary] if ep.summary else ep.captions:
            if c not in b.captions and len(b.captions) < MAX_CAPTIONS_PER_HOUR:
                b.captions.append(c)
        b.keyframe_refs.extend(ep.keyframe_refs[:1])
    return EpisodeWindow(start=start, end=end, hours=list(buckets.values()))


class FakeMemory:
    def __init__(
        self, episodes: list[Episode] | None = None, sightings: list[Sighting] | None = None, items: list[Item] | None = None
    ) -> None:
        self._episodes = episodes or []
        self._items = items or []
        self._sightings: dict[str, list[Sighting]] = defaultdict(list)
        for s in sightings or []:
            self._sightings[s.label].append(s)

    def episodes(self, start, end, camera=None, label=None):
        return [
            e
            for e in self._episodes
            if start <= e.start < end
            and (camera is None or e.camera_id == camera)
            and (label is None or label in e.labels)
        ]

    def window(self, start, end):
        return group_by_hour(start, end, self.episodes(start, end))

    def items(self, start, end, kind=None):
        return sorted(
            (i for i in self._items if i.when and start <= i.when < end and (kind is None or i.kind == kind)),
            key=lambda i: i.when,
        )

    def last_seen(self, label):
        hits = self._sightings.get(label)
        return max(hits, key=lambda s: s.ts) if hits else None

    def search(self, q, limit=5):
        words = {w for w in q.lower().split() if len(w) > 2}
        hits = []
        for e in self._episodes:
            for text in [*e.captions, *([e.summary] if e.summary else [])]:
                score = sum(w in text.lower() for w in words)
                if score:
                    ref = e.keyframe_refs[0] if e.keyframe_refs else None
                    hits.append(SearchHit(episode_id=e.id, camera_id=e.camera_id, ts=e.start, text=text, score=score, frame_ref=ref))
        return sorted(hits, key=lambda h: -h.score)[:limit]
