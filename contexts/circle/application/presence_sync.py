"""Friend presence through Neon (G5, delivery).

My own opted-in place goes to the Neon `presence` table (one row per member,
latest wins, with an expiry); friends' unexpired places come back as
circle.place.shared events for the PresenceBoard. Only a named place, its
source, and the expiry ever leave the box.

Not a CircleSharing share: presence is a single overwritten row that expires,
so only the latest unsent place is kept, and an offline box sends it later
(or never, if it expired first).
"""

import logging
from datetime import datetime
from typing import Protocol

from contexts.circle.application.sharing import CircleOffline, OutboxRow
from contracts.circle import PlaceShared

log = logging.getLogger("atlas.circle")


class PresenceRemote(Protocol):
    def push(self, row: OutboxRow) -> None: ...

    def active(self, circle_id: str, now: datetime) -> list[dict]:
        """Unexpired rows: member_id, place, source, shared_at, expires_at."""
        ...


class PresenceSync:
    def __init__(self, remote: PresenceRemote, circle_id: str, me: str) -> None:
        self.remote = remote
        self.pending: PlaceShared | None = None
        self.circle_id = circle_id
        self.me = me
        self._seen: set[tuple[str, str]] = set()  # (member, shared_at) already turned into events

    def on_shared(self, e: PlaceShared) -> None:
        """Bus handler (live only): queue my own shares; the board already checked opt-in and place names."""
        if e.member_id != self.me or e.circle_id != self.circle_id:
            return
        if self.pending is None or e.ts >= self.pending.ts:
            self.pending = e

    def sync(self, now: datetime) -> list[PlaceShared]:
        """Push my queued places, then pull friends' current places as events. Offline: try next tick."""
        mine = self.pending
        if mine is not None and mine.expires_at <= now:
            self.pending = mine = None  # expired before we could send it: nothing to share
        if mine is not None:
            payload = {"place": mine.place, "source": mine.source, "shared_at": mine.ts.isoformat(),
                       "expires_at": mine.expires_at.isoformat()}
            row = OutboxRow(mine.event_id, self.circle_id, self.me, "presence", payload, mine.ts)
            try:
                self.remote.push(row)
            except CircleOffline as exc:
                log.warning("presence sync paused, %s", exc)
                return []
            if self.pending is mine:
                self.pending = None
        try:
            rows = self.remote.active(self.circle_id, now)
        except CircleOffline as exc:
            log.warning("presence pull paused, %s", exc)
            return []
        out = []
        for r in rows:
            shared_at = r["shared_at"] if isinstance(r["shared_at"], datetime) else datetime.fromisoformat(r["shared_at"])
            expires_at = r["expires_at"] if isinstance(r["expires_at"], datetime) else datetime.fromisoformat(r["expires_at"])
            key = (r["member_id"], shared_at.isoformat())
            if r["member_id"] == self.me or key in self._seen:
                continue
            self._seen.add(key)
            out.append(PlaceShared(ts=shared_at, circle_id=self.circle_id, member_id=r["member_id"], place=r["place"],
                                   source=r["source"], expires_at=expires_at))
        return out


class InMemoryPresenceRemote:
    """Stands in for Neon in tests and on one laptop."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    def push(self, row: OutboxRow) -> None:
        self.rows[row.member_id] = {"member_id": row.member_id, "circle_id": row.circle_id, **row.payload}

    def active(self, circle_id, now):
        return [r for r in self.rows.values()
                if r["circle_id"] == circle_id and datetime.fromisoformat(r["expires_at"]) > now]
