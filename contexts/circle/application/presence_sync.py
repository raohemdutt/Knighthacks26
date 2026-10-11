"""Friend presence through Neon (G5, delivery).

My own opted-in place goes to the Neon `presence` table (one row per member,
latest wins, with an expiry); friends' unexpired places come back as
circle.place.shared events for the PresenceBoard. Only a named place, its
source, and the expiry ever leave the box.

Not a CircleSharing share: presence is a single overwritten row that expires,
so only the latest unsent place is kept, and an offline box sends it later
(or never, if it expired first). With a `queue` (PresenceQueue, on the box's
disk) the unsent place also survives a restart.

"Stop sharing" is a share that ends the moment it starts (expires_at == ts).
It is always sent, so the row in Neon ends too; a friend's box then no longer
finds that row among the unexpired ones and ends the place it was showing.
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
    def __init__(self, remote: PresenceRemote, circle_id: str, me: str, queue=None) -> None:
        self.remote = remote
        self.queue = queue  # PresenceQueue or None: where the unsent place is kept across a restart
        self.pending: PlaceShared | None = queue.load(circle_id, me) if queue else None
        self.circle_id = circle_id
        self.me = me
        self.offline = False  # Neon was down at the last sync; logged once when it changes, not on every tick
        self._seen: set[tuple[str, str]] = set()  # (member, shared_at) already turned into events
        self._shown: dict[str, PlaceShared] = {}  # each friend's latest place on this box's board

    def on_shared(self, e: PlaceShared) -> None:
        """Bus handler (live only): queue my own shares; the board already checked opt-in and place names."""
        if e.member_id != self.me or e.circle_id != self.circle_id:
            return
        if self.pending is None or e.ts >= self.pending.ts:
            self.pending = e
            if self.queue:
                self.queue.save(e)

    def on_friend(self, e: PlaceShared) -> None:
        """Bus handler (state only, replay-safe): what the board shows for each friend, so a restarted box still
        ends a place its owner stopped sharing."""
        if e.member_id == self.me or e.circle_id != self.circle_id:
            return
        prev = self._shown.get(e.member_id)
        if prev is None or e.ts >= prev.ts:
            self._shown[e.member_id] = e

    def _done(self, mine: PlaceShared) -> None:
        """Delivered, or expired unsent: no longer queued (a newer share queued meanwhile stays)."""
        if self.pending is mine:
            self.pending = None
        if self.queue:
            self.queue.clear(mine)

    def _paused(self, what: str, exc: Exception) -> list:
        if not self.offline:
            log.warning("presence %s paused, %s; an unsent place stays queued and every tick retries", what, exc)
        self.offline = True
        return []

    def sync(self, now: datetime) -> list[PlaceShared]:
        """Push my queued places, then pull friends' current places as events. Offline: try next tick."""
        mine = self.pending
        stop = mine is not None and mine.expires_at <= mine.ts  # "stop sharing": always sent, so Neon's row ends too
        if mine is not None and mine.expires_at <= now and not stop:
            self._done(mine)  # expired before we could send it: nothing to share
            mine = None
        if mine is not None:
            payload = {"place": mine.place, "source": mine.source, "shared_at": mine.ts.isoformat(),
                       "expires_at": mine.expires_at.isoformat(), "sent_at": now.isoformat()}
            row = OutboxRow(mine.event_id, self.circle_id, self.me, "presence", payload, mine.ts)
            try:
                self.remote.push(row)
            except CircleOffline as exc:
                return self._paused("sync", exc)
            self._done(mine)
        try:
            rows = self.remote.active(self.circle_id, now)
        except CircleOffline as exc:
            return self._paused("pull", exc)
        if self.offline:
            log.warning("presence sync resumed, Neon is reachable again")
            self.offline = False
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
        # A friend whose row is no longer among the unexpired ones stopped sharing (or it ran out there first):
        # end the place this box still shows. `ts` never goes back, so the board takes this as the latest.
        live = {r["member_id"] for r in rows}
        for member, shown in self._shown.items():
            if member not in live and shown.expires_at > now:
                out.append(PlaceShared(ts=max(now, shown.ts), circle_id=self.circle_id, member_id=member, place=shown.place,
                                       source=shown.source, expires_at=now))
        for e in out:
            self.on_friend(e)
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
