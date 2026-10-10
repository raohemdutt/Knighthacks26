"""Approved shares leave the box through a local outbox (N1).

    habits.day.closed / impact.count.updated / play.round.scored
        -> CircleSharing handlers (only when the member opted in)
        -> circle_outbox (SQLite, on the box)      circle.share.queued
        -> sync(now) on the app's timer -> Neon     circle.share.synced

Shares are written locally first, so nothing is lost while offline. Neon is
reached through the ShareRemote port; when it raises CircleOffline the rows
stay queued and the next tick tries again.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from contracts.circle import ShareQueued, ShareSynced
from contracts.habits import DayClosed
from contracts.impact import CountUpdated
from contracts.play import RoundScored

log = logging.getLogger("atlas.circle")

HABIT_FIELDS = {"habit_id", "day", "met", "value", "streak"}
IMPACT_FIELDS = {"items_seen", "recyclable_seen", "recycled", "contamination", "diversion_rate"}
SCORE_FIELDS = {"round_id", "session_id", "players", "item", "correct", "seconds", "path_len", "score"}


class CircleOffline(RuntimeError):
    """Neon cannot be reached right now (no network, no driver, no URL). Try again later."""


@dataclass
class OutboxRow:
    share_id: str
    circle_id: str
    member_id: str
    kind: str
    payload: dict
    created_at: datetime


class Outbox(Protocol):
    def enqueue(self, circle_id: str, member_id: str, kind: str, payload: dict, now: datetime,
                share_id: str | None = None) -> ShareQueued | None: ...

    def pending(self) -> list[OutboxRow]: ...

    def mark_synced(self, share_id: str, now: datetime) -> None: ...


class ShareRemote(Protocol):
    def push(self, row: OutboxRow) -> None: ...


class CircleSharing:
    def __init__(self, outbox: Outbox, remote: ShareRemote, circle_id: str, member_id: str, opted_in: bool) -> None:
        self.outbox = outbox
        self.remote = remote
        self.circle_id = circle_id
        self.member_id = member_id
        self.opted_in = opted_in  # nothing leaves the box without consent

    # --- bus handlers (live only): summaries, counts and scores, never raw signals ---

    def on_day_closed(self, e: DayClosed) -> ShareQueued | None:
        return self._queue("habit_summary", e, HABIT_FIELDS)

    def on_count_updated(self, e: CountUpdated) -> ShareQueued | None:
        return self._queue("impact_count", e, IMPACT_FIELDS)

    def on_round_scored(self, e: RoundScored) -> ShareQueued | None:
        return self._queue("game_score", e, SCORE_FIELDS)

    def _queue(self, kind: str, e, fields: set[str]) -> ShareQueued | None:
        if not self.opted_in:
            return None
        # share_id is this member plus the source event's id: the same event never queues twice, and two
        # boxes that close the same habit day (whose event ids are identical by design) do not collide in Neon
        return self.outbox.enqueue(self.circle_id, self.member_id, kind, e.model_dump(mode="json", include=fields),
                                   e.ts, share_id=f"{self.member_id}:{e.event_id}")

    # --- outbox worker (run on the app's timer) ---

    def sync(self, now: datetime) -> list[ShareSynced]:
        """Push queued shares to Neon, oldest first. Offline is not an error: rows stay queued."""
        synced: list[ShareSynced] = []
        for row in self.outbox.pending():
            try:
                self.remote.push(row)
            except CircleOffline as e:
                log.warning("circle sync paused, %s", e)
                break
            except Exception:
                log.exception("circle share %s (%s) was refused; it stays queued", row.share_id, row.kind)
                continue
            self.outbox.mark_synced(row.share_id, now)
            synced.append(ShareSynced(ts=now, share_id=row.share_id, circle_id=row.circle_id))
        return synced
