"""The Circle's shared calendar, as the phone reads it (G1 step 4, read side).

An event is on the calendar once everyone invited said free. Two places know
about it:

- Neon's `events` and `rsvps` tables, which every member's box can read
  (through CircleBoard.upcoming_events);
- this box's own event log, where the proposer's Atlas wrote
  circle.event.agreed. AgreedEvents rebuilds from that log, so the calendar
  still shows what this box agreed when Neon is not configured or is offline.

`upcoming` merges the two: Neon's row wins, and anything only this box knows
is added. Times go out as UTC ISO 8601; the phone shows them in its own zone.
"""

import logging
from datetime import UTC, datetime
from typing import Protocol

from contracts.circle import EventAgreed

log = logging.getLogger("atlas.circle")

MAX_EVENTS = 50


class CalendarRemote(Protocol):
    def upcoming_events(self, circle_id: str, now: datetime) -> list[dict]:
        """One row per (event, rsvp): id, thread_id, title, starts_at, ends_at, location, created_by,
        member_id, display_name, answer."""
        ...


def _utc(t: datetime | str) -> datetime:
    """A time without a zone is UTC, as everywhere else in the log."""
    if isinstance(t, str):
        t = datetime.fromisoformat(t.replace("Z", "+00:00"))
    return t.astimezone(UTC) if t.tzinfo else t.replace(tzinfo=UTC)


def _event(id_: str, thread_id: str | None, title: str, starts_at, ends_at, location: str | None,
           created_by: str | None) -> dict:
    return {"id": id_, "thread_id": thread_id, "title": title, "starts_at": _utc(starts_at).isoformat(),
            "ends_at": _utc(ends_at).isoformat(), "location": location or None, "created_by": created_by,
            "attendees": []}


def _attendee(member_id: str, display_name: str | None, answer: str) -> dict:
    return {"member_id": member_id, "display_name": display_name or member_id, "answer": answer}


def from_rows(rows: list[dict]) -> list[dict]:
    """CircleBoard.upcoming_events rows (one per RSVP) folded into one dict per event, order kept."""
    events: dict[str, dict] = {}
    for r in rows:
        ev = events.get(r["id"])
        if ev is None:
            ev = events[r["id"]] = _event(r["id"], r.get("thread_id"), r["title"], r["starts_at"], r["ends_at"],
                                          r.get("location"), r.get("created_by"))
        if r.get("member_id"):
            ev["attendees"].append(_attendee(r["member_id"], r.get("display_name"), r["answer"]))
    return list(events.values())


class AgreedEvents:
    """What this box itself saw agreed. Rebuilt from the log: subscribe with replay=True."""

    def __init__(self, names: dict[str, str] | None = None) -> None:
        self.names = names or {}  # member id -> display name, from the circle config; an unknown id shows as itself
        self.events: dict[str, EventAgreed] = {}  # thread_id -> the first agreement for it

    def on_agreed(self, e: EventAgreed) -> None:
        self.events.setdefault(e.thread_id, e)

    def upcoming(self, circle_id: str, now: datetime) -> list[dict]:
        out = []
        for e in list(self.events.values()):
            if e.circle_id != circle_id or _utc(e.event.end) <= now:
                continue
            ev = _event(e.thread_id, e.thread_id, e.event.title, e.event.start, e.event.end, e.event.location, e.created_by)
            # agreement means every attendee answered free (the proposer included)
            ev["attendees"] = [_attendee(m, self.names.get(m), "free") for m in sorted(e.attendees)]
            out.append(ev)
        return out


def upcoming(circle_id: str, now: datetime, local: AgreedEvents, remote: CalendarRemote | None,
             known_offline: bool = False) -> dict:
    """The calendar for the Circle screen. Never raises: without Neon it is what this box knows.

    `source` is "neon" (the shared table, plus anything only this box has) or "local" (this box's log only);
    `offline` is true when Neon is configured but could not be read just now. `known_offline` skips the
    read when the box's timer already found Neon down, so the screen does not wait out a connect timeout."""
    events, source, offline = [], "local", remote is not None and known_offline
    if remote is not None and not known_offline:
        try:
            events, source = from_rows(remote.upcoming_events(circle_id, now)), "neon"
        except Exception as exc:  # CircleOffline, or a database that refuses the read: show what we have
            log.warning("shared calendar read failed, showing this box's own events: %s", exc)
            offline = True
    known = {ev["thread_id"] or ev["id"] for ev in events}
    events += [ev for ev in local.upcoming(circle_id, now) if ev["thread_id"] not in known]
    events.sort(key=lambda ev: (ev["starts_at"], ev["id"]))
    return {"circle_id": circle_id, "source": source, "offline": offline, "events": events[:MAX_EVENTS]}
