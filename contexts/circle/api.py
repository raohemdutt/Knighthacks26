"""Read-only Circle endpoints for the phone's Circle screen.

    app.include_router(circle_router(atlas.bus, circle=atlas.circle, sharing=atlas.sharing, presence=atlas.presence,
                                     board=atlas.board, config_dir=atlas.settings.config_dir))

- GET /api/circle/events   the shared calendar: agreed events that have not ended, soonest first
    {"configured", "circle_id", "source": "neon"|"local"|"none", "offline",
     "events": [{"id", "thread_id", "title", "starts_at", "ends_at", "location", "created_by",
                 "attendees": [{"member_id", "display_name", "answer": "free"|"busy"|"counter"}]}]}
- GET /api/circle/sharing  what leaves this box for the Circle, and how far it got
    {"configured", "circle_id", "member_id", "share_summaries", "share_presence", "sharing_now",
     "database": "connected"|"offline"|"unknown"|"not_configured", "database_note", "last_reached_at",
     "queued", "sent", "last_synced_at", "oldest_queued_at",
     "kinds": [{"kind", "on", "queued", "sent", "last_synced_at"}], "place": {"place", "source", "expires_at"}|null}

Both answer 200 in every state (no Circle, no Neon, Neon offline), so the screen can say
which one it is. Neither writes anything or reaches Neon for the status: `database` is what
the box's own timer last saw.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import yaml
from fastapi import APIRouter

from contexts.circle.application.calendar import AgreedEvents, CalendarRemote, upcoming

SUMMARY_KINDS = ("habit_summary", "impact_count", "game_score")  # CircleSharing's outbox kinds


def _circle_config(config_dir: Path | None) -> dict:
    """config/agent.yaml `circle`, for the settings that have no object when Neon is not configured."""
    if config_dir is None:
        return {}
    try:
        cfg = (yaml.safe_load((config_dir / "agent.yaml").read_text()) or {}).get("circle") or {}
    except (OSError, yaml.YAMLError):
        return {}
    return cfg if isinstance(cfg, dict) else {}


def circle_router(bus, *, circle=None, sharing=None, presence=None, board: CalendarRemote | None = None,
                  config_dir: Path | None = None, clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> APIRouter:
    """`circle`, `sharing`, `presence` and `board` are the Atlas fields of the same names; any may be None."""
    router = APIRouter()
    agreed = AgreedEvents()
    bus.subscribe("circle.event.agreed", agreed.on_agreed, replay=True)  # state only: rebuilt from the log
    config = _circle_config(config_dir)

    @router.get("/api/circle/events")
    def events() -> dict:
        if circle is None:
            return {"configured": False, "circle_id": None, "source": "none", "offline": False, "events": []}
        down = bool(getattr(getattr(board, "db", None), "offline", None))  # the timer retries every tick and clears it
        return {"configured": True, **upcoming(circle.circle_id, clock(), agreed, board, known_offline=down)}

    @router.get("/api/circle/sharing")
    def sharing_status() -> dict:
        now = clock()
        db = getattr(board, "db", None)
        if board is None:
            database, note = "not_configured", None
        elif getattr(db, "offline", None):
            database, note = "offline", db.offline
        elif getattr(db, "last_ok", None):
            database, note = "connected", None
        else:
            database, note = "unknown", None  # configured, nothing tried yet (the first minute after start)

        # Summaries need the member's opt-in and the outbox, which only exists with Neon.
        summaries = bool(sharing.opted_in) if sharing is not None else bool(config.get("share_summaries"))
        places = bool(presence.opted_in) if presence is not None else bool(config.get("share_presence"))
        try:
            stats = {s["kind"]: s for s in sharing.outbox.stats()} if sharing is not None else {}
        except Exception:  # an unreadable outbox file must not blank the screen
            stats = {}

        kinds = []
        for kind in [*SUMMARY_KINDS, *sorted(set(stats) - set(SUMMARY_KINDS))]:
            s = stats.get(kind, {})
            kinds.append({"kind": kind, "on": sharing is not None and summaries, "queued": s.get("queued") or 0,
                          "sent": s.get("sent") or 0, "last_synced_at": s.get("last_synced_at")})
        # A named place leaves only through Neon's presence table, and only for a member who opted in.
        kinds.append({"kind": "presence", "on": presence is not None and places and board is not None,
                      "queued": None, "sent": None, "last_synced_at": None})
        # Plans: a proposal after its approval card, and one word (free / busy / another time) in reply.
        kinds.append({"kind": "event_plan", "on": circle is not None and board is not None,
                      "queued": None, "sent": None, "last_synced_at": None})

        mine = presence.current.get(presence.me) if presence is not None else None
        place = None
        if mine is not None and mine.expires_at > now:
            place = {"place": mine.place, "source": mine.source, "expires_at": mine.expires_at.isoformat()}

        return {
            "configured": circle is not None,
            "circle_id": circle.circle_id if circle is not None else None,
            "member_id": circle.member_id if circle is not None else None,
            "share_summaries": summaries,
            "share_presence": places,
            "sharing_now": sharing is not None and summaries,
            "database": database,
            "database_note": note,
            "last_reached_at": db.last_ok.isoformat() if getattr(db, "last_ok", None) else None,
            "queued": sum(k["queued"] or 0 for k in kinds),
            "sent": sum(k["sent"] or 0 for k in kinds),
            "last_synced_at": max((s["last_synced_at"] for s in stats.values() if s.get("last_synced_at")), default=None),
            "oldest_queued_at": min((s["oldest_queued_at"] for s in stats.values() if s.get("oldest_queued_at")), default=None),
            "kinds": kinds,
            "place": place,
        }

    return router
