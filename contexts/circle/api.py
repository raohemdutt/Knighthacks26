"""Circle endpoints for the phone's Circle screen: two that only read, and one that starts a plan.

    app.include_router(circle_router(atlas.bus, circle=atlas.circle, sharing=atlas.sharing, presence=atlas.presence,
                                     board=atlas.board, config_dir=atlas.settings.config_dir, guard=authorized))

- GET /api/circle/events   the shared calendar: agreed events that have not ended, soonest first
    {"configured", "circle_id", "source": "neon"|"local"|"none", "offline",
     "events": [{"id", "thread_id", "title", "starts_at", "ends_at", "location", "created_by",
                 "attendees": [{"member_id", "display_name", "answer": "free"|"busy"|"counter"}]}]}
- GET /api/circle/sharing  what leaves this box for the Circle, and how far it got
    {"configured", "circle_id", "member_id", "share_summaries", "share_presence", "sharing_now",
     "database": "connected"|"offline"|"unknown"|"not_configured", "database_note", "last_reached_at",
     "queued", "sent", "last_synced_at", "oldest_queued_at",
     "kinds": [{"kind", "on", "queued", "sent", "last_synced_at"}], "place": {"place", "source", "expires_at"}|null}
- POST /api/circle/proposals  {title, start, end, location?, invitees?} -> {"action_id", "thread_id"}
    "Find a time for all of us": CircleAgent.propose, which only raises an approval card. Nothing is
    sent to the Circle until the card is approved. 404 without a Circle; needs the token like other writes.

- GET /api/circle/threads  the agents' conversations, latest activity first (contexts/circle/application/threads.py)
    {"configured", "circle_id", "member_id", "source": "none"|"log"|"log+store", "offline",
     "members": [{"member_id", "display_name", "mine"}],
     "threads": [{"thread_id", "legs", "title", "proposer": {"member_id", "display_name", "mine"}, "mine",
                  "starts_at", "ends_at", "location", "state": "proposed"|"waiting"|"countered"|"agreed"|"declined",
                  "invitees": [{"member_id", "display_name", "mine", "answer": "free"|"busy"|"counter"|null}],
                  "waiting_on": [...], "asked_at", "last_activity", "pending_action_id",
                  "action": {"kind": "approve"|"propose"|"none", "action_id", "proposal", "text"},
                  "last_line", "messages_count"}]}
- GET /api/circle/threads/{thread_id}  one conversation (any of its `legs` ids finds it): the same fields plus
    "messages": [{"id", "thread_id", "kind": "proposal"|"reply"|"agreed"|"note", "answer", "at", "line",
                  "from_member", "display_name", "mine", "starts_at", "ends_at", "location",
                  "source": "log"|"store"|"agreement"}]
    `line` is plain language written from the structured fields, never more than the message carries.
    404 without a Circle or for an unknown thread.
- GET /api/circle/members  who is in the Circle, where they are (if shared) and whether a box of theirs was seen
    {"configured", "circle_id", "member_id", "connect": [two config lines], "members": [{"member_id",
     "display_name", "mine", "has_box": true|null, "box_seen", "place": {"place", "source", "shared_at", "expires_at"}|null}]}

The GETs for the screen (events, sharing, threads, members) answer 200 in every state (no Circle, no Neon, Neon offline), so the screen can say
which one it is. None writes anything. The status never reaches Neon (`database` is what the box's own
timer last saw); threads read the store once per request to add what the log lacks, and skip it when it is down.
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from fastapi import APIRouter, Depends, HTTPException

from contexts.circle.application.calendar import AgreedEvents, CalendarRemote, upcoming
from contexts.circle.application.threads import Members, Threads, display, summary
from contracts.circle import Proposal

log = logging.getLogger("atlas.circle")

STORE_WINDOW = timedelta(days=14)  # how far back the store is read to fill in what the log lacks
CONNECT = ["circle.member_id: hemdutt", "circle.circle_id: team"]  # the two lines, with example values

SUMMARY_KINDS = ("habit_summary", "impact_count", "game_score")  # CircleSharing's outbox kinds


def _agent_config(config_dir: Path | None) -> dict:
    if config_dir is None:
        return {}
    try:
        cfg = yaml.safe_load((config_dir / "agent.yaml").read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return cfg if isinstance(cfg, dict) else {}


def _circle_config(config_dir: Path | None) -> dict:
    """config/agent.yaml `circle`, for the settings that have no object when Neon is not configured."""
    cfg = _agent_config(config_dir).get("circle") or {}
    return cfg if isinstance(cfg, dict) else {}


def _zone(config_dir: Path | None) -> ZoneInfo:
    """The box's one time zone (agent.yaml sleep.timezone): conversation lines say "tomorrow 2:00 PM" in it."""
    sleep = _agent_config(config_dir).get("sleep")
    try:
        return ZoneInfo(str((sleep if isinstance(sleep, dict) else {}).get("timezone") or "UTC"))
    except Exception:
        return ZoneInfo("UTC")


def circle_router(bus, *, circle=None, sharing=None, presence=None, board: CalendarRemote | None = None,
                  config_dir: Path | None = None, clock: Callable[[], datetime] = lambda: datetime.now(UTC),
                  guard: Callable | None = None) -> APIRouter:
    """`circle`, `sharing`, `presence` and `board` are the Atlas fields of the same names; any may be None."""
    router = APIRouter()
    config = _circle_config(config_dir)
    names = config.get("names") if isinstance(config.get("names"), dict) else {}
    agreed = AgreedEvents({str(k): str(v) for k, v in names.items() if v})
    bus.subscribe("circle.event.agreed", agreed.on_agreed, replay=True)  # state only: rebuilt from the log
    threads, people = Threads(), Members()  # state only too: the conversations and who was heard from
    threads.subscribe(bus)
    people.subscribe(bus)
    tz = _zone(config_dir)

    def roster() -> list[str]:
        return list(getattr(circle, "members", None) or agreed.names)

    def conversations() -> tuple[list[dict], str, bool]:
        """Every conversation, with what the store adds when it can be read: (threads, source, offline)."""
        now, rows, source, offline = clock(), [], "log", False
        store = getattr(circle, "store", None)
        if store is not None:
            if getattr(getattr(store, "db", None), "offline", None):  # the timer already found it down: do not wait
                offline = True
            else:
                try:
                    rows, source = store.since(circle.circle_id, now - STORE_WINDOW), "log+store"
                except Exception as exc:  # CircleOffline, or a refused read: the log still tells the story
                    log.warning("circle threads: the store could not be read, showing this box's log: %s", exc)
                    offline = True
        return threads.view(circle.circle_id, circle.member_id, agreed.names, roster(), now, tz, extra=rows), source, offline

    @router.get("/api/circle/threads")
    def thread_list() -> dict:
        if circle is None:
            return {"configured": False, "circle_id": None, "member_id": None, "source": "none", "offline": False,
                    "members": [], "threads": []}
        found, source, offline = conversations()
        me = circle.member_id
        members = [{"member_id": m, "display_name": display(m, agreed.names), "mine": m == me}
                   for m in dict.fromkeys([me, *roster()])]
        return {"configured": True, "circle_id": circle.circle_id, "member_id": me, "source": source, "offline": offline,
                "members": members, "threads": [summary(t) for t in found]}

    @router.get("/api/circle/threads/{thread_id}")
    def thread(thread_id: str) -> dict:
        if circle is None:
            raise HTTPException(404, "the Circle is not configured")
        found, source, offline = conversations()
        hit = next((t for t in found if thread_id in t["legs"]), None)
        if hit is None:
            raise HTTPException(404, "no such conversation on this box")
        return {**summary(hit), "messages": hit["messages"], "source": source, "offline": offline,
                "circle_id": circle.circle_id, "member_id": circle.member_id}

    @router.get("/api/circle/members")
    def members() -> dict:
        if circle is None:
            return {"configured": False, "circle_id": None, "member_id": None, "connect": CONNECT, "members": []}
        return {"configured": True, "circle_id": circle.circle_id, "member_id": circle.member_id, "connect": CONNECT,
                "members": people.view(circle.circle_id, circle.member_id, agreed.names, roster(), clock())}

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

    @router.post("/api/circle/proposals", dependencies=[Depends(guard)] if guard else [])
    async def propose(body: Proposal) -> dict:
        """Start "find a time for all of us" from the phone: an approval card, nothing sent until it is approved."""
        if circle is None:
            raise HTTPException(404, "the Circle is not configured")
        if not body.title.strip():
            raise HTTPException(422, "the plan needs a title")
        if body.start.tzinfo is None or body.end.tzinfo is None:
            raise HTTPException(422, "start and end need a UTC offset, e.g. 2026-10-11T14:00:00-04:00")
        if body.end <= body.start:
            raise HTTPException(422, "the end must be after the start")
        action = circle.propose(body, f"Ask the circle about {body.title.strip()}")
        await bus.publish(action)
        return {"action_id": action.action_id, "thread_id": action.params["thread_id"]}

    return router
