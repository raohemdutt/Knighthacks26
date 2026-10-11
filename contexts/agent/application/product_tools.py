"""Ask Atlas's tools: one per thing the product can do, so the model can reach every part of it.

Each tool is a pure function of (its arguments, the ports, the time now, dry_run). It reads through a port the
composition root wires to the context that owns the data (contexts/gateway/bootstrap.py `ask_ports`), and it never
writes anything itself: a tool with an outside effect (calendar.add, reminders.add, camera.pause) RETURNS an
`agent.action.proposed` event, and the router publishes it. Workspace runs it only after the phone approves (W2,
AGENTS.md section 7). A dry run says what would happen and returns no event, so it is safe to call twice.

The model sees `schemas()`: a short description per tool, written for a 4B model to choose from. The names it sees use
underscores (calendar_list); the names everywhere else use dots (calendar.list).

The words follow the owner's rule (contexts/day/domain/ear.py): one or two short sentences, no lists read out, no ids.
"""

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from contexts.day.domain import ear
from contracts.agent import ActionProposed

CAMERA_PAUSE = "cameras.pause"  # device_control target: the privacy policy pauses every camera (resume: none)
CALENDAR_WRITE = "calendar.create_event"  # the calendar connector's declared write tool (config/connectors.yaml)
DEFAULT_MINUTES = 60
REMINDER_MINUTES = 15
FIND_BACK = timedelta(days=7)
FIND_AHEAD = timedelta(days=30)
CALENDAR_KINDS = ("calendar_event", "assignment")

# The web app's screens, by the names a person says. Paths are the Next.js routes under web/app.
SCREENS = {
    "home": "/home/", "ask": "/", "ask atlas": "/", "calendar": "/now/", "schedule": "/now/", "dailies": "/today/",
    "today": "/today/", "plan": "/dayrun/", "todays plan": "/dayrun/", "day plan": "/dayrun/", "approvals": "/approvals/",
    "room map": "/map/", "room": "/map/", "map": "/map/", "live": "/live/", "cameras": "/live/", "camera": "/live/",
    "zones": "/zones/", "timeline": "/timeline/", "memory": "/timeline/", "learning lab": "/lab/", "learning": "/lab/",
    "lab": "/lab/", "box stats": "/stats/", "stats": "/stats/", "energy": "/stats/", "power": "/stats/",
    "circle": "/circle/", "friends": "/circle/", "privacy": "/privacy/", "people": "/people/", "files": "/files/",
    "impact": "/impact/", "play": "/play/", "sort it": "/play/", "teach": "/teach/", "inspection": "/spot/",
    "plant floor": "/spot/",
}


@dataclass
class ToolResult:
    text: str  # the answer, in plain words
    data: dict = field(default_factory=dict)  # what the tool found, for the Ask tab's card
    link: str | None = None  # a screen of the web app to open ("/today")
    proposal: ActionProposed | None = None  # an approval card to publish (never published by the tool)
    ok: bool = True  # False: the tool could not do it (not set up, bad arguments)


@dataclass
class Ports:
    """What the tools read through. Every port is optional: a tool whose port is missing says so."""

    tz: ZoneInfo = field(default_factory=lambda: ZoneInfo("America/New_York"))
    items: Callable[[datetime, datetime], list] | None = None  # memory.items(start, end) -> [contracts.memory.Item]
    say_calendar: Callable[[str], str | None] | None = None  # "today" -> the owner's wording of that day (QA's fast lane)
    calendar_writes: Callable[[], bool] | None = None  # the calendar connector can write on this box
    last_seen: Callable[[str], str | None] | None = None  # "phone" -> "Your phone was last seen ..."
    search: Callable[[str], list[str]] | None = None  # "coffee" -> caption lines, newest first
    recap: Callable[[datetime, datetime], str | None] | None = None  # a window -> what the cameras saw, said
    room: Callable[[], dict | None] | None = None  # the room twin
    people: Callable[[], list[dict]] | None = None  # [{"name": str | None, "known": bool}]
    owner_signed_in: Callable[[], bool] | None = None
    cameras: Callable[[], dict] | None = None  # {"cameras": [...], "paused": bool, "running": bool}
    can_pause: Callable[[], bool] | None = None  # an executor that pauses capture is wired
    box_model: Callable[[], dict] | None = None  # {"version": ..., "deployed": ..., "source": ...}
    energy: Callable[[], dict | None] | None = None  # the last answer's {"joules", "seconds", "watts", "model"}
    learning: Callable[[], dict | None] | None = None  # {"run", "task", "score", "status"}
    shared_brain: Callable[[], dict | None] | None = None  # the Circle's shared-brain state
    # Inspection's own shift summary, in its own words: {"status": str, "do_first": [{"headline", "sentence"}],
    # "do_first_total": int, "watch": int}; None when no kit is loaded on the box
    inspection: Callable[[], dict | None] | None = None


@dataclass
class Tool:
    name: str  # "calendar.list"
    description: str  # what the model reads to choose it: short
    parameters: dict
    run: Callable[[dict, Ports, datetime, bool], ToolResult]
    writes: bool = False  # proposes an approval card

    @property
    def model_name(self) -> str:
        return self.name.replace(".", "_")

    def schema(self) -> dict:
        return {"type": "function", "function": {"name": self.model_name, "description": self.description,
                                                 "parameters": self.parameters}}


def _obj(props: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props or {}, "required": required or []}


S = {"type": "string"}
N = {"type": "number"}


# --- time words -------------------------------------------------------------------------------------------------

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
CLOCK = re.compile(r"\b(?:at\s+)?(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ap>a\.?m\.?|p\.?m\.?)?(?=\s|$|[,.?!])", re.I)
NOON = re.compile(r"\bnoon\b", re.I)
MIDNIGHT = re.compile(r"\bmidnight\b", re.I)


def day_of(text: str, now: datetime, tz: ZoneInfo) -> datetime | None:
    """The local midnight a day word names ("today", "tomorrow", "friday", "2026-10-12"); None when none."""
    t = (text or "").strip().lower()
    local = now.astimezone(tz)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    if m := re.search(r"\d{4}-\d{2}-\d{2}", t):
        try:
            d = datetime.fromisoformat(m.group(0))
            return datetime.combine(d.date(), time.min, tzinfo=tz)
        except ValueError:
            return None
    if "day after tomorrow" in t:
        return midnight + timedelta(days=2)
    if "tomorrow" in t or "tmrw" in t:
        return midnight + timedelta(days=1)
    if "yesterday" in t:
        return midnight - timedelta(days=1)
    if "today" in t or "tonight" in t or "this morning" in t or "this afternoon" in t or "this evening" in t:
        return midnight
    for i, name in enumerate(DAYS):
        if re.search(rf"\b{name[:3]}(?:{name[3:]})?\b", t):
            ahead = (i - local.weekday()) % 7
            if "next" in t and ahead == 0:
                ahead = 7
            return midnight + timedelta(days=ahead)
    return None


def parse_when(text: str, now: datetime, tz: ZoneInfo) -> datetime | None:
    """A start time from what a person (or a small model) wrote: an ISO time, or words like "3 pm tomorrow",
    "tomorrow at 15:00", "at 6" (6 PM: an hour from 1 to 7 with no am/pm is the afternoon or evening), "noon friday".
    A time that already passed today, with no day named, is tomorrow. None when no time is in it."""
    raw = (text or "").strip()
    if not raw:
        return None
    try:  # "2026-10-12T15:00:00" or "2026-10-12 15:00"
        t = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=tz)
    except ValueError:
        pass
    low = raw.lower()
    day = day_of(low, now, tz)
    hour = minute = None
    if NOON.search(low):
        hour, minute = 12, 0
    elif MIDNIGHT.search(low):
        hour, minute = 0, 0
    else:
        no_date = re.sub(r"\d{4}-\d{2}-\d{2}", " ", low)
        for m in CLOCK.finditer(no_date):
            h, mm, ap = int(m["h"]), int(m["m"] or 0), (m["ap"] or "").replace(".", "")
            if h > 23 or mm > 59:
                continue
            if ap == "pm" and h < 12:
                h += 12
            elif ap == "am" and h == 12:
                h = 0
            elif not ap and 1 <= h <= 7:
                h += 12  # "call mom at 6": the evening
            if not ap and not m["m"] and "at" not in m.group(0).lower() and not re.search(r"\b(?:o'?clock)\b", low):
                # a bare number with no "at", no minutes and no am/pm ("3 people") is not a time unless it is the only one
                if len(CLOCK.findall(no_date)) > 1:
                    continue
            hour, minute = h, mm
            break
    if hour is None:
        if day is None:
            return None
        hour, minute = 9, 0  # a day with no time: the morning
    local_now = now.astimezone(tz)
    base = day or local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    start = base.replace(hour=hour, minute=minute)
    if day is None and start <= local_now:
        start += timedelta(days=1)
    return start


def _window(args: dict, now: datetime, tz: ZoneInfo) -> tuple[datetime, datetime, str]:
    """calendar.list's window: a day word or date (default today), and how many days (default 1)."""
    word = str(args.get("day") or args.get("date") or "today")
    start = day_of(word, now, tz) or now.astimezone(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        days = max(1, min(14, int(float(args.get("days") or 1))))
    except (TypeError, ValueError):
        days = 1
    return start, start + timedelta(days=days), word


def _item_row(i, tz: ZoneInfo) -> dict:
    when = i.when.astimezone(tz).isoformat() if i.when else None
    return {"title": i.title, "when": when, "url": i.url, "connector": i.connector, "kind": i.kind}


def _action_id(*parts: str) -> str:
    """The same request in the same minute proposes the same card: a retry does not make a second one."""
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


def _and(parts: list[str]) -> str:
    """ "a", "a and b", "a, b and c" """
    parts = [p for p in parts if p]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1] if parts else ""


def _not_here(what: str) -> ToolResult:
    return ToolResult(f"{what} isn't set up on this box.", {"available": False}, ok=False)


# --- the calendar ------------------------------------------------------------------------------------------------

def calendar_list(args: dict, p: Ports, now: datetime, dry_run: bool = False) -> ToolResult:
    if p.items is None:
        return _not_here("The calendar")
    start, end, word = _window(args, now, p.tz)
    today = now.astimezone(p.tz).replace(hour=0, minute=0, second=0, microsecond=0)
    word = {today: "today", today + timedelta(days=1): "tomorrow"}.get(start, word)  # "2026-10-11" is said "today"
    rows = [_item_row(i, p.tz) for i in sorted(p.items(start, end), key=lambda i: i.when or start)
            if i.kind in CALENDAR_KINDS]
    span = word if (end - start) <= timedelta(days=1) else f"the next {(end - start).days} days"
    said = p.say_calendar(word) if (p.say_calendar and (end - start) <= timedelta(days=1)) else None
    if not said:
        if not rows:
            said = f"Nothing on your calendar {span}." if span in ("today", "tomorrow") else f"Nothing on your calendar for {span}."
        else:
            first = [r for r in rows if r["when"]][:3]
            parts = [f"{r['title']} {ear.ear_when(datetime.fromisoformat(r['when']), now.astimezone(p.tz))}" for r in first]
            more = f", and {len(rows) - len(first)} more" if len(rows) > len(first) else ""
            said = f"You have {len(rows)} thing{'s' if len(rows) != 1 else ''}: {_and(parts)}{more}."
    return ToolResult(said, {"start": start.isoformat(), "end": end.isoformat(), "events": rows}, link="/now/")


def calendar_find(args: dict, p: Ports, now: datetime, dry_run: bool = False) -> ToolResult:
    if p.items is None:
        return _not_here("The calendar")
    q = str(args.get("query") or args.get("q") or "").strip().lower()
    if not q:
        return ToolResult("What should I look for on the calendar?", {"events": []}, ok=False)
    words = [w for w in re.findall(r"[a-z0-9]+", q) if len(w) > 2] or [q]
    found = [i for i in p.items(now - FIND_BACK, now + FIND_AHEAD)
             if i.kind in CALENDAR_KINDS and all(w in i.title.lower() for w in words)]
    found.sort(key=lambda i: (i.when is None or i.when < now, abs((i.when or now) - now)))
    rows = [_item_row(i, p.tz) for i in found[:5]]
    if not rows:
        return ToolResult(f"I couldn't find {q} on your calendar.", {"query": q, "events": []}, link="/now/")
    top = found[0]
    when = ear.ear_when(top.when.astimezone(p.tz), now.astimezone(p.tz)) if top.when else "with no time set"
    return ToolResult(f"{top.title} is {when}.", {"query": q, "events": rows}, link="/now/")


def _propose_event(title: str, start: datetime, minutes: int, p: Ports, now: datetime, dry_run: bool,
                   attendees: list[str], location: str | None, what: str, said: str) -> ToolResult:
    end = start + timedelta(minutes=minutes)
    event = {"title": title, "start": start.isoformat(), "end": end.isoformat(), "location": location,
             "attendees": attendees}
    when = ear.ear_when(start, now.astimezone(p.tz))
    data = {"event": event, "target": CALENDAR_WRITE}
    if dry_run:
        return ToolResult(f"I would propose {said} {when}.", {**data, "dry_run": True}, link="/approvals/")
    reason = f"Add {what} to your calendar: {title}, {when}" + (f", with {_and(attendees)}" if attendees else "")
    if p.calendar_writes is not None and not p.calendar_writes():
        reason += ". No calendar connector on this box: approving keeps it on Atlas's calendar only"
    action_id = _action_id("calendar", title.lower(), event["start"], now.strftime("%Y-%m-%dT%H:%M"))
    proposal = ActionProposed(ts=now, action_id=action_id, kind="connector_write", target=CALENDAR_WRITE,
                              reason=reason, params={"events": [event]})
    data["action_id"] = action_id
    return ToolResult(f"I have proposed {said} {when}. Approve it on your phone.", data,
                      link=f"/approvals/?focus={action_id}", proposal=proposal)


def calendar_add(args: dict, p: Ports, now: datetime, dry_run: bool = False) -> ToolResult:
    title = str(args.get("title") or "").strip().strip("\"'")
    start = parse_when(str(args.get("start") or args.get("when") or args.get("time") or ""), now, p.tz)
    if start is None and args.get("date"):
        start = parse_when(f"{args.get('date')} {args.get('time') or ''}", now, p.tz)
    if not title or start is None:
        missing = "a title" if not title else "a time"
        return ToolResult(f"I need {missing} to add that to your calendar.", {"args": args}, ok=False)
    try:
        minutes = max(5, min(12 * 60, int(float(args.get("duration_minutes") or args.get("minutes") or DEFAULT_MINUTES))))
    except (TypeError, ValueError):
        minutes = DEFAULT_MINUTES
    if args.get("end"):
        end = parse_when(str(args["end"]), now, p.tz)
        if end and end > start:
            minutes = int((end - start).total_seconds() // 60)
    people = args.get("attendees") or args.get("with") or []
    attendees = [people] if isinstance(people, str) else [str(x) for x in people]
    with_whom = f" with {_and(attendees)}" if attendees and not any(a.lower() in title.lower() for a in attendees) else ""
    return _propose_event(title, start, minutes, p, now, dry_run, attendees, args.get("location"), "an event",
                          f"{title}{with_whom} for your calendar,")


def reminders_add(args: dict, p: Ports, now: datetime, dry_run: bool = False) -> ToolResult:
    """A reminder is a short calendar event titled "Reminder: ...": it lands where the day plan and the phone see it."""
    text = str(args.get("text") or args.get("title") or "").strip().strip("\"'")
    start = parse_when(str(args.get("at") or args.get("when") or args.get("time") or ""), now, p.tz)
    if not text or start is None:
        missing = "what to remind you of" if not text else "when"
        return ToolResult(f"I need {missing} to set that reminder.", {"args": args}, ok=False)
    text = re.sub(r"^(?:reminder:?\s*|to\s+)", "", text, flags=re.I)
    title = f"Reminder: {text}"
    return _propose_event(title, start, REMINDER_MINUTES, p, now, dry_run, [], None, "a reminder", f"a reminder to {text}")


# --- memory: what the cameras saw ----------------------------------------------------------------------------------

PARTS = {"morning": (5, 12), "afternoon": (12, 17), "evening": (17, 22), "night": (22, 29)}


def memory_last_seen(args: dict, p: Ports, now: datetime, dry_run: bool = False) -> ToolResult:
    if p.last_seen is None:
        return _not_here("Camera memory")
    label = str(args.get("label") or args.get("item") or "").strip().lower()
    label = re.sub(r"^(?:my|the|our|an?)\s+", "", label)
    if not label:
        return ToolResult("What should I look for?", ok=False)
    said = p.last_seen(label)
    return ToolResult(said or f"I haven't seen your {label}.", {"label": label, "found": bool(said)}, link="/timeline/")


def memory_search(args: dict, p: Ports, now: datetime, dry_run: bool = False) -> ToolResult:
    if p.search is None:
        return _not_here("Camera memory")
    q = str(args.get("q") or args.get("query") or "").strip()
    if not q:
        return ToolResult("What should I search for?", ok=False)
    lines = p.search(q)[:3]
    if not lines:
        return ToolResult(f"I didn't see anything about {q}.", {"q": q, "hits": []}, link="/timeline/")
    return ToolResult(lines[0], {"q": q, "hits": lines}, link="/timeline/")


def memory_recap(args: dict, p: Ports, now: datetime, dry_run: bool = False) -> ToolResult:
    if p.recap is None:
        return _not_here("Camera memory")
    part = str(args.get("part") or args.get("when") or "today").strip().lower()
    local = now.astimezone(p.tz)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    if "yesterday" in part:
        midnight -= timedelta(days=1)
    key = next((k for k in PARTS if k in part), None)
    if "hour" in part:
        start, end = now - timedelta(hours=1), now
    elif key:
        lo, hi = PARTS[key]
        start, end = midnight + timedelta(hours=lo), min(midnight + timedelta(hours=hi), now if "yesterday" not in part else midnight + timedelta(hours=hi))
    else:
        start, end = midnight, (now if "yesterday" not in part else midnight + timedelta(days=1))
    if end <= start:
        return ToolResult(f"It isn't {key or part} yet.", {"start": start.isoformat(), "end": end.isoformat()})
    said = p.recap(start, end)
    span = f"this {key}" if key and "yesterday" not in part else part
    return ToolResult(said or f"Nothing on the cameras {span}.", {"start": start.isoformat(), "end": end.isoformat()},
                      link="/timeline/")


# --- the room, people and the cameras --------------------------------------------------------------------------------

def room_map(args: dict, p: Ports, now: datetime, dry_run: bool = False) -> ToolResult:
    room = p.room() if p.room else None
    if not room:
        return ToolResult("There's no room map on this box yet. Opening the room screen.", {"available": False}, link="/map/")
    cams = [str(c) for c in room.get("cameras") or []]
    names = [str(z if isinstance(z, str) else (z.get("name") or z.get("id"))) for z in room.get("zones") or []]
    title = str(room.get("title") or room.get("name") or "room")
    said = f"Here is the map of the {title.lower()}: {len(cams)} camera{'s' if len(cams) != 1 else ''}"
    if names:
        said += f" and {len(names)} zone{'s' if len(names) != 1 else ''}, like {_and(names[:3])}"
    said += "."
    return ToolResult(said, {"title": title, "cameras": cams, "zones": names, "furniture": room.get("furniture")}, link="/map/")


def people_who_is_here(args: dict, p: Ports, now: datetime, dry_run: bool = False) -> ToolResult:
    """Privacy: names only when the owner is signed in on the box, and only of people the owner enrolled."""
    if p.people is None:
        return _not_here("Knowing who is here")
    present = p.people()
    owner = bool(p.owner_signed_in and p.owner_signed_in())
    n = len(present)
    if n == 0:
        return ToolResult("I don't see anyone right now.", {"count": 0}, link="/people/")
    named = [x["name"] for x in present if owner and x.get("known") and x.get("name")]
    if named:
        rest = n - len(named)
        said = f"I can see {_and(named)}" + (f" and {rest} other{'s' if rest != 1 else ''}" if rest else "") + "."
    else:
        said = f"I can see {n} {'person' if n == 1 else 'people'}." + ("" if owner else " Sign in as the owner to hear names.")
    return ToolResult(said, {"count": n, "names": named}, link="/people/")


def camera_status(args: dict, p: Ports, now: datetime, dry_run: bool = False) -> ToolResult:
    if p.cameras is None:
        return _not_here("The cameras")
    s = p.cameras() or {}
    cams = s.get("cameras") or []
    if s.get("paused"):
        said = "The cameras are paused" + (f" until {ear.ear_time(datetime.fromisoformat(s['until']).astimezone(p.tz))}" if s.get("until") else "") + "."
    elif s.get("running") is False:
        said = "The cameras are not running right now."
    else:
        said = f"{len(cams)} camera{'s are' if len(cams) != 1 else ' is'} on."
    return ToolResult(said, {**s}, link="/live/")


def camera_pause(args: dict, p: Ports, now: datetime, dry_run: bool = False) -> ToolResult:
    """Pausing (or resuming) the cameras changes the home: it goes through an approval card like any other outside
    effect. Approved, the privacy policy's paused cameras are set (the same change the Privacy screen makes)."""
    if p.can_pause is None or not p.can_pause():
        return ToolResult("I can't pause the cameras from here. You can stop them from the Privacy screen.",
                          {"available": False}, link="/privacy/", ok=False)
    action = str(args.get("action") or "pause").strip().lower()
    resume = action in ("resume", "on", "start", "unpause", "turn on")
    verb = "turn the cameras back on" if resume else "pause the cameras until you turn them back on"
    if dry_run:
        return ToolResult(f"I would propose to {verb}.", {"resume": resume, "dry_run": True}, link="/approvals/")
    action_id = _action_id("camera", str(resume), now.strftime("%Y-%m-%dT%H:%M"))
    proposal = ActionProposed(ts=now, action_id=action_id, kind="device_control", target=CAMERA_PAUSE,
                              reason=f"{verb[0].upper()}{verb[1:]} (asked on Ask Atlas)", params={"resume": resume})
    return ToolResult(f"I have proposed to {verb}. Approve it on your phone.", {"resume": resume, "action_id": action_id},
                      link=f"/approvals/?focus={action_id}", proposal=proposal)


# --- the box, the lab and the Circle ---------------------------------------------------------------------------------

def box_model(args: dict, p: Ports, now: datetime, dry_run: bool = False) -> ToolResult:
    if p.box_model is None:
        return _not_here("The model info")
    m = p.box_model() or {}
    version = m.get("version") or "an unknown model"
    said = f"The box is serving {version}"
    if m.get("base"):
        said += f", built on {m['base']}"
    said += "."
    if m.get("deployed_at"):
        said += f" It was deployed {ear.ear_about(datetime.fromisoformat(m['deployed_at']).astimezone(p.tz), now.astimezone(p.tz))}."
    return ToolResult(said, m, link="/lab/")


def box_energy_last_answer(args: dict, p: Ports, now: datetime, dry_run: bool = False) -> ToolResult:
    e = p.energy() if p.energy else None
    if not e or e.get("joules") is None:
        return ToolResult("I don't have a power reading for the last answer yet.", {"available": False}, link="/stats/")
    j = float(e["joules"])
    said = f"The last answer took about {j:.0f} joules" if j >= 10 else f"The last answer took about {j:.1f} joules"
    if e.get("seconds"):
        said += f" over {float(e['seconds']):.1f} seconds"
    if e.get("watts"):
        said += f", at about {float(e['watts']):.0f} watts"
    said += "."
    return ToolResult(said, e, link="/stats/")


def learning_status(args: dict, p: Ports, now: datetime, dry_run: bool = False) -> ToolResult:
    s = p.learning() if p.learning else None
    if not s:
        return ToolResult("The Learning lab has no runs on this box yet.", {"available": False}, link="/lab/")
    said = f"The latest lab run is {s.get('run') or s.get('task') or 'unnamed'}"
    if s.get("status"):
        said += f", {s['status']}"
    if s.get("score") is not None:
        said += f", scoring {s['score']}"
    said += "."
    return ToolResult(said, s, link="/lab/")


def friends_shared_brain(args: dict, p: Ports, now: datetime, dry_run: bool = False) -> ToolResult:
    s = p.shared_brain() if p.shared_brain else None
    if not s:
        return ToolResult("The shared brain isn't set up on this box.", {"available": False}, link="/circle/", ok=False)
    on = s.get("enabled") or s.get("consented") or s.get("on")
    said = "The Circle's shared brain is " + ("on" if on else "off") + (f": {s['reason']}" if s.get("reason") else "") + "."
    return ToolResult(said, s, link="/circle/")


INSPECTION_JOBS = 2  # how many do-first jobs are said; the rest are on the screen


def inspection_status(args: dict, p: Ports, now: datetime, dry_run: bool = False) -> ToolResult:
    """The plant floor in Inspection's own words (contexts/spot): its status sentence, which already counts what needs
    the operator and what to keep an eye on, then the first do-first jobs in order. No sentence is written here."""
    s = p.inspection() if p.inspection else None
    if not s or not s.get("status"):
        return _not_here("Inspection")
    jobs = [j for j in s.get("do_first") or [] if j.get("headline")]
    lines = [str(s["status"]).strip()]
    for n, j in enumerate(jobs[:INSPECTION_JOBS], 1):
        parts = [str(x).strip().rstrip(".") for x in (j["headline"], j.get("sentence") or "") if str(x).strip()]
        lines.append(f"{n}. " + ". ".join(parts) + ".")
    data = {"available": True, "status": s["status"], "do_first": jobs, "do_first_total": s.get("do_first_total", len(jobs)),
            "watch": s.get("watch", 0)}
    return ToolResult(" ".join(lines), data, link="/spot/")


def navigate_screen(args: dict, p: Ports, now: datetime, dry_run: bool = False) -> ToolResult:
    want = str(args.get("screen") or args.get("name") or "").strip().lower()
    want = re.sub(r"^(?:the|my)\s+|\s+(?:screen|page|tab)$", "", want)
    path = SCREENS.get(want) or next((v for k, v in SCREENS.items() if k in want), None)
    if path is None:
        return ToolResult(f"I don't know a {want or 'that'} screen.", {"screens": sorted(set(SCREENS))}, ok=False)
    return ToolResult(f"Opening {want}.", {"screen": want, "path": path}, link=path)


TOOLS: list[Tool] = [
    Tool("calendar.list", "The person's calendar for a day (today, tomorrow, a weekday or YYYY-MM-DD) or several days.",
         _obj({"day": S, "days": N}), calendar_list),
    Tool("calendar.add", "Add an event or meeting to the calendar (needs the phone's approval). start: when, e.g. '3 pm tomorrow'.",
         _obj({"title": S, "start": S, "duration_minutes": N, "attendees": {"type": "array", "items": S}}, ["title", "start"]),
         calendar_add, writes=True),
    Tool("calendar.find", "Find when an event is on the calendar by words in its title.", _obj({"query": S}, ["query"]),
         calendar_find),
    Tool("reminders.add", "Set a reminder: what to remind and when (e.g. 'call mom', '6 pm').",
         _obj({"text": S, "at": S}, ["text", "at"]), reminders_add, writes=True),
    Tool("memory.last_seen", "Where and when the cameras last saw an object (keys, phone, wallet).",
         _obj({"label": S}, ["label"]), memory_last_seen),
    Tool("memory.search", "Search everything the cameras saw for a word.", _obj({"q": S}, ["q"]), memory_search),
    Tool("memory.recap", "What happened or what the person did in a part of the day: morning, afternoon, evening, today, last hour, yesterday.",
         _obj({"part": S}, ["part"]), memory_recap),
    Tool("room.map", "Show the room map (cameras and zones).", _obj(), room_map),
    Tool("people.who_is_here", "Who is in the room right now.", _obj(), people_who_is_here),
    Tool("camera.status", "Whether the cameras are on or paused.", _obj(), camera_status),
    Tool("camera.pause", "Turn off (pause) the cameras, or turn them back on with action resume (needs approval).",
         _obj({"action": {"type": "string", "enum": ["pause", "resume"]}}), camera_pause, writes=True),
    Tool("box.model", "Which AI model the box is running right now.", _obj(), box_model),
    Tool("box.energy_last_answer", "How much energy (joules, watts) the last answer cost.", _obj(), box_energy_last_answer),
    Tool("learning.status", "The Learning lab's latest training run and its score.", _obj(), learning_status),
    Tool("friends.shared_brain", "The friends Circle's shared brain: on or off and what it shares.", _obj(), friends_shared_brain),
    Tool("navigate.screen", "Open a screen of the app: calendar, approvals, room map, cameras, timeline, learning lab, box stats, circle, privacy, people.",
         _obj({"screen": S}, ["screen"]), navigate_screen),
    Tool("inspection.status", "The plant floor inspection (SPOT robot): what needs the operator, what to check first, what to watch.",
         _obj(), inspection_status),
]


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self.tools = {t.name: t for t in (tools if tools is not None else TOOLS)}
        self._by_model = {t.model_name: t for t in self.tools.values()}

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name) or self._by_model.get(name)

    def names(self) -> list[str]:
        return list(self.tools)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def run(self, name: str, args: dict, ports: Ports, now: datetime, dry_run: bool = False) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(f"I don't have a {name} tool.", ok=False)
        try:
            return tool.run(dict(args or {}), ports, now, dry_run)
        except Exception as exc:  # a port failing must not fail the answer
            return ToolResult(f"I couldn't do that just now ({type(exc).__name__}).", {"error": str(exc)[:200]}, ok=False)
