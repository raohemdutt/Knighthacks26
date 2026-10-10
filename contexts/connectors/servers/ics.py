"""ICS feeds as an MCP connector: python -m contexts.connectors.servers.ics

One server reads any number of published .ics feeds: the Outlook calendar (K2
reads), the campus events calendar (K3), and Canvas's calendar feed (the K6
fallback). Adapted from the home Atlas runner's IcsCalendarClient:

- Exchange writes `TZID:Eastern Standard Time`; icalendar builds the real zone
  from the feed's VTIMEZONE blocks, so classes are not an hour off.
- Timetables are mostly RRULE plus moved RECURRENCE-ID instances; expansion
  is delegated to recurring_ical_events instead of hand-rolled.
- A published feed URL is a bearer credential, so it is read from the
  environment (`url_env`), never from the repo.

options:
  feeds:
    - {name: outlook, url_env: OUTLOOK_ICS_URL, kind: calendar_event}
    - {name: canvas, url_env: CANVAS_ICS_URL, kind: assignment}
  days_back: 1
  days_ahead: 14
"""

import datetime as dt
import os
import uuid
from pathlib import Path

import httpx
import icalendar
import recurring_ical_events
from mcp.server.fastmcp import FastMCP

from contexts.connectors.infrastructure.mcp_client import server_options

MAX_EVENTS = 300


def _text(component, key: str) -> str | None:
    value = component.get(key)
    text = str(value).strip() if value is not None else ""
    return text or None


def _as_datetime(value, tz: dt.tzinfo) -> tuple[dt.datetime, bool]:
    if isinstance(value, dt.datetime):
        return (value if value.tzinfo else value.replace(tzinfo=tz)), False
    return dt.datetime.combine(value, dt.time.min, tzinfo=tz), True  # all-day


def expand(raw: bytes, start: dt.datetime, end: dt.datetime, feed: str, kind: str) -> list[dict]:
    """Occurrences in [start, end) as connector items, earliest first."""
    calendar = icalendar.Calendar.from_ical(raw)
    items = []
    for occ in recurring_ical_events.of(calendar).between(start, end):
        begin = occ.get("DTSTART")
        if begin is None:
            continue
        begins, all_day = _as_datetime(begin.dt, start.tzinfo)
        finish = occ.get("DTEND")
        ends = _as_datetime(finish.dt, start.tzinfo)[0] if finish is not None else None
        busy = (_text(occ, "X-MICROSOFT-CDO-BUSYSTATUS") or "BUSY").upper() != "FREE"
        details = [
            "all day" if all_day else f"until {ends:%I:%M %p}".replace(" 0", " ") if ends else None,
            _text(occ, "LOCATION"),
            None if busy else "free",
        ]
        uid = _text(occ, "UID") or begins.isoformat()
        items.append({
            "external_id": f"{feed}:{uid}:{begins.isoformat()}",  # one item per occurrence
            "kind": kind,
            "title": _text(occ, "SUMMARY") or "(no title)",
            "when": begins.isoformat(),
            "url": _text(occ, "URL"),
            "summary": " · ".join(d for d in details if d) or None,
        })
    items.sort(key=lambda i: i["when"])
    return items[:MAX_EVENTS]


def read_feeds(options: dict, now: dt.datetime, fetch) -> list[dict]:
    start = now - dt.timedelta(days=options.get("days_back", 1))
    end = now + dt.timedelta(days=options.get("days_ahead", 14))
    items: list[dict] = []
    for feed in options.get("feeds", []):
        url = feed.get("url") or os.environ.get(feed.get("url_env", ""), "")
        if not url:
            raise RuntimeError(f"feed {feed['name']}: set {feed.get('url_env')} in .env")
        items += expand(fetch(url), start, end, feed["name"], feed.get("kind", "calendar_event"))
    return items


def _fetch(url: str) -> bytes:
    r = httpx.get(url, timeout=20, follow_redirects=True)
    r.raise_for_status()
    return r.content


def make_invite(events: list[dict], organizer: str | None = None, now: dt.datetime | None = None) -> bytes:
    """An .ics invite (METHOD:REQUEST) any calendar app can import: Outlook, Google, Apple.

    The fallback for creating events when the school tenant won't grant a
    Microsoft Graph token. Each event: {title, start, end, location?, attendees?}."""
    now = now or dt.datetime.now(dt.UTC)
    cal = icalendar.Calendar()
    cal.add("prodid", "-//Atlas//Knight Hacks//EN")
    cal.add("version", "2.0")
    cal.add("method", "REQUEST")
    for e in events:
        ev = icalendar.Event()
        start = dt.datetime.fromisoformat(e["start"]) if isinstance(e["start"], str) else e["start"]
        end = dt.datetime.fromisoformat(e["end"]) if isinstance(e["end"], str) else e["end"]
        ev.add("uid", f"{uuid.uuid4()}@atlas")
        ev.add("dtstamp", now)
        # UTC ("...Z") is the one form every calendar app imports correctly; offsets like
        # "UTC-04:00" from isoformat strings are not valid TZIDs.
        ev.add("dtstart", start.astimezone(dt.UTC))
        ev.add("dtend", end.astimezone(dt.UTC))
        ev.add("summary", e["title"])
        if e.get("location"):
            ev.add("location", e["location"])
        if organizer:
            ev.add("organizer", f"mailto:{organizer}")
        for who in e.get("attendees", []):
            ev.add("attendee", f"mailto:{who}" if "@" in who else who, parameters={"RSVP": "TRUE"})
        cal.add_component(ev)
    return cal.to_ical()


def write_invite(events: list[dict], out_dir: Path, organizer: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"invite-{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}.ics"
    path.write_bytes(make_invite(events, organizer))
    return path


mcp = FastMCP("atlas-ics")


@mcp.tool()
def list_events() -> list[dict]:
    """Calendar occurrences from every configured .ics feed, from yesterday to two weeks out."""
    return read_feeds(server_options(), dt.datetime.now(dt.UTC).astimezone(), _fetch)


@mcp.tool()
def create_event(
    title: str | None = None,
    start: str | None = None,
    end: str | None = None,
    location: str | None = None,
    attendees: list[str] | None = None,
    events: list[dict] | None = None,
) -> str:
    """Write tool (runs only after an approval card): create one event, or several via `events`,
    as an .ics invite file under options.invites_dir. Returns the file path to share."""
    batch = events or [{"title": title, "start": start, "end": end, "location": location, "attendees": attendees or []}]
    if not all(e.get("title") and e.get("start") and e.get("end") for e in batch):
        raise ValueError("each event needs title, start and end")
    opts = server_options()
    path = write_invite(batch, Path(opts.get("invites_dir", "data/invites")), opts.get("organizer"))
    return f"created {len(batch)} event(s) as {path}"


if __name__ == "__main__":
    mcp.run()
