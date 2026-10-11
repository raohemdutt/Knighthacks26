"""Near you: optional events for the Calendar screen, as source events (fixtures/demo/nearby.yaml).

    events = nearby_events(now)      # what `seed` adds to the night; `circle` gets them through `seed`

One `connector.item.synced` per event (`connector: calendar`, `kind: calendar_event`, `external_id`
`nearby:<slug>:<start ISO>`), and one per group chat message about an event, in the GroupMe connector's item
format. No new contract, endpoint or logic: `GET /memory/items` returns them like any other item.

`nearby:` (PREFIX) is the mark of an optional event. The consumers that turn calendar events into commitments
skip it, each in one line: the Day Run board (`ignored_event_prefixes` in config/dayrun.yaml), the leaving
checklist's "next" (contexts/agent/application/planner.py), the Circle's free/busy answer
(contexts/circle/application/protocol.py) and Habits' calendar (contexts/habits/domain/tracker.py).

Deterministic: the same `now` gives the same events with the same ids, so seeding twice adds nothing.
"""

from datetime import datetime, time, timedelta
from pathlib import Path

import yaml
from pydantic import Field

from contexts.demo.story import LOCAL, ROOT, _clock, _Strict
from contexts.play.domain.places import PlacePack, parse_pack
from contracts.connectors import ItemSynced

FILE = ROOT / "fixtures" / "demo" / "nearby.yaml"
PLACES = ROOT / "config" / "places"
PREFIX = "nearby:"
TITLE_LEN = 80  # the GroupMe connector cuts a message's title here (contexts/connectors/servers/groupme.py)
DAYS = 3  # every start is after `now` and less than this many days from it


class Near(_Strict):
    slug: str
    title: str
    day: int = Field(ge=0, lt=DAYS)  # local days from `now`
    at: time  # local start
    minutes: int = Field(gt=0)
    place: str  # a place id of the pack
    tags: list[str] = []


class Chat(_Strict):
    about: str  # an event's slug
    external_id: str
    author: str
    chat: str
    minutes_ago: int
    text: str


class Nearby(_Strict):
    pack: str = "ucf"
    synced_minutes_ago: int = 5
    events: list[Near] = []
    messages: list[Chat] = []


def load(path: str | Path = FILE) -> Nearby:
    return Nearby.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def pack(name: str, folder: Path = PLACES) -> PlacePack:
    return parse_pack((folder / f"{name}.yaml").read_text(encoding="utf-8"))


def start_of(e: Near, now: datetime) -> datetime:
    """`day` local days from `now` at `at`; a today event that has already started is tomorrow's."""
    start = datetime.combine(now.astimezone(LOCAL).date() + timedelta(days=e.day), e.at, LOCAL)
    return start if start > now else start + timedelta(days=1)


def external_id(e: Near, now: datetime) -> str:
    return f"{PREFIX}{e.slug}:{start_of(e, now).isoformat()}"


def nearby_events(now: datetime, script: Nearby | None = None) -> list[ItemSynced]:
    """The events and their chat messages as the connectors would have synced them shortly before `now`."""
    script = script or load()
    places = pack(script.pack).places
    synced = now - timedelta(minutes=script.synced_minutes_ago)
    starts: dict[str, datetime] = {}
    out: list[ItemSynced] = []
    for e in script.events:
        start = starts[e.slug] = start_of(e, now)
        name = places[e.place].name.split(" (")[0]  # "Engineering II (Knight Hacks)" is "Engineering II" in a sentence
        eid = external_id(e, now)
        out.append(ItemSynced(
            event_id=f"demo-item-calendar-{eid}-{int(synced.timestamp())}", ts=synced, connector="calendar", kind="calendar_event",
            external_id=eid, title=e.title, when=start,
            summary=" · ".join([f"until {_clock(start + timedelta(minutes=e.minutes))}", name, *e.tags])))
    for m in script.messages:
        start, posted = starts[m.about], now - timedelta(minutes=m.minutes_ago)
        text = m.text.format(weekday=f"{start:%A}".lower(), time=_clock(start).replace(" ", "").lower())
        ts = posted + timedelta(minutes=1)
        out.append(ItemSynced(
            event_id=f"demo-item-groupme-{m.external_id}-{int(ts.timestamp())}", ts=ts, connector="groupme", kind="message",
            external_id=m.external_id, title=f"{m.author}: {text}"[:TITLE_LEN], when=posted,
            summary=f"{m.author} in {m.chat}: {text}"))
    return sorted(out, key=lambda e: (e.ts, e.event_id))
