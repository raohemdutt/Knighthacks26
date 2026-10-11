"""An optional event, suggested and said yes to: one invented afternoon through the real app.

    python -m contexts.demo optional --data-dir <an empty directory> [--day 2026-10-12]
    ATLAS_DATA_DIR=<that directory> ATLAS_MEMORY_FIXTURE= ATLAS_CONNECTORS=0 ATLAS_CAPTURE=0 python -m contexts.gateway   # then open /now/

Nothing here happened to anybody. The day is invented for the demo: the friends are the story's fictional ones
(fixtures/demo/nearby.yaml), the lecture and the assignment are the check-in demo's (contexts/demo/checkin.py), and
the person at the desk is a detector track drawn into the desk zone.

What is published, as source events through the app's own bus, in event time, so every rule that fires is the real
one and nothing Atlas says is written by this file:

- 07:55, before the household's wake time: the calendar and Canvas sync (one lecture, one assignment due tomorrow)
  and "Friends' dinner" from the Near you fixture as today's, 5:15 PM at '63 South, with Riley, Sam and Alex.
- 08:05: the desk camera sees someone sit down. The Day loop makes the morning plan; nobody answers its card and it
  expires, as on a box nobody looked at.
- 3:50 PM: the desk camera sees someone sit down again.

Then the Day loop is ticked once a minute from 4:00 PM, as the gateway's ticker does, and the person does two
things, both through the calls the phone makes:

- 4:02 PM, "Add it": `OptionalEvents.answer(..., "yes")`, the function behind POST /api/day/optional/{id}. It
  publishes the answer, then a re-plan with the dinner as a block and one approval card. The plan has not changed.
- 4:03 PM, "Approve": `experience.approval.decided` for that card, published here in place of the tap (demo,
  approved locally). Only a card that stays on the box is approved: a plan card on a box with no calendar
  connector, which writes nothing. On any other card the scene stops and says so.

What Atlas then says by itself, each line from the real rules: the suggestion at 4:00 PM (once), the day with the
dinner worked in at 4:02 PM, and "Time to go" ten minutes before the time to leave. An accepted event with a walk
is announced by that leave-now line, not by a block-start line; the scene ends at the time to leave, because the
box's cameras are at home and cannot see the walk.

Deterministic: the same day gives the same lines, whatever the directory.
"""

import time as clocktime
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from contexts.day.domain import optional as rule
from contexts.demo import checkin, nearby
from contexts.demo.seed import SeedRefused, build_app, expire, settle
from contexts.demo.story import ZONES
from contexts.gateway.bootstrap import Atlas
from contexts.perception.domain.scenes import bounding_box
from contexts.perception.domain.zones import load_zones
from contracts.connectors import ItemSynced
from contracts.day import DayNarrated, DayReplanProposed
from contracts.events import Event
from contracts.experience import ApprovalDecided
from contracts.perception import ObservationRecorded
from contracts.types import Box, Track, frame_ref

LOCAL = ZoneInfo("America/New_York")
SLUG = "friends-dinner"  # the Near you fixture's event this day is about
CALENDAR_WRITE = "calendar.create_event"
SYNCED, MORNING, CARD_GONE = time(7, 55), time(8, 5), time(8, 11)
BACK, FROM, YES, APPROVE = time(15, 50), time(16, 0), time(16, 2), time(16, 3)
LAST = time(17, 15)  # the scene never runs past the event's start, whatever the walk
SIT_S, EVERY_S = 10, 2  # a few looks of the desk camera: someone sat down


def default_day(now: datetime) -> date:
    return now.astimezone(LOCAL).date()


def at(day: date, t: time) -> datetime:
    return datetime.combine(day, t, LOCAL)


def clock(t: datetime) -> str:
    t = t.astimezone(LOCAL)
    return f"{t.hour % 12 or 12}:{t:%M} {t:%p}"


def synced(day: date) -> list[ItemSynced]:
    """The morning's sync: the check-in demo's lecture and assignment, and the fixture's dinner as today's."""
    ts = at(day, SYNCED)
    items = [e.model_copy(update={"ts": ts, "event_id": f"demo-optional-{e.external_id}-{day}"}) for e in checkin.calendar(at(day, time(12)))]
    script = nearby.load()
    dinner = next(e for e in script.events if e.slug == SLUG).model_copy(update={"day": 0})
    return items + nearby.nearby_events(ts, script.model_copy(update={"events": [dinner], "messages": []}))


def offer_id(day: date) -> str:
    dinner = next(e for e in nearby.load().events if e.slug == SLUG).model_copy(update={"day": 0})
    return nearby.external_id(dinner, at(day, SYNCED))


def sits_down(start: datetime, zones: str | Path = ZONES) -> list[Event]:
    """What the desk camera would publish as someone sits down: a person's track in the desk zone, a few looks."""
    desk = bounding_box(load_zones(zones)[checkin.DESK_CAM][checkin.DESK])
    x, bottom = (desk.x1 + desk.x2) / 2, desk.y2 - 0.05  # the bottom centre of the box is what stands in the zone
    person = Track(track_id=1, label="person", confidence=0.9,
                   box=Box(x1=round(x - 0.15, 3), y1=round(bottom - 0.5, 3), x2=round(x + 0.15, 3), y2=round(bottom, 3)))
    out: list[Event] = []
    for s in range(0, SIT_S + 1, EVERY_S):
        ts = start + timedelta(seconds=s)
        ms = int(ts.timestamp() * 1000)
        out.append(ObservationRecorded(event_id=f"demo-optional-{checkin.DESK_CAM}-{ms}", ts=ts, camera_id=checkin.DESK_CAM, tracks=[person],
                                       frame_ref=frame_ref(checkin.DESK_CAM, ms)))
    return out


@dataclass
class Shown:
    app: Atlas
    day: date
    seconds: float
    lines: list[tuple[datetime, str, str]] = field(default_factory=list)  # (when, who, what), in order
    suggested: DayNarrated | None = None
    replan: DayReplanProposed | None = None
    approved: bool = False
    leave: DayNarrated | None = None
    stopped: str | None = None  # why the scene ended early, in plain words

    @property
    def ok(self) -> bool:
        return bool(self.suggested and self.replan and self.approved and self.leave and not self.stopped)


def stays_on_the_box(app: Atlas, card) -> bool:
    """A card this scene may approve in the person's place: a plan card on a box that cannot write to a calendar."""
    return card.kind == "connector_write" and card.target == CALENDAR_WRITE and not app.hub.is_write(CALENDAR_WRITE)


async def show(data_dir: str | Path, day: date, config_dir: Path | None = None) -> Shown:
    """Publish the invented day into an empty `data_dir`. Refuses a directory that already has a log."""
    data_dir = Path(data_dir)
    if list((data_dir / "events").glob("*.jsonl")) or (data_dir / "memory.sqlite").exists():
        raise SeedRefused(f"{data_dir} already has an event log or a memory file. The optional-event demo wants an empty directory.")
    started = clocktime.perf_counter()
    app = build_app(data_dir, config_dir)
    if app.day is None:
        raise RuntimeError("this config has no Day loop (no place packs): optional events live there")
    now = {"t": at(day, SYNCED)}
    app.qa.clock = lambda: now["t"]  # event time, as the seed does
    shown = Shown(app=app, day=day, seconds=0.0)
    told: list[DayNarrated] = []
    app.bus.subscribe("day.narrated", told.append)

    async def publish(events) -> None:
        for e in events:
            now["t"] = max(now["t"], e.ts)
            await app.bus.publish(e)
            await settle()

    def said_since(mark: int) -> None:
        """Every line Atlas said aloud since `mark`: not the screen's quiet "now", not a line said inside another."""
        for e in told[mark:]:
            if e.merged_into or (e.moment == "now" and not e.urgent):
                continue
            shown.lines.append((e.ts, "Atlas", e.spoken))
            if e.moment == "suggest" and e.ref == offer:
                shown.suggested = e
            if e.moment == "now" and e.urgent and shown.approved:
                shown.leave = e

    offer = offer_id(day)
    zones = app.settings.config_dir / "zones.yaml"
    await publish(synced(day))
    await publish(sits_down(at(day, MORNING), zones))  # the first sign of the day after wake: the morning plan
    await expire(app, at(day, CARD_GONE))  # nobody answered its card
    await publish(sits_down(at(day, BACK), zones))

    t, mark = at(day, FROM), len(told)
    while t <= at(day, LAST):
        now["t"] = t
        await expire(app, t)
        if t == at(day, YES) and shown.suggested is not None:
            title = app.day.optional.find(offer, t).title
            shown.lines.append((t, "You", f"Add it (yes to {title})"))
            try:
                events, body = app.day.optional.answer(t, offer, "yes")
            except Exception as refused:  # the box would answer 409 with these words
                shown.stopped = f"the yes was refused: {refused}"
                break
            await publish(events)
            shown.replan = next((e for e in events if isinstance(e, DayReplanProposed)), None)
            for change in body.get("moves", []):
                shown.lines.append((t, "Plan", f"would change: {change['title']} at {clock(at(day, time.fromisoformat(change['start'])))} ({change['change']})"))
        if t == at(day, APPROVE) and shown.replan is not None:
            card = app.approvals.requests.get(shown.replan.action_id or "")
            if card is None or card.status != "proposed" or not stays_on_the_box(app, card):
                shown.stopped = "the card is not one that stays on the box, so it was left for the phone"
                break
            shown.lines.append((t, "You", "Approve (demo, approved locally: no calendar connector on this box, nothing is written)"))
            await publish([ApprovalDecided(event_id=f"demo-optional-approved-{day}", ts=t, action_id=card.id, decision="approve")])
            shown.approved = offer in app.day.optional.accepted
            block = next((m for m in app.day.planner.current[day].moves if m.id == rule.block_id(offer)), None)
            if block is not None:
                shown.lines.append((t, "Plan", f"{block.title}, {clock(block.start)} to {clock(block.end)} at {block.place_name}, is a block of the day"))
        await publish(app.day.tick(t))
        said_since(mark)
        mark = len(told)
        status = app.day.tracker.status
        leave_by = status.next_event.leave_by if status and status.next_event and status.next_event.id == rule.block_id(offer) else None
        if shown.leave is not None and leave_by is not None and t >= leave_by:
            break  # the time to leave: the cameras are at home, the walk is the person's
        t += timedelta(minutes=1)
    shown.seconds = clocktime.perf_counter() - started
    return shown


def lines(s: Shown) -> list[str]:
    """The scene as it is printed: the same for the same day, whatever the directory and however long it took."""
    out = [f"An invented {s.day:%A} ({s.day.isoformat()}), through the real app. Nothing here happened to anybody.", ""]
    out += [f"  {clock(when):>8}  {who:<5}  {what}" for when, who, what in s.lines]
    out.append("")
    if s.ok:
        out.append("Suggested once, added only after the card was approved, and told when to leave: every line above is the app's own.")
    else:
        missing = [name for name, got in (("no suggestion", s.suggested), ("no re-plan", s.replan), ("not approved", s.approved),
                                          ("no time-to-go line", s.leave)) if not got]
        out.append("The scene did not play through: " + (s.stopped or ", ".join(missing)) + ".")
    return out


def summary(s: Shown) -> str:
    return "\n".join([*lines(s), "", "Run the app on it, then open /now/:",
                      f"  ATLAS_DATA_DIR={s.app.settings.data_dir} ATLAS_MEMORY_FIXTURE= ATLAS_CONNECTORS=0 ATLAS_CAPTURE=0 python -m contexts.gateway"])
