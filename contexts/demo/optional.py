"""An optional event, suggested and said yes to, on a day Atlas is running: one invented day through the real app.

    python -m contexts.demo optional --data-dir <an empty directory> [--day 2026-10-12]
    ATLAS_DATA_DIR=<that directory> ATLAS_MEMORY_FIXTURE= ATLAS_CONNECTORS=0 ATLAS_CAPTURE=0 python -m contexts.gateway   # then open /now/

Nothing here happened to anybody. The day is invented for the demo: the friends are the story's fictional ones
(fixtures/demo/nearby.yaml), the lecture and the assignment are the check-in demo's (contexts/demo/checkin.py), and
the person at the desk is a detector track drawn into the desk zone.

What is published, as source events through the app's own bus, in event time, so every rule that fires is the real
one and nothing Atlas says is written by this file:

- 7:55 AM, before the household's wake time: the calendar and Canvas sync (one lecture, one assignment due
  tomorrow) and "Friends' dinner" from the Near you fixture as today's, 5:15 PM at '63 South, with Riley, Sam, Alex.
- 8:05 AM: the desk camera sees someone sit down, and the Day loop makes the morning plan.
- A few minutes before the afternoon's study block at the desk, and again at 3:50 PM: the desk camera sees someone
  sit down.

The Day loop is ticked once a minute, as the gateway's ticker does, around the study block's start and from 4:00 PM
to the time to leave. The person does three things, each through the call the phone makes:

- 8:06 AM, "Approve" on the morning plan's card. Block reminders are for a plan that was approved.
- 4:02 PM, "Add it": `OptionalEvents.answer(..., "yes")`, the function behind POST /api/day/optional/{id}. It
  publishes the answer, then a re-plan with the dinner as a block and one approval card. The plan has not changed.
- 4:03 PM, "Approve" on that card.

An approval is `experience.approval.decided`, published here in place of the tap (demo, approved locally), and
only for a card that stays on the box: a plan card on a box with no calendar connector, which writes nothing. On
any other card the scene stops and says so.

What Atlas then says by itself, each line from the real rules: the morning plan; the study block as it starts; the
suggestion at 4:00 PM, once; the day with the dinner worked in; and "Time to go" ten minutes before the time to
leave. The yes comes two minutes after the suggestion: inside the arbiter's ninety seconds between two unasked
lines the answer in words would wait its turn. After the approval the next block of the day is the dinner itself,
and an accepted event with a walk is announced by its leave-now line, not by a block-start line; the block-start
line of this day is the study block's. The scene ends at the time to leave, because the box's cameras are at home
and cannot see the walk.

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
from contracts.day import DayNarrated, DayReplanProposed, PlanMove
from contracts.events import Event
from contracts.experience import ApprovalDecided
from contracts.perception import ObservationRecorded
from contracts.types import Box, Track, frame_ref

LOCAL = ZoneInfo("America/New_York")
SLUG = "friends-dinner"  # the Near you fixture's event this day is about
CALENDAR_WRITE = "calendar.create_event"
SYNCED, MORNING, PLAN_OK = time(7, 55), time(8, 5), time(8, 6)
NOON, BACK, FROM, YES, APPROVE = time(12, 0), time(15, 50), time(16, 0), time(16, 2), time(16, 3)
LAST = time(17, 15)  # the scene never runs past the event's start, whatever the walk
SIT_BEFORE, WATCH_BEFORE, WATCH_AFTER = timedelta(minutes=6), timedelta(minutes=1), timedelta(minutes=2)  # around the study block's start
SIT_S, EVERY_S = 10, 2  # a few looks of the desk camera: someone sat down
LOCALLY = "demo, approved locally: no calendar connector on this box, nothing is written"


def default_day(now: datetime) -> date:
    return now.astimezone(LOCAL).date()


def at(day: date, t: time) -> datetime:
    return datetime.combine(day, t, LOCAL)


def clock(t: datetime) -> str:
    t = t.astimezone(LOCAL)
    return f"{t.hour % 12 or 12}:{t:%M} {t:%p}"


def dinner():
    """The fixture's friends' dinner, as today's."""
    return next(e for e in nearby.load().events if e.slug == SLUG).model_copy(update={"day": 0})


def synced(day: date) -> list[ItemSynced]:
    """The morning's sync: the check-in demo's lecture and assignment, and the fixture's dinner as today's."""
    ts = at(day, SYNCED)
    items = [e.model_copy(update={"ts": ts, "event_id": f"demo-optional-{e.external_id}-{day}"}) for e in checkin.calendar(at(day, NOON))]
    return items + nearby.nearby_events(ts, nearby.load().model_copy(update={"events": [dinner()], "messages": []}))


def offer_id(day: date) -> str:
    return nearby.external_id(dinner(), at(day, SYNCED))


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
    started: DayNarrated | None = None  # the study block's start, said
    suggested: DayNarrated | None = None
    replan: DayReplanProposed | None = None
    approved: bool = False  # the yes's card was approved and the dinner is a block of the day
    leave: DayNarrated | None = None
    stopped: str | None = None  # why the scene ended early, in plain words

    @property
    def ok(self) -> bool:
        return bool(self.suggested and self.replan and self.approved and self.leave and not self.stopped)


def stays_on_the_box(app: Atlas, card) -> bool:
    """A card this scene may approve in the person's place: a plan card on a box that cannot write to a calendar."""
    return (card is not None and card.status == "proposed" and card.kind == "connector_write" and card.target == CALENDAR_WRITE
            and not app.hub.is_write(CALENDAR_WRITE))


def study_block(app: Atlas, day: date) -> PlanMove | None:
    """The plan's first block of the afternoon that Atlas chose at the desk, before the suggestion's hour: the block
    whose start the scene waits for."""
    plan = app.day.planner.current.get(day)
    desk = app.day.tracker.walks.ref(app.dayrun.cfg.desk)
    blocks = sorted(plan.moves, key=lambda m: m.start) if plan else []
    return next((m for m in blocks if not m.fixed and m.kind == "study" and m.place == desk
                 and at(day, NOON) <= m.start and m.start + WATCH_AFTER < at(day, BACK)), None)


async def show(data_dir: str | Path, day: date, config_dir: Path | None = None) -> Shown:
    """Publish the invented day into an empty `data_dir`. Refuses a directory that already has a log."""
    data_dir = Path(data_dir)
    if list((data_dir / "events").glob("*.jsonl")) or (data_dir / "memory.sqlite").exists():
        raise SeedRefused(f"{data_dir} already has an event log or a memory file. The optional-event demo wants an empty directory.")
    begun = clocktime.perf_counter()
    app = build_app(data_dir, config_dir)
    if app.day is None:
        raise RuntimeError("this config has no Day loop (no place packs): optional events live there")
    now = {"t": at(day, SYNCED)}
    app.qa.clock = lambda: now["t"]  # event time, as the seed does
    shown = Shown(app=app, day=day, seconds=0.0)
    told: list[DayNarrated] = []
    app.bus.subscribe("day.narrated", told.append)
    offer, zones = offer_id(day), app.settings.config_dir / "zones.yaml"
    mark = {"n": 0, "approvals": 0}

    def said() -> None:
        """Every line Atlas said aloud since the last look: not the screen's quiet "now", not a line said inside another."""
        for e in told[mark["n"]:]:
            if e.merged_into or (e.moment == "now" and not e.urgent):
                continue
            shown.lines.append((e.ts, "Atlas", e.spoken))
            if e.moment == "start" and shown.started is None:
                shown.started = e
            if e.moment == "suggest" and e.ref == offer:
                shown.suggested = e
            if e.moment == "now" and e.urgent and shown.approved:
                shown.leave = e
        mark["n"] = len(told)

    async def publish(events) -> None:
        for e in events:
            now["t"] = max(now["t"], e.ts)
            await app.bus.publish(e)
            await settle()
        said()

    async def minute(t: datetime) -> None:
        """One turn of the ticker at `t`: cards that ran out expire, then the Day loop looks at the day."""
        now["t"] = t
        await expire(app, t)
        await publish(app.day.tick(t))

    async def approve(action_id: str | None, t: datetime) -> bool:
        """The tap on "Approve", for a card that stays on the box; False (and nothing published) for any other."""
        card = app.approvals.requests.get(action_id or "")
        if not stays_on_the_box(app, card):
            return False
        mark["approvals"] += 1
        shown.lines.append((t, "You", f"Approve ({LOCALLY})"))
        await publish([ApprovalDecided(event_id=f"demo-optional-approved-{day}-{mark['approvals']}", ts=t, action_id=card.id, decision="approve")])
        return True

    await publish(synced(day))
    await publish(sits_down(at(day, MORNING), zones))  # the first sign of the day after wake: the morning plan
    plan = app.day.planner.current.get(day)
    if plan is None or not await approve(plan.action_id, at(day, PLAN_OK)):
        await expire(app, at(day, PLAN_OK) + timedelta(minutes=5))  # nobody answered it: no block reminders today

    block = study_block(app, day)
    if block is not None:  # the afternoon's study block, as it starts
        await publish(sits_down(block.start - SIT_BEFORE, zones))
        t = block.start - WATCH_BEFORE
        while t <= block.start + WATCH_AFTER:
            await minute(t)
            t += timedelta(minutes=1)

    await publish(sits_down(at(day, BACK), zones))
    t = at(day, FROM)
    while t <= at(day, LAST):
        now["t"] = t
        if t == at(day, YES) and shown.suggested is not None:
            shown.lines.append((t, "You", f"Add it (yes to {app.day.optional.find(offer, t).title})"))
            try:
                events, body = app.day.optional.answer(t, offer, "yes")
            except Exception as refused:  # the box would answer 409 with these words
                shown.stopped = f"the yes was refused: {refused}"
                break
            shown.replan = next((e for e in events if isinstance(e, DayReplanProposed)), None)
            for change in body.get("moves", []):
                was = clock(at(day, time.fromisoformat(change["start"])))
                shown.lines.append((t, "Plan", f"would change: {change['title']} at {was} ({change['change']})"))
            await publish(events)
        if t == at(day, APPROVE) and shown.replan is not None:
            if not await approve(shown.replan.action_id, t):
                shown.stopped = "the card is not one that stays on the box, so it was left for the phone"
                break
            shown.approved = offer in app.day.optional.accepted
            mine = next((m for m in app.day.planner.current[day].moves if m.id == rule.block_id(offer)), None)
            if mine is not None:
                shown.lines.append((t, "Plan", f"{mine.title}, {clock(mine.start)} to {clock(mine.end)} at {mine.place_name}, is a block of the day"))
        await minute(t)
        status = app.day.tracker.status
        mine = status.next_event if status and status.next_event and status.next_event.id == rule.block_id(offer) else None
        if shown.leave is not None and mine is not None and mine.leave_by is not None and t >= mine.leave_by:
            break  # the time to leave: the cameras are at home, the walk is the person's
        t += timedelta(minutes=1)
    shown.seconds = clocktime.perf_counter() - begun
    return shown


def lines(s: Shown) -> list[str]:
    """The scene as it is printed: the same for the same day, whatever the directory and however long it took."""
    out = [f"An invented {s.day:%A} ({s.day.isoformat()}), through the real app. Nothing here happened to anybody.", ""]
    out += [f"  {clock(when):>8}  {who:<5}  {what}" for when, who, what in s.lines]
    out.append("")
    if s.ok:
        out.append("Suggested once, added only after its card was approved, and told when to leave. Every line Atlas says above is the app's own.")
        if s.started is None:
            out.append("No block of the approved plan started while the scene was watching, so there is no block-start line today.")
    else:
        missing = [name for name, got in (("no suggestion", s.suggested), ("no re-plan", s.replan), ("the dinner was not added", s.approved),
                                          ("no time-to-go line", s.leave)) if not got]
        out.append("The scene did not play through: " + (s.stopped or ", ".join(missing)) + ".")
    return out


def summary(s: Shown) -> str:
    return "\n".join([*lines(s), "", "Run the app on it, then open /now/:",
                      f"  ATLAS_DATA_DIR={s.app.settings.data_dir} ATLAS_MEMORY_FIXTURE= ATLAS_CONNECTORS=0 ATLAS_CAPTURE=0 python -m contexts.gateway"])
