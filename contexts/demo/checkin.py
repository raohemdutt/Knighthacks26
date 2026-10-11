"""The check-in, shown (D4): one ordinary afternoon that ends in Atlas speaking up, through the real app.

    python -m contexts.demo checkin --data-dir <an empty directory> [--end 2026-10-10T15:40:00-04:00]
    ATLAS_DATA_DIR=<that directory> ATLAS_MEMORY_FIXTURE= ATLAS_CONNECTORS=0 python -m contexts.gateway   # then open /now/

Nothing here was measured on anybody. Every value is invented for the demo: the nights are written in the
health connector's own item format (contexts/connectors/domain/health.py `to_item`), the looks are
`perception.expression.observed` events, the bottle is a detector track on the desk camera. The real watch
export (fixtures/watch) is not read.

What is published, as source events through the app's own bus, so the rule that fires is the real one
(contexts/day/domain/checkin.py) and nothing about the check-in is written by this file:

- the morning (`end`'s local day at 08:05): the calendar and Canvas sync (one lecture, one assignment due
  tomorrow) and the watch's sync: ten nights, the last one short with the resting heart rate up, three of the
  earlier ones the same. The Day loop makes the morning plan from them.
- the afternoon: someone sits down at the desk 105 minutes before `end` and stays; a bottle is on the desk,
  picked up once, 100 minutes before `end`, and never again; from 21 minutes before `end` the looks read as
  strained.
- before any of it, three earlier days in the log (6, 4 and 2 days back: the days of the three short nights):
  the coach's notes of an afternoon block that got under half its planned minutes at the desk. They are
  written straight to the log and replayed, the way a box that has been running for a week would have them,
  so nothing is said or pushed about them now.

At `end` the Day loop's tick looks at the rule once: it raises `day.checkin.raised` with all six signals, the
narrator says it, and "Move the block" is offered because the re-plan really can move the block to after dinner.
`end` defaults to now when it is afternoon, else to today 15:40 local, so "since two" and "after dinner" read
right; any time of day works with --end, and the sentence then says what that day really holds.
"""

import time as clocktime
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from bus import EventBus
from contexts.connectors.domain.health import Night, to_item
from contexts.demo.seed import SeedRefused, build_app, settle
from contexts.demo.story import ZONES
from contexts.gateway.bootstrap import Atlas
from contexts.perception.domain.scenes import bounding_box, scene
from contexts.perception.domain.zones import load_zones
from contracts.connectors import ItemSynced
from contracts.day import DayCheckinRaised, DayCoachNoted, DayNarrated, SessionReview
from contracts.events import Event
from contracts.perception import ExpressionObserved, ObservationRecorded
from contracts.types import Box, Track, frame_ref

LOCAL = ZoneInfo("America/New_York")
DESK_CAM, DESK = "cam-desk", "desk"
MEMBER = "m-demo"  # an enrolled member who switched "read my expressions" on, in this story only
SIT_MIN, DRINK_MIN, STRAINED_MIN = 105, 100, 21
EVERY_S = 2  # under the zone engine's 3 s exit: one unbroken stay
LIKE_THIS = (6, 4, 2)  # days back that had a short night with the resting heart rate up
FELL_AT = {6: time(15, 0), 4: time(15, 30), 2: time(16, 0)}  # when that day's afternoon block began


def default_end(now: datetime) -> datetime:
    """Now when it is afternoon; else today at 15:40 local, where the story's words fit."""
    local = now.astimezone(LOCAL).replace(microsecond=0)
    return local if time(13, 30) <= local.time() < time(19, 0) else local.replace(hour=15, minute=40, second=0)


def _box(x: float, bottom: float, w: float, h: float) -> Box:
    return Box(x1=round(x - w / 2, 3), y1=round(bottom - h, 3), x2=round(x + w / 2, 3), y2=round(bottom, 3))


def nights(end: datetime) -> list[ItemSynced]:
    """Ten invented nights as the health connector would sync them in the morning: one item per night."""
    day, synced = end.astimezone(LOCAL).date(), datetime.combine(end.astimezone(LOCAL).date(), time(8, 6), LOCAL)
    out = []
    for back in range(9, -1, -1):
        d = day - timedelta(days=back)
        short = back == 0 or back in LIKE_THIS
        asleep = 302 if short else 440 + (back * 7) % 30
        bed = datetime.combine(d, time(7, 20), LOCAL) - timedelta(minutes=asleep + 20)
        night = Night(day=d, bed=bed, wake=bed + timedelta(minutes=asleep + 20), asleep_min=asleep, in_bed_min=asleep + 20,
                      stage_min={"core": asleep - 150, "deep": 60, "rem": 90, "awake": 20}, resting_hr=(63 if short else 54 + back % 3),
                      sleeping_hr=58, hrv_ms=60, resp=14.5)
        out.append(ItemSynced(event_id=f"demo-checkin-night-{d}-{int(synced.timestamp())}", ts=synced, connector="health", **to_item(night, LOCAL)))
    return out


def calendar(end: datetime) -> list[ItemSynced]:
    day = end.astimezone(LOCAL).date()
    synced, lecture = datetime.combine(day, time(8, 5), LOCAL), datetime.combine(day, time(10, 40), LOCAL)
    due = datetime.combine(day + timedelta(days=1), time(23, 59), LOCAL)
    return [
        ItemSynced(event_id=f"demo-checkin-lecture-{day}", ts=synced, connector="calendar", kind="calendar_event",
                   external_id=f"outlook:cop4533-lecture:{lecture.isoformat()}", title="COP4533 lecture", when=lecture,
                   summary="until 11:55 AM · Classroom Building I"),
        ItemSynced(event_id=f"demo-checkin-milestone-{day}", ts=synced, connector="canvas", kind="assignment",
                   external_id="cop4533:assignment:88213", title="Project - Milestone1 Fall2026 [COP4533]", when=due,
                   url="https://canvas.example.edu/courses/5123/assignments/88213",
                   summary="Dynamic programming: code plus a runtime analysis writeup. 100 points."),
    ]


def history(end: datetime) -> list[Event]:
    """The coach's notes from three earlier days like this one: the morning block held, the afternoon one did not."""
    day = end.astimezone(LOCAL).date()
    out: list[Event] = []
    for back in LIKE_THIS:
        d = day - timedelta(days=back)
        plan_id = f"plan-{d}-0810"
        for block_id, start, planned, desk in (("m1", time(9, 30), 60, 52), ("m4", FELL_AT[back], 90, 14 + 2 * back)):
            begin = datetime.combine(d, start, LOCAL)
            done = begin + timedelta(minutes=planned)
            session = SessionReview(block_id=block_id, title="Study at Desk", start=begin, end=done, place="apartment.desk",
                                    place_name="Desk", planned_minutes=planned, measured=True, desk_minutes=desk,
                                    late_start_minutes=0, ended_early_minutes=planned - desk, longest_stretch_minutes=desk,
                                    longest_stretch_start=begin, text=f"You did {desk} of {planned} minutes.")
            out.append(DayCoachNoted(event_id=f"day-coach-{plan_id}-{block_id}", ts=done, day=d, plan_id=plan_id, block_id=block_id,
                                     text=session.text, session=session))
    return sorted(out, key=lambda e: e.ts)


def afternoon(end: datetime, zones: str | Path = ZONES) -> list[Event]:
    """What the desk camera would have published: someone at the desk, a bottle picked up once, strained looks at the end."""
    desk = bounding_box(load_zones(zones)[DESK_CAM][DESK])
    x = (desk.x1 + desk.x2) / 2
    person = Track(track_id=1, label="person", confidence=0.9, box=_box(x, desk.y2 - 0.05, 0.3, 0.5))
    rest = Track(track_id=2, label="bottle", confidence=0.85, box=_box(x + 0.08, desk.y1 + 0.30, 0.05, 0.08))
    raised = Track(track_id=2, label="bottle", confidence=0.8, box=_box(x + 0.02, person.box.y1 + 0.14, 0.05, 0.08))
    sat, drink = end - timedelta(minutes=SIT_MIN), end - timedelta(minutes=DRINK_MIN)
    events: list[Event] = []
    for s in range(0, SIT_MIN * 60 + 1, EVERY_S):
        ts = sat + timedelta(seconds=s)
        since_drink = (ts - drink).total_seconds()
        if 0 <= since_drink <= 4:
            things = [raised]  # two looks with the bottle at head height: picked up
        elif s <= 8 or 4 < since_drink <= 12 or s % 600 == 0 or s >= SIT_MIN * 60 - 60:
            things = [rest]  # on the desk: when the scene starts, when it is put back, now and then, and at the end
        else:
            things = []  # behind an arm or the laptop: the detector does not always hold a small thing
        ms = int(ts.timestamp() * 1000)
        events.append(ObservationRecorded(event_id=f"demo-checkin-{DESK_CAM}-{ms}", ts=ts, camera_id=DESK_CAM, tracks=[person, *things],
                                          frame_ref=frame_ref(DESK_CAM, ms)))
    seen = end - timedelta(minutes=2)
    events.append(scene(DESK_CAM, frame_ref(DESK_CAM, int(seen.timestamp() * 1000)), seen,
                        "A person at the desk with a laptop open and a water bottle beside it.", "demo-script"))
    for m in range(STRAINED_MIN, 0, -3):
        a, b = end - timedelta(minutes=m), end - timedelta(minutes=m - 3)
        events.append(ExpressionObserved(event_id=f"demo-checkin-looks-{int(b.timestamp())}", ts=b, camera_id=DESK_CAM, member_id=MEMBER,
                                         window_start=a, window_end=b, dominant="sad", samples=40, confidence=0.82, looks=52,
                                         shares={"sad": 0.36, "angry": 0.28, "neutral": 0.36}))
    return sorted(events, key=lambda e: e.ts)


@dataclass
class Shown:
    app: Atlas
    end: datetime
    seconds: float
    checkins: list[DayCheckinRaised]
    told: list[DayNarrated]
    published: int


async def show(data_dir: str | Path, end: datetime, config_dir: Path | None = None) -> Shown:
    """Publish the story into an empty `data_dir`, ending at `end`. Refuses a directory that already has a log."""
    data_dir = Path(data_dir)
    if list((data_dir / "events").glob("*.jsonl")) or (data_dir / "memory.sqlite").exists():
        raise SeedRefused(f"{data_dir} already has an event log or a memory file. The check-in demo wants an empty directory.")
    started = clocktime.perf_counter()
    # The earlier days go into the log first and are replayed: state, with nothing said or pushed about them.
    await EventBus(data_dir / "events").publish_all(history(end))
    app = build_app(data_dir, config_dir)
    await app.bus.replay_into()
    if app.day is None:
        raise RuntimeError("this config has no Day loop (no place packs): the check-in lives there")

    events = sorted(calendar(end) + nights(end) + afternoon(end, app.settings.config_dir / "zones.yaml"), key=lambda e: e.ts)
    clock = {"now": events[0].ts}
    app.qa.clock = lambda: clock["now"]  # event time, as the seed does
    raised: list[DayCheckinRaised] = []
    told: list[DayNarrated] = []
    app.bus.subscribe("day.checkin.raised", raised.append)
    app.bus.subscribe("day.narrated", lambda e: told.append(e) if e.moment == "checkin" else None)
    # The rule is looked at once, at `end`, the way a box does on the tick after everything is in: the story is
    # published first with the live look switched off, so a part of it never fires an earlier, thinner check-in.
    enabled, app.day.checkin.enabled = app.day.checkin.enabled, False
    for e in events:
        clock["now"] = e.ts
        await app.bus.publish(e)
        await settle()
    app.day.checkin.enabled = enabled
    clock["now"] = end
    for job in (lambda: app.projection.flush(end), lambda: app.habits.flush(end) + app.nudger.check(end) if app.habits else [],
                lambda: app.day.tick(end)):
        await app.bus.publish_all(job())
        await settle()
    return Shown(app=app, end=end, seconds=clocktime.perf_counter() - started, checkins=raised, told=told, published=len(events))


def summary(s: Shown) -> str:
    lines = [f"Seeded {s.app.settings.data_dir}: {s.published} source events (all invented) in {s.seconds:.1f} s; the story ends "
             f"{s.end.astimezone(LOCAL):%a %Y-%m-%d %H:%M %Z}."]
    if not s.checkins:
        watching = s.app.day.checkin.watching(s.end)
        lines += ["", "No check-in was raised.", f"  signals: {', '.join(watching['signals']) or 'none'}", f"  held: {watching['held']}",
                  "  (outside 08:00 to 20:00, or in the lecture, the rule stays quiet: try --end with an afternoon time)"]
        return "\n".join(lines)
    e = s.checkins[-1]
    said = next((n for n in s.told if n.ref == e.event_id), None)
    lines += ["", f"Check-in raised at {e.at.astimezone(LOCAL):%H:%M} by the rule in contexts/day/domain/checkin.py:", "",
              f"  {said.spoken if said else e.text}", "", "  chips: " + " · ".join(x.summary for x in e.signals),
              "  suggestions: " + "; ".join(x.text for x in e.suggestions),
              f"  \"Move the block\" offered: {'yes' if e.proposes_replan else 'no (nothing to move to after a meal at this time of day)'}"]
    missing = {"sleep", "resting_hr", "expression", "sitting", "drink", "history"} - {x.kind for x in e.signals}
    if missing:
        lines.append(f"  not in it at this time of day: {', '.join(sorted(missing))}")
    lines += ["", "Run the app on it, then open /now/:",
              f"  ATLAS_DATA_DIR={s.app.settings.data_dir} ATLAS_MEMORY_FIXTURE= ATLAS_CONNECTORS=0 ATLAS_CAPTURE=0 python -m contexts.gateway"]
    return "\n".join(lines)
