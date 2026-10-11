"""The check-in, shown (D4): one ordinary afternoon that ends in Atlas speaking up, through the real app.

    python -m contexts.demo checkin --data-dir <an empty directory> [--end 2026-10-10T15:40:00-04:00]
    python -m contexts.demo checkin --data-dir <a `circle`-seeded directory> --onto     # the same afternoon, on the demo night
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

- and one more earlier afternoon, three days back (`earlier`), for the look memory (contexts/day/application/looks.py)
  to call back to. It is the demo owner's looks and nothing else (`perception.expression.observed`): two hours of them
  without a gap, the last twenty minutes strained, then none for ten minutes, then calmer ones. No desk or drink event
  is written for that day, on purpose: Habits counts from those, and the demo night's numbers must stay what they
  are (so the look memory reads "at the desk" from the owner's own windows there, as it does on any box whose desk
  zone has no record). The remembered moment, its outcome ("ten minutes away from the desk helped") and the callback
  sentence are the look memory's own, and the weekday it names is whatever day that really was. The demo owner
  (`m-demo`, made-up vectors marked as demo data: contexts/demo/people.py) is written to the identity store, so the
  People screen lists "Looks Atlas remembers".
- and, for what Atlas LEARNS from its remembered looks (stage two, contexts/day/domain/look_model.py), nine more
  early afternoons over the two weeks before (`THEN`), written the same way: ten remembered looks with a known
  outcome in all. On six the owner got up for a few minutes (the look eased on five; on one, a four-minute break, it
  was still there after: the counter-example), on four they stayed at the desk (still there each time). Every count
  Atlas then says ("five of the last six times") is the look memory's own count of these; nothing about the model
  is written by this file. Looks only, again: a drink seen picked up on one of those days would have to be a camera
  event, and the first camera event of a day is where Habits starts counting it (the log is replayed day by day), so
  the night's streaks would change. So this story has no "water alone": the box never saw a drink on those days,
  and what Atlas says of them is "staying at the desk hasn't any of the four times". The tests of the look memory
  itself cover a person a drink helps (contexts/day/tests/test_look_model.py).

`--onto` adds the afternoon to a directory that already holds the demo night (`python -m contexts.demo circle`):
its log is replayed, the earlier days are appended to it, and the afternoon is published live from after the
night's last event. `end` then has to leave room for the sitting after that event; the command says so if not.

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
from contexts.demo.seed import SeedRefused, build_app, settle  # noqa: F401  (SeedRefused is this command's refusal too)
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
STRAINED = {"sad": 0.36, "angry": 0.28, "neutral": 0.36}
CALMER = {"neutral": 0.7, "happy": 0.2, "sad": 0.1}
# The earlier afternoon the look memory calls back to (`earlier`): days back, and when the owner sat down. Not one of
# the LIKE_THIS days (those afternoons fell apart: nobody stayed at the desk), and clear of the hours the demo people's
# week of looks covers on an ordinary day (contexts/demo/people.py MORNING, and AFTERNOON from 15:00 on).
THEN_BACK, THEN_SAT = 3, time(12, 44)
THEN_SIT_MIN = 120  # the look had held for the rule's twenty minutes two hours after sitting down
THEN_UP_MIN, THEN_AWAY_MIN = 6, 10  # the last look before getting up came six minutes after that; then ten minutes away
# Every earlier afternoon the look memory learns from, the callback's included: (days back, when the look had held
# for the rule's twenty minutes, minutes at the desk by then, what the owner did, minutes away, whether the look
# eased). Never a LIKE_THIS day, never yesterday (the night's), always early afternoon: clear of the people's week
# of looks, and far enough from the story's own hour that the callback stays the most recent one, three days back.
UP, STAYED = "up", "stayed"
THEN = (
    (14, time(13, 30), 114, UP, 12, True),
    (13, time(13, 42), 102, STAYED, 0, False),
    (12, time(13, 48), 126, UP, 8, True),
    (11, time(13, 36), 96, STAYED, 0, False),
    (10, time(13, 30), 105, UP, 4, False),  # the counter-example: four minutes away, and it was still there after
    (9, time(13, 51), 111, UP, 15, True),
    (8, time(13, 45), 90, STAYED, 0, False),
    (7, time(13, 39), 108, STAYED, 0, False),
    (5, time(13, 54), 93, UP, 7, True),
    (THEN_BACK, time(14, 44), THEN_SIT_MIN, UP, THEN_AWAY_MIN, True),  # the one the check-in calls back to
)
STILL_MIN = 33  # staying put: the strained looks go on this long after the look held, past the half hour an outcome takes
WINDOW_MIN = 3  # a window of looks
ONTO_GUARD = timedelta(minutes=45)  # `--onto`: stay this clear of a fixed block of the night (the rule's own guard is 30, plus a walk)


def default_end(now: datetime) -> datetime:
    """Now when it is afternoon; else today at 15:40 local, where the story's words fit."""
    local = now.astimezone(LOCAL).replace(microsecond=0)
    return local if time(13, 30) <= local.time() < time(19, 0) else local.replace(hour=15, minute=40, second=0)


def onto_end(data_dir: str | Path, now: datetime) -> datetime:
    """For `--onto` with no `--end`: the first afternoon time after the seeded night's last event at which the night's
    own plan has nothing fixed on or about to start (the rule stays quiet in a class and before a time to leave).
    `default_end` when the directory has no log, or when that time is already clear."""
    log = list(EventBus(Path(data_dir) / "events").read_log())
    if not log:
        return default_end(now)
    last = max(e.ts for e in log).astimezone(LOCAL)
    fixed = [(m.start - ONTO_GUARD, m.end) for e in log if e.type in ("day.plan.proposed", "day.replan.proposed") for m in e.moves if m.fixed]
    first = default_end(now)
    for day in (first.date() + timedelta(days=n) for n in range(3)):
        t = max(first if day == first.date() else datetime.combine(day, time(15, 40), LOCAL), datetime.combine(day, time(13, 30), LOCAL))
        while t.date() == day and t.time() < time(19, 0):
            if t - timedelta(minutes=SIT_MIN + 5) >= last and not any(a <= t < b for a, b in fixed):
                return t
            t = (t + timedelta(minutes=10)).replace(minute=(t.minute // 10 * 10 + 10) % 60) if t.minute % 10 else t + timedelta(minutes=10)
    return first


def _box(x: float, bottom: float, w: float, h: float) -> Box:
    return Box(x1=round(x - w / 2, 3), y1=round(bottom - h, 3), x2=round(x + w / 2, 3), y2=round(bottom, 3))


def nights(end: datetime, synced: datetime | None = None) -> list[ItemSynced]:
    """Ten invented nights as the health connector would sync them in the morning: one item per night."""
    day, synced = end.astimezone(LOCAL).date(), synced or datetime.combine(end.astimezone(LOCAL).date(), time(8, 6), LOCAL)
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


def calendar(end: datetime, synced: datetime | None = None) -> list[ItemSynced]:
    day = end.astimezone(LOCAL).date()
    synced, lecture = synced or datetime.combine(day, time(8, 5), LOCAL), datetime.combine(day, time(10, 40), LOCAL)
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


def _looks(start: datetime, minutes: int, shares: dict[str, float], name: str) -> list[ExpressionObserved]:
    """`minutes` of the demo owner's looks from `start`, in three-minute windows."""
    out = []
    for m in range(0, minutes, WINDOW_MIN):
        a, b = start + timedelta(minutes=m), start + timedelta(minutes=m + WINDOW_MIN)
        out.append(ExpressionObserved(event_id=f"demo-checkin-{name}-{int(b.timestamp())}", ts=b, camera_id=DESK_CAM, member_id=MEMBER,
                                      window_start=a, window_end=b, dominant=max(shares, key=shares.get), samples=40, confidence=0.82,
                                      looks=52, shares=dict(shares)))
    return out


def earlier(end: datetime) -> list[ExpressionObserved]:
    """The earlier afternoons (THEN), as the log of a box that had been running would hold the owner's looks: calm
    from sitting down, strained for the twenty minutes up to the mark and after. Then, by what the owner did: up six
    minutes later, no looks while away, and calmer ones after (or, once, strained ones still); or the strained looks
    going on for half an hour and more. What the look memory remembers of them, and what it learns from them, is
    the look memory's."""
    day, out = end.astimezone(LOCAL).date(), []
    for back_days, at, sit_min, did, away_min, eased in THEN:
        held = datetime.combine(day - timedelta(days=back_days), at, LOCAL)  # the rule's twenty minutes hold from here
        sat = held - timedelta(minutes=sit_min)
        strained = held - timedelta(minutes=18)  # ... because by then six of its seven windows are strained ones
        out += _looks(sat, sit_min - 18, CALMER, "then-looks")
        if did == UP:
            up = held + timedelta(minutes=THEN_UP_MIN)
            back = up + timedelta(minutes=away_min)
            out += _looks(strained, 18 + THEN_UP_MIN, STRAINED, "then-looks")
            out += _looks(back, 18, CALMER, "then-looks") if eased else [*_looks(back, 21, STRAINED, "then-looks"),
                                                                           *_looks(back + timedelta(minutes=21), 15, CALMER, "then-looks")]
            continue
        out += _looks(strained, 18 + STILL_MIN, STRAINED, "then-looks") + _looks(held + timedelta(minutes=STILL_MIN), 15, CALMER, "then-looks")
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


async def show(data_dir: str | Path, end: datetime, config_dir: Path | None = None, onto: bool = False) -> Shown:
    """Publish the story into an empty `data_dir`, ending at `end`. Refuses a directory that already has a log, unless
    `onto`: then the directory is one the demo night was seeded into (`python -m contexts.demo circle`), its log is
    replayed first and the afternoon is added after its last event."""
    data_dir = Path(data_dir)
    log_files = sorted((data_dir / "events").glob("*.jsonl"))
    if (log_files or (data_dir / "memory.sqlite").exists()) and not onto:
        raise SeedRefused(f"{data_dir} already has an event log or a memory file. The check-in demo wants an empty directory "
                          "(or --onto, to add the afternoon to a directory seeded with `python -m contexts.demo circle`).")
    if onto and not log_files:
        raise SeedRefused(f"{data_dir} has no event log to add the afternoon to. Seed it first: python -m contexts.demo circle --data-dir {data_dir}")
    if onto and config_dir is None and (data_dir / "config").is_dir():
        config_dir = data_dir / "config"  # the Circle demo's own copy
    last, known, taken = None, set(), []
    if onto:
        before = list(EventBus(data_dir / "events").read_log())
        last, known = max(e.ts for e in before), {e.event_id for e in before}
        taken = [(e.window_start, e.window_end) for e in before if isinstance(e, ExpressionObserved) and e.member_id == MEMBER]
        if end - timedelta(minutes=SIT_MIN + 5) < last:
            raise SeedRefused(f"The night in {data_dir} ends at {last.astimezone(LOCAL):%a %H:%M}: the afternoon needs {SIT_MIN + 5} minutes "
                              f"after that. Pass --end with a later afternoon time (it is {end.astimezone(LOCAL):%a %H:%M}), or leave "
                              "--end out and one is picked.")
    started = clocktime.perf_counter()
    # The earlier days go into the log first and are replayed: state, with nothing said or pushed about them.
    # Onto a night that has the demo people's week of looks, a window of the earlier afternoon that one of theirs
    # already covers is left to theirs (the calmer looks after three o'clock): no stretch is in the log twice.
    await EventBus(data_dir / "events").publish_all([e for e in history(end) + earlier(end) if e.event_id not in known
                                                     and not any(a < e.window_end and e.window_start < b for a, b in taken
                                                                 if isinstance(e, ExpressionObserved))])
    app = build_app(data_dir, config_dir)
    await app.bus.replay_into()
    if app.day is None:
        raise RuntimeError("this config has no Day loop (no place packs): the check-in lives there")
    if app.identity is not None:  # the demo owner whose looks the story reads: made-up vectors, marked as demo data
        from contexts.demo import people  # here, not at the top: it imports this module

        people.write(app.identity, end, circle=app.circle is not None)

    morning = datetime.combine(end.astimezone(LOCAL).date(), time(8, 5), LOCAL)
    synced = last + timedelta(minutes=1) if last is not None and last >= morning else None  # onto a night that ends later in the day
    events = sorted(calendar(end, synced) + nights(end, synced and synced + timedelta(minutes=1))
                    + afternoon(end, app.settings.config_dir / "zones.yaml"), key=lambda e: e.ts)
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
    # the habit nudge at `end` took the slot: the check-in's words follow on the tick after the arbiter's gap. Onto a
    # night whose own lines are due at the same minute (a re-plan, "you're behind") they wait one gap more, as on a box.
    for gaps in range(1, 6):
        if raised and any(n.ref == raised[-1].event_id for n in told):
            break
        await app.bus.publish_all(app.day.tick(end + gaps * timedelta(seconds=app.day.arbiter.rules.gap_s)))
        await settle()
    return Shown(app=app, end=end, seconds=clocktime.perf_counter() - started, checkins=raised, told=told, published=len(events))


def learned_lines(s: Shown, e: DayCheckinRaised) -> list[str]:
    """What Atlas has learned from its remembered looks, as the command says it: the look memory's own words and counts."""
    back, looks = e.callback, s.app.day.looks
    if back is None or not back.evidence:
        return ["  learned: not enough remembered looks with a known outcome yet, so the check-in is stage one's"]
    found = looks.view(looks.clock_now(s.end), member_id=MEMBER)["learned"]
    return [f"  learned: {back.learned}" + (f" {back.pattern}" if back.pattern else "") + f" ({back.evidence})",
            *(f"    - {x['text']}" for x in found["findings"]),
            f"  on the card only (never said, never pushed): {e.ahead}" if e.ahead else "  on the card only: nothing (not near the usual time)"]


def summary(s: Shown) -> str:
    lines = [f"Seeded {s.app.settings.data_dir}: {s.published} source events (all invented) in {s.seconds:.1f} s; the story ends "
             f"{s.end.astimezone(LOCAL):%a %Y-%m-%d %H:%M %Z}."]
    if not s.checkins:
        watching = s.app.day.checkin.watching(s.end)
        lines += ["", "No check-in was raised.", f"  signals: {', '.join(watching['signals']) or 'none'}", f"  held: {watching['held']}",
                  "  (outside 08:00 to 20:00, in a class or close to a time to leave the rule stays quiet: try --end with another "
                  "afternoon time; with --onto, leave --end out and a clear one is picked)"]
        return "\n".join(lines)
    e = s.checkins[-1]
    said = next((n for n in s.told if n.ref == e.event_id), None)
    back = e.callback
    lines += ["", f"Check-in raised at {e.at.astimezone(LOCAL):%H:%M} by the rule in contexts/day/domain/checkin.py:", "",
              f"  {said.spoken if said else e.text}", "",
              "  chips: " + " · ".join([x.summary for x in e.signals] + ([back.chip] if back else [])),
              (f"  look memory: the earlier look was on {back.day:%A %Y-%m-%d}. " + " ".join(back.lines)) if back and back.said
              else "  look memory: no earlier look was similar enough, so nothing is said about the past",
              *learned_lines(s, e),
              "  suggestions: " + "; ".join(x.text for x in e.suggestions),
              f"  \"Move the block\" offered: {'yes' if e.proposes_replan else 'no (nothing to move to after a meal at this time of day)'}"]
    missing = {"sleep", "resting_hr", "expression", "sitting", "drink", "history"} - {x.kind for x in e.signals}
    if missing:
        lines.append(f"  not in it at this time of day: {', '.join(sorted(missing))}")
    own = s.app.settings.data_dir / "config"  # the Circle demo's own config, when the afternoon was added onto that night
    config = f"ATLAS_CONFIG_DIR={own} " if s.app.settings.config_dir == own else ""
    lines += ["", "Run the app on it, then open /now/ (the card, and its \"like <day>\" chip) and /people/ (\"Looks Atlas remembers\", and "
              "\"What Atlas has learned about your looks\" under it):",
              f"  ATLAS_DATA_DIR={s.app.settings.data_dir} {config}ATLAS_MEMORY_FIXTURE= ATLAS_CONNECTORS=0 ATLAS_CAPTURE=0 python -m contexts.gateway"]
    return "\n".join(lines)
