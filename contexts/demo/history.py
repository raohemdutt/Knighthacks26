"""Twelve earlier days for Today's Plan, lived through the real Day loop (demo data, clearly invented).

    python -m contexts.demo seed --history        # `python -m contexts.demo circle` always includes it

The demo night starts two days before its `now`. This adds the twelve days before that (now - 15 days to
now - 4 days; the day in between is left empty on purpose, so the date strip has one day with nothing to open),
so the strip on Today's Plan has days to look back at, review and replay.

What is invented is only what a connector or a person would have supplied:

- each day's calendar and what is due (an invented student's week, below), handed to the planner through the
  port Day Run already reads agreed plans through (`DayRun.agreed`), for the length of this seeding only;
- what Habits would have measured that evening (desk minutes, meals, sleep), handed to the planner's own review;
- on three days, what the person did: replaced Atlas's plan with their own, approved a card, or had the day
  change under them (the calendar changed mid-morning).

Everything else is the app's own code, called the way the app calls it: `DayPlanner.plan` builds the board and
chooses the morning plan (`day.plan.proposed` with its approval card), `edited` takes the person's own moves,
`replan` builds the new plan when the day changes, `review` scores the evening (`day.reviewed`); the tracker and
the narrator react to those events as they do on a live box. So the three presenter presets that used to be
buttons are days in the log:

    now - 6 days   your own plan, which did not work (a lunch run that misses the lecture, a short night)
    now - 5 days   Atlas's plan, approved and followed
    now - 4 days   two things changed mid-morning and Atlas re-planned; still made it

No connector or camera event is published for these days, on purpose: Habits, Memory, Waste Watch and the League
count from those, and the story's own numbers (streaks, standings, patterns) must stay exactly what they are
without this. The fixed blocks of each day are in its plan event, which is where Today's Plan reads them back.

Deterministic: times are relative to the story's `now`, plans and reviews carry the loop's own ids
(`day-plan-plan-<day>-<hhmm>`), and seeding twice adds nothing (the loop already has a plan and a review for
each day, so nothing is asked of it again).
"""

from collections.abc import Awaitable, Callable
from datetime import date, datetime, time, timedelta, tzinfo

from contexts.demo.story import LOCAL, _clock
from contexts.play.dayrun.models import Measured, Move
from contracts.events import Event
from contracts.experience import ApprovalDecided
from contracts.memory import Item

DAYS = 12  # how many days are added
FIRST = 15  # the oldest is this many days before the story's `now`
OWN_PLAN, ON_PLAN, CHANGED = 6, 5, 4  # days before `now` of the three days that carry the old presets

# An invented student's week: (title, start, minutes, place as the calendar writes it), by weekday (Monday = 0).
WEEK: dict[int, list[tuple[str, time, int, str]]] = {
    0: [("COP4533 lecture", time(10, 40), 50, "HEC 101"), ("PHY2049 lecture", time(14, 0), 75, "Classroom Building I")],
    1: [("MAC2313 lecture", time(9, 30), 75, "Classroom Building I"), ("Senior design meeting", time(15, 0), 60, "Engineering II")],
    2: [("COP4533 lecture", time(10, 40), 50, "HEC 101"), ("PHY2049 lecture", time(14, 0), 75, "Classroom Building I")],
    3: [("MAC2313 lecture", time(9, 30), 75, "Classroom Building I"), ("Senior design meeting", time(15, 0), 60, "Engineering II")],
    4: [("COP4533 lecture", time(10, 40), 50, "HEC 101"), ("Robotics club", time(17, 0), 60, "Student Union")],
    5: [("Intramural soccer", time(16, 0), 60, "Recreation and Wellness Center")],
    6: [("Brunch with roommates", time(11, 0), 60, "'63 South")],
}
# Canvas: (title, weekday due, time due), every week.
DUE = [("Homework {n} [MAC2313]", 3, time(23, 59)), ("Quiz {n} - Graphs [COP4533]", 2, time(17, 0))]
# The day that changes under you: the second fixed block moves here and starts this much earlier, and this is added.
MOVED_TO, MOVED_BY = "Engineering II", timedelta(minutes=30)
ADDED = ("COP4533 review session", time(12, 15), 45, "Library, 3rd floor")
# What Habits would have measured each evening, oldest day first: (desk minutes, meals, hours asleep that night).
# None: nothing was measured that day (the box was off), which the review then says in so many words.
MEASURED: list[tuple[float, int, float] | None] = [
    (150, 3, 7.6), (95, 2, 6.4), (170, 3, 8.0), (135, 2, 7.2), (60, 2, 7.1), (140, 3, 7.8),
    (120, 2, 6.9), (180, 3, 8.1), (110, 3, 7.4), (45, 1, 4.6), (175, 3, 7.9), (130, 3, 7.5),
]


def _at(day: date, t: time) -> datetime:
    return datetime.combine(day, t, LOCAL)


def first_day(now: datetime) -> date:
    return now.astimezone(LOCAL).date() - timedelta(days=FIRST)


def history_days(now: datetime) -> list[date]:
    return [first_day(now) + timedelta(days=n) for n in range(DAYS)]


class Diary:
    """The invented calendar of the history days, in the shape of `AgreedPlans` (the port Day Run reads extra
    calendar items through). `inner` is the real one, when the box has a Circle: its items are passed on."""

    def __init__(self, now: datetime, inner=None) -> None:
        self.days, self.inner = set(history_days(now)), inner
        self.changed: set[date] = set()  # days whose calendar has changed (the re-sync at 11:40)

    def of(self, day: date) -> list[Item]:
        out: list[Item] = []
        for n, (title, at, minutes, where) in enumerate(WEEK[day.weekday()]):
            start = _at(day, at)
            if day in self.changed and n == 1:
                start, where = start - MOVED_BY, MOVED_TO
            out.append(self._event(day, title, start, minutes, where))
        if day in self.changed:
            title, at, minutes, where = ADDED
            out.append(self._event(day, title, _at(day, at), minutes, where))
        for title, weekday, at in DUE:
            if day.weekday() == weekday:
                due = _at(day, at)
                out.append(Item(connector="canvas", kind="assignment", external_id=f"hist:assignment:{title.split()[0].lower()}-{day.isoformat()}",
                                title=title.format(n=day.isocalendar().week % 9 + 1), when=due, summary=f"Due {due:%a %b} {due.day}, {_clock(due)}."))
        return out

    @staticmethod
    def _event(day: date, title: str, start: datetime, minutes: int, where: str) -> Item:
        slug = "".join(c if c.isalnum() else "-" for c in title.lower()).strip("-")
        return Item(connector="calendar", kind="calendar_event", external_id=f"outlook:hist-{slug}:{day.isoformat()}", title=title,
                    when=start, summary=f"until {_clock(start + timedelta(minutes=minutes))} · {where}")

    def items(self, lo: datetime, hi: datetime, tz: tzinfo, known: list[Item] | None = None) -> list[Item]:
        out = list(self.inner.items(lo, hi, tz, known)) if self.inner is not None else []
        for day in sorted(self.days):
            out += [i for i in self.of(day) if i.when is not None and lo <= i.when < hi]
        return out


def own_plan(board) -> list[Move]:
    """The plan a person might make for themselves: a long lunch across campus that cannot make it back, the
    evening at the desk, and a short night. Built from the board's own slots, so it stays valid if the week changes."""
    moves: list[Move] = []
    before_class = next((s for s in board.slots if s.to_place and s.to_place.startswith("ucf.")), None)
    if before_class:
        far = max((o for o in before_class.moves if o.kind == "eat" and o.place and o.place.startswith("ucf.")),
                  key=lambda o: o.walk_minutes + o.back_minutes, default=None)
        if far:
            moves.append(Move(slot=before_class.id, kind="eat", place=far.place, minutes=max(far.max_minutes, 15) + 11))
    last = board.slots[-1]
    moves += [Move(slot=last.id, kind="rest", minutes=180), Move(slot=last.id, kind="study", place="apartment.desk", minutes=420),
              Move(slot=last.id, kind="sleep")]
    return moves


async def publish(app, now: datetime, clock: dict, expire: Callable[..., Awaitable], settle: Callable[[], Awaitable]) -> int:
    """Live the twelve days through `app`, oldest first. Returns how many events were handed to the bus.

    `clock`, `expire` and `settle` are the seed's own (contexts/demo/seed.py): event time for the agent, the approvals
    ticker in story time, and waiting for the handlers the bus started."""
    if app.day is None or app.dayrun is None:
        return 0
    planner, dayrun, published = app.day.planner, app.dayrun, 0
    diary = Diary(now, dayrun.agreed)

    async def put_all(events: list[Event]) -> None:
        nonlocal published
        for e in events:
            clock["now"] = e.ts
            await expire(app, e.ts)
            await app.bus.publish(e)
            await settle()
            published += 1

    async def approve(action_id: str | None, at: datetime, name: str) -> None:
        if action_id:
            await put_all([ApprovalDecided(event_id=f"demo-history-{name}-approved", ts=at, action_id=action_id, decision="approve")])

    real_agreed, real_measured = dayrun.agreed, dayrun.measured
    dayrun.agreed = diary
    try:
        today = now.astimezone(LOCAL).date()
        for n, day in enumerate(history_days(now)):
            ago = (today - day).days
            if day not in planner.plans:  # the morning: Atlas plans the day (what POST /api/day/plan {} asks for)
                clock["now"] = _at(day, time(8, 5))
                await put_all(planner.plan(_at(day, time(8, 5))))
            plan = planner.current.get(day)
            if plan is None:
                continue
            if ago == OWN_PLAN and getattr(plan, "by", "atlas") == "atlas" and plan.plan_id == planner.plans[day].plan_id:
                at = _at(day, time(8, 20))  # the person replaces it with their own (POST /api/day/plan {moves})
                clock["now"] = at
                await put_all(planner.edited(at, own_plan(dayrun.board(at))))
            if ago in (ON_PLAN, CHANGED) and planner.plans[day].plan_id not in planner.committed:
                await approve(planner.plans[day].action_id, _at(day, time(8, 7)), f"plan-{day.isoformat()}")
            if ago == CHANGED and len(WEEK[day.weekday()]) > 1:
                diary.changed.add(day)  # from 11:40 the calendar reads differently, for the rest of this seeding too
                if day not in planner.pending and not planner.earlier.get(day):
                    title, was = WEEK[day.weekday()][1][0], _at(day, WEEK[day.weekday()][1][1])
                    why = (f"Two things changed: {title} moved to {MOVED_TO} and now starts at {_clock(was - MOVED_BY)}, "
                           f"and a {ADDED[0]} was added at {_clock(_at(day, ADDED[1]))}.")
                    at = _at(day, time(11, 41))
                    clock["now"] = at
                    events = planner.replan(at, None, why)
                    await put_all(events)
                    if events:
                        await approve(events[0].action_id, at + timedelta(minutes=2), f"replan-{day.isoformat()}")
            # The evening: the planner's own review. These days have no camera events, so what Habits would have
            # measured is handed to it; a day with nothing measured is reviewed as exactly that.
            if day not in planner.reviews:
                at = _at(day, time(20, 5))
                clock["now"] = at
                if MEASURED[n] is not None:
                    desk, meals, sleep = MEASURED[n]
                    dayrun.measured = lambda d, scenario=None, m=Measured(focus_minutes=desk, meals=meals, sleep_hours=sleep): m
                try:
                    await put_all([planner.review(day, at)])
                finally:
                    dayrun.measured = real_measured
    finally:
        dayrun.agreed, dayrun.measured = real_agreed, real_measured
    return published
