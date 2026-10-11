"""Block-start reminders on the box: on each tick, the planned block that starts now is said once, through the arbiter.

    reminders = Reminders(planner, tracker, narrator, clock, enabled=True)   # agent.yaml `day: {block_reminders: false}` to switch off
    reminders.subscribe(bus)      # what was already said, from `day.narrated` moment "start" (replay=True)
    ... DayLoop.tick calls reminders.tick(now) before the arbiter's tick

The rule is contexts/day/domain/reminders.py; the words are the "start" template (contexts/day/domain/narration.py).
`tick` hands the facts to `Narrator.tell`, which queues them in the arbiter: the arbiter decides whether, when and
with what the line goes out. Nothing is emitted on replay; times are the tick's event time.
"""

import logging
from datetime import datetime

from contexts.day.domain import reminders as rule
from contexts.day.domain.narration import hhmm, short_place, short_title
from contracts.events import Event

log = logging.getLogger("atlas.day.reminders")


class Reminders:
    def __init__(self, planner, tracker, narrator, clock, enabled: bool = True) -> None:
        self.planner, self.tracker, self.narrator, self.clock = planner, tracker, narrator, clock
        self.enabled = enabled
        self.said = rule.Said(blocks=set(), starts=set())

    def subscribe(self, bus) -> None:
        bus.subscribe("day.narrated", self.on_narrated, replay=True)

    def on_narrated(self, e) -> None:
        if e.moment == "start":
            self.said.note(e.ref, e.ts, e.day)

    def walk(self, a: str, b: str) -> int:
        try:
            return self.tracker.walks.walk(a, b)[0]
        except Exception:
            return 0

    def tick(self, now: datetime) -> list[Event]:
        if not self.enabled or self.narrator.mode == "off":
            return []
        try:
            return self.check(now)
        except Exception:  # a reminder must never stop the day
            log.exception("the block reminder failed")
            return []

    def check(self, now: datetime) -> list[Event]:
        day = self.clock.day_of(now)
        plan = self.planner.current.get(day)
        if plan is None:
            return []
        seen = self.tracker.where(now)
        awake = self.clock.wake_at(day) <= now < self.clock.bed_at(day)
        d = rule.due(plan.moves, plan.plan_id, plan.plan_id in self.planner.committed, now, seen[0] if seen else None,
                     self.walk, self.said, self.clock.tz, awake=awake)
        if d is None:
            return []
        return self.narrator.tell("start", self.facts(d, now), now, day, plan.plan_id, d.ref, about=[m.id for m in d.run])

    def facts(self, d: rule.Due, now: datetime) -> dict:
        m, tz = d.first, self.clock.tz
        f: dict = {"moment": "start", "block": self.narrator.label(m, m.title, m.start), "kind": m.kind, "at": hhmm(m.start, tz)}
        if d.count > 1:
            f["count"], f["until"] = d.count, hhmm(d.until, tz)
        else:
            f["min"] = m.minutes
        place = short_place(m.place_name) if m.place_name else None
        if place:
            f["place"] = place
        if d.walk:
            f["walk_min"] = d.walk
        elif d.there:
            f["there"] = True
        title = self.deadline_title(m.deadline, now) if m.deadline else None
        if title:
            f["deadline"] = short_title(title)
        return f

    def deadline_title(self, deadline_id: str, now: datetime) -> str | None:
        try:
            return next((x.title for x in self.planner.dayrun.board(now).deadlines if x.id == deadline_id), None)
        except Exception:
            return None
