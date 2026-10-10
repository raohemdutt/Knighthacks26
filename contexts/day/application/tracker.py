"""The live day (D1): the plan against what is happening, from events as they arrive.

    tracker = DayTracker(planner, dayrun, clock, places, habits=habits, waste=waste, presence=presence)

Where you should be comes from the day's plan (the morning plan, or a re-plan once the phone approved it).
Where the box thinks you are comes from the zone events (a person at the desk, someone at the door) and
from your own shared place in the Circle. The next fixed event gets a "leave by" from the place pack's
walking time, and `PlaceSet.feasible` says whether you still make it from where you are. The day's meters
are read from Habits (meals, desk minutes, last night's sleep) and Waste Watch (recycled today).

`day.status.changed` goes out when something material changes (a block starts, drift crosses a threshold,
the risk changes, you were seen somewhere else), never more than once a minute of event time. When the
next fixed event is no longer feasible, the planner builds a re-plan from where you are and one card.

What rebuilds from the log (replay=True): where you were last seen (zone events) and the last published
status (`day.status.changed`), so a restart mid-day shows the same status. Computing and publishing is
live only, and nothing here runs a model.
"""

import logging
from datetime import datetime, timedelta

from contexts.day.application.planner import DayPlanner
from contexts.day.domain.clock import DayClock
from contexts.play.dayrun.board import minutes_between
from contexts.play.dayrun.service import DayRun
from contexts.play.domain.places import PlaceSet
from contracts.day import DayMeters, DayStatusChanged, NextEvent, PlanMove
from contracts.events import Event
from contracts.perception import ZoneEvent

log = logging.getLogger("atlas.day")
MIN_GAP = timedelta(minutes=1)  # between two statuses, in event time
DRIFT_STEP = 10  # minutes: a status when drift crosses a multiple of this
LEAVE_SOON = 10  # minutes of slack or less: "leave now"
REPLAN_GAP = timedelta(minutes=15)  # between two re-plans of one day (a card lives five minutes)


class DayTracker:
    def __init__(self, planner: DayPlanner, dayrun: DayRun, clock: DayClock, places: PlaceSet, habits=None, waste=None,
                 presence=None, desk_zone: str = "desk", door_zone: str = "door", desk_camera: str = "cam-desk") -> None:
        self.planner, self.dayrun, self.clock, self.places = planner, dayrun, clock, places
        self.habits, self.waste, self.presence = habits, waste, presence
        self.desk_zone, self.door_zone, self.desk_camera = desk_zone, door_zone, desk_camera
        self.walks = dayrun.walks
        # --- state, rebuilt from the log ---
        self.seen: tuple[str, str, datetime] | None = None  # (place, how, ts): the cameras' last word on where you are
        self.status: DayStatusChanged | None = None
        self.dirty = False  # a material change was held back by the minute gap

    # --- state handlers (replay=True) ---

    def on_zone_state(self, e: ZoneEvent) -> None:
        if e.label != "person":
            return
        if e.zone_id == self.desk_zone:
            place, how = self.walks.ref(self.dayrun.cfg.desk), "desk camera"
        elif e.zone_id == self.door_zone:
            home = self.dayrun.cfg.home.split(".")[0]
            place, how = f"{home}.door" if self._known(f"{home}.door") else self.walks.ref(self.dayrun.cfg.home), "door camera"
        else:
            return
        if self.seen is None or e.ts >= self.seen[2]:
            self.seen = (place, how, e.ts)

    def on_status_changed(self, e: DayStatusChanged) -> None:
        if self.status is None or e.ts >= self.status.ts:
            self.status = e
            self.dirty = False

    # --- live handlers ---

    def on_zone_live(self, e: ZoneEvent) -> list[Event]:
        return self.update(e.ts, "seen") if e.label == "person" else []

    def on_change(self, e: Event) -> list[Event]:
        """Something the status is built from changed: a connector item, a plan, a habit signal, a shared place."""
        return self.update(e.ts, e.type.split(".")[0])

    def tick(self, now: datetime) -> list[Event]:
        return self.update(now, "tick")

    # --- the status ---

    def update(self, now: datetime, cause: str) -> list[Event]:
        try:
            status = self.compute(now)
        except Exception:
            log.exception("the day status failed")
            return []
        if status is None:
            return []
        out: list[Event] = []
        if self.material(status) or self.dirty:
            if self.status is not None and now - self.status.ts < MIN_GAP:
                self.dirty = True  # held back: the next tick or event publishes it
            else:
                out.append(status)
        event = status.next_event.id if status.next_event else None
        if status.risk in ("infeasible", "late") and self.planner.can_replan(now, REPLAN_GAP, event):
            try:
                out += self.planner.replan(now, status.seen_at, self.why(status), for_event=event)
            except Exception:
                log.exception("the re-plan failed")
        return out

    def compute(self, now: datetime) -> DayStatusChanged | None:
        day = self.clock.day_of(now)
        plan = self.planner.current.get(day)
        if plan is None:
            return None
        blocks = sorted(plan.moves, key=lambda m: m.start)
        cur = next((m for m in blocks if m.start <= now < m.end), None)
        nxt = next((m for m in blocks if m.start > now), None)
        seen = self.where(now)
        should = cur.place if cur else None
        drift = 0
        if cur and should and seen:
            if seen[0] != should:
                drift = minutes_between(cur.start, now) + self.walks.walk(seen[0], should)[0]
        elif cur is None and nxt and nxt.place and seen and seen[0] == nxt.place:
            drift = -minutes_between(now, nxt.start)
        fixed = next((m for m in blocks if m.fixed and m.place and m.end > now and (m.start > now or not seen or seen[0] != m.place)), None)
        next_event, risk = None, "ok"
        if fixed:
            origin = (seen[0] if seen else None) or should or self.walks.ref(self.dayrun.cfg.start_place)
            walk = self.walks.walk(origin, fixed.place)[0]
            next_event = NextEvent(id=fixed.id, title=fixed.title, start=fixed.start, end=fixed.end, place=fixed.place,
                                   place_name=fixed.place_name, walk_minutes=walk, leave_by=fixed.start - timedelta(minutes=walk))
            if seen and seen[0] == fixed.place:
                risk = "ok"
            else:
                slack = self.places.feasible(origin, fixed.place, now, fixed.start)["slack_minutes"]
                risk = "ok" if slack > LEAVE_SOON else "leave_now" if slack >= 0 else "late" if now >= fixed.start else "infeasible"
        deadline_risk = self.deadline_risk(now)
        meters = self.meters(now)
        status = DayStatusChanged(
            ts=now, day=day, plan_id=plan.plan_id, now=now, block_id=cur.id if cur else None, block_title=cur.title if cur else None,
            block_end=cur.end if cur else None, should_be=should, should_be_name=cur.place_name if cur else None,
            seen_at=seen[0] if seen else None, seen_at_name=self.walks.name(seen[0]) if seen else None,
            seen_how=seen[1] if seen else None, seen_ts=seen[2] if seen else None, drift_minutes=drift, next_event=next_event,
            next_block=NextEvent(id=nxt.id, title=nxt.title, start=nxt.start, end=nxt.end, place=nxt.place, place_name=nxt.place_name) if nxt else None,
            risk=risk, deadline_risk=deadline_risk, meters=meters)
        status.reason = self.reason(status)
        status.event_id = f"day-status-{day}-{now.timestamp():.0f}"
        return status

    def where(self, now: datetime) -> tuple[str, str, datetime] | None:
        """The cameras' last word, unless your own shared place in the Circle is newer."""
        seen = self.seen
        if self.presence is not None:
            mine = self.presence.current.get(self.presence.me)
            if mine and mine.expires_at > now and (seen is None or mine.ts > seen[2]):
                place = self.walks.shared(mine.place)
                if place:
                    seen = (self.walks.ref(place), "shared place", mine.ts)
        return seen

    def material(self, s: DayStatusChanged) -> bool:
        p = self.status
        if p is None or p.day != s.day or p.plan_id != s.plan_id:
            return True
        return (p.block_id != s.block_id or p.risk != s.risk or p.seen_at != s.seen_at
                or (p.next_event.id if p.next_event else None) != (s.next_event.id if s.next_event else None)
                or p.drift_minutes // DRIFT_STEP != s.drift_minutes // DRIFT_STEP
                or p.meters.meals != s.meters.meals or p.meters.recycled != s.meters.recycled)

    def reason(self, s: DayStatusChanged) -> str:
        p = self.status
        if p is None or p.day != s.day or p.plan_id != s.plan_id:
            return "plan started" if s.plan_id else "no plan"
        if p.block_id != s.block_id:
            return f"{s.block_title} starts" if s.block_title else "free until the next block"
        if p.risk != s.risk and s.next_event:
            return {"ok": f"on time for {s.next_event.title}", "leave_now": f"leave now for {s.next_event.title}",
                    "late": f"late for {s.next_event.title}", "infeasible": f"you will miss {s.next_event.title}"}[s.risk]
        if p.seen_at != s.seen_at:
            return f"seen at {s.seen_at_name}" if s.seen_at_name else "not seen"
        if p.drift_minutes // DRIFT_STEP != s.drift_minutes // DRIFT_STEP:
            return f"{abs(s.drift_minutes)} minutes {'behind' if s.drift_minutes > 0 else 'ahead'}"
        return "meters moved"

    def why(self, s: DayStatusChanged) -> str:
        e = s.next_event
        at = f"At {s.seen_at_name} at {s.now.astimezone(self.clock.tz):%H:%M}" if s.seen_at_name else "From here"
        if not e:
            return f"{at}, the plan no longer fits."
        verb = "started" if s.risk == "late" else "starts"
        return f"{at}; {e.title} {verb} at {e.start.astimezone(self.clock.tz):%H:%M} at {e.place_name}, a {e.walk_minutes} minute walk away."

    # --- the meters ---

    def deadline_risk(self, now: datetime) -> float:
        try:
            return round(max((d.risk for d in self.dayrun.board(now).deadlines), default=0.0), 3)
        except Exception:
            return 0.0

    def meters(self, now: datetime) -> DayMeters:
        m = DayMeters()
        if self.habits is not None:
            try:
                rows = {r["id"]: r for r in self.habits.digest(now=now).get("habits", [])}
                if "meals" in rows:
                    m.meals = int(rows["meals"]["value"])
                if "focus" in rows:
                    m.desk_minutes = round(rows["focus"]["value"] * 60)
                sleep = rows.get("sleep")
                if sleep and sleep.get("applies", True):
                    m.sleep_hours, m.sleep_source = sleep["value"], sleep.get("source")
            except Exception:
                log.debug("habits unavailable for the day's meters", exc_info=True)
        if self.waste is not None:
            try:
                m.recycled = self.waste.metrics(since=self.clock.day_start(self.clock.day_of(now)))["recycled"]
            except Exception:
                log.debug("waste watch unavailable for the day's meters", exc_info=True)
        return m

    def _known(self, ref: str) -> bool:
        try:
            self.walks.ref(ref)
            return True
        except ValueError:
            return False


def block_at(moves: list[PlanMove], now: datetime) -> PlanMove | None:
    return next((m for m in sorted(moves, key=lambda m: m.start) if m.start <= now < m.end), None)
