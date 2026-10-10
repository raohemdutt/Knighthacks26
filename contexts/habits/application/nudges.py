"""Nudges (H2): when to say something about a habit.

Live only: a nudge notifies phones, so none of these handlers may run on
replay. They read the tracker's rebuilt state and decide; the tracker itself
never nudges. Cooldowns are per habit and measured in event time.

- `on_signal`: one drink too many, a watch item left behind, a wrong bin.
- `check(now)`: the ticker. Sleep, meals, and the two that join the camera with
  Canvas: focus and start-early against the next due date. Quiet while asleep.
- `on_day_closed`: celebrate a streak.
"""

from datetime import datetime, timedelta
from uuid import uuid4

from contexts.habits.domain.config import Habit, HabitsConfig
from contexts.habits.domain.tracker import HabitTracker
from contracts.habits import DayClosed, NudgeRaised, SignalRecorded
from contracts.types import Citation


class _Blank(dict):
    def __missing__(self, key: str) -> str:
        return ""


def _n(x: float) -> str:
    """2.0 -> "2", 4.25 -> "4.2": numbers the way a person would say them."""
    return str(int(x)) if float(x).is_integer() else f"{x:.1f}"


def _hours_left(when: datetime, now: datetime) -> int:
    return round((when - now).total_seconds() / 3600)


class Nudger:
    def __init__(self, cfg: HabitsConfig, tracker: HabitTracker) -> None:
        self.cfg = cfg
        self.tracker = tracker
        self._last: dict[str, datetime] = {}
        self._last_check: datetime | None = None

    # --- on a signal ---

    def on_signal(self, e: SignalRecorded) -> list[NudgeRaised]:
        t = self.tracker
        h = self.cfg.by_signal(e.kind) if e.kind in ("drink", "forgot", "disposal") else None
        if h is None:
            return []
        value, _ = t.measure(h, t.sched.local_day(e.ts), e.ts)
        if h.met(value) or (e.kind == "disposal" and e.value):  # only the wrong throw gets a word
            return []
        fields, evidence = {}, []
        if e.kind == "forgot" and t.forgotten:
            f = t.forgotten
            fields = {"items": " and ".join(f["items"]), "title": f["title"],
                      "minutes": round((f["starts"] - e.ts).total_seconds() / 60)}
            evidence = [Citation(frame_ref=f["frame_ref"], ts=f["ts"], title=f"Still on the {self.cfg.vocabulary.workplace}"),
                        Citation(url=f["url"], ts=f["starts"], title=f["title"])]
        return self._warn(h, e.ts, value, fields, evidence)

    # --- on the ticker ---

    def check(self, now: datetime) -> list[NudgeRaised]:
        """At most one nudge per call, and none while the household sleeps."""
        gap = timedelta(minutes=self.cfg.nudges.min_gap_minutes)
        if self.tracker.sched.asleep(now) or (self._last_check and now - self._last_check < gap):
            return []
        local = now.astimezone(self.tracker.sched.tz).time()
        for h in self.cfg.habits:
            if h.nudge_after and local < h.nudge_after:
                continue
            out = self._check(h, now)
            if out:
                self._last_check = now
                return out
        return []

    def _check(self, h: Habit, now: datetime) -> list[NudgeRaised]:
        t = self.tracker
        value, applies = t.measure(h, t.sched.local_day(now), now)
        if h.signal == "start_lead":  # judged on the due day, so warn while it can still be met
            a = t.unstarted(now, h)
            return self._warn(h, now, 0, {"title": a.title, "hours": _hours_left(a.when, now)},
                              [Citation(url=a.url, ts=a.when, title=a.title)]) if a else []
        if h.signal not in ("away_window", "meal", "desk_time") or not applies or h.met(value):
            return []
        if h.signal != "desk_time":
            measured = t.source(h, t.sched.local_day(now)) == "watch"  # the night was measured, not read off the desk
            return self._warn(h, now, value, text=h.nudge_measured if measured else None)
        a = t.next_deadline(now, h.deadline_hours)
        if a is None:
            return []
        fields = {"title": a.title, "hours": _hours_left(a.when, now), "minutes": round(value * 60)}
        evidence = [Citation(url=a.url, ts=a.when, title=a.title),
                    Citation(frame_ref=t.desk_frame, ts=t.active_until, title=f"Last seen at the {self.cfg.vocabulary.workplace}")]
        return self._warn(h, now, value, fields, evidence)

    # --- on a closed day ---

    def on_day_closed(self, e: DayClosed) -> list[NudgeRaised]:
        h = self.cfg.by_id(e.habit_id)
        closed = next((c for c in self.tracker.history.get(e.habit_id, []) if c.day == e.day), None)
        if h is None or not e.met or e.streak != self.cfg.nudges.celebrate_streak or not (closed and closed.applies):
            return []  # a day with nothing to judge holds the streak; it does not earn a second cheer
        message = self.cfg.nudges.celebrate.format_map(_Blank(self.cfg.vocabulary.words(), name=h.name, streak=e.streak))
        return [NudgeRaised(ts=e.ts, nudge_id=uuid4().hex, habit_id=h.id, message=message, tone="celebrate")]

    # --- cooldown ---

    def _warn(self, h: Habit, ts: datetime, value: float, fields: dict | None = None,
              evidence: list[Citation] | None = None, text: str | None = None) -> list[NudgeRaised]:
        logged = self.tracker.latest_nudge.get(h.id)  # survives a restart, unlike `_last`
        last = self._last.get(h.id) or (logged.ts if logged else None)
        cooldown = h.cooldown_minutes if h.cooldown_minutes is not None else self.cfg.nudges.cooldown_minutes
        if last is not None and ts - last < timedelta(minutes=cooldown):
            return []
        self._last[h.id] = ts
        # The profile's words ({workplace}, {event}, {deadline}, ...) are available to every nudge text.
        message = (text or h.nudge).format_map(_Blank(self.cfg.vocabulary.words(), **(fields or {}), name=h.name,
                                                      value=_n(value), target=_n(h.target)))
        return [NudgeRaised(ts=ts, nudge_id=uuid4().hex, habit_id=h.id, message=message, evidence=evidence or [])]
