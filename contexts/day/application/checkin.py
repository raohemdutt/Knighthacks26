"""The check-in (D4): Atlas looks back across what it recorded and speaks up unasked.

    checkin = CheckIn(planner, tracker, coach, clock, memory=memory)
    checkin.subscribe(bus)             # DayLoop does both
    ... every tick: checkin.tick(now)

"You slept about five hours, your resting heart rate is up, and you've looked strained for the last twenty
minutes. The last three times a day looked like this, your study blocks fell apart by four. I haven't seen you
pick up a drink since two. Drink some water, take ten minutes outside, and I can move the study block to after
dinner if you say yes." Every clause comes from a signal that is there and is left out when it is not; the rule
that decides when to speak, and every threshold, is contexts/day/domain/checkin.py.

Where the signals come from (each handler below is a projection of the log, replay=True):
- the watch: `connector.item.synced` of kind `health` (one item per night: time asleep, resting heart rate);
- how the person looked: `perception.expression.observed` on the desk camera, for an enrolled member who
  switched "read my expressions" on; a member no longer enrolled is forgotten (`perception.identity.changed`);
- at the desk, and for how long: the coach's desk presence (`perception.zone.event`), read, not kept twice;
- a drink picked up: `perception.observation.recorded` on the desk camera (a tracked bottle or cup raised to a
  person's head) and `perception.scene.described` (a caption that says someone is drinking; the desk's object
  list for "a bottle is in view");
- earlier days: the coach's notes and the evening reviews this context already keeps (their measured blocks);
- the plan: the planner's current plan (fixed blocks) and the tracker's time to leave.

Deciding and publishing `day.checkin.raised` is live only (events as they arrive, and the ticker), looked at
once every few minutes of event time: a restart replays the check-ins already raised, with their cooldown, and
raises none of them again. The narrator says it (`day.narrated`, moment "checkin"); the phone's push names the
suggestions only. "Move the block" never changes the plan: it builds a re-plan (`day.replan.proposed`) with
the same approval card every re-plan gets, and only when the phone asks for it.

Box-only: `day.checkin.raised` and `day.checkin.answered` are in contracts.day.BOX_ONLY_EVENTS. A resting heart
rate never leaves this object as a number except through `view(numbers=True)`, the phone's explicit "show numbers".
"""

import logging
from datetime import date, datetime, timedelta

from contexts.connectors.domain.health import night_day, parse_summary
from contexts.day.domain import checkin as rule
from contexts.day.domain import looks as look_rule
from contexts.day.domain import narration as words
from contexts.day.domain.checkin import DrinkWatch, Gate, Look, Night, PastSession, Reading
from contexts.day.domain.clock import DayClock
from contracts.connectors import ItemSynced
from contracts.day import CheckinCallback, DayCheckinAnswered, DayCheckinRaised, DayReplanProposed
from contracts.events import Event
from contracts.perception import ExpressionObserved, ObservationRecorded, SceneDescribed

log = logging.getLogger("atlas.day.checkin")
KEEP = timedelta(days=3)  # looks and drinks, in event time
KEEP_NIGHTS = timedelta(days=rule.HISTORY_DAYS + rule.BASELINE_DAYS + 7)
KEEP_RAISED = 50
FRESH = timedelta(hours=4)  # an answered check-in older than this is history, not something to show on the Now screen
HEALTH = "health"


class CheckIn:
    def __init__(self, planner, tracker, coach, clock: DayClock, memory=None, enabled: bool = True) -> None:
        self.planner, self.tracker, self.coach, self.clock = planner, tracker, coach, clock
        self.memory, self.enabled = memory, enabled
        self.narrator = None  # DayLoop sets it: the words of a check-in, for the card
        self.look_memory = None  # DayLoop sets it: remembered looks to call back to (contexts/day/application/looks.py)
        v = planner.dayrun.cfg.vocabulary
        self.words = words.Words(activity=v.activity, workplace=v.workplace, event=v.event, deadline=v.deadline)
        # --- state, rebuilt from the log ---
        self.nights: dict[date, Night] = {}
        self.looks: dict[str, list[Look]] = {}  # enrolled member -> how they looked at the desk camera
        self.drinks = DrinkWatch()
        self.raised: list[DayCheckinRaised] = []  # oldest first
        self.answers: dict[str, DayCheckinAnswered] = {}
        # --- live only ---
        self.looked_at: datetime | None = None  # when the rule was last looked at
        self.sent_at: datetime | None = None  # a check-in handed to the bus (the log's own copy arrives a moment later)

    def subscribe(self, bus) -> None:
        bus.subscribe("connector.item.synced", self.on_item, replay=True)
        bus.subscribe("perception.expression", self.on_expression, replay=True)
        bus.subscribe("perception.identity", self.on_identity, replay=True)
        bus.subscribe("perception.observation", self.on_observation, replay=True)
        bus.subscribe("perception.scene", self.on_scene, replay=True)
        bus.subscribe("day.checkin.raised", self.on_raised, replay=True)
        bus.subscribe("day.checkin.answered", self.on_answered, replay=True)
        if self.enabled:  # live only: it decides and publishes
            for prefix in ("perception.expression", "perception.zone", "connector.item.synced"):
                bus.subscribe(prefix, self.on_event)

    # --- state handlers (replay=True) ---

    def on_item(self, e: ItemSynced) -> None:
        if e.kind != HEALTH:
            return
        day, night = night_day(e.external_id), parse_summary(e.summary)
        if day is None or night is None:
            return
        self.nights[day] = Night(day=day, asleep_min=night.asleep_min, resting_hr=night.resting_hr, ref=f"{e.connector}:{e.external_id}",
                                 in_bed_min=night.in_bed_min, stages=dict(night.stage_min))
        old = max(self.nights) - KEEP_NIGHTS
        if min(self.nights) < old:
            self.nights = {d: n for d, n in self.nights.items() if d >= old}

    def on_expression(self, e: ExpressionObserved) -> None:
        """How an enrolled member looked at the desk camera over a stretch. The anonymous room count is nobody's."""
        if e.member_id is None or e.camera_id != self.coach.desk_camera or e.samples <= 0:
            return
        rows = self.looks.setdefault(e.member_id, [])
        rows.append(Look(e.window_start, e.window_end, e.samples, rule.strained_looks(e.shares, e.samples), e.event_id))
        if rows[0].end < e.window_end - KEEP - timedelta(days=1):
            self.looks[e.member_id] = [x for x in rows if x.end >= e.window_end - KEEP]

    def on_identity(self, e: Event) -> None:
        """Someone was forgotten, or switched "read my expressions" off: their looks go from here too."""
        enrolled = set(getattr(e, "members", ()))
        self.looks = {m: rows for m, rows in self.looks.items() if m in enrolled}

    def on_observation(self, e: ObservationRecorded) -> None:
        if e.camera_id != self.coach.desk_camera or not any(t.label.lower() in rule.DRINK_LABELS for t in e.tracks):
            return
        self.drinks.see(e.ts, e.tracks, e.frame_ref)
        if self.drinks.moments and self.drinks.moments[0][0] < e.ts - KEEP - timedelta(days=1):
            self.drinks.prune(e.ts - KEEP)

    def on_scene(self, e: SceneDescribed) -> None:
        if e.camera_id != self.coach.desk_camera:
            return
        if e.zone_id is None:
            self.drinks.caption(e.ts, e.caption, e.frame_ref)
        elif e.zone_id == self.coach.desk_zone and e.objects:
            self.drinks.listed(e.ts, [o.label for o in e.objects], e.frame_ref)

    def on_raised(self, e: DayCheckinRaised) -> None:
        if any(x.event_id == e.event_id for x in self.raised):
            return
        self.raised = sorted([*self.raised, e], key=lambda x: x.ts)[-KEEP_RAISED:]

    def on_answered(self, e: DayCheckinAnswered) -> None:
        self.answers[e.checkin_id] = e

    # --- live: look at the rule ---

    def on_event(self, e: Event) -> list[Event]:
        return self.check(e.ts)

    def tick(self, now: datetime) -> list[Event]:
        return self.check(now)

    def check(self, now: datetime) -> list[Event]:
        if not self.enabled:
            return []
        if self.looked_at is not None and abs(now - self.looked_at) < timedelta(minutes=rule.EVERY_MIN):
            return []  # looked at a moment ago (an event that arrives late, with an older time, counts as a moment ago too)
        self.looked_at = now
        try:
            event = self.evaluate(now)
        except Exception:  # a check-in must never stop the day
            log.exception("the check-in failed")
            return []
        if event is None:
            return []
        self.sent_at = now
        return [event]

    def evaluate(self, now: datetime) -> DayCheckinRaised | None:
        """The check-in to raise at `now`, or None: the signals have to agree and nothing may hold it back."""
        r = self.reading(now)
        found = rule.signals(r)
        if not rule.agree(found) or rule.held(now, self.gate(now, r)) is not None:
            return None
        past = rule.history(r, found)
        move = self.offer(now)
        suggestions = rule.suggestions(found, move)
        back = self.look_back(now, found)  # an earlier look like this one, when the look memory has one that is similar enough
        if back is not None:
            suggestions = look_rule.rerank(suggestions, back.lead)  # the same advice; what the look eased after last time comes first
        signals = found + ([past[1]] if past else [])
        said = words.render(rule.facts(signals, suggestions, past[0] if past else None, self.clock.tz, callback=back), self.words)
        plan = self.planner.current.get(r.day)
        return DayCheckinRaised(event_id=f"day-checkin-{r.day}-{now.astimezone(self.clock.tz):%H%M}", ts=now, day=r.day, at=now,
                                signals=signals, suggestions=suggestions, history=past[0] if past else None,
                                proposes_replan=move is not None, plan_id=plan.plan_id if plan else None, text=said.display,
                                callback=back)

    def look_back(self, now: datetime, found: list) -> CheckinCallback | None:
        """The look memory's callback, only when a look is one of the signals. Never a reason for a check-in to fail."""
        if self.look_memory is None or "expression" not in rule.kinds(found):
            return None
        try:
            return self.look_memory.callback(now)
        except Exception:
            log.exception("the look memory's callback failed")
            return None

    def reading(self, now: datetime) -> Reading:
        window = timedelta(minutes=rule.EXPR_WINDOW_MIN)
        looks = max(self.looks.values(), default=[], key=lambda rows: sum(x.samples for x in rows if now - window < x.end <= now))
        # At the desk means a stay that is still open: someone who just stood up and left is not sitting there.
        here = self.coach.presence
        desk = here.intervals(now - timedelta(hours=12), now + timedelta(seconds=1)) if here.open else []
        return Reading(now=now, tz=self.clock.tz, day=self.clock.day_of(now), nights=self.nights, looks=looks, desk=desk,
                       drinks=self.drinks, sessions=self.sessions(), activity=self.words.activity, workplace=self.words.workplace)

    def sessions(self) -> dict[date, list[PastSession]]:
        """Every measured block of the main activity the coach noted or a review carried, by day."""
        rows: dict[date, dict[tuple, PastSession]] = {}
        for day, review in self.planner.reviews.items():
            for s in (review.sessions if review else []):
                if s.measured:
                    rows.setdefault(day, {})[s.start, s.block_id] = PastSession(s.start, s.planned_minutes, s.desk_minutes or 0, review.event_id)
        for day, notes in self.coach.notes.items():
            for n in notes.values():
                if n.session.measured:
                    rows.setdefault(day, {})[n.session.start, n.session.block_id] = PastSession(
                        n.session.start, n.session.planned_minutes, n.session.desk_minutes or 0, n.event_id)
        return {day: list(found.values()) for day, found in rows.items()}

    def gate(self, now: datetime, r: Reading) -> Gate:
        day = r.day
        plan = self.planner.current.get(day)
        fixed = next((m.title for m in plan.moves if m.fixed and m.start <= now < m.end), None) if plan else None
        leave_by = event_start = None
        try:
            status = self.tracker.compute(now)
            if status is not None and status.next_event is not None:
                leave_by, event_start = status.next_event.leave_by, status.next_event.start
        except Exception:
            log.debug("no status for the check-in's gate", exc_info=True)
        last = max((t for t in (self.raised[-1].ts if self.raised else None, self.sent_at) if t is not None), default=None)
        return Gate(awake=self.clock.wake_at(day) <= now < self.clock.bed_at(day), at_desk=rule.sitting(r.desk, now) is not None,
                    fixed=fixed, leave_by=leave_by, event_start=event_start, last=last)

    # --- "move the block": a re-plan behind the usual card, never a change by itself ---

    def replan(self, now: datetime, why: str = "", draft: bool = True) -> tuple[list[Event], str] | None:
        """(the re-plan's events, the offer in words) with today's blocks of the main activity moved to after the
        next meal, or None when there is nothing to move or the engine would not take it. `draft`: no card."""
        seen = self.tracker.where(now)
        moved: list[int] = []

        def adjust(moves):
            got = rule.after_meal(moves, now, self.clock.tz)
            if got is None:
                return None
            moved[:] = got[1]
            return got[0]

        events = self.planner.replan(now, seen[0] if seen else None, why, asked=True, adjust=adjust, draft=draft)
        if not events or not isinstance(events[0], DayReplanProposed):
            return None
        meal = rule.moved_after(events[0].moves, now, self.clock.tz)  # read back from what the engine really placed
        if meal is None:
            return None
        return events, rule.move_text(meal, len(moved), self.words.activity)

    def offer(self, now: datetime) -> str | None:
        try:
            got = self.replan(now)
        except Exception:
            log.exception("the check-in's re-plan draft failed")
            return None
        return got[1] if got else None

    def move(self, now: datetime, checkin: DayCheckinRaised) -> list[Event]:
        """"Move the block": the re-plan and its approval card, and the answer. Empty when it can no longer be made."""
        draft = self.replan(now)
        if draft is None:
            return []
        got = self.replan(now, why=f"You asked to {words.lower_first(draft[1])}.", draft=False)
        if got is None:
            return []
        events = got[0]
        return events + [DayCheckinAnswered(event_id=f"day-checkin-answer-{checkin.event_id}-move-{now.timestamp():.0f}", ts=now, day=checkin.day,
                                            checkin_id=checkin.event_id, answer="move", action_id=events[0].action_id)]

    def done(self, now: datetime, checkin: DayCheckinRaised) -> DayCheckinAnswered:
        return DayCheckinAnswered(event_id=f"day-checkin-answer-{checkin.event_id}-done", ts=now, day=checkin.day,
                                  checkin_id=checkin.event_id, answer="done")

    # --- for the phone (GET /api/day/checkin) ---

    def find(self, checkin_id: str | None) -> DayCheckinRaised | None:
        if checkin_id is None:
            return self.raised[-1] if self.raised else None
        return next((e for e in self.raised if e.event_id == checkin_id), None)

    def latest(self, now: datetime) -> DayCheckinRaised | None:
        """The check-in to show on the Now screen: the newest, for the rest of its day while it waits for an answer,
        and for FRESH after it once it has one."""
        e = self.raised[-1] if self.raised else None
        if e is None or now < e.ts or self.clock.day_of(now) != e.day:
            return None
        return e if e.event_id not in self.answers or now - e.ts <= FRESH else None

    def card(self, e: DayCheckinRaised, numbers: bool = False) -> dict:
        answer = self.answers.get(e.event_id)
        said = self.narrator.latest.get("checkin") if self.narrator is not None else None
        told = said if said is not None and said.ref == e.event_id else None
        body = e.model_dump(mode="json")
        if e.callback is not None and (self.look_memory is None or e.callback.moment_id not in self.look_memory.moments):
            body["callback"] = None  # the remembered moment was forgotten since: its chip goes with it
        return {**body, "answered": answer.answer if answer else None,
                "spoken": told.spoken if told else None, "display": told.display if told else e.text,
                "evidence": self.evidence(e, numbers)}

    def view(self, now: datetime, numbers: bool = False) -> dict:
        latest = self.latest(now)
        return {"now": now.isoformat(), "enabled": self.enabled, "latest": self.card(latest, numbers) if latest else None,
                "history": [{"id": e.event_id, "at": e.at.isoformat(), "day": e.day.isoformat(), "text": e.text,
                             "signals": [s.summary for s in e.signals],
                             "answered": self.answers[e.event_id].answer if e.event_id in self.answers else None}
                            for e in reversed(self.raised[-10:])],
                "watching": self.watching(now), "rule": RULE, "limits": list(rule.LIMITS)}

    def watching(self, now: datetime) -> dict:
        """What the rule sees right now, and why it is quiet: so a silent box can be told from a broken one."""
        try:
            r = self.reading(now)
            found = rule.signals(r)
            return {"signals": [s.summary for s in found], "agree": rule.agree(found), "held": rule.held(now, self.gate(now, r)),
                    "nights": len(self.nights), "expressions": bool(self.looks), "drink_in_view": self.drinks.container is not None}
        except Exception as exc:
            return {"signals": [], "agree": False, "held": f"unavailable: {type(exc).__name__}"}

    def evidence(self, e: DayCheckinRaised, numbers: bool = False) -> dict:
        """What is behind each chip, looked up on the box when the phone asks. As of the check-in, not of now."""
        out: dict = {}
        have = {s.kind: s for s in e.signals}
        night = self.nights.get(e.day)
        if "sleep" in have and night:
            usual = rule.own(self.nights, e.day, "asleep_min")
            out["sleep"] = {"asleep_min": night.asleep_min, "in_bed_min": night.in_bed_min, "stages": night.stages,
                            "usual_min": round(usual.median) if usual else None, "nights": usual.n if usual else 0}
        if "resting_hr" in have and night:
            base = rule.own(self.nights, e.day, "resting_hr")
            days = sorted(d for d, n in self.nights.items() if n.resting_hr is not None and 0 <= (e.day - d).days <= rule.BASELINE_DAYS)
            if base and base.median:
                # The line's shape only: each day against the person's own baseline (1.0). The values themselves are
                # sent only when the phone asked to show numbers.
                series = [{"day": d.isoformat(), "rel": round(self.nights[d].resting_hr / base.median, 3)} for d in days]
                out["resting_hr"] = {"series": series, "days": base.n}
                if numbers:
                    out["resting_hr"] |= {"baseline": round(base.median, 1), "today": self.nights[e.day].resting_hr,
                                          "values": [self.nights[d].resting_hr for d in days]}
        if "expression" in have:
            rows = max(self.looks.values(), default=[], key=lambda xs: sum(x.samples for x in xs if e.at - timedelta(hours=1) < x.end <= e.at))
            out["expression"] = {"windows": [{"start": x.start.isoformat(), "end": x.end.isoformat(), "samples": x.samples,
                                              "strained": round(x.strained / x.samples, 2)}
                                             for x in rows if e.at - timedelta(hours=1) < x.end <= e.at and x.samples > 0]}
        if "sitting" in have:
            out["sitting"] = {"since": have["sitting"].since.isoformat() if have["sitting"].since else None, "minutes": have["sitting"].minutes}
        if "drink" in have:
            s = have["drink"]
            frames = [ref for ref in s.evidence_refs if ref.startswith("frames:")]
            seen = None
            label = self.drinks.container[1] if self.drinks.container else None
            if self.memory is not None and label:
                try:
                    hit = self.memory.last_seen(label)
                    seen = {"label": hit.label, "ts": hit.ts.isoformat(), "frame_ref": hit.frame_ref or None, "zone_id": hit.zone_id,
                            "position": hit.position} if hit else None
                except Exception:
                    log.debug("memory's last-seen unavailable for the check-in", exc_info=True)
            if seen is None and self.drinks.container:
                at, label, frame = self.drinks.container
                seen = {"label": label, "ts": at.isoformat(), "frame_ref": frame or None, "zone_id": None, "position": None}
            out["drink"] = {"since": s.since.isoformat() if s.since else None, "picked_up": s.since is not None,
                            "frame_ref": frames[0] if frames else None, "last_seen": seen}
        if e.history:
            out["history"] = {"days": [d.model_dump(mode="json") for d in e.history.days], "outcome": e.history.outcome,
                              "by": words.clock(e.history.by) if e.history.by else None}
        return out


RULE = {
    "speaks_when": "two independent groups of signals agree and at least one is about right now; the sitting and the drink count as one group",
    "baseline_days": rule.BASELINE_DAYS, "min_baseline_nights": rule.MIN_BASELINE, "sleep_short_min": rule.SLEEP_SHORT_MIN,
    "resting_hr_margin_bpm": rule.RHR_MARGIN, "resting_hr_spreads": rule.RHR_SPREADS, "expression_window_min": rule.EXPR_WINDOW_MIN,
    "expression_share": rule.EXPR_SHARE, "expression_min_looks": rule.EXPR_MIN_SAMPLES, "sitting_min": rule.SIT_MIN,
    "drink_gap_min": rule.DRINK_GAP_MIN, "cooldown_min": rule.COOLDOWN_MIN, "leave_guard_min": rule.LEAVE_GUARD_MIN,
    "history_days": rule.HISTORY_MIN, "every_min": rule.EVERY_MIN,
}
