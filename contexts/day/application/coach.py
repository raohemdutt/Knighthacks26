"""The study coach (D2): each planned block of the main activity against what the box recorded.

    coach = DayCoach(planner, dayrun, clock, memory=memory, habits=habits)

"The camera watches you study and gives you improvement: you did this and wanted to do that, but you got
distracted because of this, maybe I suggest this." For every block of the profile's main activity
("study" for a student; the words come from the profile's vocabulary through Day Run's config), the coach
works out the minutes actually at the desk, the breaks, the longest unbroken stretch, the late start, and
for each break the most likely trigger with its evidence (the rules are in contexts/day/domain/coach.py).

Where the inputs come from:
- at the desk, and someone at the door: `perception.zone.event` for a person in the desk and door zones;
- the phone, a drink or food: `perception.scene.described` on the desk camera, its captions and the desk
  zone's object lists (an object counts when it was not in the list before: appearing, not lying there);
- chat messages with their times, and the episode a scene or a door entry belongs to (the id the screen
  cites): Memory's reader (`items`, `episodes`);
- the day's desk minutes: Habits' digest; the planned blocks: the planner's current plan;
- how the person looked (optional): `perception.expression.observed` on the desk camera for an enrolled member who
  switched "read my expressions" on. With no such events the coach says exactly what it said before. A member who
  is no longer enrolled (`perception.identity.changed`) is forgotten here too.

When a block ends, and the person was at the desk for at least five minutes of it, the coach publishes
`day.coach.noted` (one sentence for the phone and the voice; a skipped block is not announced); the evening review
carries the day's sessions; the next morning's plan reads the advice back as a `review` source and sizes
its blocks to the longest unbroken stretch. Every measured block also feeds Day Run's forecast calibration
(planned minutes are the estimate, desk minutes the result).

State rebuilds from the log (those handlers replay): desk presence, door entries, desk scenes that could
pull someone away, and the notes already published. Publishing a note is live only. Nothing here runs a model
or reads the wall clock.
"""

import logging
from datetime import date, datetime, timedelta

from contexts.day.domain.clock import DayClock
from contexts.day.domain.coach import (
    FOOD_LABELS,
    FOOD_WORDS,
    LIMITS,
    NOTE_MIN_MINUTES,
    PHONE_WORDS,
    Advice,
    STRAINED_LOOKS,
    DeskPresence,
    Looks,
    Signal,
    Words,
    day_text,
    mentions,
    note_text,
    review_session,
    suggest,
)
from contexts.play.dayrun.models import Estimate
from contexts.play.dayrun.service import DayRun
from contracts.day import CoachCalibration, DayCoachNoted, DayPlanProposed, DayReplanProposed, NextEvent, PlanMove, SessionReview
from contracts.events import Event
from contracts.perception import ExpressionObserved, SceneDescribed, ZoneEvent

log = logging.getLogger("atlas.day")
KEEP = timedelta(days=3)  # how far back presence and scenes are kept, in event time
NOTE_WITHIN = timedelta(hours=3)  # a block that ended longer ago than this gets no note (the box was off)
MAIN_KIND = "study"  # Day Run's block kind for the profile's main activity, whatever the profile calls it
MESSAGE = "message"


class DayCoach:
    def __init__(self, planner, dayrun: DayRun, clock: DayClock, memory=None, habits=None, desk_zone: str = "desk",
                 door_zone: str = "door", desk_camera: str = "cam-desk") -> None:
        self.planner, self.dayrun, self.clock = planner, dayrun, clock
        self.memory, self.habits = memory, habits
        self.desk_zone, self.door_zone, self.desk_camera = desk_zone, door_zone, desk_camera
        v = dayrun.cfg.vocabulary
        self.words = Words(activity=v.activity, workplace=v.workplace)
        # --- state, rebuilt from the log ---
        self.presence = DeskPresence()
        self.seen: list[tuple[str, datetime, str]] = []  # (kind, at, camera): door entries and desk scenes, in time order
        self.desk_labels: set[str] | None = None  # the desk zone's last object list; None until the first one
        self.looks: dict[str, list[Looks]] = {}  # enrolled member -> how they looked at the desk camera, in time order
        self.notes: dict[date, dict[tuple[str, str], DayCoachNoted]] = {}  # day -> (plan id, block id) -> note
        self.latest: DayCoachNoted | None = None
        self._noting: set[tuple[str, str]] = set()  # handed to the bus, not yet seen back (live only)

    # --- state handlers (replay=True) ---

    def on_zone(self, e: ZoneEvent) -> None:
        if e.label != "person":
            return
        if e.zone_id == self.desk_zone:
            self.presence.add(e.ts, (e.camera_id, e.track_id), e.kind, e.dwell_s)
            if e.kind == "exit":
                self._prune(e.ts)
        elif e.zone_id == self.door_zone and e.kind == "enter":
            self.seen.append(("door", e.ts, e.camera_id))

    def on_scene(self, e: SceneDescribed) -> None:
        if e.zone_id == self.desk_zone and e.objects:  # the desk's object list: what appeared since the last one
            labels = {o.label.lower() for o in e.objects}
            new = labels - self.desk_labels if self.desk_labels is not None else set()
            self.desk_labels = labels
            if any(mentions(label, PHONE_WORDS) for label in new):
                self.seen.append(("phone", e.ts, e.camera_id))
            if any(label in FOOD_LABELS for label in new):
                self.seen.append(("food", e.ts, e.camera_id))
        elif e.zone_id is None and e.camera_id == self.desk_camera:  # a caption of the whole desk frame
            if mentions(e.caption, PHONE_WORDS):
                self.seen.append(("phone", e.ts, e.camera_id))
            if mentions(e.caption, FOOD_WORDS):
                self.seen.append(("food", e.ts, e.camera_id))

    def on_expression(self, e: ExpressionObserved) -> None:
        """How an enrolled member looked at the desk camera over a stretch. The anonymous room count is not a person's."""
        if e.member_id is None or e.camera_id != self.desk_camera or e.samples <= 0:
            return
        strained = sum(share for label, share in e.shares.items() if label in STRAINED_LOOKS) * e.samples
        rows = self.looks.setdefault(e.member_id, [])
        rows.append(Looks(e.window_start, e.window_end, e.samples, strained))
        if rows[0].end < e.window_end - KEEP - timedelta(days=1):
            self.looks[e.member_id] = [x for x in rows if x.end >= e.window_end - KEEP]

    def on_identity(self, e: Event) -> None:
        """Someone was forgotten: their looks go from here as well."""
        enrolled = set(getattr(e, "members", ()))
        self.looks = {m: rows for m, rows in self.looks.items() if m in enrolled}

    def looks_in(self, block: PlanMove) -> list[Looks]:
        """The looks inside the block of the member seen most in it (the desk is one person's at a time)."""
        inside = [[x for x in rows if x.end > block.start and x.start < block.end] for rows in self.looks.values()]
        return max(inside, key=lambda rows: sum(x.samples for x in rows), default=[])

    def on_noted(self, e: DayCoachNoted) -> None:
        self.notes.setdefault(e.day, {})[e.plan_id, e.block_id] = e
        self._noting.discard((e.plan_id, e.block_id))
        if self.latest is None or e.ts >= self.latest.ts:
            self.latest = e

    def _prune(self, now: datetime) -> None:
        self.presence.prune(now - KEEP)
        if self.seen and self.seen[0][1] < now - KEEP - timedelta(days=1):
            self.seen = [s for s in self.seen if s[1] >= now - KEEP]

    # --- live: a note when a block ends ---

    def on_event(self, e: Event) -> list[Event]:
        return self.check(e.ts)

    def tick(self, now: datetime) -> list[Event]:
        return self.check(now)

    def check(self, now: datetime) -> list[Event]:
        out: list[Event] = []
        for day, plan in sorted(self.planner.current.items()):
            if now - plan.start > KEEP:
                continue
            for block in self.blocks(day, plan):
                key = (plan.plan_id, block.id)
                if not (block.end <= now <= block.end + NOTE_WITHIN) or key in self._noting or key in self.notes.get(day, {}):
                    continue
                self._noting.add(key)  # once, measured or not
                try:
                    session = self.session(block)
                    if session.measured and (session.desk_minutes or 0) >= NOTE_MIN_MINUTES:
                        out.append(self.note(day, plan, block, session, now))
                except Exception:  # Memory or Habits being down must not stop the box
                    log.exception("the coach's note for %s failed", block.id)
        return out

    def note(self, day: date, plan, block: PlanMove, session: SessionReview, now: datetime) -> DayCoachNoted:
        nxt = next((b for b in sorted(plan.moves, key=lambda m: m.start) if b.kind == MAIN_KIND and b.start >= block.end), None)
        advice = suggest([session], self.words)  # about this block; the day's pattern is the evening review's
        return DayCoachNoted(
            event_id=f"day-coach-{plan.plan_id}-{block.id}", ts=now, day=day, plan_id=plan.plan_id, block_id=block.id,
            text=note_text(session, nxt, self.words), session=session, suggestions=advice.suggestions,
            calibration=self.calibration([session]),
            next_block=NextEvent(id=nxt.id, title=nxt.title, start=nxt.start, end=nxt.end, place=nxt.place, place_name=nxt.place_name) if nxt else None)

    # --- the sessions of a day ---

    def blocks(self, day: date, plan: DayPlanProposed | DayReplanProposed) -> list[PlanMove]:
        """The plan's blocks of the main activity that are this day's to judge: a plan runs into the next
        morning, and from the moment the next day has its own plan, that plan owns what follows."""
        nxt = self.planner.plans.get(day + timedelta(days=1))
        return [m for m in sorted(plan.moves, key=lambda m: m.start)
                if m.kind == MAIN_KIND and m.minutes > 0 and (nxt is None or m.end <= nxt.ts)]

    def session(self, block: PlanMove) -> SessionReview:
        spans = self.presence.intervals(block.start, block.end)
        desk = self.dayrun.walks.ref(self.dayrun.cfg.desk)
        return review_session(block, spans, self.signals(block) if spans else [], desk, self.words, self.clock.tz,
                              looks=self.looks_in(block) if spans and self.looks else None)

    def sessions(self, day: date, now: datetime) -> list[SessionReview]:
        """Every block of the main activity in the day's plan that has ended by `now`, measured or said not to be."""
        plan = self.planner.current.get(day)
        if plan is None:
            return []
        noted = {k[1]: n.session for k, n in self.notes.get(day, {}).items() if k[0] == plan.plan_id}
        out = []
        for block in self.blocks(day, plan):
            if block.end > now:
                continue
            # past what the box still holds in memory, the note written at the time is the record
            old = now - block.start > KEEP - timedelta(hours=1)
            out.append(noted[block.id] if old and block.id in noted else self.session(block))
        return out

    def signals(self, block: PlanMove) -> list[Signal]:
        """Everything recorded inside the block that could pull someone away from the desk."""
        lo, hi = block.start - timedelta(minutes=5), block.end
        out: list[Signal] = []
        if self.memory is not None:
            try:
                out += [Signal("chat", i.when, f"{i.connector}:{i.external_id}") for i in self.memory.items(lo, hi, kind=MESSAGE) if i.when]
            except Exception:
                log.debug("memory items unavailable for the coach", exc_info=True)
        for kind, at, camera in self.seen:
            if lo <= at < hi:
                out.append(Signal(kind, at, self.episode_id(camera, at)))
        return out

    def episode_id(self, camera: str, at: datetime) -> str:
        """The Memory episode that holds this moment on this camera: the id the screen cites."""
        if self.memory is not None:
            try:
                eps = [e for e in self.memory.episodes(at - timedelta(hours=1), at + timedelta(seconds=1), camera=camera)
                       if e.start <= at and (e.end is None or e.end >= at - timedelta(minutes=1))]
                if eps:
                    return eps[-1].id
            except Exception:
                log.debug("memory episodes unavailable for the coach", exc_info=True)
        return f"{camera}@{at.isoformat()}"

    # --- the day, for the evening review, the API and tomorrow's plan ---

    def habits_desk_minutes(self, day: date, now: datetime) -> int | None:
        if self.habits is None:
            return None
        try:
            rows = {r["id"]: r for r in self.habits.digest(day=day, now=now).get("habits", [])}
            return round(rows["focus"]["value"] * 60) if "focus" in rows and rows["focus"].get("applies", True) else None
        except Exception:
            return None

    def calibration(self, extra: list[SessionReview] = ()) -> CoachCalibration | None:
        """Day Run's forecast calibration over every measured block so far: planned minutes against desk minutes."""
        rows: dict[str, tuple[int, int]] = {}
        noted = [n.session for notes in self.notes.values() for n in notes.values()]
        for x in [*noted, *(x for x in extra if x.measured)]:  # a block already noted counts once
            rows.setdefault(f"{x.start.isoformat()}/{x.block_id}", (x.planned_minutes, x.desk_minutes or 0))
        if not rows:
            return None
        result = self.dayrun.forecast([Estimate(block=k, minutes=p) for k, (p, _) in rows.items()], {k: float(m) for k, (_, m) in rows.items()})
        planned, done = sum(p for p, _ in rows.values()), sum(m for _, m in rows.values())
        n = len(rows)
        if done <= 0:
            return CoachCalibration(score=result.score, ratio=0.0, blocks=n,
                                    text=f"None of the {planned} planned minutes were measured at the {self.words.workplace} yet (calibration {result.score}/100).")
        ratio = round(planned / done, 1)
        lean = "match what happens" if 0.95 <= ratio <= 1.05 else f"run {ratio:g}x what happens"
        return CoachCalibration(score=result.score, ratio=ratio, blocks=n,
                                text=f"Your plans {lean} (calibration {result.score}/100 over {n} measured block{'' if n == 1 else 's'}).")

    def day_review(self, day: date, now: datetime) -> dict:
        """The DayReviewed fields the coach fills, and what GET /api/day/coach returns."""
        sessions = self.sessions(day, now)
        advice = suggest(sessions, self.words)
        calibration = self.calibration(sessions)
        text = day_text(sessions, self.words, self.habits_desk_minutes(day, now) if sessions else None, self.clock.tz)
        if text and calibration and advice.block_minutes:
            text += f" {calibration.text[:-1]}; I'm planning {advice.block_minutes}-minute blocks."
        elif text and calibration:
            text += f" {calibration.text}"
        return {"sessions": sessions, "coach_text": text, "suggestions": advice.suggestions if sessions else [],
                "calibration": calibration if sessions else None, "block_minutes": advice.block_minutes}

    def advice(self, day: date) -> tuple[Advice, str, str] | None:
        """What yesterday taught, for the plan of `day`: the advice, the plan it came from, and one line to read back."""
        before = day - timedelta(days=1)
        sessions: dict[tuple[datetime, str], SessionReview] = {}
        review = self.planner.reviews.get(before)
        for s in (review.sessions if review else []):
            sessions[s.start, s.block_id] = s
        plan_id = review.plan_id if review else None
        for (pid, _), n in self.notes.get(before, {}).items():  # a block that ended after the review is in its note
            sessions[n.session.start, n.session.block_id] = n.session
            plan_id = plan_id or pid
        advice = suggest(list(sessions.values()), self.words)
        if not advice.suggestions or plan_id is None or not any(s.measured for s in sessions.values()):
            return None
        calibration = self.calibration(list(sessions.values()))
        text = "coach: " + " ".join(advice.suggestions)
        if calibration and calibration.ratio and advice.block_minutes:
            text += f" Your plans run {calibration.ratio:g}x what happens."
        return advice, plan_id, text

    def recent(self, now: datetime, within: timedelta = timedelta(hours=24)) -> list[DayCoachNoted]:
        """The notes of the last day, newest first: what the Now screen shows after a block ends."""
        notes = [n for day in self.notes.values() for n in day.values() if timedelta(0) <= now - n.ts <= within]
        return sorted(notes, key=lambda n: n.ts, reverse=True)

    def limits(self) -> list[str]:
        out = [s.format(workplace=self.words.workplace) for s in LIMITS]
        if self.looks:  # only where expressions are read at all
            out.append('"Looked strained" is how a face looked to a small model over many looks: a noisy signal, '
                       'not an emotion and nothing about health.')
        return out
