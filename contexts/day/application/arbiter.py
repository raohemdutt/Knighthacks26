"""The arbiter on the box: every `day.narrated` Atlas says unasked passes through here (rules: contexts/day/domain/arbiter.py).

    arbiter = Arbiter(narrator, planner, clock, rules=Rules.of(agent_yaml_day))
    narrator.arbiter = arbiter          # Narrator.on_event and Narrator.tell go through it
    arbiter.subscribe(bus)              # after the narrator: what was said, what is open, rebuilt from the log
    ... DayLoop.tick ends with arbiter.tick(now): the queued lines that are due now, merged or one by one

The public interface (what another part of the day context calls):

- `offer(job) -> list[Event]`: an event-driven narration (the narrator's `on_event`): said now, said silent inside
  the plan narration it belongs to, held for a free slot, or dropped.
- `queue(job) -> list[Event]`: a tick-driven narration (`Narrator.tell`: a block start, a suggestion). Always
  returns []; the line is decided on the next `tick`, together with whatever else is due, so two lines due in
  the same minute are merged into one sentence.
- `tick(now) -> list[Event]`: what the held and queued lines come to at `now`.
- `quiet(moment, now, about=()) -> str | None`: why a line of `moment` would not be said at `now` (a suppression
  rule), or None. A producer may ask before it builds its facts.
- `speaker`: who says a line aloud: a callable returning "box" or None (the phone's voice). Default None.

State that rebuilds from the log (replay=True): the newest speak-up (`day.narrated` not merged, urgent "now",
`habits.nudge.raised`, `agent.alert.raised`), leave-now / late lines by the event they are about, check-ins and
their answers, open approval cards. Live only: the held lines, the answer being said. Nothing here is emitted on
replay, and all times are event times.
"""

import logging
from dataclasses import replace
from datetime import datetime, timedelta

from contexts.day.domain import arbiter as rule
from contexts.day.domain.arbiter import Line, Rules, Scene
from contexts.day.domain.narration import URGENT_RISKS
from contracts.events import Event

log = logging.getLogger("atlas.day.arbiter")
ANSWER_SPEAK = timedelta(seconds=30)  # after an answer is written, the voice may still be saying it this long
# Moments that are screen state, not speak-ups: a non-urgent "now" is never pushed or said unasked.
SCREEN_ONLY = ("now",)


class Arbiter:
    def __init__(self, narrator, planner, clock, rules: Rules | None = None) -> None:
        self.narrator, self.planner, self.clock = narrator, planner, clock
        self.rules = rules or Rules()
        self.ledger = rule.Ledger()
        self.speaker = lambda: None  # "box" when the box's own speaker should say the lines (bootstrap sets it)
        # --- live only ---
        self.held: list = []  # Jobs waiting for a slot, oldest first
        self.answered_at: datetime | None = None
        self.questions: set[str] = set()
        self.log: list[tuple[str, str, str]] = []  # (event id or moment, verdict, reason): what was decided, for tests

    def subscribe(self, bus) -> None:
        bus.subscribe("day.narrated", self.on_narrated, replay=True)
        bus.subscribe("day.status.changed", self.on_status, replay=True)
        bus.subscribe("day.checkin.raised", self.on_checkin, replay=True)
        bus.subscribe("day.checkin.answered", self.on_checkin_answered, replay=True)
        bus.subscribe("habits.nudge.raised", self.on_other, replay=True)
        bus.subscribe("agent.alert.raised", self.on_other, replay=True)
        bus.subscribe("workspace.approval.requested", self.on_card, replay=True)
        bus.subscribe("experience.approval.decided", self.on_card_closed, replay=True)
        bus.subscribe("workspace.action.completed", self.on_card_closed, replay=True)
        bus.subscribe("experience.question.asked", self.on_question)
        bus.subscribe("agent.answer.generated", self.on_answer)

    # --- state (replay=True) ---

    def on_narrated(self, e) -> None:
        if e.merged_into or (e.moment in SCREEN_ONLY and not e.urgent):
            return
        self.ledger.note_said(e.moment, e.ts, urgent=e.urgent, plan_id=e.plan_id, event_id=e.event_id)

    def on_status(self, e) -> None:
        if e.risk in URGENT_RISKS:
            self.ledger.note_urgent([e.next_event.id if e.next_event else None, e.block_id], e.ts)

    def on_checkin(self, e) -> None:
        self.ledger.note_checkin(e.event_id, e.ts)

    def on_checkin_answered(self, e) -> None:
        self.ledger.note_checkin_answered(e.checkin_id)

    def on_other(self, e) -> None:
        """A habit nudge or a room alert: another context's speak-up. It takes the slot; it is never held here."""
        moment = "alert" if e.type == "agent.alert.raised" else "nudge"
        self.ledger.note_said(moment, e.ts, urgent=moment == "alert", event_id=e.event_id)

    def on_card(self, e) -> None:
        self.ledger.note_card(e.action_id, e.expires_at)

    def on_card_closed(self, e) -> None:
        self.ledger.note_card_closed(e.action_id)

    # --- live ---

    def on_question(self, e) -> None:
        self.questions.add(e.question_id)

    def on_answer(self, e) -> None:
        self.questions.discard(e.question_id)
        if self.answered_at is None or e.ts >= self.answered_at:
            self.answered_at = e.ts

    def scene(self, now: datetime) -> Scene:
        day = self.clock.day_of(now)
        awake = self.clock.wake_at(day) <= now < self.clock.bed_at(day)
        plan = self.planner.current.get(day)
        fixed = bool(plan) and any(m.fixed and m.start <= now < m.end for m in plan.moves)
        answering = bool(self.questions) or (self.answered_at is not None and timedelta(0) <= now - self.answered_at < ANSWER_SPEAK)
        return Scene(awake=awake, fixed_running=fixed, answering=answering)

    @staticmethod
    def line(job) -> Line:
        return Line(moment=job.moment, ts=job.ts, ref=job.ref, urgent=job.urgent, plan_id=job.plan_id, about=frozenset(job.about))

    def quiet(self, moment: str, now: datetime, about=()) -> str | None:
        """Why a line of `moment` would not be said at `now` (suppression only: a line in the gap is held, not quiet)."""
        return rule.suppressed(Line(moment=moment, ts=now, about=frozenset(about)), self.scene(now), self.ledger, now, self.rules)

    def offer(self, job) -> list[Event]:
        if job.moment in SCREEN_ONLY and not job.urgent:
            return self.narrator.say(job)
        now = job.ts
        verdict = rule.decide(self.line(job), self.scene(now), self.ledger, now, self.rules)
        self.log.append((job.event_id, verdict.action, verdict.reason))
        if verdict.action == "say":
            return self.out(job, now)
        if verdict.action == "companion":
            job.merged_into = verdict.into
            return [self.narrator.event(job, job.draft.spoken, "template")]
        if verdict.action == "hold":
            self.held.append(job)
        return []

    def queue(self, job) -> list[Event]:
        if not any(j.event_id == job.event_id for j in self.held):
            self.held.append(job)
        return []

    def tick(self, now: datetime) -> list[Event]:
        """The held lines at `now`: the highest that may be said goes out (with a second merged into it when it
        can be), the suppressed and stale ones are dropped, the rest keep waiting."""
        if not self.held:
            return []
        scene, keep, ready = self.scene(now), [], []
        for job in sorted(self.held, key=lambda j: (self.line(j).rank, j.ts)):
            verdict = rule.decide(self.line(job), scene, self.ledger, now, self.rules)
            if verdict.action == "drop":
                self.log.append((job.event_id, "drop", verdict.reason))
            elif verdict.action == "say":
                ready.append(job)
            else:
                keep.append(job)
        out: list[Event] = []
        if ready:
            first, rest = ready[0], ready[1:]
            second = next((j for j in rest if rule.mergeable(self.line(first), self.line(j))), None)
            if second is not None:
                out = self.merged(first, second, now)
                rest = [j for j in rest if j is not second]
            else:
                self.log.append((first.event_id, "say", "the slot is free"))
                out = self.out(replace(first, ts=now), now)
            keep = rest + keep
        self.held = keep
        return out

    def merged(self, first, second, now: datetime) -> list[Event]:
        facts = dict(first.facts, also=second.facts)
        job = self.narrator.job(first.moment, facts, now, first.day, first.plan_id, first.ref, urgent=first.urgent,
                                event_id=first.event_id)
        if job.draft.spoken == first.draft.spoken:  # the second has no short form: one at a time after all
            self.held.append(second)
            self.log.append((first.event_id, "say", "the slot is free"))
            return self.out(replace(first, ts=now), now)
        job.about = first.about | second.about
        self.log.append((first.event_id, "say", f"merged with {second.event_id}"))
        self.log.append((second.event_id, "merged", first.event_id))
        said = self.out(job, now, fast=True)
        silent = replace(second, ts=now, merged_into=first.event_id)
        return said + [self.narrator.event(silent, silent.draft.spoken, "template")]

    def out(self, job, now: datetime, fast: bool = False) -> list[Event]:
        job.ts = now
        job.spoken_by = self.speaker()
        self.ledger.note_said(job.moment, now, urgent=job.urgent, plan_id=job.plan_id, event_id=job.event_id)
        return self.narrator.say(job, fast=fast)
