"""The narrator (D3): the day's events said the way a person who knows your day would say them.

    narrator = Narrator(planner, tracker, coach, clock, runtime=runtime, config=NarrationConfig.of(raw, runtime))

The planner, the tracker and the coach publish structured events (`day.plan.proposed`, `day.status.changed`,
`day.replan.proposed`, `day.coach.noted`, `day.reviewed`) and approval cards. For each of those moments the
narrator gathers a few facts, turns them into words (contexts/day/domain/narration.py) in two registers,
`spoken` for the ear and `display` for the screen, and publishes `day.narrated`. The phone's Now screen, its
voice and the push notification all use that one event, so they say the same thing.

Two engines behind one call, chosen per narration:

- the template engine: deterministic, no model, always available. It is the answer whenever the model is not
  used, and the draft the model is asked to improve.
- the model engine: the on-device model through `runtime.chat` at the lowest priority ("backfill": the
  runtime's gate lets a waiting question go first). It rewrites the template's spoken text, and its reply is
  used only when `narration.check` finds every time, number and name in it grounded in the facts. A rejected
  reply, a timeout (`timeout_s`, default 6 s) or a runtime error falls back to the template.

The model is never called: on replay (the handlers that narrate are live only); with the fake runtime; while a
question is waiting for its answer; while another narration call is in flight; within `gap_s` (default 60 s of
event time) of the last one; for an approval ask or an urgent status (leave now, late), which must not wait;
or for facts it already rewrote or failed on. The same facts are never narrated twice in a row.

State that rebuilds from the log (replay=True): the latest narration per moment, the asks by action id, and
the model's accepted rewrites by facts hash, so a restart says the same words without calling the model.

Config (config/agent.yaml, layered by the profile; ATLAS_DAY_NARRATION overrides the mode):

    day:
      narration: model        # model | template | off. Default: model with a real runtime, template with the fake one
      narration_timeout_s: 6
      narration_gap_s: 60
      narration_tone: ""      # one more sentence of tone for the model, after the profile's vocabulary line
"""

import asyncio
import logging
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from contexts.day.domain import narration as words
from contexts.day.domain.clock import DayClock
from contexts.day.domain.narration import Narration, hhmm, short_place, short_title
from contracts.day import (
    DayCoachNoted,
    DayNarrated,
    DayPlanProposed,
    DayReplanProposed,
    DayReviewed,
    DayStatusChanged,
    PlanMove,
)
from contracts.events import Event
from contracts.types import ChatMessage
from contracts.workspace import ApprovalRequested
from runtime.fake import FakeRuntime

log = logging.getLogger("atlas.day.narrator")
MODES = ("model", "template", "off")
MODE_ENV = "ATLAS_DAY_NARRATION"
CALENDAR_WRITE = "calendar.create_event"
QUESTION_TTL = timedelta(seconds=120)  # a question with no answer after this long is not waiting any more
KEEP_ASKS = 20
KEEP_CACHE = 200
PRIORITY = "backfill"  # the runtime's lowest: never ahead of a person's question


@dataclass(frozen=True)
class NarrationConfig:
    mode: str = "template"
    timeout_s: float = 6.0
    gap_s: float = 60.0
    tone: str = ""

    @classmethod
    def of(cls, raw: dict | None = None, runtime=None) -> "NarrationConfig":
        """`raw` is agent.yaml's `day:` block. With no mode set, a real model narrates and the fake one does not."""
        raw = raw or {}
        real = runtime is not None and not isinstance(runtime, FakeRuntime)
        mode = (os.environ.get(MODE_ENV) or raw.get("narration") or ("model" if real else "template")).strip().lower()
        if mode not in MODES:
            log.warning("day.narration is %r; one of %s. Using the templates.", mode, ", ".join(MODES))
            mode = "template"
        return cls(mode=mode, timeout_s=float(raw.get("narration_timeout_s", 6.0)), gap_s=float(raw.get("narration_gap_s", 60.0)),
                   tone=str(raw.get("narration_tone") or ""))


@dataclass
class Job:
    """One narration on its way out: the facts, the template's words, and where it came from."""

    moment: str
    facts: dict
    draft: Narration
    hash: str
    ts: datetime
    day: date
    plan_id: str | None
    ref: str | None
    urgent: bool
    event_id: str


class Narrator:
    def __init__(self, planner, tracker, coach, clock: DayClock, runtime=None, config: NarrationConfig | None = None,
                 description: str = "") -> None:
        self.planner, self.tracker, self.coach, self.clock = planner, tracker, coach, clock
        self.runtime = runtime
        self.config = config or NarrationConfig.of(None, runtime)
        v = planner.dayrun.cfg.vocabulary
        self.words = words.Words(activity=v.activity, workplace=v.workplace, event=v.event, deadline=v.deadline)
        self.tone = words.tone_line(self.words, description, self.config.tone)
        # --- state, rebuilt from the log ---
        self.latest: dict[str, DayNarrated] = {}  # moment -> the newest narration
        self.asks: OrderedDict[str, DayNarrated] = OrderedDict()  # action id -> how to ask for it
        self.cache: OrderedDict[str, str] = OrderedDict()  # facts hash -> the model's accepted rewrite
        # --- live only ---
        self.questions: dict[str, datetime] = {}  # questions waiting for an answer
        self.inflight = False
        self.last_model: datetime | None = None
        self.failed: set[str] = set()  # facts the model was already asked about and did not improve
        self.jobs: dict[str, Job] = {}  # handed to the background task, by source event id
        self.background = None  # bus.in_background(self.rewrite): set when subscribed
        self.stats = {"calls": 0, "used": 0, "rejected": 0, "timeouts": 0, "errors": 0, "last_problem": None}

    @property
    def mode(self) -> str:
        return self.config.mode

    @property
    def engine(self) -> str:
        """What actually writes the words: "model" only with the mode on and a real runtime behind it."""
        if self.mode == "off":
            return "off"
        real = self.runtime is not None and not isinstance(self.runtime, FakeRuntime)
        return "model" if self.mode == "model" and real else "template"

    def subscribe(self, bus) -> None:
        bus.subscribe("day.narrated", self.on_narrated, replay=True)
        if self.mode == "off":
            return
        self.background = bus.in_background(self.rewrite)  # the model call runs as its own task, off the bus
        bus.subscribe("experience.question.asked", self.on_question)
        bus.subscribe("agent.answer.generated", self.on_answer)
        for prefix in ("day.plan.proposed", "day.status.changed", "day.replan.proposed", "day.coach.noted", "day.reviewed",
                       "workspace.approval.requested"):
            bus.subscribe(prefix, self.on_event)  # live only: it publishes, and may call the model

    # --- state handlers (replay=True) ---

    def on_narrated(self, e: DayNarrated) -> None:
        if e.moment == "approval" and e.ref:
            self.asks[e.ref] = e
            while len(self.asks) > KEEP_ASKS:
                self.asks.popitem(last=False)
        last = self.latest.get(e.moment)
        if last is None or e.ts >= last.ts:
            self.latest[e.moment] = e
        if e.engine == "model":
            self.remember(e.facts_hash, e.spoken)

    def remember(self, facts_hash: str, spoken: str) -> None:
        self.cache[facts_hash] = spoken
        self.cache.move_to_end(facts_hash)
        while len(self.cache) > KEEP_CACHE:
            self.cache.popitem(last=False)

    # --- live handlers ---

    def on_question(self, e: Event) -> None:
        self.questions[e.question_id] = e.ts

    def on_answer(self, e: Event) -> None:
        self.questions.pop(e.question_id, None)

    def question_waiting(self, now: datetime) -> bool:
        for qid, asked in list(self.questions.items()):
            if abs(now - asked) > QUESTION_TTL:
                del self.questions[qid]
        return bool(self.questions)

    def on_event(self, e: Event) -> list[Event]:
        try:
            job = self.job_for(e)
        except Exception:  # a narration must never stop the day
            log.exception("could not narrate %s", e.type)
            return []
        if job is None or self.same_as_last(job):
            return []
        return self.say(job)

    def say(self, job: Job, fast: bool = False) -> list[Event]:
        """The narration now (the template's words, or the model's accepted rewrite of these facts), or nothing
        yet when the model was asked: its words, or the template's, follow from the background task."""
        if job.hash in self.cache:
            return [self.event(job, self.cache[job.hash], "model")]
        if fast or not self.may_ask_model(job) or self.background is None:
            return [self.event(job, job.draft.spoken, "template")]
        self.inflight = True  # taken here, before the task starts, so the next event sees it
        self.last_model = job.ts
        self.jobs[job.event_id] = job
        self.background(_Ticket(job.event_id))
        return []

    def same_as_last(self, job: Job) -> bool:
        if job.moment == "approval":
            return job.ref in self.asks
        last = self.latest.get(job.moment)
        waiting = any(j.moment == job.moment and j.hash == job.hash for j in self.jobs.values())
        # the same facts, or different facts that come out as the same words (seen at the desk, then at the door)
        return waiting or (last is not None and (last.facts_hash == job.hash or last.display == job.draft.display))

    def may_ask_model(self, job: Job) -> bool:
        if self.engine != "model" or job.moment == "approval" or job.urgent or self.inflight or job.hash in self.failed:
            return False
        if self.question_waiting(job.ts):
            return False
        return self.last_model is None or abs((job.ts - self.last_model).total_seconds()) >= self.config.gap_s

    # --- the model engine (live only, in the background) ---

    async def rewrite(self, ticket) -> list[Event]:
        job = self.jobs.get(ticket.event_id)
        if job is None:
            return []
        try:
            spoken = await self.ask_model(job)
        except Exception:  # anything unexpected: the template's words still go out
            log.exception("narration (%s) failed; using the template", job.moment)
            self.inflight, spoken = False, None
        finally:
            self.jobs.pop(ticket.event_id, None)
        return [self.event(job, spoken, "model") if spoken else self.event(job, job.draft.spoken, "template")]

    async def ask_model(self, job: Job) -> str | None:
        """The model's rewrite of the draft when it is grounded and came in time; None for the template."""
        self.inflight = True
        self.stats["calls"] += 1
        call = asyncio.get_running_loop().run_in_executor(None, self.call, job)
        late = {"timed_out": False}

        def done(f: asyncio.Future) -> None:
            self.inflight = False  # the one slot is free only when the runtime has really returned
            if late["timed_out"] and not f.cancelled() and f.exception() is None:
                self.accept(job, f.result(), count=False)  # too late to say now; kept for the next time these facts come up

        call.add_done_callback(done)
        try:
            content = await asyncio.wait_for(asyncio.shield(call), self.config.timeout_s)
        except asyncio.TimeoutError:
            late["timed_out"] = True
            self.stats["timeouts"] += 1
            self.stats["last_problem"] = f"no reply within {self.config.timeout_s:g} s"
            log.info("narration (%s): no reply within %g s; using the template", job.moment, self.config.timeout_s)
            return None
        except Exception as exc:
            self.stats["errors"] += 1
            self.stats["last_problem"] = f"runtime error: {type(exc).__name__}"
            self.failed.add(job.hash)
            log.warning("narration (%s): the runtime failed (%s); using the template", job.moment, exc)
            return None
        return self.accept(job, content)

    def call(self, job: Job) -> str:
        """The one model call, in a worker thread: a short system prompt, the facts as compact JSON, the draft."""
        messages = [ChatMessage(role="system", content=words.system_prompt(job.moment, self.tone)),
                    ChatMessage(role="user", content=words.user_prompt(job.facts, job.draft.spoken))]
        return self.runtime.chat(messages, priority=PRIORITY, json_schema=words.reply_schema(job.moment)).content

    def accept(self, job: Job, content: str, count: bool = True) -> str | None:
        text = words.reply_text(content)
        if text == job.draft.spoken:
            self.failed.add(job.hash)  # nothing to improve: the template stands, and is not asked about again
            return None
        problems = words.check(text, job.facts, job.draft.spoken)
        if problems:
            self.failed.add(job.hash)
            if count:
                self.stats["rejected"] += 1
                self.stats["last_problem"] = problems[0]
            log.info("narration (%s) rejected: %s | %r", job.moment, "; ".join(problems), text)
            return None
        self.remember(job.hash, text)
        if count:
            self.stats["used"] += 1
        return text

    def event(self, job: Job, spoken: str, engine: str) -> DayNarrated:
        return DayNarrated(event_id=job.event_id, ts=job.ts, day=job.day, moment=job.moment, plan_id=job.plan_id, ref=job.ref,
                           spoken=spoken, display=job.draft.display, engine=engine, facts_hash=job.hash, urgent=job.urgent)

    # --- on demand (POST /api/day/brief) ---

    async def regenerate(self, moment: str, now: datetime, fast: bool = False) -> DayNarrated | None:
        """The moment narrated again from the state as it is at `now`. The same facts as the latest narration
        return that narration (no new event) unless the model can still improve on a template. `fast` never
        waits for the model."""
        job = self.job_now(moment, now)
        if job is None:
            return None
        last = self.latest.get(moment)
        if last is not None and last.facts_hash == job.hash:
            if last.engine == "model" or fast or not self.may_ask_model(job) or job.hash in self.cache:
                return last
        if job.hash in self.cache:
            return self.event(job, self.cache[job.hash], "model")
        if fast or not self.may_ask_model(job):
            return self.event(job, job.draft.spoken, "template")
        self.last_model = job.ts
        spoken = await self.ask_model(job)
        return self.event(job, spoken, "model") if spoken else self.event(job, job.draft.spoken, "template")

    def job_now(self, moment: str, now: datetime) -> Job | None:
        p, day = self.planner, self.clock.day_of(now)
        if day not in p.current and p.current:
            day = max(p.current)
        stamp = f"{now.timestamp():.0f}"
        if moment == "morning":
            plan = p.current.get(day)
            return self.job("morning", self.plan_facts(plan, now), now, day, plan.plan_id, plan.event_id, event_id=f"day-narrated-brief-{plan.plan_id}-{stamp}") if plan else None
        if moment == "now":
            status = None
            try:
                status = self.tracker.compute(now)
            except Exception:
                log.debug("no fresh status for the brief", exc_info=True)
            status = status or self.tracker.status
            return self.status_job(status, event_id=f"day-narrated-now-{stamp}") if status else None
        if moment == "replan":
            replan = p.pending.get(day)
            return self.job_for(replan, event_id=f"day-narrated-replan-{stamp}") if replan else None
        if moment == "coach":
            note = self.coach.latest
            return self.job_for(note, event_id=f"day-narrated-coach-{stamp}") if note else None
        if moment == "evening":
            reviews = [r for r in p.reviews.values() if r is not None]
            review = p.reviews.get(day) or (max(reviews, key=lambda r: r.day) if reviews else None)
            return self.job_for(review, event_id=f"day-narrated-evening-{stamp}") if review else None
        return None

    # --- the facts of each moment ---

    def job(self, moment: str, facts: dict | None, ts: datetime, day: date, plan_id: str | None, ref: str | None,
            urgent: bool = False, event_id: str | None = None) -> Job | None:
        if facts is None:
            return None
        return Job(moment=moment, facts=facts, draft=words.render(facts, self.words), hash=words.facts_hash(facts, self.tone), ts=ts,
                   day=day, plan_id=plan_id, ref=ref, urgent=urgent, event_id=event_id or f"day-narrated-{ref}")

    def job_for(self, e: Event, event_id: str | None = None) -> Job | None:
        if isinstance(e, DayPlanProposed):
            return self.job("morning", self.plan_facts(e, e.ts), e.ts, e.day, e.plan_id, e.event_id, event_id=event_id)
        if isinstance(e, DayStatusChanged):
            return self.status_job(e, event_id=event_id)
        if isinstance(e, DayReplanProposed):
            return self.job("replan", self.replan_facts(e), e.ts, e.day, e.plan_id, e.event_id, event_id=event_id)
        if isinstance(e, DayCoachNoted):
            return self.job("coach", self.coach_facts(e), e.ts, e.day, e.plan_id, e.event_id, event_id=event_id)
        if isinstance(e, DayReviewed):
            return self.job("evening", self.review_facts(e), e.ts, e.day, e.plan_id, e.event_id, event_id=event_id)
        if isinstance(e, ApprovalRequested):
            facts = self.approval_facts(e)
            day = self.planner.cards[e.action_id][0] if e.action_id in self.planner.cards else self.clock.day_of(e.ts)
            return self.job("approval", facts, e.ts, day, self.planner.cards.get(e.action_id, (None, None))[1], e.action_id,
                            event_id=f"day-narrated-ask-{e.action_id}")
        return None

    def status_job(self, s: DayStatusChanged, event_id: str | None = None) -> Job | None:
        return self.job("now", self.status_facts(s), s.ts, s.day, s.plan_id, s.event_id, urgent=s.risk in words.URGENT_RISKS, event_id=event_id)

    def local(self, t: datetime) -> str:
        return hhmm(t, self.clock.tz)

    def writes_calendar(self) -> bool:
        hub = self.planner.hub
        try:
            return hub is not None and hub.is_write(CALENDAR_WRITE)
        except Exception:
            return False

    def card_open(self, plan, now: datetime) -> bool:
        p = self.planner
        if not plan.action_id or plan.plan_id in p.committed or p.card_status.get(plan.action_id):
            return False
        expires = p.expires.get(plan.action_id)
        return expires is None or now < expires

    def walk(self, origin: str | None, place: str | None) -> int:
        try:
            return self.tracker.walks.walk(origin, place)[0] if origin and place else 0
        except Exception:
            return 0

    def plan_facts(self, plan: DayPlanProposed | DayReplanProposed, now: datetime, skip: str | None = None) -> dict:
        """The rest of the plan's day from `now`: how you slept, a few beats, the first fixed event and when to
        leave for it, the nearest deadline, and whether there is a card to say yes to."""
        tz = self.clock.tz
        part = words.part_of_day(now.astimezone(tz))
        f: dict = {"moment": "morning", "greeting": "evening" if part == "night" else part, "day": words.WEEKDAYS[self.clock.day_of(now).weekday()]}
        if getattr(plan, "by", "atlas") == "person":
            f["by"] = "person"
        try:
            sleep = self.tracker.meters(now).sleep_hours
        except Exception:
            sleep = None
        if sleep is not None:
            f["sleep_hours"] = round(float(sleep), 1)
        f.update(words.shape(plan.moves, now, tz, skip=skip))
        self.leave_by(f, plan.moves, now)
        deadline = self.deadline(plan, now)
        if deadline:
            f["deadline"] = deadline
        chosen = sum(1 for m in plan.moves if not m.fixed)
        if chosen and self.card_open(plan, now):
            f["blocks"] = chosen
            f["ask"] = "calendar" if self.writes_calendar() else "plan"
        return f

    def leave_by(self, f: dict, moves: list[PlanMove], now: datetime) -> None:
        """When to leave for the first fixed event: from where the plan has you just before it."""
        first = f.get("first")
        if not first:
            return
        blocks = sorted(moves, key=lambda m: m.start)
        move = next((m for m in blocks if m.fixed and m.start > now and short_title(m.title) == first["title"] and self.local(m.start) == first["at"]), None)
        if move is None or not move.place:
            return
        # the same origin the live status uses (where you were last seen), so the brief and the status agree
        before = [m for m in blocks if m.end <= move.start and m.place and m.end > now - timedelta(hours=12)]
        seen = self.tracker.where(now)
        origin = seen[0] if seen else (before[-1].place if before else None)
        walk = self.walk(origin, move.place)
        if walk:
            first["walk_min"] = walk
            first["leave_by"] = self.local(move.start - timedelta(minutes=walk))

    def deadline(self, plan, now: datetime) -> dict | None:
        """The one deadline that matters most: the nearest with work left."""
        try:
            open_ = [d for d in self.planner.dayrun.board(now).deadlines if d.work_left > 0]
        except Exception:
            return None
        if not open_:
            return None
        d = min(open_, key=lambda d: d.due)
        due = d.due.astimezone(self.clock.tz)
        planned = sum(m.minutes for m in plan.moves if m.deadline == d.id and m.end > now)
        out = {"title": short_title(d.title), "due_day": words.say_day(due.date(), self.clock.day_of(now)), "due": self.local(due), "left_min": round(d.work_left)}
        if planned:
            out["planned_min"] = planned
        return out

    def label(self, move: PlanMove | None, title: str | None, at: datetime | None) -> str:
        """A block in a sentence: "lunch", "your study block", "the mentor check-in"."""
        if move is None:
            return words.the(short_title(title or "the next block"))
        if move.fixed:
            return words.the(short_title(move.title))
        if move.kind == "eat":
            return words.meal_name(self.local(move.start))
        if move.kind == "study":
            return f"your {self.words.activity} block"
        if move.kind == "rest":
            return "a rest" if move.minutes >= words.LONG_REST else "a break"
        if move.kind == "sleep":
            return "a nap" if move.minutes < words.NAP_MAX else "sleep"
        return words.the(short_title(move.title))

    def status_facts(self, s: DayStatusChanged) -> dict:
        plan = self.planner.current.get(s.day)
        moves = {m.id: m for m in plan.moves} if plan else {}
        f: dict = {"moment": "now", "risk": s.risk}
        if s.block_id:
            f["block"] = self.label(moves.get(s.block_id), s.block_title, s.now)
            if s.block_end:
                f["until"] = self.local(s.block_end)
        if s.should_be_name:
            f["should_be"] = short_place(s.should_be_name)
        if s.seen_at_name:
            f["seen_at"] = short_place(s.seen_at_name)
        current = moves.get(s.block_id) if s.block_id else None
        resting = current is not None and not current.fixed and current.kind in ("rest", "sleep")
        if abs(s.drift_minutes) >= 10 and s.block_id and not resting:  # nobody is behind on a rest
            f["drift_min"] = int(s.drift_minutes / abs(s.drift_minutes)) * (abs(s.drift_minutes) // 5 * 5)
        e = s.next_event
        if e:
            nxt = {"title": short_title(e.title), "at": self.local(e.start)}
            if e.place_name:
                nxt["place"] = short_place(e.place_name)
            if e.walk_minutes and e.leave_by:
                nxt["walk_min"] = e.walk_minutes
                nxt["leave_by"] = self.local(e.leave_by)
                nxt["leave_in_min"] = max(0, int((e.leave_by - s.now).total_seconds() // 300 * 5))
            f["next"] = nxt
        b = s.next_block
        if b and not s.block_id:
            move = moves.get(b.id)
            if move is not None and not move.fixed and move.kind == "rest" and plan:  # free until the next thing that is not a rest
                move = next((m for m in sorted(plan.moves, key=lambda m: m.start) if m.start > b.start and (m.fixed or m.kind != "rest")), move)
            if move is not None:
                f["then"] = {"title": self.label(move, move.title, move.start), "at": self.local(move.start)}
            else:
                f["then"] = {"title": self.label(None, b.title, b.start), "at": self.local(b.start)}
        return f

    def replan_facts(self, e: DayReplanProposed) -> dict:
        current = self.planner.current.get(e.day)
        seen = self.tracker.where(e.ts)
        seen_name = None
        if seen:
            try:
                seen_name = short_place(self.tracker.walks.name(seen[0]))
            except Exception:
                seen_name = None
        f = self.plan_facts(e, e.ts, skip=e.for_event)
        f = {k: v for k, v in f.items() if k not in ("greeting", "sleep_hours", "deadline", "day", "by")}
        f["moment"] = "replan"
        if e.asked:
            f["asked"] = True
        if seen_name:
            f["seen_at"] = seen_name
        subject = next((m for m in current.moves if m.id == e.for_event), None) if current and e.for_event else None
        if subject is not None and not e.asked:
            ev = {"title": short_title(subject.title), "at": self.local(subject.start)}
            if subject.place_name:
                ev["place"] = short_place(subject.place_name)
            walk = self.walk(seen[0] if seen else None, subject.place)
            if walk:
                ev["walk_min"] = walk
            if e.ts >= subject.start:
                ev["started"] = True
            if any(m.id == e.for_event for m in e.moves):
                ev["kept"] = True
            f["for"] = ev
        return f

    def coach_facts(self, e: DayCoachNoted) -> dict:
        s = e.session
        f: dict = {"moment": "coach", "title": short_title(s.title), "planned_min": s.planned_minutes, "desk_min": s.desk_minutes or 0,
                   "breaks": len(s.breaks)}
        if s.breaks:
            long = max(s.breaks, key=lambda b: (b.minutes, -b.start.timestamp()))
            f["long_break"] = {"min": long.minutes, "trigger": long.trigger}
        if (s.late_start_minutes or 0) >= 5:
            f["late_start_min"] = s.late_start_minutes
        if s.longest_stretch_minutes:
            f["longest_min"] = s.longest_stretch_minutes
        for text in e.suggestions:
            size = re.search(r"(\d+)-minute blocks", text)
            if size:
                f["block_min"] = int(size.group(1))
            elif "tip" not in f and not text.startswith("Keep this shape"):
                f["tip"] = text.split(":")[0].strip().rstrip(".")
        if e.next_block:
            f["next"] = {"min": round((e.next_block.end - e.next_block.start).total_seconds() / 60), "at": self.local(e.next_block.start)}
        return f

    def review_facts(self, e: DayReviewed) -> dict:
        f: dict = {"moment": "evening", "score": e.score, "stars": e.stars, "points": e.points, "plan_stars": e.plan_stars,
                   "followed": len(e.followed), "slipped": len(e.slipped)}
        best = max((s for s in e.sessions if s.measured and (s.longest_stretch_minutes or 0) >= 15),
                   key=lambda s: s.longest_stretch_minutes or 0, default=None)
        plan = self.planner.current.get(e.day)
        if best is not None:
            f["win"] = {"kind": "stretch", "min": best.longest_stretch_minutes, "title": short_title(best.title)}
        elif e.followed:
            move = next((m for m in (plan.moves if plan else []) if m.title == e.followed[0]), None)
            title = short_title(e.followed[0])
            if move is not None and move.kind == "eat" and not move.fixed:
                title = words.meal_name(self.local(move.start)) + (f" {words.at_place(short_place(move.place_name))}" if move.place_name else "")
            f["win"] = {"kind": "kept", "title": title}
        elif e.streak >= 2:
            f["win"] = {"kind": "streak", "days": e.streak}
        if e.change_tomorrow:
            f["tomorrow"] = e.change_tomorrow
        waste = self.planner.sorting(e.day, e.ts)  # Waste Watch's clause for that day, as the review's text has it
        if waste:
            f["waste"] = waste
        return f

    def approval_facts(self, e: ApprovalRequested) -> dict | None:
        """The asks this context can put in its own words: its own plan cards, and the League's three."""
        p = self.planner
        if e.action_id in p.cards:
            day, plan_id = p.cards[e.action_id]
            plan = next((x for x in (p.pending.get(day), p.current.get(day), p.plans.get(day)) if x is not None and x.plan_id == plan_id), None)
            if plan is None:
                return None
            today = self.clock.day_of(e.ts)
            return {"moment": "approval", "kind": "calendar", "count": sum(1 for m in plan.moves if not m.fixed),
                    "day": words.say_day(day, today), "replace": isinstance(plan, DayReplanProposed) or e.reason.startswith("Replace"),
                    "writes": self.writes_calendar()}
        if m := re.match(r"Set the League stake from the week of [^:]+: (.+)$", e.reason):
            return {"moment": "approval", "kind": "league_stake", "text": m.group(1).strip().rstrip(".")}
        if m := re.match(r"Start a League dare: (.+) for (\d+) days?, inviting (.+)$", e.reason):
            who = m.group(3).strip().rstrip(".")
            f = {"moment": "approval", "kind": "league_dare", "title": m.group(1).strip(), "days": int(m.group(2))}
            if who != "nobody else yet":
                f["who"] = who
            return f
        if e.reason.startswith("Post the dare's result to the group chat"):
            return {"moment": "approval", "kind": "league_post"}
        return None


@dataclass(frozen=True)
class _Ticket:
    """What the bus's background runner is handed: enough to find the job and to name it in a log line."""

    event_id: str
    type: str = "day.narration"
