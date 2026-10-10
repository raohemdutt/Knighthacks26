"""The coach feeds the learning loop: "Was that right?" about a break becomes calibration, then training data.

    feedback = CoachFeedback(day.coach, memory=memory, learning=learning_board)
    feedback.subscribe(bus)
    app.include_router(coach_feedback_router(day, feedback, publish=bus.publish, guard=authorized))

The coach guesses why a break began (chat, phone, door, food, or no visible reason) from what the box
recorded in the two minutes before. It is a guess. Here the person answers, and three things follow:

1. The answer is the event `day.coach.corrected`. The review of that session reflects it: the break's reason
   becomes the person's, and the suggestions are drawn again from the corrected reasons.
2. Per-trigger precision is counted from the answers: "The coach is right about phone breaks 4 of 5 times."
3. After enough answers about one trigger (`min_corrections`, default 5, of which at least `min_negatives`,
   default 2, said "no"), the coach proposes a TEACH TASK through the learning context's own `teach()`: the
   breaks' keyframes as examples, thumbs up where the person said that was the reason, thumbs down where the
   coach said so and the person said no. `teach()` returns `learning.task.created` and the normal
   `export_training_data` approval card; this module publishes exactly those two and nothing else. It never
   approves anything: no picture leaves the box unless the person approves the card on the phone.

Which teach task came from the coach is read back from the log: its slug starts with `desk-distraction-`.
The task's stage comes from the learning board (GET /api/learning's data). When the board says the model
trained for that task was deployed, the view carries one note: "Atlas now recognises phone in hand; break
reasons should be more accurate."

State rebuilds from the log (`on_corrected`, `on_task` replay). Proposing is live only. Times are event times.
"""

from collections.abc import Awaitable, Callable
from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from contexts.day.domain.coach import AFTER, PRIORITY, hm, suggest
from contexts.learning.application.export import DEFAULT_LOOKBACK, span, teach
from contracts.day import CoachAnswer, DayCoachCorrected, SessionReview
from contracts.events import Event
from contracts.learning import TaskCreated, TaskSpec, TeachExample

TASK_PREFIX = "desk-distraction-"
YES_NO = {"type": "object", "properties": {"present": {"type": "boolean"}, "where": {"type": "string"}}, "required": ["present"]}
# What the box can be taught to SEE. A chat message is not in a picture, so there is no task for it.
TEACHABLE: dict[str, dict] = {
    "phone": {"name": "phone in hand",
              "instruction": "Is a phone in the person's hand or being looked at? Answer yes or no and say where it is."},
    "food": {"name": "food or a drink at the desk",
             "instruction": "Is the person eating or drinking, or has food or a drink just been put on the desk? Answer yes or no and say what it is."},
    "door": {"name": "someone coming in",
             "instruction": "Has someone other than the person at the desk come in or is standing at the door? Answer yes or no and say where they are."},
}
# The quick answers on the phone, in order. "Yes" is not one of them: it is `right: true`.
OPTIONS: list[tuple[str, str]] = [("phone", "Phone"), ("chat", "Chat"), ("door", "Someone came in"), ("food", "Food or drink"),
                                  ("bathroom", "Bathroom"), ("done", "I was done"), ("other", "Something else")]
SAID = {"chat": "a message in the chat", "phone": "the phone", "door": "someone coming in", "food": "food or a drink",
        "bathroom": "a bathroom break", "done": "being done for now", "other": "something else", "none": "nothing in particular"}
GUESS = {"chat": "chat", "phone": "phone", "door": "door", "food": "food or drink", "none": "no visible reason"}


def _key(day: date, block_id: str, start: datetime) -> tuple[date, str, str]:
    return day, block_id, start.astimezone(timezone.utc).isoformat()


class CoachFeedback:
    def __init__(self, coach, memory=None, learning=None, min_corrections: int = 5, min_negatives: int = 2,
                 door_camera: str = "cam-door") -> None:
        self.coach, self.memory, self.learning = coach, memory, learning
        self.min_corrections, self.min_negatives = int(min_corrections), int(min_negatives)
        self.door_camera = door_camera
        # --- state, rebuilt from the log ---
        self.corrections: dict[tuple[date, str, str], DayCoachCorrected] = {}  # one per break: the latest answer
        self.tasks: dict[str, dict] = {}  # trigger -> {"task": slug, "created": ts}: the teach tasks the coach proposed
        # --- live only ---
        self._proposing: set[str] = set()  # handed to the bus, not yet seen back
        self.proposed: dict[str, str] = {}  # trigger -> the approval card's action id, for the POST's answer

    def subscribe(self, bus) -> None:
        bus.subscribe("day.coach.corrected", self.on_corrected, replay=True)
        bus.subscribe("learning.task.created", self.on_task, replay=True)
        bus.subscribe("day.coach.corrected", self.propose)  # live only: it proposes an approval card

    # --- state handlers (replay=True) ---

    def on_corrected(self, e: DayCoachCorrected) -> None:
        key = _key(e.day, e.block_id, e.break_start)
        old = self.corrections.get(key)
        if old is None or e.ts >= old.ts:
            self.corrections[key] = e

    def on_task(self, e: TaskCreated) -> None:
        if e.task.startswith(TASK_PREFIX) and e.task[len(TASK_PREFIX):] in TEACHABLE:
            trigger = e.task[len(TASK_PREFIX):]
            self.tasks.setdefault(trigger, {"task": e.task, "created": e.ts})
            self._proposing.discard(trigger)

    # --- calibration ---

    def precision(self) -> list[dict]:
        """How often each kind of guess was right, by the person's own answers."""
        out = []
        for trigger in (*PRIORITY, "none"):
            rows = [c for c in self.corrections.values() if c.guess == trigger]
            if not rows:
                continue
            right, n = sum(1 for c in rows if c.right), len(rows)
            what = "breaks with no visible reason" if trigger == "none" else f"{GUESS[trigger]} breaks"
            out.append({"trigger": trigger, "right": right, "total": n, "precision": round(right / n, 3),
                        "text": f"The coach is right about {what} {right} of {n} time{'' if n == 1 else 's'}."})
        return out

    def pool(self, trigger: str) -> tuple[list[DayCoachCorrected], list[DayCoachCorrected]]:
        """The answers that say something about one trigger: (it was this, the coach said this and it was not)."""
        rows = sorted(self.corrections.values(), key=lambda c: (c.break_start, c.block_id))
        return [c for c in rows if c.answer == trigger], [c for c in rows if c.guess == trigger and not c.right]

    def ready(self, trigger: str) -> bool:
        ups, downs = self.pool(trigger)
        return len(ups) + len(downs) >= self.min_corrections and len(downs) >= self.min_negatives

    # --- live: the teach task ---

    def propose(self, e: DayCoachCorrected) -> list[Event]:
        """Enough answers about one trigger: one teach task, as an approval card. Never approved here."""
        out: list[Event] = []
        for trigger in dict.fromkeys((e.guess, e.answer)):
            if trigger not in TEACHABLE or trigger in self.tasks or trigger in self._proposing or not self.ready(trigger):
                continue
            made = self.teach_task(trigger, e.ts)
            if made is None:  # no picture of any of those breaks on this box: nothing to teach from
                continue
            self._proposing.add(trigger)
            self.proposed[trigger] = made[1].action_id
            out += made
        return out

    def teach_task(self, trigger: str, now: datetime) -> tuple[Event, Event] | None:
        """What `teach()` gives for this trigger: the task and its export card, with the corrected breaks as examples."""
        ups, downs = self.pool(trigger)
        examples: dict[str, TeachExample] = {}
        oldest = now
        for verdict, rows in (("up", ups), ("down", downs)):
            for c in rows:
                ref = self.frame_for(c, trigger)
                if ref and ref not in examples:
                    examples[ref] = TeachExample(frame_ref=ref, verdict=verdict)
                    oldest = min(oldest, c.break_start)
        if not examples:
            return None
        cfg = TEACHABLE[trigger]
        spec = TaskSpec(task=TASK_PREFIX + trigger, instruction=cfg["instruction"], output_schema=YES_NO,
                        camera=self.door_camera if trigger == "door" else self.coach.desk_camera, examples=list(examples.values()))
        lookback = max(DEFAULT_LOOKBACK, now - oldest + timedelta(hours=1))  # the export window must reach the oldest break
        if lookback > DEFAULT_LOOKBACK:  # in whole days, so the card reads "the last 3 days"
            lookback = timedelta(days=-(-lookback.total_seconds() // 86400))
        created, proposal = teach(spec, now, lookback=lookback)
        n, yes = len(examples), sum(1 for x in examples.values() if x.verdict == "up")
        where = "door" if trigger == "door" else self.coach.words.workplace
        reason = (f"Teach Atlas to spot '{cfg['name']}' from {n} of your own blurred example{'' if n == 1 else 's'}? "
                  f"({yes} where you said that was it, {n - yes} where you said it was not.) Approving sends those keyframes, and "
                  f"look-alike {where} frames with their captions from the last {span(lookback)}, to the lab.")
        return created, proposal.model_copy(update={"reason": reason})

    def moment(self, c: DayCoachCorrected, trigger: str) -> datetime:
        """When to look: the evidence of that kind just before the break when the coach recorded one, else the exit."""
        try:
            for s in self.coach.sessions(c.day, c.ts):
                for b in (s.breaks if s.block_id == c.block_id else []):
                    if b.start == c.break_start:
                        return next((ev.at for ev in reversed(b.evidence) if ev.kind == trigger), c.break_start)
        except Exception:
            pass
        return c.break_start

    def frame_for(self, c: DayCoachCorrected, trigger: str | None = None) -> str | None:
        """The stored keyframe closest to that break, from the Memory episode that holds the moment."""
        if self.memory is None:
            return None
        trigger = trigger or (c.guess if c.guess != "none" else c.answer)
        at = self.moment(c, trigger)
        camera = self.door_camera if trigger == "door" else self.coach.desk_camera
        try:
            eps = [ep for ep in self.memory.episodes(at - timedelta(hours=1), at + timedelta(seconds=1), camera=camera)
                   if ep.start <= at and (ep.end is None or ep.end >= at - timedelta(minutes=1)) and ep.keyframe_refs]
        except Exception:
            return None
        if not eps:
            return None
        refs = eps[-1].keyframe_refs
        try:
            return min(refs, key=lambda r: abs(int(r.rsplit(":", 1)[1]) / 1000 - at.timestamp()))
        except ValueError:
            return refs[-1]

    # --- the review, corrected ---

    def corrected(self, day: date, session: SessionReview) -> SessionReview:
        """The session with each answered break carrying the person's reason instead of the coach's guess."""
        breaks, said = [], []
        for b in session.breaks:
            c = self.corrections.get(_key(day, session.block_id, b.start))
            if c is None or c.right:
                breaks.append(b)
                continue
            breaks.append(b.model_copy(update={"trigger": c.answer if c.answer in PRIORITY else "none"}))
            guess = f", not {SAID[c.guess]}" if c.guess != "none" else ""
            said.append(f"You said the {hm(b.start, self.coach.clock.tz)} break was {SAID[c.answer]}{guess}.")
        if not said:
            return session
        return session.model_copy(update={"breaks": breaks, "text": " ".join([session.text, *said])})

    def session_view(self, day: date, session: SessionReview) -> dict:
        fixed = self.corrected(day, session)
        body = fixed.model_dump(mode="json")
        body["day"] = day.isoformat()
        for i, (b, raw) in enumerate(zip(body["breaks"], session.breaks, strict=True)):
            c = self.corrections.get(_key(day, session.block_id, raw.start))
            b.update(index=i, guess=raw.trigger, guess_text=AFTER[raw.trigger], frame_ref=None,
                     correction=None if c is None else {"right": c.right, "answer": c.answer, "answer_text": SAID[c.answer],
                                                        "note": c.note, "ts": c.ts.isoformat()})
            if self.memory is not None:
                probe = DayCoachCorrected(ts=session.end, day=day, block_id=session.block_id, break_index=i, break_start=raw.start,
                                          guess=raw.trigger, right=True, answer=raw.trigger)
                b["frame_ref"] = self.frame_for(probe, raw.trigger if raw.trigger in TEACHABLE else "phone")
        return body

    def learning_view(self) -> list[dict]:
        """One row per teach task the coach proposed, with its stage from the learning board."""
        board = {}
        if self.learning is not None:
            try:
                board = {t["task"]: t for t in self.learning.snapshot()["tasks"]}
            except Exception:
                board = {}
        out = []
        for trigger, mine in self.tasks.items():
            t, name = board.get(mine["task"], {}), TEACHABLE[trigger]["name"]
            stage, learned = t.get("stage", "waiting for approval"), bool(t.get("deployed"))
            if learned:
                text = f"Atlas now recognises {name}; break reasons should be more accurate."
            elif stage == "waiting for approval":
                text = f"Atlas can learn to spot {name}: the card is waiting for your OK on the Approvals screen. Nothing has left the box."
            elif stage in ("denied", "expired"):
                text = f"You did not approve sending examples of {name}, so Atlas is not learning it."
            elif stage == "failed":
                text = f"Learning to spot {name} failed in the lab."
            else:
                text = f"Atlas is learning this: {name} ({stage})."
            updated = t.get("updated")
            out.append({"trigger": trigger, "task": mine["task"], "name": name, "stage": stage, "progress": t.get("progress", 0.0),
                        "learned": learned, "model_version": t.get("model_version"), "examples": t.get("examples"),
                        "proposed_at": mine["created"].isoformat(),
                        "updated": updated.isoformat() if isinstance(updated, datetime) else updated, "text": text})
        return out

    def progress(self) -> list[dict]:
        """How far each teachable trigger is from a teach task."""
        out = []
        for trigger, cfg in TEACHABLE.items():
            ups, downs = self.pool(trigger)
            n = len(ups) + len(downs)
            if not n and trigger not in self.tasks:
                continue
            out.append({"trigger": trigger, "name": cfg["name"], "answers": n, "negatives": len(downs),
                        "needed": self.min_corrections, "negatives_needed": self.min_negatives,
                        "ready": self.ready(trigger), "proposed": trigger in self.tasks})
        return out

    def view(self, now: datetime, days: int = 3) -> dict:
        """GET /api/day/coach/feedback: recent sessions with their breaks, the answers, precision and what Atlas is learning."""
        c = self.coach
        sessions, fixed = [], []
        for day in sorted(c.planner.current, reverse=True)[:max(1, days)]:
            try:
                todays = c.sessions(day, now)
            except Exception:
                todays = []
            for s in sorted(todays, key=lambda s: s.start, reverse=True):
                sessions.append(self.session_view(day, s))
                fixed.append(self.corrected(day, s))
        learning = self.learning_view()
        return {
            "now": now.isoformat(),
            "words": {"activity": c.words.activity, "workplace": c.words.workplace},
            "options": [{"id": k, "label": label} for k, label in OPTIONS],
            "sessions": sessions,
            "suggestions": suggest(fixed, c.words).suggestions if fixed else [],
            "precision": self.precision(),
            "answers": len(self.corrections),
            "progress": self.progress(),
            "threshold": {"min_corrections": self.min_corrections, "min_negatives": self.min_negatives},
            "learning": learning,
            "notes": [t["text"] for t in learning if t["learned"]],  # once per task the box has learned
        }


class CorrectionBody(BaseModel):
    day: date
    block_id: str = Field(min_length=1, max_length=200)
    break_index: int = Field(ge=0, le=500)
    right: bool
    answer: CoachAnswer | None = None  # needed when `right` is false
    note: str | None = Field(default=None, max_length=280)


def coach_feedback_router(day, feedback: CoachFeedback, publish: Callable[[object], Awaitable] | None = None,
                          guard: Callable | None = None,
                          clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> APIRouter:
    """GET /api/day/coach/feedback?days=   and   POST /api/day/coach/corrections {day, block_id, break_index, right, answer?, note?}.

    The POST publishes `day.coach.corrected`. If that answer completes the threshold for a trigger, the same
    publish carries on to `learning.task.created` and the export approval card; `proposed` in the answer is
    then that card's action id."""
    router = APIRouter(prefix="/api/day/coach", tags=["day"])
    writes = [Depends(guard)] if guard else []

    def event_now() -> datetime:
        now = clock()
        latest = max((e.ts for e in (day.tracker.status, *day.planner.current.values()) if e), default=None)
        return now if latest is None or now >= latest else latest

    @router.get("/feedback")
    async def get_feedback(days: int = Query(3, ge=1, le=14)) -> dict:
        return feedback.view(event_now(), days)

    @router.post("/corrections", dependencies=writes)
    async def correct(body: CorrectionBody) -> dict:
        now = event_now()
        session = next((s for s in day.coach.sessions(body.day, now) if s.block_id == body.block_id), None)
        if session is None or body.break_index >= len(session.breaks):
            raise HTTPException(404, "no such break in that day's sessions")
        b = session.breaks[body.break_index]
        answer = b.trigger if body.right else body.answer
        if answer is None:
            raise HTTPException(422, "say what it was: `answer` is needed when `right` is false")
        plan = day.planner.current.get(body.day)
        note = (body.note or "").strip() or None
        e = DayCoachCorrected(ts=clock(), day=body.day, block_id=body.block_id, break_index=body.break_index, break_start=b.start,
                              guess=b.trigger, right=body.right or answer == b.trigger, answer=answer, note=note,
                              plan_id=plan.plan_id if plan else None)
        before = dict(feedback.proposed)
        if publish is None:
            feedback.on_corrected(e)
        else:
            await publish(e)
        proposed = {t: a for t, a in feedback.proposed.items() if before.get(t) != a}  # the card this answer raised, if any
        return {"correction": e.model_dump(mode="json"), "session": feedback.session_view(body.day, session),
                "precision": feedback.precision(), "progress": feedback.progress(), "learning": feedback.learning_view(),
                "proposed": proposed or None}

    return router
