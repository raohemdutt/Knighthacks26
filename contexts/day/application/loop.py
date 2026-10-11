"""The Day loop wired together: one object the composition root builds, subscribes and ticks.

    day = DayLoop(dayrun, places, config_dir=s.config_dir, habits=habits, memory=memory, waste=waste, presence=presence, hub=hub)
    day.subscribe(bus)
    ... every tick: await bus.publish_all(day.tick(now))
"""

from datetime import datetime
from pathlib import Path

from contexts.day.application.arbiter import Arbiter
from contexts.day.application.checkin import CheckIn
from contexts.day.application.coach import DayCoach
from contexts.day.application.narrator import NarrationConfig, Narrator
from contexts.day.application.optional import OptionalEvents
from contexts.day.application.planner import DayPlanner
from contexts.day.application.reminders import Reminders
from contexts.day.application.tracker import DayTracker
from contexts.day.domain import arbiter as arbiter_rule
from contexts.day.domain import narration as words
from contexts.day.domain.arbiter import Rules
from contexts.day.domain.clock import DayClock
from contexts.play.dayrun.service import DayRun
from contexts.play.domain.places import PlaceSet
from contracts.day import DayCheckinRaised, DayCoachNoted, DayNarrated, DayPlanProposed, DayReplanProposed, DayReviewed
from contracts.events import Event


class DayLoop:
    def __init__(self, dayrun: DayRun, places: PlaceSet, config_dir: str | Path = "config", habits=None, memory=None,
                 waste=None, presence=None, hub=None, clock: DayClock | None = None, runtime=None, narration: dict | None = None,
                 description: str = "") -> None:
        self.config_dir = Path(config_dir)
        self.clock = clock or DayClock.load(self.config_dir / "agent.yaml")
        self.planner = DayPlanner(dayrun, self.clock, habits=habits, memory=memory, presence=presence, hub=hub)
        self.tracker = DayTracker(self.planner, dayrun, self.clock, places, habits=habits, waste=waste, presence=presence)
        self.coach = DayCoach(self.planner, dayrun, self.clock, memory=memory, habits=habits, desk_zone=self.tracker.desk_zone,
                              door_zone=self.tracker.door_zone, desk_camera=self.tracker.desk_camera)
        self.planner.coach = self.coach
        self.memory = memory
        # The narrator (D3): `narration` is agent.yaml's `day:` block; `runtime` is the model it may ask (never the fake one).
        self.narrator = Narrator(self.planner, self.tracker, self.coach, self.clock, runtime=runtime,
                                 config=NarrationConfig.of(narration, runtime), description=description)
        # The check-in (D4): speaks up when enough independent signals agree. agent.yaml `day: {checkin: false}` turns it off.
        self.checkin = CheckIn(self.planner, self.tracker, self.coach, self.clock, memory=memory,
                               enabled=bool((narration or {}).get("checkin", True)))
        self.checkin.narrator = self.narrator
        # The arbiter: one gate for everything said unasked (order, a global gap, merging, suppression, one speaker).
        # agent.yaml `day: {speak_gap_s: 90}`.
        self.arbiter = Arbiter(self.narrator, self.planner, self.clock, rules=Rules.of(narration))
        self.narrator.arbiter = self.arbiter
        # Block-start reminders: said once when a planned block starts. agent.yaml `day: {block_reminders: false}` turns them off.
        self.reminders = Reminders(self.planner, self.tracker, self.narrator, self.clock,
                                   enabled=bool((narration or {}).get("block_reminders", True)))
        # Optional events ("Near you"): a yes raises a re-plan behind the usual card; approved, the event is a block of the day.
        # Atlas brings one up once, 90 to 30 minutes before it, through the arbiter. agent.yaml `day: {suggest_optional: false}`.
        self.optional = OptionalEvents(self.planner, self.tracker, self.clock, memory=memory, config=narration)
        self.optional.narrator = self.narrator

    def subscribe(self, bus) -> None:
        p, t, c = self.planner, self.tracker, self.coach
        # The coach's state first, so the planner's review (same event, later handler) sees the block that just ended.
        bus.subscribe("perception.zone", c.on_zone, replay=True)
        bus.subscribe("perception.scene", c.on_scene, replay=True)
        bus.subscribe("day.coach.noted", c.on_noted, replay=True)
        bus.subscribe("perception.expression", c.on_expression, replay=True)  # optional input: how an enrolled member looked
        bus.subscribe("perception.identity", c.on_identity, replay=True)  # a forgotten member's looks are dropped
        # State that rebuilds from the log.
        bus.subscribe("day.plan.proposed", p.on_plan_proposed, replay=True)
        bus.subscribe("day.replan.proposed", p.on_replan_proposed, replay=True)
        bus.subscribe("day.plan.committed", p.on_plan_committed, replay=True)
        bus.subscribe("day.reviewed", p.on_reviewed, replay=True)
        bus.subscribe("workspace.approval.requested", p.on_approval_requested, replay=True)
        bus.subscribe("workspace.action.completed", p.on_action_completed, replay=True)
        bus.subscribe("perception.zone", t.on_zone_state, replay=True)
        bus.subscribe("day.status.changed", t.on_status_changed, replay=True)
        # Live only: deciding, proposing cards, publishing statuses.
        bus.subscribe("experience.approval.decided", p.on_decided)
        bus.subscribe("habits.day.closed", p.on_day_closed)
        for prefix in ("perception.zone", "connector.item.synced", "habits.signal"):
            bus.subscribe(prefix, p.on_event)  # the morning plan at the first sign of the day after wake
        bus.subscribe("perception.zone", t.on_zone_live)
        for prefix in ("perception.zone", "connector.item.synced", "habits.signal"):
            bus.subscribe(prefix, c.on_event)  # a block of the main activity ended: one sentence for the phone
        for prefix in ("connector.item.synced", "circle.place.shared", "day.plan.proposed", "day.plan.committed"):
            bus.subscribe(prefix, t.on_change)

        # Optional events: answers, which plan carries which, and which are accepted, all from the log (replay=True inside).
        # After the planner: an approval counts only for the plan the day is now measured against.
        self.optional.subscribe(bus)

        # The check-in: its signals rebuild from the log (replay=True inside); looking at the rule and raising one is
        # live only, so a restart never raises an old check-in again. After the coach: it reads the coach's desk presence.
        self.checkin.subscribe(bus)

        # Last, so the planner, the tracker and the coach have already taken the event in: the day in words.
        self.narrator.subscribe(bus)
        self.arbiter.subscribe(bus)  # what was said and what is open, from the log; the held lines are live only
        self.reminders.subscribe(bus)  # which starts were said, from the log; saying one is the tick's (live)

    def tick(self, now: datetime) -> list[Event]:
        out = self.planner.tick(now) + self.tracker.tick(now) + self.coach.tick(now) + self.checkin.tick(now)
        out += self.reminders.tick(now)  # queues its line in the arbiter
        out += self.optional.tick(now)  # ... and so does the suggestion of an optional event coming up
        return out + self.arbiter.tick(now)  # last: the lines queued by this tick and the held ones

    def in_words(self, now: datetime) -> dict[str, str]:
        """Today in a few plain sentences, for whoever is talking with the person (the agent's conversation):
        `now` from the live state (and `now_spoken`, its wording for the ear), `plan`, `coach` and `review` from
        the latest narration of each (each with its `_spoken` wording too: the conversation is heard, and the
        screen's text has brackets and exact times a voice should not read out). Only today's;
        empty when there is no plan yet. Reads state, publishes nothing, never calls a model."""
        n, day, out = self.narrator, self.clock.day_of(now), {}
        try:
            job = n.job_now("now", now)
        except Exception:
            job = None
        if job is not None and job.day == day:
            out["now"] = job.draft.display
            out["now_spoken"] = job.draft.spoken  # the same moment for the ear: no brackets, times as they are said
        for key, moment in (("plan", "morning"), ("coach", "coach"), ("review", "evening")):
            said = n.latest.get(moment)
            if said is not None and said.day == day:
                out[key] = said.display
                out[f"{key}_spoken"] = said.spoken
        return out

    def blocks(self, now: datetime) -> list[dict] | None:
        """Today's plan from `now`, for whoever answers a question about the day (the agent's rundown and the facts
        it gives its model): the blocks the plan chose, in the narrator's words and in time order, as
        {"kind": "study", "title": "a study block", "start", "end"}, then at most one
        {"kind": "leave", "title": <the event's title>, "start": <when to leave>} for the next fixed event that needs
        a walk. Fixed events are the calendar's own and are left out; short breaks inside the main activity and the
        night's sleep are not said. None when there is no plan for today. Reads state, publishes nothing, no model."""
        plan = self.planner.current.get(self.clock.day_of(now))
        if plan is None:
            return None
        w, tz = self.narrator.words, self.clock.tz
        ahead = sorted((m for m in plan.moves if m.start >= now), key=lambda m: m.start)
        night = next((m for m in ahead if m.kind == "sleep" and m.minutes >= words.NAP_MAX), None)
        out: list[dict] = []
        for m in ahead:
            if m.fixed or m is night or (night is not None and m.start >= night.start):
                continue
            kind = "nap" if m.kind == "sleep" else "meal" if m.kind == "eat" else m.kind
            if kind == "rest" and m.minutes <= words.BREAK_MAX:
                continue  # a breather, not something a person lists
            if kind == "study" and out and out[-1]["kind"] == "study" and (m.start - out[-1]["end"]).total_seconds() <= words.BREAK_MAX * 60:
                out[-1].update(end=m.end, min=out[-1]["min"] + m.minutes)  # one sitting, said once
                out[-1]["title"] = words.beat_words({"kind": "study", "count": 1, "min": out[-1]["min"]}, w)
                continue
            beat = {"kind": kind, "count": 1, "min": m.minutes, "at": words.hhmm(m.start, tz), "title": words.short_title(m.title)}
            out.append({"kind": kind, "title": words.beat_words(beat, w), "start": m.start, "end": m.end, "min": m.minutes})
        try:
            status = self.tracker.compute(now) or self.tracker.status
        except Exception:
            status = self.tracker.status
        nxt = status.next_event if status is not None else None
        if nxt is not None and nxt.leave_by is not None and nxt.walk_minutes > 0 and nxt.leave_by >= now and nxt.start > now:
            out.append({"kind": "leave", "title": nxt.title, "start": nxt.leave_by})
        return out


MOMENT_PUSH = {"morning": ("Your day", "sunrise"), "replan": ("Re-plan", "arrows_counterclockwise"),
               "coach": ("Coach", "dart"), "evening": ("Your day, reviewed", "moon"), "start": ("Next up", "bell")}


def checkin_push(e: DayCheckinRaised) -> str:
    """What a check-in's push may say: the plain nudges, and where to read the rest. Nothing about the person."""
    nudges = [s.text.rstrip(".") for s in e.suggestions if s.kind != "move_block"]
    todo = f" {nudges[0]}{''.join(', ' + n[:1].lower() + n[1:] for n in nudges[1:])}." if nudges else ""
    return f"Atlas has a check-in for you.{todo} Open Now to see why."


def suggest_push(e: DayNarrated) -> str:
    """What the push of an optional-event suggestion says: the event, its time and place and how it fits (the
    narration's screen line, which names nobody), and where to answer it. Never who is going."""
    return f"Atlas has a suggestion near you: {e.display.removesuffix('Add it?').strip()} Open Atlas to add it."


def box_voice(say):
    """A live bus handler that hands the lines the arbiter gave to the box's speaker (`spoken_by == "box"`) to
    `say(text)` (the voice service's POST /say); the phone stays silent for those. A line said inside another
    (`merged_into`) is never said again. Off unless agent.yaml `day: {speaker: box}` (contexts/gateway/bootstrap.py)."""

    async def on(e: Event) -> None:
        if isinstance(e, DayNarrated) and e.spoken_by == "box" and not e.merged_into and e.spoken:
            await say(e.spoken)

    return on


def day_phone(ntfy, topic: str, narrated: bool = False):
    """A bus handler that pushes the plan, a re-plan, the review and the coach's sentence after a block to the
    phones (the card itself goes out as every approval does, through PhoneNotifier.on_approval).

    With the narrator on (`narrated`), the push is the narration's paragraph (`day.narrated`), the same words
    the screen shows and the voice says, plus an urgent "now" (leave now, late); the raw events are not pushed
    as well. With it off, the events' own texts go out as before."""

    pushed: set[str] = set()  # one push per moment: day + moment + ref (contexts/day/domain/arbiter.py `key`)

    async def push(e: Event) -> None:
        if isinstance(e, DayNarrated):
            if e.merged_into:  # said inside another narration, which was pushed
                return
            k = arbiter_rule.key(e.day, e.moment, e.ref)
            if k in pushed:
                return
            if narrated and e.moment in MOMENT_PUSH:
                pushed.add(k)
                title, tag = MOMENT_PUSH[e.moment]
                await ntfy.publish(topic, e.display, title=title, tags=[tag])
            elif narrated and e.moment == "now" and e.urgent:
                pushed.add(k)
                await ntfy.publish(topic, e.display, title="Time to move", tags=["walking"])
            elif narrated and e.moment == "suggest":
                pushed.add(k)
                await ntfy.publish(topic, suggest_push(e), title="Near you", tags=["calendar"])
        elif isinstance(e, DayCheckinRaised):
            # Box-only: the push goes through a notification service, so it names what to do and never why. The
            # signals (a night, a heart rate, a face) are read on the phone, from the box.
            await ntfy.publish(topic, checkin_push(e), title="Check-in", tags=["wave"])
        elif narrated:
            return
        elif isinstance(e, DayPlanProposed):
            await ntfy.publish(topic, e.text, title="Your day", tags=["sunrise"])
        elif isinstance(e, DayReplanProposed):
            await ntfy.publish(topic, f"{e.why} New plan: {e.outcome.summary}", title="Re-plan", tags=["arrows_counterclockwise"])
        elif isinstance(e, DayReviewed):
            coach = f" Coach: {' '.join(e.suggestions)}" if e.suggestions else ""
            await ntfy.publish(topic, f"{e.text} {e.change_tomorrow}{coach}", title="Your day, reviewed", tags=["moon"])
        elif isinstance(e, DayCoachNoted):
            await ntfy.publish(topic, e.text, title="Coach", tags=["dart"])

    return push
