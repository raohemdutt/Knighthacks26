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
from contexts.day.application.planner import DayPlanner
from contexts.day.application.reminders import Reminders
from contexts.day.application.tracker import DayTracker
from contexts.day.domain import arbiter as arbiter_rule
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
        return out + self.arbiter.tick(now)  # last: the lines queued by this tick and the held ones

    def in_words(self, now: datetime) -> dict[str, str]:
        """Today in a few plain sentences, for whoever is talking with the person (the agent's conversation):
        `now` from the live state (and `now_spoken`, its wording for the ear), `plan`, `coach` and `review` from
        the latest narration of each. Only today's;
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
        return out


MOMENT_PUSH = {"morning": ("Your day", "sunrise"), "replan": ("Re-plan", "arrows_counterclockwise"),
               "coach": ("Coach", "dart"), "evening": ("Your day, reviewed", "moon"), "start": ("Next up", "bell")}


def checkin_push(e: DayCheckinRaised) -> str:
    """What a check-in's push may say: the plain nudges, and where to read the rest. Nothing about the person."""
    nudges = [s.text.rstrip(".") for s in e.suggestions if s.kind != "move_block"]
    todo = f" {nudges[0]}{''.join(', ' + n[:1].lower() + n[1:] for n in nudges[1:])}." if nudges else ""
    return f"Atlas has a check-in for you.{todo} Open Now to see why."


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
