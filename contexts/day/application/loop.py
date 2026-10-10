"""The Day loop wired together: one object the composition root builds, subscribes and ticks.

    day = DayLoop(dayrun, places, config_dir=s.config_dir, habits=habits, memory=memory, waste=waste, presence=presence, hub=hub)
    day.subscribe(bus)
    ... every tick: await bus.publish_all(day.tick(now))
"""

from datetime import datetime
from pathlib import Path

from contexts.day.application.coach import DayCoach
from contexts.day.application.narrator import NarrationConfig, Narrator
from contexts.day.application.planner import DayPlanner
from contexts.day.application.tracker import DayTracker
from contexts.day.domain.clock import DayClock
from contexts.play.dayrun.service import DayRun
from contexts.play.domain.places import PlaceSet
from contracts.day import DayCoachNoted, DayNarrated, DayPlanProposed, DayReplanProposed, DayReviewed
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

    def subscribe(self, bus) -> None:
        p, t, c = self.planner, self.tracker, self.coach
        # The coach's state first, so the planner's review (same event, later handler) sees the block that just ended.
        bus.subscribe("perception.zone", c.on_zone, replay=True)
        bus.subscribe("perception.scene", c.on_scene, replay=True)
        bus.subscribe("day.coach.noted", c.on_noted, replay=True)
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

        # Last, so the planner, the tracker and the coach have already taken the event in: the day in words.
        self.narrator.subscribe(bus)

    def tick(self, now: datetime) -> list[Event]:
        return self.planner.tick(now) + self.tracker.tick(now) + self.coach.tick(now)


MOMENT_PUSH = {"morning": ("Your day", "sunrise"), "replan": ("Re-plan", "arrows_counterclockwise"),
               "coach": ("Coach", "dart"), "evening": ("Your day, reviewed", "moon")}


def day_phone(ntfy, topic: str, narrated: bool = False):
    """A bus handler that pushes the plan, a re-plan, the review and the coach's sentence after a block to the
    phones (the card itself goes out as every approval does, through PhoneNotifier.on_approval).

    With the narrator on (`narrated`), the push is the narration's paragraph (`day.narrated`), the same words
    the screen shows and the voice says, plus an urgent "now" (leave now, late); the raw events are not pushed
    as well. With it off, the events' own texts go out as before."""

    async def push(e: Event) -> None:
        if isinstance(e, DayNarrated):
            if narrated and e.moment in MOMENT_PUSH:
                title, tag = MOMENT_PUSH[e.moment]
                await ntfy.publish(topic, e.display, title=title, tags=[tag])
            elif narrated and e.moment == "now" and e.urgent:
                await ntfy.publish(topic, e.display, title="Time to move", tags=["walking"])
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
