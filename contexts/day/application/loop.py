"""The Day loop wired together: one object the composition root builds, subscribes and ticks.

    day = DayLoop(dayrun, places, config_dir=s.config_dir, habits=habits, memory=memory, waste=waste, presence=presence, hub=hub)
    day.subscribe(bus)
    ... every tick: await bus.publish_all(day.tick(now))
"""

from datetime import datetime
from pathlib import Path

from contexts.day.application.planner import DayPlanner
from contexts.day.application.tracker import DayTracker
from contexts.day.domain.clock import DayClock
from contexts.play.dayrun.service import DayRun
from contexts.play.domain.places import PlaceSet
from contracts.day import DayPlanProposed, DayReplanProposed, DayReviewed
from contracts.events import Event


class DayLoop:
    def __init__(self, dayrun: DayRun, places: PlaceSet, config_dir: str | Path = "config", habits=None, memory=None,
                 waste=None, presence=None, hub=None, clock: DayClock | None = None) -> None:
        self.clock = clock or DayClock.load(Path(config_dir) / "agent.yaml")
        self.planner = DayPlanner(dayrun, self.clock, habits=habits, memory=memory, presence=presence, hub=hub)
        self.tracker = DayTracker(self.planner, dayrun, self.clock, places, habits=habits, waste=waste, presence=presence)
        self.memory = memory

    def subscribe(self, bus) -> None:
        p, t = self.planner, self.tracker
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
        for prefix in ("connector.item.synced", "circle.place.shared", "day.plan.proposed", "day.plan.committed"):
            bus.subscribe(prefix, t.on_change)

    def tick(self, now: datetime) -> list[Event]:
        return self.planner.tick(now) + self.tracker.tick(now)


def day_phone(ntfy, topic: str):
    """A bus handler that pushes the plan, a re-plan and the review to the phones (the card itself goes out as
    every approval does, through PhoneNotifier.on_approval)."""

    async def push(e: Event) -> None:
        if isinstance(e, DayPlanProposed):
            await ntfy.publish(topic, e.text, title="Your day", tags=["sunrise"])
        elif isinstance(e, DayReplanProposed):
            await ntfy.publish(topic, f"{e.why} New plan: {e.outcome.summary}", title="Re-plan", tags=["arrows_counterclockwise"])
        elif isinstance(e, DayReviewed):
            await ntfy.publish(topic, f"{e.text} {e.change_tomorrow}", title="Your day, reviewed", tags=["moon"])

    return push
