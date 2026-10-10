"""Always-on agent loop (A7).

Every tick (10 minutes), the agent looks at what happened since the last
tick and flags what is unusual. Today that means activity during the
household's sleep hours ("someone at the desk at 3:12 AM"), raised as an
alert with the episode as evidence. Proposing actions goes through Workspace
like everything else.
"""

from datetime import datetime, timedelta

from contexts.agent.application.memory_port import MemoryReader
from contexts.agent.application.qa import SleepSchedule
from contracts.agent import AlertRaised
from contracts.types import Citation

TICK = timedelta(minutes=10)
ODD_HOURS_RULE = "odd-hours-activity"


class AgentLoop:
    def __init__(self, memory: MemoryReader, sleep: SleepSchedule, watch_labels: tuple[str, ...] = ("person",)) -> None:
        self.memory = memory
        self.sleep = sleep
        self.watch_labels = watch_labels
        self._since: datetime | None = None
        self._flagged: set[str] = set()

    def due(self, now: datetime) -> bool:
        return self._since is None or now - self._since >= TICK

    def tick(self, now: datetime) -> list[AlertRaised]:
        since = self._since or now - TICK
        self._since = now
        alerts = []
        night_start, night_end = self.sleep.last_night(now)
        for ep in self.memory.episodes(since, now):
            if ep.id in self._flagged or not night_start <= ep.start < night_end:
                continue
            if not any(label in ep.labels for label in self.watch_labels):
                continue
            self._flagged.add(ep.id)
            at = ep.start.astimezone(self.sleep.tz).strftime("%I:%M %p").lstrip("0")
            alerts.append(
                AlertRaised(
                    ts=now,
                    alert_id=f"odd-{ep.id}",
                    rule_id=ODD_HOURS_RULE,
                    message=f"Activity on {ep.camera_id} at {at}, during sleep hours",
                    evidence=[Citation(episode_id=ep.id, frame_ref=(ep.keyframe_refs or [None])[0], ts=ep.start)],
                )
            )
        return alerts
