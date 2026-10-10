"""Always-on agent loop (A7).

Runs 24/7 from the app's timer:
- every 10 minutes: flag unusual activity, i.e. activity during the
  household's sleep hours, as an alert with the episode as evidence;
- every hour: write a short note on the past hour (agent.note.written) so
  memory search can find "what happened around 3 PM" without reading every caption;
- on a package alert when nobody has been home for a while: propose messaging
  the household, through an approval card like every outside action.
"""

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta
from uuid import uuid4

from contexts.agent.application.memory_port import MemoryReader
from contexts.agent.application.qa import SleepSchedule
from contracts.agent import ActionProposed, AlertRaised, NoteWritten
from contracts.types import Citation

TICK = timedelta(minutes=10)
ODD_HOURS_RULE = "odd-hours-activity"
AWAY_FOR = timedelta(minutes=30)
MAX_NOTE_CAPTIONS = 12


ODD_HOURS_DESCRIPTION = "Activity during sleep hours (agent.yaml sleep window): a person on any camera"


def place(camera_id: str) -> str:
    """ "cam-door" -> "door": how a person names a camera."""
    return camera_id.removeprefix("cam-").replace("-", " ")


class AgentLoop:
    def __init__(self, memory: MemoryReader, sleep: SleepSchedule, watch_labels: tuple[str, ...] = ("person",),
                 summarize: Callable[[list[str]], str] | None = None, notify: str | None = None,
                 enabled: Callable[[str], bool] = lambda rule_id: True) -> None:
        self.memory = memory
        self.sleep = sleep
        self.watch_labels = watch_labels
        self.summarize = summarize  # runtime.summarize; None writes notes from captions without a model
        self.notify = notify  # person to message about packages when nobody is home (config: agent.yaml)
        self._since: datetime | None = None
        self._flagged: set[str] = set()
        self._noted_hour: datetime | None = None
        self.enabled = enabled  # the Rules screen's switch (RuleEvaluator.enabled)

    def due(self, now: datetime) -> bool:
        return self._since is None or now - self._since >= TICK

    async def tick(self, now: datetime) -> list:
        return self.odd_hours(now) + await self.hourly_note(now)

    async def hourly_note(self, now: datetime) -> list[NoteWritten]:
        """Once per hour, a note on the previous hour (skipped when nothing happened)."""
        hour = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
        if self._noted_hour is not None and hour <= self._noted_hour:
            return []
        self._noted_hour = hour
        episodes = self.memory.episodes(hour, hour + timedelta(hours=1))
        captions = list(dict.fromkeys(c for ep in episodes for c in ([ep.summary] if ep.summary else ep.captions)))
        if not captions:
            return []
        captions = captions[:MAX_NOTE_CAPTIONS]
        text = await asyncio.to_thread(self.summarize, captions) if self.summarize else " ".join(captions[:3])
        return [NoteWritten(ts=now, note_id=uuid4().hex, hour=hour, text=text.strip(), episode_ids=[e.id for e in episodes])]

    def on_alert(self, e: AlertRaised) -> list[ActionProposed]:
        """A package at the door while nobody has been home: offer to tell the household."""
        if self.notify is None or e.rule_id != "package-at-door":
            return []
        recent = self.memory.episodes(e.ts - AWAY_FOR, e.ts + timedelta(seconds=1))
        if any("person" in ep.labels and ep.camera_id != "cam-door" for ep in recent):
            return []  # someone is home and will see it
        return [ActionProposed(ts=e.ts, action_id=uuid4().hex, kind="send_message", target=self.notify,
                               reason="A package was left at the door and nobody has been home for 30 minutes",
                               params={"text": f"{e.message}. Nobody's been home for a while; can someone grab it?"})]

    def odd_hours(self, now: datetime) -> list[AlertRaised]:
        since = self._since or now - TICK
        self._since = now
        alerts = []
        if not self.enabled(ODD_HOURS_RULE):
            return []
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
                    message=f"Activity at the {place(ep.camera_id)} at {at}, during sleep hours",
                    evidence=[Citation(episode_id=ep.id, frame_ref=(ep.keyframe_refs or [None])[0], ts=ep.start)],
                )
            )
        return alerts
