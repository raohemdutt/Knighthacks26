"""The household's day (config/agent.yaml `sleep`): local days, wake and bedtime, in event time."""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


@dataclass(frozen=True)
class DayClock:
    tz: ZoneInfo = ZoneInfo("America/New_York")
    wake: time = time(8)
    bedtime: time = time(20)

    @classmethod
    def load(cls, agent_yaml: str | Path = "config/agent.yaml") -> "DayClock":
        path = Path(agent_yaml)
        raw = ((yaml.safe_load(path.read_text()) or {}) if path.exists() else {}).get("sleep") or {}
        return cls(tz=ZoneInfo(raw.get("timezone", "America/New_York")),
                   wake=time.fromisoformat(raw.get("wake", "08:00")), bedtime=time.fromisoformat(raw.get("bedtime", "20:00")))

    def local(self, ts: datetime) -> datetime:
        return ts.astimezone(self.tz) if ts.tzinfo else ts.replace(tzinfo=self.tz)

    def day_of(self, ts: datetime) -> date:
        return self.local(ts).date()

    def day_start(self, day: date) -> datetime:
        return datetime.combine(day, time(0), self.tz)

    def wake_at(self, day: date) -> datetime:
        return datetime.combine(day, self.wake, self.tz)

    def bed_at(self, day: date) -> datetime:
        """When `day` ends for the loop: its bedtime, or the next morning's bedtime when bedtime is after midnight."""
        bed = datetime.combine(day, self.bedtime, self.tz)
        return bed if self.bedtime > self.wake else bed + timedelta(days=1)
