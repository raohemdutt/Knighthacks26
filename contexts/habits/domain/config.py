"""Habit definitions (config/habits.yaml) and the household's day (config/agent.yaml `sleep`)."""

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel

Signal = Literal["away_window", "meal", "drink", "desk_time", "start_lead", "forgot", "disposal"]


class Habit(BaseModel):
    id: str
    name: str
    signal: Signal
    target: float
    direction: Literal["at_least", "at_most"]
    nudge: str
    unit: str = ""
    cooldown_minutes: float | None = None  # None: nudges.cooldown_minutes
    nudge_after: time | None = None  # local time before which the ticker stays quiet about it
    keywords: list[str] = []  # meal, drink
    gap_minutes: float = 60  # meal, drink: sightings closer than this are the same one
    count_each: bool = False  # drink: every matching object counts
    deadline_hours: float = 48  # desk_time, start_lead
    min_minutes: float = 15  # start_lead
    watch_items: list[str] = []  # forgot
    event_minutes: float = 30  # forgot

    def met(self, value: float) -> bool:
        return value >= self.target if self.direction == "at_least" else value <= self.target

    def pattern(self) -> re.Pattern:
        return word_pattern(self.keywords)


def word_pattern(words: list[str]) -> re.Pattern:
    """Whole words, any case, plural allowed: "can" matches "cans", not "candle"."""
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")s?\b", re.IGNORECASE)


class Desk(BaseModel):
    zone: str = "desk"
    camera: str = "cam-desk"


class CoverageSettings(BaseModel):
    """When a night or a day was watched enough to judge (contexts/habits/domain/coverage.py)."""

    gap_minutes: float = 5  # the desk camera silent for longer than this was not watching
    steady_minutes: float = 30  # ... once it has reported this long without such a silence
    min_fraction: float = 0.8  # sleep and focus are judged only when this much of their window was watched


class NudgeSettings(BaseModel):
    cooldown_minutes: float = 180
    min_gap_minutes: float = 20
    celebrate_streak: int = 3
    celebrate: str = "{name}: {streak} days in a row. Keep it going."


@dataclass(frozen=True)
class Schedule:
    """Local days and the sleep window. Defaults match config/agent.yaml."""

    tz: ZoneInfo = ZoneInfo("America/New_York")
    bedtime: time = time(20)
    wake: time = time(8)

    def local_day(self, ts: datetime) -> date:
        return ts.astimezone(self.tz).date()

    def day_start(self, day: date) -> datetime:
        return datetime.combine(day, time(0), self.tz)

    def sleep_window(self, day: date) -> tuple[datetime, datetime]:
        """The night that ends on `day`: bedtime the evening before to wake."""
        end = datetime.combine(day, self.wake, self.tz)
        start = datetime.combine(day, self.bedtime, self.tz)
        return (start - timedelta(days=1) if start >= end else start), end

    def asleep(self, ts: datetime) -> bool:
        day = self.local_day(ts)
        return any(start <= ts < end for start, end in (self.sleep_window(day), self.sleep_window(day + timedelta(days=1))))


class HabitsConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    habits: list[Habit]
    desk: Desk = Desk()
    door_zone: str = "door"
    close_grace_minutes: float = 2
    coverage: CoverageSettings = CoverageSettings()
    nudges: NudgeSettings = NudgeSettings()
    schedule: Schedule = Schedule()

    def by_signal(self, signal: Signal) -> Habit | None:
        return next((h for h in self.habits if h.signal == signal), None)

    def by_id(self, habit_id: str) -> Habit | None:
        return next((h for h in self.habits if h.id == habit_id), None)


def load_schedule(path: str | Path = "config/agent.yaml") -> Schedule:
    path = Path(path)
    raw = ((yaml.safe_load(path.read_text()) or {}) if path.exists() else {}).get("sleep") or {}
    return Schedule(
        tz=ZoneInfo(raw.get("timezone", "America/New_York")),
        bedtime=time.fromisoformat(raw.get("bedtime", "20:00")),
        wake=time.fromisoformat(raw.get("wake", "08:00")),
    )


def load_config(path: str | Path = "config/habits.yaml", agent_path: str | Path = "config/agent.yaml") -> HabitsConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return HabitsConfig(**raw, schedule=load_schedule(agent_path))
