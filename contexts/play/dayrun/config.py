"""Day Run's rates and settings, read from config/dayrun.yaml (every number is explained there)."""

from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, ConfigDict

from contexts.profiles import Profile, Vocabulary


class SleepCfg(BaseModel):
    bedtime: time = time(23, 30)
    wake: time = time(7, 30)
    need_hours: float = 8
    fixed: bool = False
    min_block_minutes: int = 90


class StartCfg(BaseModel):
    energy_rested: float = 90
    energy_per_missed_hour: float = 8
    default_sleep_hours: float = 7
    no_meal_penalty: float = 6
    no_meal_after: time = time(9)
    focus: float = 85
    hungry_minutes_no_meal: float = 240
    hungry_minutes_default: float = 120


class WorkEstimate(BaseModel):
    match: str
    minutes: float


class WinCfg(BaseModel):
    sleep_hours: float = 6
    meals: int = 2


class StarsCfg(BaseModel):
    energy_floor: float = 25
    sleep_hours: float = 7.5
    risk: float = 0.35


class ReviewCfg(BaseModel):
    weights: dict[str, float] = {"study": 0.5, "sleep": 0.3, "meals": 0.2}


class DayRunConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tz: ZoneInfo = ZoneInfo("America/New_York")
    tick_minutes: int = 15
    window_hours: int = 24
    deadline_horizon_hours: int = 72
    pack: str = "ucf"
    home: str = "apartment.bedroom"
    desk: str = "apartment.desk"
    start_place: str = "apartment.desk"
    kinds: dict[str, list[str]] = {"study": ["library", "desk", "class"], "eat": ["dining", "kitchen"]}
    max_options: int = 5
    sleep: SleepCfg = SleepCfg()
    default_event_minutes: int = 60
    optional_event_prefixes: list[str] = ["campus:"]
    ignored_event_prefixes: list[str] = ["nearby:"]  # offered on the Calendar screen, not by the board: never a block or a move
    meal_words: list[str] = ["breakfast", "brunch", "lunch", "dinner", "pizza", "meal"]
    study_words: list[str] = ["study", "review", "office hours", "lab", "tutoring", "workshop"]
    start: StartCfg = StartCfg()
    energy_per_hour: dict[str, float] = {"idle": -2.5, "study": -4.5, "class": -4.0, "walk": -5.0, "rest": -0.5,
                                         "event": -2.0, "eat": 0.0, "sleep": 11.0}
    meal_energy: float = 15
    snack_factor: float = 0.3
    meal_min_minutes: int = 20
    meal_gap_minutes: int = 90
    hunger_after_minutes: float = 300
    hunger_energy_per_hour: float = -3.0
    debt_drain_per_hour: float = 0.04
    max_sleep_debt_hours: float = 24
    focus_per_hour: dict[str, float] = {"study": -14.0, "class": -6.0, "walk": 10.0, "idle": 15.0, "rest": 30.0,
                                        "eat": 30.0, "event": 20.0, "sleep": 25.0}
    energy_full_focus: float = 50
    focus_floor: float = 0.15
    friend_boost: float = 1.2
    friend_decay: float = 0.7
    study_place_factor: dict[str, float] = {"library": 1.0, "desk": 0.9, "class": 0.9, "default": 0.7}
    work_estimates: list[WorkEstimate] = []
    default_work_minutes: float = 120
    capacity_minutes_per_day: float = 180
    on_track_risk: float = 0.5
    win: WinCfg = WinCfg()
    stars: StarsCfg = StarsCfg()
    min_move_minutes: int = 15
    review: ReviewCfg = ReviewCfg()

    vocabulary: Vocabulary = Vocabulary()  # the profile's words: "Study at", "Work at"; block kinds stay "study"

    @classmethod
    def load(cls, path: str | Path = "config/dayrun.yaml", agent_yaml: str | Path | None = "config/agent.yaml",
             profile: Profile | None = None) -> "DayRunConfig":
        """dayrun.yaml; with a profile, its dayrun.yaml layered over it and its vocabulary for the titles."""
        raw = (yaml.safe_load(Path(path).read_text()) or {}) if Path(path).exists() else {}
        agent = (yaml.safe_load(Path(agent_yaml).read_text()) or {}) if agent_yaml and Path(agent_yaml).exists() else {}
        if profile is not None:
            raw = profile.layer("dayrun.yaml", raw)
            agent = profile.layer("agent.yaml", agent)
            raw["vocabulary"] = profile.vocabulary
        zone = (agent.get("sleep") or {}).get("timezone")  # one time zone for the whole box
        if zone:
            raw.setdefault("tz", ZoneInfo(zone))
        if isinstance(raw.get("tz"), str):
            raw["tz"] = ZoneInfo(raw["tz"])
        return cls.model_validate(raw)
