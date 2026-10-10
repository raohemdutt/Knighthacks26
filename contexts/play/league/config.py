"""config/league.yaml: the rule set. Sane defaults when the file or a key is missing."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from contracts.play import Stake

Source = Literal["habit", "dayrun", "impact"]


class TargetRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weight: float = Field(1, ge=0)
    label: str | None = None
    unit: str = "days"
    source: Source = "habit"
    min_stars: int = 2  # dayrun: a committed plan with at least this many stars counts as met
    min_rate: float = 0.8  # impact: the week's diversion rate that counts as met


class Preset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    habit: str
    direction: Literal["at_least", "at_most"]
    target: float
    title: str


DEFAULT_TARGETS: dict[str, TargetRule] = {
    "sleep": TargetRule(weight=2, label="sleep", unit="nights"),
    "meals": TargetRule(label="meals"),
    "focus": TargetRule(label="focus"),
    "energy_drinks": TargetRule(label="energy drinks"),
    "leave_ready": TargetRule(label="leave ready"),
    "sort_right": TargetRule(label="sort right"),
    "dayrun_stars": TargetRule(label="Day Run stars", unit="plans", source="dayrun"),
    "recycling": TargetRule(label="recycling", unit="week", source="impact"),
}

DEFAULT_PRESETS: dict[str, Preset] = {
    "no_energy_drinks": Preset(habit="energy_drinks", direction="at_most", target=0, title="No energy drinks"),
    "sleep_six": Preset(habit="sleep", direction="at_least", target=6, title="Six hours of sleep"),
    "three_meals": Preset(habit="meals", direction="at_least", target=3, title="Three meals a day"),
}


class LeagueConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    me: str = "me"
    week_starts: Literal["monday"] = "monday"
    close_grace_hours: float = Field(3, ge=0, le=24)  # after local Monday 00:00, so the last summaries arrive first
    targets: dict[str, TargetRule] = Field(default_factory=lambda: dict(DEFAULT_TARGETS))
    stake: Stake = Stake()
    team_diversion_target: float = Field(0.8, ge=0, le=1)
    max_days: int = Field(14, ge=1)
    presets: dict[str, Preset] = Field(default_factory=lambda: dict(DEFAULT_PRESETS))

    def label(self, target: str) -> str:
        rule = self.targets.get(target)
        return (rule.label if rule and rule.label else target).replace("_", " ")


def load_config(path: str | Path | None) -> LeagueConfig:
    """The file, or the defaults when it is missing. A bad file is a bug (ValueError), not a silent default."""
    raw = {}
    if path and Path(path).exists():
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    team = raw.pop("team", None) or {}
    challenges = raw.pop("challenges", None) or {}
    stake = raw.pop("stake", None) or {}
    return LeagueConfig(
        **raw,
        stake=Stake(kind=stake.get("kind") or "pays_for_food", text=stake.get("text")),
        team_diversion_target=team.get("diversion_target", 0.8),
        max_days=challenges.get("max_days", 14),
        presets={k: Preset(**v) for k, v in (challenges.get("presets") or DEFAULT_PRESETS).items()}
        if "presets" in challenges else dict(DEFAULT_PRESETS),
    )
