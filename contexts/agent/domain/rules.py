"""Alert rules (A3 YAML half, A4 evaluator).

Pure code on zone events, no model and no network, so alerts keep working with
the GPUs and the internet gone. Time comes from event `ts`, never the wall
clock, so replaying the log gives the same alerts as the live run.
"""

from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import yaml
from pydantic import BaseModel, field_validator

from contracts.agent import AlertRaised, RuleAdded
from contracts.perception import ZoneEvent
from contracts.types import Citation


class Condition(BaseModel):
    label: list[str]
    zone: str
    kind: Literal["enter", "exit", "dwell"]
    seconds: float = 0  # minimum dwell, only for kind == "dwell"
    camera: str | None = None

    @field_validator("label", mode="before")
    @classmethod
    def _one_or_many(cls, v: str | list[str]) -> list[str]:
        return [v] if isinstance(v, str) else v


class Rule(BaseModel):
    id: str
    when: Condition
    notify: str
    cooldown_s: float = 60

    def matches(self, e: ZoneEvent) -> bool:
        w = self.when
        if e.label not in w.label or e.zone_id != w.zone or e.kind != w.kind:
            return False
        if w.camera and e.camera_id != w.camera:
            return False
        return w.kind != "dwell" or (e.dwell_s or 0) >= w.seconds


def load_rules(path: str | Path = "config/rules.yaml") -> list[Rule]:
    raw = yaml.safe_load(Path(path).read_text()) or []
    return [Rule.model_validate(r) for r in raw]


class RuleEvaluator:
    def __init__(self, rules: list[Rule]) -> None:
        self.rules = rules
        self._last_fired: dict[str, datetime] = {}

    def on_rule_added(self, e: RuleAdded) -> None:
        rule = Rule.model_validate(e.rule)
        self.rules = [r for r in self.rules if r.id != rule.id] + [rule]

    def on_zone_event(self, e: ZoneEvent) -> list[AlertRaised]:
        alerts = []
        for rule in self.rules:
            if not rule.matches(e):
                continue
            last = self._last_fired.get(rule.id)
            if last is not None and (e.ts - last).total_seconds() < rule.cooldown_s:
                continue
            self._last_fired[rule.id] = e.ts
            alerts.append(
                AlertRaised(
                    ts=e.ts,
                    alert_id=uuid4().hex,
                    rule_id=rule.id,
                    message=rule.notify,
                    evidence=[Citation(frame_ref=e.frame_ref, ts=e.ts)],
                )
            )
        return alerts
