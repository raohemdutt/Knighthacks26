"""Alert rules (A3 YAML half, A4 evaluator).

Pure code on zone events, no model and no network, so alerts keep working with
the GPUs and the internet gone. Time comes from event `ts`, never the wall
clock, so replaying the log gives the same alerts as the live run.
"""

from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import yaml
from pydantic import BaseModel, field_validator

from contracts.agent import AlertRaised, RuleAdded, RuleToggled
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

    def describe(self) -> str:
        """In words, for the phone: "Package left at the door: package, box... stays in the door zone 5 s"."""
        w = self.when
        what = {"enter": "enters", "exit": "leaves", "dwell": "stays in"}[w.kind]
        how_long = f" {w.seconds:g} s" if w.kind == "dwell" and w.seconds else ""
        where = f" on {w.camera}" if w.camera else ""
        return f"{self.notify}: {', '.join(w.label)} {what} the {w.zone} zone{how_long}{where}"

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
    def __init__(self, rules: list[Rule], builtin: dict[str, str] | None = None) -> None:
        self.rules = rules
        self.builtin = builtin or {}  # rules that live in code (the loop's odd-hours check): id -> description
        self.disabled: set[str] = set()
        self.sources: dict[str, str] = {}  # rule id -> the plain English it came from (rules added from the phone)
        self._last_fired: dict[str, datetime] = {}

    def on_rule_added(self, e: RuleAdded) -> None:
        rule = Rule.model_validate(e.rule)
        self.rules = [r for r in self.rules if r.id != rule.id] + [rule]
        self.sources[rule.id] = e.source_text

    def on_rule_toggled(self, e: RuleToggled) -> None:
        (self.disabled.discard if e.enabled else self.disabled.add)(e.rule_id)

    def enabled(self, rule_id: str) -> bool:
        return rule_id not in self.disabled

    def known(self, rule_id: str) -> bool:
        return rule_id in self.builtin or any(r.id == rule_id for r in self.rules)

    def listing(self) -> list[dict]:
        """Every rule with a short description and whether it is on (GET /api/rules)."""
        rows = [{"id": r.id, "description": r.describe(), "notify": r.notify, "enabled": self.enabled(r.id),
                 "source": "phone" if r.id in self.sources else "config", "text": self.sources.get(r.id)} for r in self.rules]
        rows += [{"id": i, "description": d, "notify": None, "enabled": self.enabled(i), "source": "built-in", "text": None}
                 for i, d in self.builtin.items()]
        return rows

    def on_zone_event(self, e: ZoneEvent) -> list[AlertRaised]:
        alerts = []
        for rule in self.rules:
            if rule.id in self.disabled or not rule.matches(e):
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


class AlertHistory:
    """Recent alerts for the phone (GET /api/alerts). State only: rebuilt from the log on start."""

    def __init__(self, keep: int = 200) -> None:
        self.alerts: deque[AlertRaised] = deque(maxlen=keep)

    def on_alert(self, e: AlertRaised) -> None:
        self.alerts.append(e)

    def recent(self, limit: int = 50) -> list[dict]:
        return [{"alert_id": a.alert_id, "rule_id": a.rule_id, "message": a.message, "ts": a.ts.isoformat(),
                 "evidence": [c.model_dump(mode="json", exclude_none=True) for c in a.evidence]}
                for a in list(self.alerts)[::-1][:max(0, limit)]]
