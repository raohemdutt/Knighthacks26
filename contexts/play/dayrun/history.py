"""The days around today, for the date strip on Today's Plan (GET /api/dayrun/days, GET /api/dayrun/day).

    book = DayBook(data_dir / "events")
    book.days()      # {date: DayRecord}: every day the log has a plan for, with its re-plans and its review

Past days are read from the event log, which is the source of truth: `day.plan.proposed`, `day.replan.proposed`,
`day.plan.committed` and `day.reviewed` (contracts/day.py), as the Day loop published them. Nothing is kept here
but a cache of that reading, redone when a log file changes. Read-only: this never writes, publishes or calls a model.

The log is read as plain JSON (the four event types above, by their documented fields) so that Day Run needs no
subscription of its own in the composition root.
"""

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

TYPES = ("day.plan.proposed", "day.replan.proposed", "day.plan.committed", "day.reviewed")


@dataclass
class DayRecord:
    day: date
    plan: dict | None = None  # the morning plan (the last one proposed for the day)
    replans: list[dict] = field(default_factory=list)  # oldest first
    committed: set[str] = field(default_factory=set)  # plan ids the phone approved
    review: dict | None = None

    @property
    def lived(self) -> dict | None:
        """The plan the day was measured against: the last re-plan that was approved, else the morning plan."""
        approved = [r for r in self.replans if r.get("plan_id") in self.committed]
        return approved[-1] if approved else self.plan


class DayBook:
    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self._seen: tuple = ()
        self._days: dict[date, DayRecord] = {}

    def _signature(self) -> tuple:
        out = []
        for p in sorted(self.log_dir.glob("*.jsonl")):
            st = p.stat()
            out.append((p.name, st.st_size, st.st_mtime_ns))
        return tuple(out)

    def days(self) -> dict[date, DayRecord]:
        sig = self._signature()
        if sig == self._seen:
            return self._days
        days: dict[date, DayRecord] = {}
        for name, _, _ in sig:
            try:
                with (self.log_dir / name).open(encoding="utf-8") as f:
                    for line in f:
                        if '"day.' not in line:
                            continue
                        try:
                            e = json.loads(line)
                        except ValueError:
                            continue
                        if e.get("type") not in TYPES or not e.get("day"):
                            continue
                        rec = days.setdefault(date.fromisoformat(e["day"]), DayRecord(day=date.fromisoformat(e["day"])))
                        if e["type"] == "day.plan.proposed":
                            rec.plan = e
                        elif e["type"] == "day.replan.proposed":
                            rec.replans.append(e)
                        elif e["type"] == "day.plan.committed":
                            rec.committed.add(e.get("plan_id", ""))
                        else:
                            rec.review = e
            except OSError:
                continue
        self._seen, self._days = sig, days
        return days
