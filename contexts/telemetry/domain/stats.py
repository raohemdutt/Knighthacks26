"""The figures GET /runtime/stats serves, each with the record it came from, or null and the reason nothing has.

A Figure is one number the endpoint serves: its value and unit, the node it describes, when it was MEASURED (the
record's own time, never the time the endpoint was read), the record it came from (`metrics.jsonl:<line>`,
`ledger seq <n>`, or the telemetry events it was computed from) and that record's low-confidence flag. When nothing
has recorded it, value is null and `reason` says what was looked for. lab/slide_numbers.py reads value, ts, source and
low_confidence.

Two kinds of figure are computed from telemetry records rather than copied from a lab row:
- a decode rate from request windows: the output tokens after the first, over the time from the first token to the
  end of the answer, summed over the windows (so each answer counts for its tokens, not once);
- a training run's minutes from its steps: from the first step to the last, plus one step's own length, since the
  first step's event is stamped when that step ended.
"""

from collections.abc import Iterable
from datetime import datetime

from pydantic import BaseModel, model_validator

from contracts.telemetry import RequestWindow


class Figure(BaseModel):
    value: float | None
    unit: str
    node: str | None = None  # the machine the number describes, as its record names it ("jetson", "pod-a", "gpu-a")
    ts: datetime | None = None  # when the record says it was measured
    source: str | None = None  # the record it came from
    low_confidence: bool = False  # the record's own flag: derived, or not stated as such by a primary record
    reason: str | None = None  # why value is null

    @model_validator(mode="after")
    def _a_value_with_its_record_or_a_reason(self) -> "Figure":
        if self.value is None and not self.reason:
            raise ValueError("a figure with no value says why")
        if self.value is not None and (self.ts is None or not self.source):
            raise ValueError("a figure with a value names its record and when it was measured")
        return self


class EvalRow(BaseModel):
    """Before and after eval scores of one deployed model, from its newest `learning.run.evaluated`."""

    model_version: str
    deployed_ts: datetime  # the newest learning.model.deployed of this version
    previous_version: str | None
    task: str | None = None
    run_id: str | None = None
    decision: str | None = None
    before: float | None = None  # scores["current"]: the model that was serving when the candidate was judged
    after: float | None = None  # scores["candidate"]: this model
    teacher: float | None = None  # scores["teacher"], when the eval scored the teacher too
    ts: datetime | None = None  # when the evaluation was recorded
    source: str | None = None  # learning.run.evaluated <event_id>
    reason: str | None = None  # why before or after is null


def missing(unit: str, reason: str) -> Figure:
    return Figure(value=None, unit=unit, reason=reason)


def newest(candidates: Iterable[Figure | None]) -> Figure | None:
    """The candidate measured last; on a tie the one listed first wins, so callers list the primary record first."""
    best: Figure | None = None
    for c in candidates:
        if c is not None and c.value is not None and (best is None or c.ts > best.ts):
            best = c
    return best


def decode_rate(windows: Iterable[RequestWindow]) -> tuple[float, int, datetime] | None:
    """(tokens per second after the first token, windows used, end of the newest window used), or None.

    Only windows that report a first-token time and two or more output tokens, and whose last token came after the
    first, can be timed: without the first-token time prefill and decode are mixed, and one token has no decode."""
    tokens, seconds, used, last = 0, 0.0, 0, None
    for w in windows:
        if w.first_token is None or w.tokens_out is None or w.tokens_out < 2:
            continue
        span = (w.ended - w.first_token).total_seconds()
        if span <= 0:
            continue
        tokens += w.tokens_out - 1
        seconds += span
        used += 1
        last = w.ended if last is None or w.ended > last else last
    return (tokens / seconds, used, last) if used else None


def run_minutes(first_ts: datetime, last_ts: datetime, step_ms: float | None) -> float:
    """A training run's length in minutes from its first and last step events and one step's length in ms."""
    return ((last_ts - first_ts).total_seconds() + (step_ms or 0.0) / 1000.0) / 60.0
