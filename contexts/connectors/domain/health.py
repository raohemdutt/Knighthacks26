"""Nights of sleep from a watch's sleep stages (K9). Pure: no database, no clock, no files.

    stages = watch_stages(rows, ["Apple Watch"])            # rows shaped like FreeReps' sleep_stages table
    for night in nights(stages, tz, bedtime, wake):
        attach_metrics(night, metric_rows, tz)              # rows shaped like health_metrics, value = COALESCE(qty, avg_val)
        item = to_item(night, tz)                           # {external_id, title, when, summary, kind: "health"}
    reading = parse_summary(item["summary"])                # the numbers back out of the item, for Habits

What the data's owner said about it, and what this does with it:

- Only the watch's rows count. The phone writes `In Bed` rows and nothing else: they are dropped, whatever
  their source.
- FreeReps' own `sleep_sessions` splits a night into fragments, so it is not used. Stages are clustered
  again here: a pause of up to two hours between stages is still the same sleep (`GAP`).
- A night belongs to the local day it ends on: the cluster has to touch that day's sleep window (bedtime the
  evening before to wake, from config/agent.yaml), and the longest such cluster is the night. An afternoon
  nap touches no sleep window, so it is never a night, not even on a day with no other data.
- The watch is not worn every night. A night with no stages has no item: no data, not no sleep.
- All-day metrics (steps, rings) are not read.

`connector.item.synced` has no field for numbers, so they travel in `summary` in one fixed, readable format;
`summary_line` writes it and `parse_summary` is the only reader.
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, tzinfo

GAP = timedelta(hours=2)
NOT_SLEEP = {"awake", "in bed"}
STAGE_ORDER = ["core", "deep", "rem", "asleep", "awake"]
METRICS = ("resting_heart_rate", "heart_rate", "heart_rate_variability", "respiratory_rate")
ID_PREFIX = "sleep:"


@dataclass(frozen=True)
class Stage:
    start: datetime
    end: datetime
    stage: str  # lower case: core, deep, rem, awake, asleep


@dataclass
class Night:
    day: date  # the local day this night ends on (the Habits day it counts for)
    bed: datetime  # first stage of the cluster
    wake: datetime  # end of its last stage
    asleep_min: int  # every stage but Awake, overlaps counted once
    in_bed_min: int  # bed to wake
    stage_min: dict[str, int] = field(default_factory=dict)
    resting_hr: int | None = None  # bpm, the day's resting heart rate
    sleeping_hr: int | None = None  # bpm, mean heart rate between bed and wake
    hrv_ms: int | None = None  # mean heart rate variability between bed and wake
    resp: float | None = None  # breaths per minute, mean between bed and wake


@dataclass(frozen=True)
class SleepReading:
    """The numbers of one night, read back from an item's summary."""

    asleep_min: int
    in_bed_min: int | None = None
    stage_min: dict[str, int] = field(default_factory=dict)
    resting_hr: int | None = None
    sleeping_hr: int | None = None
    hrv_ms: int | None = None
    resp: float | None = None

    @property
    def asleep_h(self) -> float:
        return round(self.asleep_min / 60, 2)


def as_time(value) -> datetime:
    """A timestamptz as the driver gives it, or the same thing written in a fixture file."""
    ts = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if ts.tzinfo is None:
        raise ValueError(f"timestamp without a UTC offset: {value!r}")
    return ts


def is_watch(source: str | None, watch_sources: list[str]) -> bool:
    """True when any configured name is part of the row's source, whatever the case or the apostrophe."""
    plain = (source or "").replace("’", "'").lower()
    return any(name.replace("’", "'").lower() in plain for name in watch_sources if name)


def watch_stages(rows: list[dict], watch_sources: list[str]) -> list[Stage]:
    """The watch's sleep stages, oldest first. `In Bed` rows and every other source are dropped."""
    out = []
    for row in rows:
        stage = str(row.get("stage") or "").strip().lower()
        if stage == "in bed" or not stage or not is_watch(row.get("source"), watch_sources):
            continue
        start, end = as_time(row["start_time"]), as_time(row["end_time"])
        if end > start:
            out.append(Stage(start, end, stage))
    return sorted(out, key=lambda s: (s.start, s.end))


def clusters(stages: list[Stage], gap: timedelta = GAP) -> list[list[Stage]]:
    """Stages (oldest first) grouped into sleeps: a new one starts after more than `gap` with no stage."""
    out: list[list[Stage]] = []
    last_end: datetime | None = None
    for s in stages:
        if last_end is None or s.start - last_end > gap:
            out.append([])
            last_end = s.end
        out[-1].append(s)
        last_end = max(last_end, s.end)
    return out


def _minutes(spans: list[tuple[datetime, datetime]]) -> int:
    """Total length of the spans in minutes, overlapping ones counted once."""
    total, upto = timedelta(0), None
    for a, b in sorted(spans):
        a = max(a, upto) if upto else a
        if b > a:
            total += b - a
            upto = b
    return round(total.total_seconds() / 60)


def sleep_window(day: date, tz: tzinfo, bedtime: time, wake: time) -> tuple[datetime, datetime]:
    """The night that ends on `day`: bedtime the evening before to wake (as Habits' Schedule has it)."""
    end = datetime.combine(day, wake, tz)
    start = datetime.combine(day, bedtime, tz)
    return (start - timedelta(days=1) if start >= end else start), end


def _night_day(bed: datetime, wake: datetime, tz: tzinfo, bedtime: time, wake_at: time) -> date | None:
    """The day whose sleep window this sleep overlaps most; None for one that touches no window (a nap)."""
    first = wake.astimezone(tz).date()
    best, most = None, timedelta(0)
    for day in (first, first + timedelta(days=1)):
        start, end = sleep_window(day, tz, bedtime, wake_at)
        overlap = min(wake, end) - max(bed, start)
        if overlap > most:
            best, most = day, overlap
    return best


def nights(stages: list[Stage], tz: tzinfo, bedtime: time = time(20), wake: time = time(8),
           gap: timedelta = GAP) -> list[Night]:
    """One night per local day that has one, oldest first: the longest sleep that touches its sleep window."""
    best: dict[date, Night] = {}
    for group in clusters(stages, gap):
        bed, up = group[0].start, max(s.end for s in group)
        day = _night_day(bed, up, tz, bedtime, wake)
        if day is None:
            continue
        per_stage: dict[str, list] = {}
        for s in group:
            per_stage.setdefault(s.stage, []).append((s.start, s.end))
        asleep = _minutes([span for stage, spans in per_stage.items() if stage not in NOT_SLEEP for span in spans])
        night = Night(day, bed, up, asleep, _minutes([(bed, up)]), {k: _minutes(v) for k, v in per_stage.items()})
        if asleep and (day not in best or asleep > best[day].asleep_min):
            best[day] = night
    return [best[d] for d in sorted(best)]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def attach_metrics(night: Night, rows: list[dict], tz: tzinfo) -> Night:
    """Heart numbers for this night. `rows`: {time, metric_name, value}, value already COALESCE(qty, avg_val)."""
    during: dict[str, list[float]] = {}
    resting = []
    for row in rows:
        if row.get("value") is None:
            continue
        ts, name, value = as_time(row["time"]), row["metric_name"], float(row["value"])
        if name == "resting_heart_rate":  # one figure a day: the one dated the day the night ended
            if ts.astimezone(tz).date() == night.day:
                resting.append(value)
        elif night.bed <= ts <= night.wake:
            during.setdefault(name, []).append(value)
    hr, hrv, resp = (_mean(during.get(k, [])) for k in ("heart_rate", "heart_rate_variability", "respiratory_rate"))
    night.resting_hr = round(_mean(resting)) if resting else None
    night.sleeping_hr = round(hr) if hr is not None else None
    night.hrv_ms = round(hrv) if hrv is not None else None
    night.resp = round(resp, 1) if resp is not None else None
    return night


# --- the item, and its summary line ---


def hm(minutes: int) -> str:
    return f"{minutes // 60} h {minutes % 60} min"


def _clock(ts: datetime, tz: tzinfo) -> str:
    return ts.astimezone(tz).strftime("%I:%M %p").lstrip("0")


def summary_line(night: Night, tz: tzinfo) -> str:
    """"asleep 7 h 12 min · in bed 7 h 40 min · 11:48 PM to 7:28 AM · core 240 min · ... · resting HR 52 bpm"."""
    order = [s for s in STAGE_ORDER if s in night.stage_min] + sorted(set(night.stage_min) - set(STAGE_ORDER))
    parts = [f"asleep {hm(night.asleep_min)}", f"in bed {hm(night.in_bed_min)}",
             f"{_clock(night.bed, tz)} to {_clock(night.wake, tz)}"]
    parts += [f"{'REM' if s == 'rem' else s} {night.stage_min[s]} min" for s in order]
    parts += [text for value, text in (
        (night.resting_hr, f"resting HR {night.resting_hr} bpm"),
        (night.sleeping_hr, f"sleeping HR {night.sleeping_hr} bpm"),
        (night.hrv_ms, f"HRV {night.hrv_ms} ms"),
        (night.resp, f"breathing {night.resp}/min"),
    ) if value is not None]
    return " · ".join(parts)


def to_item(night: Night, tz: tzinfo) -> dict:
    """The night as a connector item. `external_id` is stable per night, so a later upload updates it."""
    return {
        "external_id": f"{ID_PREFIX}{night.day.isoformat()}",
        "kind": "health",
        "title": f"Sleep {hm(night.asleep_min)}",
        "when": night.wake.astimezone(tz).isoformat(),
        "summary": summary_line(night, tz),
    }


_DURATION = r"(\d+) h (\d+) min"
_ASLEEP = re.compile(rf"^asleep {_DURATION}$")
_IN_BED = re.compile(rf"^in bed {_DURATION}$")
_STAGE = re.compile(r"^([A-Za-z][A-Za-z ()]*?) (\d+) min$")
_NUMBERS = {"resting_hr": (re.compile(r"^resting HR (\d+) bpm$"), int),
            "sleeping_hr": (re.compile(r"^sleeping HR (\d+) bpm$"), int),
            "hrv_ms": (re.compile(r"^HRV (\d+) ms$"), int),
            "resp": (re.compile(r"^breathing (\d+(?:\.\d+)?)/min$"), float)}


def parse_summary(summary: str | None) -> SleepReading | None:
    """The numbers in a line `summary_line` wrote. None when it is not one (no "asleep" part)."""
    fields: dict = {"stage_min": {}}
    for part in (p.strip() for p in (summary or "").split("·")):
        if m := _ASLEEP.match(part):
            fields["asleep_min"] = int(m[1]) * 60 + int(m[2])
        elif m := _IN_BED.match(part):
            fields["in_bed_min"] = int(m[1]) * 60 + int(m[2])
        elif m := _STAGE.match(part):
            fields["stage_min"][m[1].lower()] = int(m[2])
        else:
            for name, (pattern, cast) in _NUMBERS.items():
                if m := pattern.match(part):
                    fields[name] = cast(m[1])
    return SleepReading(**fields) if "asleep_min" in fields else None


def night_day(external_id: str) -> date | None:
    """The local day a health item's night ends on, from its id ("sleep:2026-10-10")."""
    if not external_id.startswith(ID_PREFIX):
        return None
    try:
        return date.fromisoformat(external_id[len(ID_PREFIX):])
    except ValueError:
        return None
