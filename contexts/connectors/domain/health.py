"""Nights of sleep from a watch's sleep samples (K9). Pure: no database, no clock, no files.

    stages = watch_stages(rows, ["Apple Watch"])            # rows shaped like FreeReps' category_samples table
    for night in nights(stages, tz, bedtime, wake):
        attach_metrics(night, metric_rows, tz)              # rows shaped like health_metrics, value = COALESCE(qty, avg_val)
        item = to_item(night, tz)                           # {external_id, title, when, summary, kind: "health"}
    reading = parse_summary(item["summary"])                # the numbers back out of the item, for Habits

What the data's owner said about it, checked against his real export (fixtures/watch), and what this does:

- Sleep is in `category_samples`, type HKCategoryTypeIdentifierSleepAnalysis: one row per interval, the stage
  in `value_label` ("Asleep Core", "Asleep Deep", "Asleep REM", "Awake", "In Bed"). Timestamps are UTC; every
  local day and clock time here is worked out in the household's time zone.
- What tells measured sleep from the rest is the label, not the source. Stage rows only ever come from the
  watch, so one with an empty source counts; one that names something not in `watch_sources` does not.
  `In Bed` rows are written by more than the watch (the phone, an older device label), so they count only
  when their source is named in `watch_sources`, and only for a night that has no stages (below).
- FreeReps' own `sleep_sessions` is not used: in the real export it keeps only the first stretch of about one
  night in three and runs some pairs of nights together. Stages are clustered again here: a pause of up to
  two hours between stages is still the same sleep (`GAP`).
- A night belongs to the local day it ends on: the cluster has to touch that day's sleep window (bedtime the
  evening before to wake, from config/agent.yaml), and the longest such cluster is the night. An afternoon
  nap touches no sleep window, so it is never a night, not even on a day with no other data.
- A night with no stages but with the watch's `In Bed` rows (every night before the watch recorded stages) is
  still a night, marked as one: its item says "In bed", gives no time asleep, and `parse_summary` reads no
  measurement out of it, so Habits keeps its own estimate. Time in bed is not time asleep. One under an hour
  or over fourteen is a fragment or two nights run together, and is dropped.
- The watch is not worn every night. A night with no rows has no item: no data, not no sleep.
- All-day metrics (steps, rings) are not read: the watch is worn for runs and sleep, so they undercount.

`connector.item.synced` has no field for numbers, so they travel in `summary` in one fixed, readable format;
`summary_line` writes it and `parse_summary` is the only reader.
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, tzinfo

GAP = timedelta(hours=2)
IN_BED = "in bed"
NOT_SLEEP = {"awake", IN_BED}
IN_BED_ONLY_MIN, IN_BED_ONLY_MAX = 60, 14 * 60  # minutes: outside this, In Bed rows alone are not a night
NO_STAGES = "no sleep stages: time in bed, not measured sleep"
STAGE_ORDER = ["core", "deep", "rem", "asleep", "awake"]
METRICS = ("resting_heart_rate", "heart_rate", "heart_rate_variability", "respiratory_rate")
ID_PREFIX = "sleep:"


@dataclass(frozen=True)
class Stage:
    start: datetime
    end: datetime
    stage: str  # lower case: core, deep, rem, awake, asleep, in bed


@dataclass
class Night:
    day: date  # the local day this night ends on (the Habits day it counts for)
    bed: datetime  # first stage of the cluster
    wake: datetime  # end of its last stage
    asleep_min: int  # every stage but Awake, overlaps counted once; 0 when there are no stages
    in_bed_min: int  # bed to wake; with no stages, the In Bed rows themselves, overlaps counted once
    stage_min: dict[str, int] = field(default_factory=dict)
    resting_hr: int | None = None  # bpm, the day's resting heart rate
    sleeping_hr: int | None = None  # bpm, mean heart rate between bed and wake
    hrv_ms: int | None = None  # mean heart rate variability between bed and wake
    resp: float | None = None  # breaths per minute, mean between bed and wake
    staged: bool = True  # False: put together from In Bed rows only, so there is no time asleep


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
    """A timestamptz as the driver gives it, as Postgres prints it in an export ("2026-01-26 05:50:25+00"), or
    as a fixture file writes it. Python 3.10 (the Jetson) reads neither "Z", "+00" nor ".5" seconds as they are."""
    if isinstance(value, datetime):
        ts = value
    else:
        text = re.sub(r"([+-]\d\d)$", r"\1:00", str(value).strip().replace("Z", "+00:00"))
        text = re.sub(r"\.(\d+)", lambda m: "." + m[1][:6].ljust(6, "0"), text)
        ts = datetime.fromisoformat(text)
    if ts.tzinfo is None:
        raise ValueError(f"timestamp without a UTC offset: {value!r}")
    return ts


def _plain(name: str | None) -> str:
    """A device name for comparing: Apple Health writes a curly apostrophe and a no-break space in "Apple Watch"."""
    return " ".join((name or "").replace("’", "'").lower().split())


def is_watch(source: str | None, watch_sources: list[str]) -> bool:
    """True when any configured name is part of the row's source, whatever the case, apostrophe or kind of space."""
    plain = _plain(source)
    return any(_plain(name) in plain for name in watch_sources if _plain(name))


def stage_name(value_label: str | None) -> str:
    """"Asleep Core" -> "core", "Asleep REM" -> "rem", "Awake" -> "awake", "In Bed" -> "in bed", "Asleep" -> "asleep"."""
    label = " ".join(str(value_label or "").lower().split())
    if not label.startswith("asleep"):
        return label
    rest = label[len("asleep"):].strip()
    return "asleep" if rest in ("", "unspecified") else rest


def watch_stages(rows: list[dict], watch_sources: list[str]) -> list[Stage]:
    """The watch's sleep rows, oldest first: stages from the watch or with no source, `In Bed` only from the watch.

    `rows`: {start_date, end_date, value_label, source}, as in `category_samples`."""
    out = []
    for row in rows:
        stage, source = stage_name(row.get("value_label")), row.get("source")
        named = is_watch(source, watch_sources)
        if not stage or not (named if stage == IN_BED else named or not _plain(source)):
            continue
        start, end = as_time(row["start_date"]), as_time(row["end_date"])
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
    """One night per local day that has one, oldest first: the longest sleep that touches its sleep window.
    A day with no staged sleep gets its longest `In Bed` stretch instead, marked `staged=False`."""
    best: dict[date, Night] = {}
    for group in clusters([s for s in stages if s.stage != IN_BED], gap):
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
    unstaged: dict[date, Night] = {}
    for group in clusters([s for s in stages if s.stage == IN_BED], gap):
        bed, up = group[0].start, max(s.end for s in group)
        day = _night_day(bed, up, tz, bedtime, wake)
        in_bed = _minutes([(s.start, s.end) for s in group])
        if day is None or day in best or not IN_BED_ONLY_MIN <= in_bed <= IN_BED_ONLY_MAX:
            continue
        if day not in unstaged or in_bed > unstaged[day].in_bed_min:
            unstaged[day] = Night(day, bed, up, 0, in_bed, staged=False)
    best.update(unstaged)
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
    """"asleep 7 h 12 min · in bed 7 h 40 min · 11:48 PM to 7:28 AM · core 240 min · ... · resting HR 52 bpm";
    with no stages, "in bed 7 h 40 min · 11:48 PM to 7:28 AM · no sleep stages: time in bed, not measured sleep · ..."."""
    order = [s for s in STAGE_ORDER if s in night.stage_min] + sorted(set(night.stage_min) - set(STAGE_ORDER))
    parts = [f"asleep {hm(night.asleep_min)}"] if night.staged else []
    parts += [f"in bed {hm(night.in_bed_min)}", f"{_clock(night.bed, tz)} to {_clock(night.wake, tz)}"]
    parts += [f"{'REM' if s == 'rem' else s} {night.stage_min[s]} min" for s in order] if night.staged else [NO_STAGES]
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
        "title": f"Sleep {hm(night.asleep_min)}" if night.staged else f"In bed {hm(night.in_bed_min)} (no sleep stages)",
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
    """The numbers in a line `summary_line` wrote. None when it is not one, and for a night with no stages
    (no "asleep" part): time in bed is not a measurement of sleep."""
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
