"""python -m lab metrics: each metric's newest reading against the one before it (report), or one metric across a
time window (compare), with a verdict from lab/machine/verdict.py.

    python -m lab metrics report
    python -m lab metrics report --metric teacher.decode_tok_s --metric teacher.joules_per_token --format md
    python -m lab metrics compare teacher.decode_tok_s --before 2026-10-10T12:00:00Z --after 2026-10-10T15:00:00Z
    python -m lab metrics compare pod.idle_watts --before 2026-10-11

Readings. A reading is one stored value of one metric with its unit, node, conditions, source, ts and
low_confidence: a line of lab/refine/metrics.jsonl (read with lab/machine/metrics.py), or one metric of a ledger row
(read with lab/machine/ledger.py and shaped the way ledger.latest_metric() shapes it). Every reading carries the place
it was read from, written the way a pass cites it: metrics.jsonl:<line> or ledger:<seq>. --source picks the stores:
metrics (the file alone), ledger (the ledger alone), or auto, the default: the file, plus the ledger when one exists.
In auto, a ledger reading that copies a file line (same metric, node, value and conditions, and the same ts or the
same source, which is how a backfill copies a line) is counted once, as the file line, with its ledger seq in
also_in.

Series. A reading is only compared with readings measured the same way: a series is one metric on one node, in one
unit, under one set of conditions. Some condition keys describe the number rather than the setup, so they do not
split a series (NOT_CONDITIONS): the recorded spread, how a figure was stated (stated_as, value_is, range_w, approx),
why it is low confidence, a derived figure's inputs, and a note. Within a series, readings are ordered by ts as an
instant; a tie keeps file order, then ledger order, with the ledger after the file.

The recorded spread. A reading that summarizes repeated runs records them in conditions.spread: {"runs": 5,
"sd": 0.03}, sd being the sample standard deviation (the n - 1 divisor), plus "mean" when the value is not the mean
(a median, say), and any of "median", "min" and "max" for whoever reads the row. A ledger row with several metrics
keys it by metric: {"spread": {"teacher.decode_tok_s": {"runs": 5, "sd": 0.03}, ...}}. A reading with no spread is
one run. A spread that cannot be read (runs that are not a whole number of at least 1, several runs without a usable
sd, a flat spread on a row with several metrics) is reported as a warning, and the reading counts as one run.

report. For every series: the latest reading, the reading before it (previous), the change between their values,
and the verdict of the latest (arm B) against the previous (arm A).

compare METRIC. For every series of METRIC: the readings strictly before --before (the before arm, A) against the
readings at or after --after (the after arm, B). Each arm pools the runs of its readings: their count, their mean and
the exact pooled standard deviation. One time given splits the history at that time; readings between two given
times fall in neither arm, and an --after earlier than --before is refused.

Verdicts. verdict.verdict(A, B) reads two measurements with Welch's t test, so an arm needs MIN_RUNS_PER_ARM runs
(five) before anything is called. The verdict is too_early, no_difference, better or worse. Better and worse follow
the metric's direction (DIRECTION by name, else UNIT_DIRECTION by unit); a metric with no better side (a VRAM share
that restates a setting, a credit balance, idle watts, an error that is best at zero) reads higher or lower instead.
The verdict's own fields (statistic, p, df, runs still needed) come along under stats, with A and B as above.

Output. One JSON line by default (the dispatcher's contract); --format md prints a markdown report instead, one table
row per series and a note for each row with both arms. Each answer carries read_at, the UTC time of the reading,
which a later compare can take as --before to see what has moved since. Nothing is written anywhere.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import difflib
import json
import math
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lab.machine import ledger, metrics, verdict

__all__ = [
    "BETTER", "DIRECTION", "FORMATS", "HIGHER", "LOWER", "NOT_CONDITIONS", "NO_DIFFERENCE", "Reading", "SOURCES",
    "SPREAD", "TOO_EARLY", "UNIT_DIRECTION", "VERDICTS", "WORSE",
    "comparable", "compare", "direction_of", "file_readings", "ledger_readings", "load_readings", "main", "merge",
    "parse_time", "pooled", "render_md", "report", "series_of",
]

CLI_VERB = "metrics"   # registers python -m lab metrics
REPO_ROOT = Path(__file__).resolve().parents[2]

HIGHER = "higher"
LOWER = "lower"
BETTER = "better"
WORSE = "worse"
TOO_EARLY = verdict.TOO_EARLY
NO_DIFFERENCE = verdict.NO_DIFFERENCE
#: Every verdict a row can carry: higher and lower stand in for better and worse when a metric has no better side.
VERDICTS = (TOO_EARLY, NO_DIFFERENCE, BETTER, WORSE, HIGHER, LOWER)

SOURCE_AUTO = "auto"
SOURCE_METRICS = "metrics"
SOURCE_LEDGER = "ledger"
SOURCES = (SOURCE_AUTO, SOURCE_METRICS, SOURCE_LEDGER)
FORMATS = ("json", "md")

#: The conditions key a reading records its runs and their spread under.
SPREAD = "spread"
#: Condition keys that describe the number rather than how it was measured, so readings that differ only in them
#: belong to one series.
NOT_CONDITIONS = frozenset({SPREAD, "stated_as", "value_is", "range_w", "approx", "low_confidence_reason", "inputs",
                            "note"})

#: Which way is better, by metric name, from the "Better" columns and metric sets of the track rubrics
#: (lab/refine/<track>/RUBRIC.md). None means the metric has no better side, so it reads higher or lower.
DIRECTION: dict[str, str | None] = {
    "teacher.decode_tok_s": HIGHER,
    "teacher.ttft_s": LOWER,
    "teacher.watts_avg": LOWER,
    "teacher.watts_peak": LOWER,
    "teacher.joules_per_token": LOWER,
    "teacher.engine_init_s": LOWER,
    "teacher.max_concurrency": HIGHER,
    "teacher.output_match_rate": HIGHER,
    "teacher.vram_pct": None,  # restates the --gpu-memory-utilization setting
    "trainer.samples_per_s": HIGHER,
    "trainer.peak_mem_gb": LOWER,
    "trainer.smoke_loss_first": LOWER,
    "trainer.smoke_loss_last": LOWER,
    "pod.idle_watts": None,  # tracked, not a target: a move means a condition changed
    "jetson.idle_watts": None,
    "pod.start_s": LOWER,
    "pod.recovery_s": LOWER,
    "pod.launch_to_serving_s": LOWER,
    "pod.credits_per_min": LOWER,
    "pod.credits_per_hour": LOWER,
    "pod.balance_credits": None,  # a state, not a result
    "link.push_mb_s": HIGHER,
    "link.pull_mb_s": HIGHER,
    "link.failed_call_rate": LOWER,
    "telemetry.joules_per_answer": LOWER,
    "telemetry.sampler_cpu_pct": LOWER,
    "telemetry.sampler_interference_pct": LOWER,
    "telemetry.samples_per_window": HIGHER,
    "telemetry.attribution_error_pct": None,  # best at zero, from either side
    "learning.heldout.current": HIGHER,
    "learning.heldout.candidate": HIGHER,
    "learning.heldout.teacher": HIGHER,
    "learning.label_agreement": HIGHER,
    "learning.loop_minutes": LOWER,
    "learning.package_mb": LOWER,
}

#: The direction of a metric DIRECTION does not name, by its unit. Any other unit ending in "/s" is a rate: higher.
UNIT_DIRECTION: dict[str, str] = {
    "s": LOWER, "ms": LOWER, "min": LOWER,
    "W": LOWER, "J": LOWER, "J/token": LOWER,
    "GiB": LOWER, "GB": LOWER, "MiB": LOWER, "MB": LOWER,
    "loss": LOWER, "credits/min": LOWER, "credits/h": LOWER,
}

_ARM_NAMES = {"report": ("previous", "latest"), "compare": ("before", "after")}
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_REFS_SHOWN = 3  # refs a markdown cell lists per arm before it counts the rest
_CELL_NONE = "-"


# --- readings ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Reading:
    """One stored value of one metric, with where it was read from (ref, store, position) and the runs it stands for:
    runs, mean and sd come from conditions.spread, or are 1, the value and None for a reading with no spread."""

    metric: str
    value: int | float
    unit: str | None
    node: str | None
    conditions: Mapping[str, Any]
    source: str
    ts: str
    instant: dt.datetime
    low_confidence: bool
    ref: str
    store: str
    position: int
    runs: int
    mean: float
    sd: float | None
    spread_problem: str | None = None
    also_in: str | None = None


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _instant(text: Any) -> dt.datetime | None:
    """ISO 8601 text with Z or a UTC offset as an aware time; None for anything else. Handles Z itself, which
    datetime.fromisoformat() before Python 3.11 refuses."""
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if cleaned[-1:] in ("Z", "z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        moment = dt.datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    return moment if moment.utcoffset() is not None else None


def _utc_text(moment: dt.datetime) -> str:
    text = moment.astimezone(dt.timezone.utc).isoformat()
    return text[:-len("+00:00")] + "Z" if text.endswith("+00:00") else text


def _now_text() -> str:
    return _utc_text(dt.datetime.now(dt.timezone.utc).replace(microsecond=0))


def parse_time(text: str) -> dt.datetime:
    """A --before or --after time: ISO 8601 with Z or an offset (2026-10-10T12:00:00Z), or a date (2026-10-10), which
    means midnight UTC. A time with no offset is refused rather than guessed."""
    cleaned = text.strip()
    if _DATE_RE.fullmatch(cleaned):
        try:
            return dt.datetime.combine(dt.date.fromisoformat(cleaned), dt.time(), tzinfo=dt.timezone.utc)
        except ValueError:
            pass
    moment = _instant(cleaned)
    if moment is None:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a time with a UTC offset: give ISO 8601 such as 2026-10-10T12:00:00Z, or a date such "
            "as 2026-10-10 (midnight UTC)")
    return moment


def _spread(conditions: Mapping[str, Any], metric: str, value: int | float,
            metric_count: int) -> tuple[int, float, float | None, str | None]:
    """(runs, mean, sd, problem) of one reading from its conditions.spread. No spread is one run of the value; a
    spread that cannot be read is one run of the value too, with the problem named."""
    one = (1, float(value), None)
    raw = conditions.get(SPREAD)
    if raw is None:
        return (*one, None)
    if not isinstance(raw, Mapping):
        return (*one, f"conditions.spread must be an object, not {type(raw).__name__}")
    if "runs" not in raw and "sd" not in raw:
        raw = raw.get(metric)
        if raw is None:
            return (*one, None)
        if not isinstance(raw, Mapping):
            return (*one, f"conditions.spread[{metric!r}] must be an object, not {type(raw).__name__}")
    elif metric_count > 1:
        return (*one, f"conditions.spread is not keyed by metric on a row with {metric_count} metrics")
    runs = raw.get("runs")
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        return (*one, f"spread runs must be a whole number of at least 1, not {runs!r}")
    mean = raw.get("mean", value)
    if not _finite(mean):
        return (*one, f"spread mean must be a finite number, not {mean!r}")
    if runs == 1:
        return 1, float(mean), None, None
    sd = raw.get("sd")
    if not _finite(sd) or sd < 0:
        return (*one, f"a spread of {runs} runs needs sd, a finite number of at least 0, not {sd!r}")
    return runs, float(mean), float(sd), None


def _line_numbers(path: Path, count: int) -> list[int]:
    """The line numbers of the first `count` rows of the metrics file, counted the way metrics.load() counts them
    (from 1, blank lines skipped but counted)."""
    if count == 0:
        return []
    numbers = [number for number, raw in enumerate(path.read_bytes().split(b"\n"), start=1) if raw.strip()]
    if len(numbers) < count:
        raise metrics.MetricsFormatError(f"{path} lost lines while it was being read; read it again")
    return numbers[:count]


def file_readings(path: Path | str = metrics.DEFAULT_PATH) -> list[Reading]:
    """Every row of the metrics file as a reading, in file order, each with its ref <file name>:<line>. A missing file
    is no readings. metrics.load() refuses a corrupt file (MetricsFormatError) and a lock held too long
    (MetricsLocked)."""
    path = Path(path)
    rows = metrics.load(path)
    out = []
    for position, (row, line) in enumerate(zip(rows, _line_numbers(path, len(rows)))):
        runs, mean, sd, problem = _spread(row["conditions"], row["metric"], row["value"], 1)
        out.append(Reading(
            metric=row["metric"], value=row["value"], unit=row["unit"], node=row["node"], conditions=row["conditions"],
            source=row["source"], ts=row["ts"], instant=_instant(row["ts"]), low_confidence=row["low_confidence"],
            ref=f"{path.name}:{line}", store=SOURCE_METRICS, position=position, runs=runs, mean=mean, sd=sd,
            spread_problem=problem))
    return out


def ledger_readings(state_dir: Path | str | None = None) -> list[Reading]:
    """One reading per metric of every ledger row that carries metrics, in ledger order, each with its ref
    ledger:<seq>. A reading has the fields ledger.latest_metric() gives a measurement: unit from the row's units,
    conditions {} when the row has none, source "ledger seq <n>" when it names none, low_confidence only when true.
    A row whose ts is not an aware time, and a metric whose value is not a finite number, are skipped as
    latest_metric() skips them."""
    out = []
    for row in ledger.read(state_dir=state_dir, metrics=lambda found: isinstance(found, dict) and bool(found)):
        instant = _instant(row.get("ts"))
        if instant is None:
            continue
        seq = row["seq"]
        units = row.get("units") if isinstance(row.get("units"), dict) else {}
        conditions = dict(row["conditions"]) if isinstance(row.get("conditions"), dict) else {}
        names = row["metrics"]
        for name, value in names.items():
            if not _finite(value):
                continue
            runs, mean, sd, problem = _spread(conditions, name, value, len(names))
            out.append(Reading(
                metric=name, value=value, unit=units.get(name), node=row.get("node"), conditions=conditions,
                source=row.get("source") or f"ledger seq {seq}", ts=row["ts"], instant=instant,
                low_confidence=row.get("low_confidence") is True, ref=f"ledger:{seq}", store=SOURCE_LEDGER,
                position=seq, runs=runs, mean=mean, sd=sd, spread_problem=problem))
    return out


def comparable(conditions: Mapping[str, Any]) -> dict[str, Any]:
    """conditions without the keys that describe the number rather than the setup (NOT_CONDITIONS)."""
    return {key: value for key, value in conditions.items() if key not in NOT_CONDITIONS}


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _twin_key(reading: Reading) -> tuple:
    return (reading.metric, reading.node or "", reading.value, _canon(comparable(reading.conditions)))


def merge(from_file: Sequence[Reading], from_ledger: Sequence[Reading]) -> tuple[list[Reading], int]:
    """The file's readings and the ledger's, a ledger reading that copies a file line counted once: (readings,
    copies merged). A copy has the line's metric, node, value and conditions (NOT_CONDITIONS aside) and its ts or its
    source; the file line stays, with the copy's ref in also_in. Each line absorbs at most one copy."""
    lines: dict[tuple, list[int]] = {}
    for index, reading in enumerate(from_file):
        lines.setdefault(_twin_key(reading), []).append(index)
    kept = list(from_file)
    taken: set[int] = set()
    extra = []
    for reading in from_ledger:
        twin = next((index for index in lines.get(_twin_key(reading), ())
                     if index not in taken and (from_file[index].instant == reading.instant
                                                or from_file[index].source == reading.source)), None)
        if twin is None:
            extra.append(reading)
        else:
            taken.add(twin)
            kept[twin] = dataclasses.replace(from_file[twin], also_in=reading.ref)
    return kept + extra, len(taken)


def _shown(path: Path) -> str:
    """A path as the report prints it: relative to the repo root when it is inside it."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_readings(source: str = SOURCE_AUTO, metrics_path: Path | str = metrics.DEFAULT_PATH,
                  state_dir: Path | str | None = None) -> tuple[list[Reading], dict[str, Any], list[str]]:
    """(readings, what was read, warnings about the stores themselves) for a --source: metrics, ledger, or auto (the
    file plus the ledger when it exists, copies merged)."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {', '.join(SOURCES)}, not {source!r}")
    metrics_path = Path(metrics_path)
    info: dict[str, Any] = {"mode": source}
    warnings: list[str] = []
    from_file: list[Reading] = []
    from_ledger: list[Reading] = []
    if source in (SOURCE_AUTO, SOURCE_METRICS):
        from_file = file_readings(metrics_path)
        info["metrics_file"] = {"path": _shown(metrics_path), "present": metrics_path.exists(),
                                "readings": len(from_file)}
        if not metrics_path.exists():
            warnings.append(f"no metrics file at {_shown(metrics_path)}")
    if source in (SOURCE_AUTO, SOURCE_LEDGER):
        ledger_file = ledger.ledger_path(state_dir)
        present = ledger_file.exists()
        from_ledger = ledger_readings(state_dir) if present else []
        info["ledger"] = {"path": _shown(ledger_file), "present": present, "readings": len(from_ledger)}
        if source == SOURCE_LEDGER and not present:
            warnings.append(f"no ledger at {_shown(ledger_file)}")
    readings, copies = merge(from_file, from_ledger)
    info["copies_merged"] = copies
    info["readings"] = len(readings)
    return readings, info, warnings


def reading_warnings(readings: Iterable[Reading]) -> list[str]:
    """What a reader should know about these readings: a spread that could not be read (the reading counted as one
    run), and a metric stored with no unit (a series of its own, apart from the same metric with one)."""
    warnings = []
    no_unit: dict[str, list[str]] = {}
    for reading in readings:
        if reading.spread_problem:
            warnings.append(f"{reading.ref} {reading.metric}: {reading.spread_problem}; read as one run")
        if reading.unit is None:
            no_unit.setdefault(reading.metric, []).append(reading.ref)
    for metric, refs in no_unit.items():
        warnings.append(f"{metric} has no unit at {_listed(refs)}, so those readings are a series of their own")
    return warnings


def _listed(refs: Sequence[str]) -> str:
    shown = ", ".join(refs[:_REFS_SHOWN])
    return shown + (f" and {len(refs) - _REFS_SHOWN} more" if len(refs) > _REFS_SHOWN else "")


# --- series, arms and verdicts ----------------------------------------------------------------------------------------

def _series_key(reading: Reading) -> tuple[str, str, str, str]:
    return (reading.metric, reading.node or "", reading.unit or "", _canon(comparable(reading.conditions)))


def series_of(readings: Iterable[Reading]) -> list[list[Reading]]:
    """The readings grouped into series (metric, node, unit, comparable conditions), series sorted by those, each
    series oldest first: by ts as an instant, then the file before the ledger, then file or ledger order."""
    groups: dict[tuple[str, str, str, str], list[Reading]] = {}
    for reading in readings:
        groups.setdefault(_series_key(reading), []).append(reading)
    return [sorted(groups[key], key=lambda r: (r.instant, r.store != SOURCE_METRICS, r.position))
            for key in sorted(groups)]


def pooled(readings: Sequence[Reading]) -> verdict.Samples:
    """One arm: the runs of every reading taken together. A reading brings its recorded runs, mean and sd, or one run
    of its value; the arm's sd is the exact pooled sample sd (within-reading plus between-reading sums of squares,
    over n - 1)."""
    if len(readings) == 1:
        only = readings[0]
        return verdict.Samples(only.runs, only.mean, only.sd)
    n = sum(reading.runs for reading in readings)
    if n == 0:
        return verdict.Samples(0, None, None)
    mean = math.fsum(reading.runs * reading.mean for reading in readings) / n
    if n == 1:
        return verdict.Samples(1, mean, None)
    within = math.fsum((reading.runs - 1) * reading.sd ** 2 for reading in readings if reading.sd is not None)
    between = math.fsum(reading.runs * (reading.mean - mean) ** 2 for reading in readings)
    return verdict.Samples(n, mean, math.sqrt((within + between) / (n - 1)))


def direction_of(metric: str, unit: str | None) -> str | None:
    """HIGHER or LOWER when one side of the metric is better, None when neither is: DIRECTION by name first, then the
    unit (UNIT_DIRECTION, or any unit ending in "/s", a rate)."""
    if metric in DIRECTION:
        return DIRECTION[metric]
    if unit is None:
        return None
    if unit in UNIT_DIRECTION:
        return UNIT_DIRECTION[unit]
    return HIGHER if unit.endswith("/s") else None


def _word(label: str, direction: str | None) -> str:
    if label == verdict.B_AHEAD:
        return BETTER if direction else HIGHER
    if label == verdict.A_AHEAD:
        return WORSE if direction else LOWER
    return label


def _tidy(number: float | None) -> float | None:
    """A computed number without float noise (12 significant digits): 7.8 - 7.7 reads 0.1."""
    return None if number is None else float(f"{number:.12g}")


def _arm(readings: Sequence[Reading], samples: verdict.Samples) -> dict[str, Any] | None:
    if not readings:
        return None
    single = len(readings) == 1
    sd = readings[0].sd if single else _tidy(samples.sd)
    return {
        "value": readings[0].value if single else _tidy(samples.mean),
        "runs": samples.n,
        "sd": sd,
        "readings": len(readings),
        "ts": readings[-1].ts,
        "refs": [reading.ref for reading in readings],
        "also_in": [reading.also_in for reading in readings if reading.also_in],
        "low_confidence": any(reading.low_confidence for reading in readings),
    }


def _row(series: Sequence[Reading], arm_a: Sequence[Reading], arm_b: Sequence[Reading], command: str,
         missing: dict[str, str]) -> dict[str, Any]:
    """One series read as A against B. `missing` holds the note for an empty arm: keys "a", "b" and "both"."""
    newest = series[-1]
    direction = direction_of(newest.metric, newest.unit)
    samples_a, samples_b = pooled(arm_a), pooled(arm_b)
    read = verdict.verdict(samples_a, samples_b, higher_is_better=direction != LOWER)
    summary_a, summary_b = _arm(arm_a, samples_a), _arm(arm_b, samples_b)
    change = change_pct = None
    if summary_a and summary_b:
        difference = summary_b["value"] - summary_a["value"]
        change = _tidy(difference)
        if summary_a["value"] != 0:
            change_pct = _tidy(100.0 * difference / abs(summary_a["value"]))
    if not arm_a and not arm_b:
        note = missing["both"]
    elif not arm_a:
        note = missing["a"]
    elif not arm_b:
        note = missing["b"]
    elif direction is None and read.label in (verdict.A_AHEAD, verdict.B_AHEAD):
        note = f"no side of this metric is better, so this reads higher or lower: {read.note}"
    else:
        note = read.note
    flagged = [reading.ref for reading in (*arm_a, *arm_b) if reading.low_confidence]
    if flagged:
        note += f"; low confidence: {_listed(flagged)}"
    name_a, name_b = _ARM_NAMES[command]
    return {
        "metric": newest.metric, "node": newest.node, "unit": newest.unit,
        "conditions": comparable(newest.conditions), "direction": direction,
        name_a: summary_a, name_b: summary_b, "change": change, "change_pct": change_pct,
        "verdict": _word(read.label, direction), "note": note, "low_confidence": bool(flagged),
        "readings": len(series), "stats": read.as_dict(),
    }


def _matches(reading: Reading, names: set[str], node: str | None, where: Mapping[str, Any]) -> bool:
    if names and reading.metric not in names:
        return False
    if node is not None and reading.node != node:
        return False
    return all(key in reading.conditions and reading.conditions[key] == wanted for key, wanted in where.items())


def _filters(names: Sequence[str], node: str | None, where: Mapping[str, Any]) -> dict[str, Any]:
    return {"metric": list(names), "node": node, "where": dict(where)}


def report(*, metric_names: Sequence[str] = (), node: str | None = None, where: Mapping[str, Any] | None = None,
           source: str = SOURCE_AUTO, metrics_path: Path | str = metrics.DEFAULT_PATH,
           state_dir: Path | str | None = None) -> dict[str, Any]:
    """Every series (narrowed by metric names, node and where, a dict every reading's conditions must contain): its
    latest reading against the previous one. Asked-for metrics with no reading are listed under no_readings."""
    where = dict(where or {})
    names = list(dict.fromkeys(metric_names))
    read_at = _now_text()
    readings, info, warnings = load_readings(source, metrics_path, state_dir)
    chosen = [reading for reading in readings if _matches(reading, set(names), node, where)]
    missing = {"a": "one reading only: nothing earlier to compare it with", "b": "", "both": ""}
    rows = [_row(series, series[-2:-1], series[-1:], "report", missing) for series in series_of(chosen)]
    found = {reading.metric for reading in chosen}
    return {
        "ok": True, "outcome": "report", "command": "report", "read_at": read_at, "source": info,
        "filters": _filters(names, node, where), "count": len(rows), "rows": rows,
        "no_readings": [name for name in names if name not in found],
        "warnings": warnings + reading_warnings(chosen),
    }


def compare(metric: str, *, before: dt.datetime | None = None, after: dt.datetime | None = None,
            node: str | None = None, where: Mapping[str, Any] | None = None, source: str = SOURCE_AUTO,
            metrics_path: Path | str = metrics.DEFAULT_PATH, state_dir: Path | str | None = None) -> dict[str, Any]:
    """Every series of `metric`: the readings strictly before `before` against the readings at or after `after`,
    each arm pooled. One of the two times stands for both. ValueError when neither is given, when one has no UTC
    offset, or when after is earlier than before. A metric with no matching reading answers ok false, outcome
    no_readings, with the closest metric names on record."""
    if before is None and after is None:
        raise ValueError("compare needs --before, --after or both")
    before = after if before is None else before
    after = before if after is None else after
    for name, moment in (("before", before), ("after", after)):
        if moment.utcoffset() is None:
            raise ValueError(f"{name} must carry a UTC offset, not the naive {moment.isoformat()}")
    if after < before:
        raise ValueError(f"--after {_utc_text(after)} is earlier than --before {_utc_text(before)}, so the two arms "
                         "would share readings")
    where = dict(where or {})
    read_at = _now_text()
    readings, info, warnings = load_readings(source, metrics_path, state_dir)
    chosen = [reading for reading in readings if _matches(reading, {metric}, node, where)]
    warnings += reading_warnings(chosen)
    window = {"before": _utc_text(before), "after": _utc_text(after)}
    if not chosen:
        known = sorted({reading.metric for reading in readings})
        narrowed = " that match --node and --where" if node is not None or where else ""
        return {
            "ok": False, "outcome": "no_readings", "command": "compare", "read_at": read_at, "metric": metric,
            "error": f"no readings of {metric}{narrowed} in {info['readings']} readings",
            "close": difflib.get_close_matches(metric, known, n=3), "window": window, "source": info,
            "filters": _filters([metric], node, where), "warnings": warnings,
        }
    missing = {
        "a": f"no reading before {window['before']}",
        "b": f"no reading at or after {window['after']}",
        "both": f"every reading falls between {window['before']} and {window['after']}, so neither arm has one",
    }
    rows = []
    for series in series_of(chosen):
        arm_a = [reading for reading in series if reading.instant < before]
        arm_b = [reading for reading in series if reading.instant >= after]
        rows.append(_row(series, arm_a, arm_b, "compare", missing))
    return {
        "ok": True, "outcome": "compare", "command": "compare", "read_at": read_at, "metric": metric, "window": window,
        "source": info,
        "filters": _filters([metric], node, where), "count": len(rows), "rows": rows, "warnings": warnings,
    }


# --- markdown ---------------------------------------------------------------------------------------------------------

def _cell(text: Any) -> str:
    """One table cell: a pipe escaped, newlines and runs of spaces flattened, nothing shown as a dash."""
    if text is None or text == "":
        return _CELL_NONE
    return " ".join(str(text).replace("|", "\\|").split())


def _num(value: Any) -> str:
    """A number as a cell shows it: an int as stored, a float to six significant digits, never in e-notation at a
    million or more."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int):
        return str(value)
    text = f"{value:.6g}"
    return f"{value:.0f}" if "e" in text and abs(value) >= 1 else text


def _conditions_text(conditions: Mapping[str, Any]) -> str:
    parts = []
    for key, value in conditions.items():
        shown = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        parts.append(f"{key}={shown}")
    return ", ".join(parts)


def _arm_text(arm: Mapping[str, Any] | None, unit: str | None) -> str:
    if arm is None:
        return _CELL_NONE
    text = _num(arm["value"]) + (f" {unit}" if unit else "")
    runs, sd = arm["runs"], arm["sd"]
    text += ", 1 run" if runs == 1 else f", {runs} runs, sd {_num(sd) if sd is not None else 'unknown'}"
    if arm["readings"] > 1:
        text += f" ({arm['readings']} readings)"
    if arm["low_confidence"]:
        text += ", low confidence"
    return text


def _change_text(row: Mapping[str, Any]) -> str:
    if row["change"] is None:
        return _CELL_NONE
    sign = "+" if row["change"] > 0 else ""
    text = f"{sign}{_num(row['change'])}" + (f" {row['unit']}" if row["unit"] else "")
    if row["change_pct"] is not None:
        text += f" ({sign}{row['change_pct']:.1f}%)"
    return text


def _refs_text(arm: Mapping[str, Any] | None) -> str:
    if arm is None:
        return "none"
    return _listed(arm["refs"])


def _source_text(doc: Mapping[str, Any]) -> str:
    info = doc["source"]
    read, absent = [], []
    for key, label in (("metrics_file", "metrics file"), ("ledger", "ledger")):
        if key not in info:
            continue
        found = info[key]
        if found["present"]:
            read.append(f"{found['path']} ({found['readings']} readings)")
        else:
            absent.append(f"no {label} at {found['path']}")
    text = f"Read at {doc['read_at']} from " + (" and ".join(read) if read else "nothing")
    text += f", --source {info['mode']}" + "".join(f"; {item}" for item in absent) + "."
    if info.get("copies_merged"):
        text += f" {info['copies_merged']} ledger copies of metrics-file lines were counted once."
    return text


def render_md(doc: Mapping[str, Any]) -> str:
    """The report or comparison as markdown: what was read, how to read it, one table row per series, then, for each
    row with both arms, a note with the verdict's statistic and the runs still needed. A document that is not ok
    shows its error instead of a table."""
    command = doc["command"]
    name_a, name_b = _ARM_NAMES[command]
    if command == "report":
        lines = ["# Metrics report", "", _source_text(doc), "",
                 "One row per series: a metric on one node, in one unit, under one set of conditions. Latest is the "
                 "newest reading (B), previous the one before it (A); a dash is no reading."]
    else:
        window = doc["window"]
        lines = [f"# Metrics compare: {doc['metric']}", "", _source_text(doc), "",
                 f"Before (A) pools the readings before {window['before']}; after (B) pools the readings at or after "
                 f"{window['after']}. One row per series of the metric; a dash is no reading."]
    lines[-1] += (f" A verdict needs {verdict.MIN_RUNS_PER_ARM} runs an arm (Welch's t test in "
                  "lab/machine/verdict.py); a reading counts the runs in its conditions.spread, or one.")
    lines.append("")
    for warning in doc.get("warnings") or []:
        lines += [f"Warning: {warning}", ""]
    if not doc.get("ok", True):
        lines += [f"Nothing to show: {doc.get('error', doc.get('outcome'))}.", ""]
        return "\n".join(lines)
    rows = doc["rows"]
    if rows:
        lines += [f"| # | metric | node | conditions | {name_a} | {name_b} | change | verdict | read from |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for number, row in enumerate(rows, start=1):
            cells = [str(number), row["metric"], row["node"], _conditions_text(row["conditions"]),
                     _arm_text(row[name_a], row["unit"]), _arm_text(row[name_b], row["unit"]), _change_text(row),
                     row["verdict"].replace("_", " "), f"{_refs_text(row[name_a])} -> {_refs_text(row[name_b])}"]
            lines.append("| " + " | ".join(_cell(cell) for cell in cells) + " |")
        notes = [f"{number}. {row['metric']} on {row['node'] or 'no node'}: {row['note']}"
                 for number, row in enumerate(rows, start=1) if row[name_a] and row[name_b]]
        if notes:
            lines += ["", "Notes (rows with both readings):", *notes]
    else:
        lines.append("No readings match.")
    if doc.get("no_readings"):
        lines += ["", "No readings, so no baseline yet: " + ", ".join(doc["no_readings"]) + "."]
    lines.append("")
    return "\n".join(lines)


# --- the verb ---------------------------------------------------------------------------------------------------------

def _where_arg(text: str) -> tuple[str, Any]:
    key, sep, value = text.partition("=")
    if not sep or not key.strip():
        raise argparse.ArgumentTypeError(f"{text!r} is not KEY=VALUE")
    try:
        parsed = json.loads(value)
    except ValueError:
        parsed = value
    return key.strip(), parsed


def _common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--node", default=None, help="only this node's readings (pod-a, pod-b, jetson)")
    parser.add_argument("--where", action="append", default=[], type=_where_arg, metavar="KEY=VALUE",
                        help="only readings whose conditions hold KEY with this value; VALUE is read as JSON when it "
                             "parses (batch=1 is the number 1), else as text; repeatable")
    parser.add_argument("--source", choices=SOURCES, default=SOURCE_AUTO,
                        help="metrics: the metrics file; ledger: the ledger; auto (default): the file plus the "
                             "ledger when one exists")
    parser.add_argument("--format", choices=FORMATS, default="json",
                        help="json (default): one JSON line; md: a markdown report")
    parser.add_argument("--metrics-file", type=Path, default=metrics.DEFAULT_PATH, metavar="PATH",
                        help="the metrics file (default lab/refine/metrics.jsonl)")
    parser.add_argument("--state-dir", type=Path, default=None, metavar="DIR",
                        help="the state dir holding ledger.jsonl (default $LAB_STATE_DIR, else lab/state)")


def build_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    """(the metrics parser, its compare sub-parser)."""
    parser = argparse.ArgumentParser(prog="python -m lab metrics", description=main.__doc__)
    commands = parser.add_subparsers(dest="command", required=True, metavar="{report,compare}")
    report_parser = commands.add_parser(
        "report", help="every series: the latest reading against the previous one",
        description="Every series: the latest reading, the previous one, the change and a verdict.")
    report_parser.add_argument("--metric", action="append", default=[], metavar="NAME",
                               help="only this metric (repeatable); one with no reading is listed as having none")
    _common_args(report_parser)
    compare_parser = commands.add_parser(
        "compare", help="one metric: the readings before a time against the readings after one",
        description="Every series of METRIC: readings before --before against readings at or after --after.")
    compare_parser.add_argument("metric", metavar="METRIC")
    compare_parser.add_argument("--before", type=parse_time, default=None, metavar="TS",
                                help="the before arm is every reading strictly before this time (default: --after); "
                                     "ISO 8601 with Z or an offset, or a date meaning midnight UTC")
    compare_parser.add_argument("--after", type=parse_time, default=None, metavar="TS",
                                help="the after arm is every reading at or after this time (default: --before)")
    _common_args(compare_parser)
    return parser, compare_parser


def _print(text: str) -> None:
    """Print text whatever the console's encoding: a character it cannot show becomes a replacement mark."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except UnicodeEncodeError:
        text = text.encode(encoding, "replace").decode(encoding)
    sys.stdout.write(text)
    sys.stdout.flush()


def main(argv: Sequence[str]) -> dict[str, Any] | int:
    """Report each metric's latest reading against its previous one, or compare a metric across a time window."""
    parser, compare_parser = build_parser()
    args = parser.parse_args(list(argv))
    where = dict(args.where)
    common = {"node": args.node, "where": where, "source": args.source, "metrics_path": args.metrics_file,
              "state_dir": args.state_dir}
    if args.command == "compare":
        if args.before is None and args.after is None:
            compare_parser.error("give --before, --after or both")
        if args.before is not None and args.after is not None and args.after < args.before:
            compare_parser.error("--after is earlier than --before, so the two arms would share readings")
    try:
        if args.command == "report":
            doc = report(metric_names=args.metric, **common)
        else:
            doc = compare(args.metric, before=args.before, after=args.after, **common)
    except (metrics.MetricsFormatError, metrics.MetricsLocked) as exc:
        return {"ok": False, "outcome": "error", "command": args.command, "error": f"{type(exc).__name__}: {exc}"}
    if args.format == "md" and doc.get("ok", True):
        _print(render_md(doc))
        return 0
    return doc
