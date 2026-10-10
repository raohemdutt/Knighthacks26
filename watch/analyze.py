"""Analysis helpers over the loaded rows. Pure functions on iterables of dicts; no plotting.

Things to know about this data (learned building the atlas-health pipeline):
- heart_rate rows carry min/avg/max and a NULL qty, so value() falls back to avg_val.
- The watch is worn for runs and sleep only, so all-day totals (steps, calories, stand) undercount.
- source is "" for companion-app rows; watch rows name the watch.
- sleep_sessions fields are hours; stages exist only from 2026-01-26, earlier nights only have in_bed.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/New_York")

# How a day of each metric collapses: cumulative metrics add up, the rest average.
SUM_METRICS = {"step_count", "distance_walking_running", "active_energy", "basal_energy_burned",
               "flights_climbed", "apple_exercise_time", "apple_stand_time", "distance_swimming",
               "swimming_stroke_count", "distance_cycling", "distance_downhill_snow_sports"}

AGGS: dict[str, Callable[[list[float]], float]] = {
    "mean": statistics.fmean, "sum": sum, "min": min, "max": max, "median": statistics.median,
    "count": len,
}


def value(row: dict) -> float | None:
    """COALESCE(qty, avg_val): the one number a health_metrics row stands for."""
    return row["qty"] if row.get("qty") is not None else row.get("avg_val")


def local_day(ts: datetime, tz: ZoneInfo = TZ) -> date:
    return ts.astimezone(tz).date()


def default_agg(metric_name: str) -> str:
    return "sum" if metric_name in SUM_METRICS else "mean"


def inventory(metric_rows: Iterable[dict]) -> list[dict]:
    """One line per metric: rows, units, first and last day, days with data."""
    acc: dict[str, dict] = {}
    for r in metric_rows:
        a = acc.get(r["metric_name"])
        if a is None:
            a = acc[r["metric_name"]] = {"metric": r["metric_name"], "rows": 0, "units": r["units"],
                                         "first": r["time"], "last": r["time"], "days": set()}
        a["rows"] += 1
        a["first"] = min(a["first"], r["time"])
        a["last"] = max(a["last"], r["time"])
        a["days"].add(local_day(r["time"]))
    return [{"metric": a["metric"], "rows": a["rows"], "units": a["units"],
             "first": a["first"].date().isoformat(), "last": a["last"].date().isoformat(),
             "days_with_data": len(a["days"])}
            for a in sorted(acc.values(), key=lambda a: -a["rows"])]


def daily(metric_rows: Iterable[dict], agg: str = "mean", tz: ZoneInfo = TZ) -> dict[date, float]:
    """Collapse a metric to one number per local day."""
    fn = AGGS[agg]
    by_day: dict[date, list[float]] = defaultdict(list)
    for r in metric_rows:
        v = value(r)
        if v is not None:
            by_day[local_day(r["time"], tz)].append(v)
    return {d: fn(vs) for d, vs in sorted(by_day.items())}


def rolling_mean(series: dict[date, float], days: int = 7, min_points: int = 1) -> dict[date, float]:
    """Mean over the trailing `days` calendar days (not the trailing N points, so gaps stay gaps)."""
    keys = sorted(series)
    out = {}
    for d in keys:
        win = [series[k] for k in keys if d - timedelta(days=days - 1) <= k <= d]
        if len(win) >= min_points:
            out[d] = statistics.fmean(win)
    return out


def weekly(series: dict[date, float], agg: str = "mean") -> dict[date, float]:
    """Collapse a daily series to ISO weeks, keyed by the Monday."""
    weeks: dict[date, list[float]] = defaultdict(list)
    for d, v in series.items():
        weeks[d - timedelta(days=d.weekday())].append(v)
    return {w: AGGS[agg](vs) for w, vs in sorted(weeks.items())}


def window(series: dict[date, float], days: int, end: date | None = None) -> dict[date, float]:
    """The last `days` calendar days of a series, ending at `end` (default: its last day)."""
    end = end or max(series, default=date.today())
    return {d: v for d, v in series.items() if end - timedelta(days=days - 1) <= d <= end}


def describe(values: Iterable[float]) -> dict:
    vs = sorted(values)
    if not vs:
        return {"n": 0}
    q = statistics.quantiles(vs, n=4, method="inclusive") if len(vs) > 1 else [vs[0]] * 3
    return {"n": len(vs), "mean": statistics.fmean(vs), "sd": statistics.stdev(vs) if len(vs) > 1 else 0.0,
            "min": vs[0], "p25": q[0], "median": q[1], "p75": q[2], "max": vs[-1]}


def trend(series: dict[date, float]) -> dict:
    """Least-squares slope per 30 days, with the span it was fitted over."""
    if len(series) < 2:
        return {"n": len(series)}
    d0 = min(series)
    xs = [(d - d0).days for d in series]
    slope, intercept = statistics.linear_regression(xs, list(series.values()))
    return {"n": len(xs), "from": d0.isoformat(), "to": max(series).isoformat(),
            "slope_per_30d": slope * 30, "start_fit": intercept, "end_fit": intercept + slope * max(xs)}


# --- sleep ---------------------------------------------------------------------------------

def sleep_nights(sessions: Iterable[dict], staged_only: bool = False, tz: ZoneInfo = TZ) -> list[dict]:
    """One dict per night: hours asleep (in-bed hours for nights recorded before staging), stages,
    efficiency, local bed and wake times. staged_only keeps nights with core/deep/REM recorded."""
    out = []
    for s in sessions:
        staged = bool((s["core"] or 0) + (s["deep"] or 0) + (s["rem"] or 0))
        if staged_only and not staged:
            continue
        asleep = s["total_sleep"] or s["asleep"] or 0.0
        start = s["sleep_start"] or s["in_bed_start"]
        end = s["sleep_end"] or s["in_bed_end"]
        out.append({
            "date": s["date"], "hours": asleep or (s["in_bed"] or 0.0), "staged": staged,
            "core": s["core"], "deep": s["deep"], "rem": s["rem"], "in_bed": s["in_bed"],
            "efficiency": asleep / s["in_bed"] if asleep and s["in_bed"] else None,
            "bed": start.astimezone(tz).time() if start else None,
            "wake": end.astimezone(tz).time() if end else None,
        })
    return out


def _hours_after(t: time, pivot: int) -> float:
    """Hours since `pivot` o'clock, so clock times that straddle midnight average sensibly."""
    return (t.hour + t.minute / 60 - pivot) % 24


def _clock(h: float, pivot: int) -> str:
    minutes = round(((h + pivot) % 24) * 60) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def sleep_summary(nights: list[dict], goal_h: float = 8.0) -> dict:
    hours = [n["hours"] for n in nights if n["hours"]]
    staged = [n for n in nights if n["staged"] and n["hours"]]
    beds = [_hours_after(n["bed"], 12) for n in nights if n["bed"]]
    wakes = [_hours_after(n["wake"], 0) for n in nights if n["wake"]]
    return {
        "nights": len(nights),
        "hours": describe(hours),
        "debt_vs_goal_h": sum(max(goal_h - h, 0) for h in hours),
        "share_meeting_goal": sum(h >= goal_h for h in hours) / len(hours) if hours else None,
        "stage_share": {k: statistics.fmean([(n[k] or 0) / n["hours"] for n in staged])
                        for k in ("core", "deep", "rem")} if staged else None,
        "mean_bedtime": _clock(statistics.fmean(beds), 12) if beds else None,
        "bedtime_sd_min": statistics.stdev(beds) * 60 if len(beds) > 1 else None,
        "mean_wake": _clock(statistics.fmean(wakes), 0) if wakes else None,
    }


# --- workouts ------------------------------------------------------------------------------

def _km(w: dict) -> float | None:
    if not w["distance"]:
        return None
    return w["distance"] / 1000 if w["distance_units"] == "m" else w["distance"]


def pace_min_per_km(w: dict) -> float | None:
    km = _km(w)
    return (w["duration_sec"] / 60) / km if km and w["duration_sec"] else None


def workout_summary(workouts: Iterable[dict]) -> list[dict]:
    """Per workout type: count, hours, km, kcal, mean of the workouts' average HR, last date."""
    acc: dict[str, dict] = {}
    for w in workouts:
        a = acc.get(w["name"])
        if a is None:
            a = acc[w["name"]] = {"type": w["name"], "count": 0, "hours": 0.0, "km": 0.0,
                                  "kcal": 0.0, "hr": [], "last": w["start_time"]}
        a["count"] += 1
        a["hours"] += (w["duration_sec"] or 0) / 3600
        a["km"] += _km(w) or 0
        a["kcal"] += w["active_energy_burned"] or 0
        if w["avg_heart_rate"]:
            a["hr"].append(w["avg_heart_rate"])
        a["last"] = max(a["last"], w["start_time"])
    return [{"type": a["type"], "count": a["count"], "hours": a["hours"], "km": a["km"], "kcal": a["kcal"],
             "avg_hr": statistics.fmean(a["hr"]) if a["hr"] else None, "last": a["last"].date().isoformat()}
            for a in sorted(acc.values(), key=lambda a: -a["count"])]


def hr_zones(hr_rows: Iterable[dict], max_hr: float,
             bounds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)) -> dict[str, int]:
    """Minutes in each %-of-max-HR zone from workout_heart_rate (one row per minute, avg_bpm).
    z1 is bounds[0]..bounds[1] of max, z5 is bounds[4] and up; under bounds[0] is 'below'."""
    zones: Counter[str] = Counter()
    for r in hr_rows:
        if r["avg_bpm"] is None:
            continue
        z = sum(r["avg_bpm"] / max_hr >= b for b in bounds)
        zones[f"z{z}" if z else "below"] += 1
    return dict(sorted(zones.items()))
