"""The analysis verbs of `python -m watch`. Each prints JSON to stdout."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from watch import analyze, load


def _dump(obj) -> None:
    print(json.dumps(obj, indent=2, default=lambda o: o.isoformat() if hasattr(o, "isoformat") else str(o)))


def _series_out(series: dict[date, float]) -> dict[str, float]:
    return {d.isoformat(): round(v, 3) for d, v in series.items()}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m watch")
    ap.add_argument("--data", type=Path, default=load.DATA_DIR)
    sub = ap.add_subparsers(dest="verb", required=True)
    sub.add_parser("summary", help="what is in the export, per metric")
    d = sub.add_parser("daily", help="one number per day for a metric, with stats and trend")
    d.add_argument("metric")
    d.add_argument("--agg", choices=sorted(analyze.AGGS), help="default: sum for cumulative metrics, else mean")
    d.add_argument("--days", type=int, help="only the last N calendar days that end at the latest data")
    d.add_argument("--rolling", type=int, help="replace values with an N-day trailing mean")
    d.add_argument("--weekly", action="store_true", help="collapse to ISO weeks (Monday keys)")
    s = sub.add_parser("sleep", help="nightly sleep summary")
    s.add_argument("--days", type=int, default=90)
    s.add_argument("--goal", type=float, default=8.0)
    s.add_argument("--nights", action="store_true", help="also list each night")
    w = sub.add_parser("workouts", help="per workout type; --max-hr adds HR-zone minutes")
    w.add_argument("--max-hr", type=float)
    args = ap.parse_args(argv)

    if args.verb == "summary":
        _dump(analyze.inventory(load.rows("health_metrics", args.data)))
    elif args.verb == "daily":
        series = analyze.daily(load.metric(args.metric, args.data), args.agg or analyze.default_agg(args.metric))
        if not series:
            _dump({"metric": args.metric, "error": "no rows; see `python -m watch summary`"})
            return 1
        if args.days:
            series = analyze.window(series, args.days)
        stats = {"describe": analyze.describe(series.values()), "trend": analyze.trend(series)}
        if args.rolling:
            series = analyze.rolling_mean(series, args.rolling)
        if args.weekly:
            series = analyze.weekly(series)
        _dump({"metric": args.metric, **stats, "series": _series_out(series)})
    elif args.verb == "sleep":
        nights = analyze.sleep_nights(load.rows("sleep_sessions", args.data))
        if nights and args.days:
            end = max(n["date"] for n in nights)
            nights = [n for n in nights if (end - n["date"]).days < args.days]
        out = {"days": args.days, **analyze.sleep_summary(nights, args.goal)}
        if args.nights:
            out["each"] = nights
        _dump(out)
    elif args.verb == "workouts":
        out: dict = {"by_type": analyze.workout_summary(load.rows("workouts", args.data))}
        if args.max_hr:
            out["hr_zone_minutes"] = analyze.hr_zones(load.rows("workout_heart_rate", args.data), args.max_hr)
        _dump(out)
    return 0
