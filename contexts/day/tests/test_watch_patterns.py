"""Patterns from the watch's history: counted statements with their date span, or nothing when there is too little."""

import csv
import gzip
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml

from contexts.connectors.domain.health import Night
from contexts.day.application.watch_history import export_patterns
from contexts.day.domain.clock import DayClock
from contexts.day.domain.watch_patterns import MIN_N, watch_patterns

CLOCK = DayClock()
NY = CLOCK.tz
FIRST = date(2026, 3, 2)  # a Monday


def night(n: int, asleep_min: int, bed: str = "23:30") -> Night:
    """The night that ended `n` days after FIRST, begun at `bed` the evening before (or that morning, after midnight)."""
    day = FIRST + timedelta(days=n)
    h, m = (int(x) for x in bed.split(":"))
    start = datetime(day.year, day.month, day.day, h, m, tzinfo=NY) - timedelta(days=1 if h >= 12 else 0)
    return Night(day, start, start + timedelta(minutes=asleep_min + 10), asleep_min, asleep_min + 10)


def test_each_statement_is_counted_and_dated():
    # four weeks: 6 h before the Tuesday and Thursday workouts, 7.5 h otherwise; in bed late on the nights into Sunday
    nights = [night(n, 360 if n % 7 in (1, 3) else 450, "02:00" if n % 7 == 6 else "23:30") for n in range(28)]
    nights += [night(28 + n, 300 if n % 2 else 480) for n in range(28)]  # four more weeks, no workouts
    workouts = {FIRST + timedelta(days=n) for n in range(28) if n % 7 in (1, 3)}
    resting = {n.day: 50.0 if n.asleep_min < 360 else 44.0 for n in nights}
    out = watch_patterns(nights, workouts, resting, NY)
    assert [p["text"] for p in out] == [
        "On the 8 nights before a workout day, median sleep was 6 h 0 min, against 7 h 30 min on the other 48 "
        "(watch, 2026-03-02 to 2026-04-26).",
        "Resting heart rate after the 14 nights under 6 h had a median of 50 bpm, against 44 bpm after the 34 nights over 7 h "
        "(watch, 2026-03-02 to 2026-04-26).",
        "Bedtime was latest on nights into Sunday (median 12:45 AM, 8 nights) and earliest on nights into Monday (11:30 PM, 8 nights): "
        "1 h 15 min apart (watch, 2026-03-02 to 2026-04-26).",
    ]
    assert [p["n"] for p in out] == [56, 48, 56]
    assert all((p["from"], p["to"], p["source"]) == ("2026-03-02", "2026-04-26", "watch") for p in out)


def test_a_statement_with_too_few_nights_on_either_side_is_not_made():
    nights = [night(n, 330 if n < 7 else 450) for n in range(21)]  # three weeks: 7 short nights, 3 nights per weekday
    workouts = {n.day for n in nights[:7]}
    assert MIN_N == 8 and watch_patterns(nights, workouts, {n.day: 50.0 for n in nights}, NY) == []
    one_more = nights + [night(21, 330)]
    [only] = watch_patterns(one_more, workouts | {one_more[-1].day}, {}, NY)  # 8 against 14; no resting heart rate at all
    assert only["text"].startswith("On the 8 nights before a workout day, median sleep was 5 h 30 min, against 7 h 30 min on the other 14")
    assert watch_patterns(nights[:15], workouts, {}, NY) == []  # under sixteen nights altogether: nothing


# --- read from an export, when the health connector is configured to read one ---


def pg(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00")


def write(folder: Path, table: str, columns: list[str], rows: list[dict]) -> None:
    with gzip.open(folder / f"{table}.csv.gz", "wt", encoding="utf-8", newline="") as f:
        out = csv.DictWriter(f, fieldnames=columns, restval="")
        out.writeheader()
        out.writerows(rows)


def config(tmp_path: Path, **options) -> Path:
    folder = tmp_path / "config"
    folder.mkdir(exist_ok=True)
    (folder / "connectors.yaml").write_text(yaml.safe_dump({"health": {"module": "contexts.connectors.servers.health", "options": options}}))
    return folder


def test_the_export_is_read_only_when_the_health_connector_names_one(tmp_path):
    export = tmp_path / "watch"
    export.mkdir()
    made = [night(n, 360 if n % 2 else 450) for n in range(20)]
    write(export, "category_samples", ["id", "type", "value_label", "start_date", "end_date", "source"],
          [{"id": str(i), "type": "HKCategoryTypeIdentifierSleepAnalysis", "value_label": "Asleep Core", "start_date": pg(n.bed),
            "end_date": pg(n.bed + timedelta(minutes=n.asleep_min)), "source": "Dominick’s Apple Watch"} for i, n in enumerate(made)]
          + [{"id": "phone", "type": "HKCategoryTypeIdentifierSleepAnalysis", "value_label": "In Bed", "start_date": pg(made[0].bed - timedelta(days=30)),
              "end_date": pg(made[0].bed - timedelta(days=30) + timedelta(hours=8)), "source": "Dominick Dupuy"}])
    write(export, "workouts", ["id", "name", "start_time", "end_time", "duration_sec", "raw_json"],
          [{"id": str(i), "name": "Running", "start_time": pg(n.wake + timedelta(hours=2)), "end_time": pg(n.wake + timedelta(hours=3)),
            "duration_sec": 3600, "raw_json": "{}"} for i, n in enumerate(made) if n.asleep_min == 360])
    write(export, "health_metrics", ["time", "metric_name", "source", "qty", "avg_val"],
          [{"time": pg(n.wake), "metric_name": "resting_heart_rate", "qty": 48 if n.asleep_min == 360 else 45} for n in made])

    assert export_patterns(config(tmp_path, export_dir=None), CLOCK) == []
    assert export_patterns(tmp_path / "no-config-here", CLOCK) == []
    out = export_patterns(config(tmp_path, export_dir=str(export), export_rebase=True), CLOCK)  # the rebase does not move history
    assert out[0]["text"] == ("On the 10 nights before a workout day, median sleep was 6 h 0 min, against 7 h 30 min on the other 10 "
                              "(watch, 2026-03-02 to 2026-03-21).")  # the phone's In Bed row a month earlier is not a night
    assert len(out) == 1  # no night was under 6 h, and three nights per weekday say nothing about weekdays
    assert export_patterns(config(tmp_path, export_dir=str(tmp_path / "gone")), CLOCK) == []
