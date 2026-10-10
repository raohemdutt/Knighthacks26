"""watch/: the Apple Watch export loader and analysis helpers."""

from __future__ import annotations

import gzip
from datetime import date, datetime, time, timezone

import pytest

from watch import analyze, load, pull


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def hm(when: str, name: str, qty=None, avg=None) -> dict:
    return {"time": ts(when), "metric_name": name, "source": "", "units": "x", "qty": qty, "avg_val": avg}


def test_parse_row_types():
    row = load.parse_row({"time": "2017-11-09 10:00:00+00", "metric_name": "step_count", "qty": "14.5",
                          "avg_val": "", "date": "2020-12-16", "is_indoor": "f", "source": ""})
    assert row["time"] == ts("2017-11-09T10:00:00")
    assert row["qty"] == 14.5 and row["avg_val"] is None
    assert row["date"] == date(2020, 12, 16) and row["is_indoor"] is False and row["source"] == ""


def test_value_falls_back_to_avg_for_heart_rate():
    assert analyze.value(hm("2026-01-01T12:00", "heart_rate", avg=61.0)) == 61.0
    assert analyze.value(hm("2026-01-01T12:00", "step_count", qty=0.0, avg=5.0)) == 0.0


def test_daily_buckets_by_new_york_day_and_sums_or_averages():
    rows = [hm("2026-03-02T03:00", "step_count", qty=100),  # 22:00 on Mar 1 in New York
            hm("2026-03-02T15:00", "step_count", qty=50),
            hm("2026-03-02T16:00", "step_count", qty=25)]
    assert analyze.daily(rows, "sum") == {date(2026, 3, 1): 100, date(2026, 3, 2): 75}
    assert analyze.default_agg("step_count") == "sum" and analyze.default_agg("resting_heart_rate") == "mean"


def test_rolling_weekly_window_trend():
    series = {date(2026, 1, d): float(d) for d in (1, 2, 3, 10)}
    assert analyze.rolling_mean(series, 3)[date(2026, 1, 3)] == 2.0
    assert analyze.rolling_mean(series, 3)[date(2026, 1, 10)] == 10.0  # gap: only itself in the window
    assert analyze.weekly(series) == {date(2025, 12, 29): 2.0, date(2026, 1, 5): 10.0}
    assert list(analyze.window(series, 2)) == [date(2026, 1, 10)]
    assert analyze.trend(series)["slope_per_30d"] == pytest.approx(30.0)


def test_sleep_nights_and_summary():
    sessions = [
        {"date": date(2026, 9, 1), "total_sleep": 7.0, "asleep": 7.0, "core": 4.0, "deep": 1.5, "rem": 1.5,
         "in_bed": 7.5, "sleep_start": ts("2026-09-01T04:00"), "sleep_end": ts("2026-09-01T11:00"),
         "in_bed_start": None, "in_bed_end": None},
        {"date": date(2026, 9, 2), "total_sleep": 0.0, "asleep": 0.0, "core": 0.0, "deep": 0.0, "rem": 0.0,
         "in_bed": 9.0, "sleep_start": ts("2026-09-02T02:00"), "sleep_end": ts("2026-09-02T11:00"),
         "in_bed_start": None, "in_bed_end": None},
    ]
    nights = analyze.sleep_nights(sessions)
    assert [n["hours"] for n in nights] == [7.0, 9.0]  # an unstaged night counts its in-bed time
    assert nights[0]["bed"] == time(0, 0) and nights[0]["efficiency"] == pytest.approx(7 / 7.5)
    s = analyze.sleep_summary(nights, goal_h=8)
    assert s["debt_vs_goal_h"] == 1.0 and s["share_meeting_goal"] == 0.5
    assert s["mean_bedtime"] == "23:00"  # midnight and 22:00 average across midnight, not to noon
    assert s["stage_share"]["deep"] == pytest.approx(1.5 / 7)
    assert len(analyze.sleep_nights(sessions, staged_only=True)) == 1


def test_workouts_and_zones():
    w = {"name": "Running", "duration_sec": 1800.0, "distance": 5000.0, "distance_units": "m",
         "active_energy_burned": 300.0, "avg_heart_rate": 160.0, "start_time": ts("2026-09-28T12:00")}
    assert analyze.pace_min_per_km(w) == 6.0
    [summary] = analyze.workout_summary([w, {**w, "avg_heart_rate": None}])
    assert summary["count"] == 2 and summary["km"] == 10.0 and summary["avg_hr"] == 160.0
    hr = [{"avg_bpm": b} for b in (80, 110, 150, 185, None)]
    assert analyze.hr_zones(hr, max_hr=200) == {"below": 1, "z1": 1, "z3": 1, "z5": 1}


def test_pull_sql_is_a_utc_copy_and_drops_user_id():
    for stem, select in pull.TABLES.items():
        sql = pull.copy_sql(select)
        assert sql.startswith("SET TIME ZONE 'UTC';") and "TO STDOUT WITH CSV HEADER" in sql
        assert "user_id" not in sql, stem


def test_write_gz_is_deterministic(tmp_path):
    a, b = tmp_path / "a.csv.gz", tmp_path / "b.csv.gz"
    pull.write_gz(a, b"x,y\n1,2\n")
    pull.write_gz(b, b"x,y\n1,2\n")
    assert a.read_bytes() == b.read_bytes() and gzip.decompress(a.read_bytes()) == b"x,y\n1,2\n"


@pytest.mark.skipif(not (load.DATA_DIR / "sleep_sessions.csv.gz").exists(), reason="fixture not pulled")
def test_committed_fixture_loads():
    nights = analyze.sleep_nights(load.rows("sleep_sessions"))
    assert len(nights) > 100
    assert analyze.workout_summary(load.rows("workouts"))[0]["count"] > 0
    assert len(analyze.daily(load.metric("resting_heart_rate"))) > 100
