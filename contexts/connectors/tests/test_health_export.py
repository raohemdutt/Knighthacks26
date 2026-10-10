"""The health connector on an export of FreeReps' tables (K9): a folder of gzipped CSVs, as `python -m watch pull` writes.

Most tests write a small export of made-up rows in the real files' shapes. The last ones read the real export in
fixtures/watch (one teammate's own data, in the repo by his decision) and are skipped where it is not there.
"""

import csv
import gzip
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from contexts.connectors.domain.health import (
    SESSION_CAP, nights, parse_summary, session_nights, stage_name, watch_stages, with_sessions,
)
from contexts.connectors.domain.spec import load_specs
from contexts.connectors.infrastructure.health_export import SESSION_COLUMNS, SLEEP_TYPE, ExportSource, parse_utc
from contexts.connectors.servers.health import read_nights
from contracts.connectors import ItemSynced

NY = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[3]
REAL = ROOT / "fixtures" / "watch"
FIXTURE = ROOT / "fixtures" / "health" / "freereps.yaml"
AGENT = str(ROOT / "config" / "agent.yaml")
WATCH = "Dominick’s Apple Watch"  # the export's spelling: a curly apostrophe and a no-break space
PHONE = "Dominick Dupuy"  # what the phone's In Bed rows are labelled with
NOW = datetime(2026, 10, 10, 12, 0, tzinfo=NY)
real = pytest.mark.skipif(not (REAL / "category_samples.csv.gz").exists(), reason="the watch export is not in this checkout")


def at(day: int, clock: str) -> datetime:
    h, m = clock.split(":")
    return datetime(2026, 10, day, int(h), int(m), tzinfo=NY)


def pg(ts: datetime) -> str:
    """A timestamp the way the export prints it: UTC, "2026-10-10 03:40:00+00"."""
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00")


def write(folder: Path, table: str, columns: list[str], rows: list[dict]) -> None:
    with gzip.open(folder / f"{table}.csv.gz", "wt", encoding="utf-8", newline="") as f:
        out = csv.DictWriter(f, fieldnames=columns, restval="")
        out.writeheader()
        out.writerows(rows)


def stage(start: datetime, minutes: int, label: str, source: str = WATCH) -> dict:
    return {"id": f"{label}-{pg(start)}", "type": SLEEP_TYPE, "value_label": label, "start_date": pg(start),
            "end_date": pg(start + timedelta(minutes=minutes)), "source": source}


def session(day: date, start: datetime, minutes: int, asleep_h: float, **stages) -> dict:
    end = start + timedelta(minutes=minutes)
    return {"date": day.isoformat(), "total_sleep": asleep_h, "asleep": asleep_h, "in_bed": minutes / 60,
            "sleep_start": pg(start), "sleep_end": pg(end), "in_bed_start": pg(start), "in_bed_end": pg(end),
            **{k: stages.get(k, 0) for k in ("core", "deep", "rem")}}


def metric(ts: datetime, name: str, qty=None, avg=None) -> dict:
    return {"time": pg(ts), "metric_name": name, "source": "", "qty": "" if qty is None else qty, "avg_val": "" if avg is None else avg}


def export(folder: Path, stages=(), sessions=(), metrics=()) -> dict:
    """A small export in `folder`; returns the connector options that read it."""
    write(folder, "category_samples", ["id", "type", "value", "value_label", "start_date", "end_date", "source"], list(stages))
    write(folder, "sleep_sessions", list(SESSION_COLUMNS), list(sessions))
    write(folder, "health_metrics", ["time", "metric_name", "source", "units", "qty", "min_val", "avg_val", "max_val"], list(metrics))
    return {"export_dir": str(folder), "agent_config": AGENT}


def a_week(folder: Path) -> dict:
    """Stages for the nights ending Oct 9 and 10, session rows only for Oct 6 to 8, and the quirks of the real files."""
    stages = [
        stage(at(8, "23:30"), 30, "In Bed", PHONE), stage(at(8, "23:40"), 400, "In Bed"),  # never a night
        stage(at(8, "23:40"), 200, "Asleep Core"), stage(at(9, "03:00"), 10, "Awake"), stage(at(9, "03:10"), 190, "Asleep REM"),
        stage(at(9, "14:00"), 60, "Asleep Core"),  # a nap
        stage(at(9, "23:00"), 120, "Asleep Core"), stage(at(10, "01:00"), 60, "Asleep Deep"),
        stage(at(10, "03:30"), 150, "Asleep REM"),  # after 90 minutes with no stage at all: the same night
        {"id": "stand", "type": "HKCategoryTypeIdentifierAppleStandHour", "value_label": "Stood", "start_date": pg(at(9, "23:00")),
         "end_date": pg(at(10, "00:00")), "source": WATCH},
    ]
    sessions = [
        session(date(2026, 10, 6), at(5, "23:00"), 420, 6.5, core=4, deep=1, rem=1.5),
        session(date(2026, 10, 7), at(6, "23:30"), 90, 1.4, core=1.4),  # one night that FreeReps split in two:
        session(date(2026, 10, 7), at(7, "01:30"), 270, 4.4, core=2.4, rem=2),  # 30 minutes apart
        session(date(2026, 10, 7), at(7, "14:00"), 45, 0.7, core=0.7),  # a nap
        session(date(2026, 10, 8), at(7, "02:00"), 28 * 60, 15.6, core=10),  # two nights merged into one row: no data
        session(date(2026, 10, 9), at(8, "23:40"), 60, 0.9, core=0.9),  # a fragment of a night the stages already give
        session(date(2026, 10, 4), at(3, "23:00"), 480, 0),  # the phone's In Bed only: nothing was measured
    ]
    metrics = [
        metric(at(10, "00:01"), "resting_heart_rate", qty=44), metric(at(10, "01:00"), "heart_rate", avg=46),
        metric(at(10, "03:00"), "heart_rate", avg=50), metric(at(10, "12:00"), "heart_rate", avg=120),
        metric(at(10, "02:00"), "heart_rate_variability", qty=80.4), metric(at(10, "02:00"), "respiratory_rate", qty=17),
        metric(at(10, "02:00"), "step_count", qty=9000),
    ]
    return export(folder, stages, sessions, metrics)


# --- the source rule, as the real files need it ---


def test_the_watchs_name_matches_through_the_no_break_space_the_export_writes():
    rows = [{"start_time": at(9, "23:00"), "end_time": at(10, "05:00"), "stage": "Asleep Core", "source": WATCH}]
    assert [(s.stage, s.end - s.start) for s in watch_stages(rows, ["Apple Watch"])] == [("core", timedelta(hours=6))]
    assert len(watch_stages(rows, ["Dominick's Apple Watch"])) == 1 and watch_stages(rows, ["Garmin"]) == []


def test_a_stage_with_no_source_is_kept_and_in_bed_or_another_source_is_not():
    span = {"start_time": at(9, "23:00"), "end_time": at(10, "05:00")}
    kept = lambda stage, source, names=("Apple Watch",): len(watch_stages([span | {"stage": stage, "source": source}], list(names)))  # noqa: E731
    assert [kept(s, "") for s in ("Core", "Deep", "REM", "Awake", "Asleep Core", "Asleep")] == [1] * 6  # only a watch writes these
    assert kept("Core", None) == 1
    assert kept("In Bed", "") == kept("In Bed", PHONE) == kept("In Bed", WATCH) == 0  # In Bed is never a stage
    assert kept("Core", PHONE) == kept("Core", "Somebody's sleep ring") == 0  # it names a source that is not the watch
    assert kept("Core", "Somebody's sleep ring", names=()) == 1 and kept("In Bed", PHONE, names=()) == 0  # [] takes any source


def test_healthkit_and_freereps_stage_labels_read_the_same():
    assert [stage_name(s) for s in ("Asleep Core", "Core", "Asleep REM", "REM", "Asleep Deep", "Awake", "In Bed", "Asleep",
                                    "Asleep Unspecified", None)] == ["core", "core", "rem", "rem", "deep", "awake", "in bed",
                                                                     "asleep", "asleep", ""]


def test_postgres_style_timestamps_are_read_without_python_3_11():
    want = datetime(2026, 9, 29, 9, 29, 49, tzinfo=timezone.utc)
    assert [parse_utc(t) for t in ("2026-09-29 09:29:49+00", "2026-09-29T09:29:49Z", "2026-09-29 09:29:49+00:00")] == [want] * 3
    assert parse_utc("2026-09-29 09:29:49.5+00") == want + timedelta(milliseconds=500)
    with pytest.raises(ValueError):
        parse_utc("2026-09-29 09:29:49-04")  # the export is UTC; anything else is not this file format


# --- a small export, through the connector's read ---


def test_an_export_gives_nights_from_stages_and_from_session_rows_where_there_are_none(tmp_path):
    items = read_nights(a_week(tmp_path), NOW, env={})
    assert [(i["external_id"], i["title"]) for i in items] == [
        ("sleep:2026-10-06", "Sleep 6 h 30 min"),  # a session row
        ("sleep:2026-10-07", "Sleep 5 h 48 min"),  # two session rows half an hour apart: one night; the nap is not one
        ("sleep:2026-10-09", "Sleep 6 h 30 min"),  # stages; the watch's own In Bed row and the session fragment are ignored
        ("sleep:2026-10-10", "Sleep 5 h 30 min"),
    ]  # Oct 8: only a 28 hour row, so no night. Oct 4: only the phone's In Bed, so no night
    assert items[1]["summary"] == "asleep 5 h 48 min · in bed 6 h 30 min · 11:30 PM to 6:00 AM · core 228 min · REM 120 min"
    assert items[2]["summary"] == "asleep 6 h 30 min · in bed 6 h 40 min · 11:40 PM to 6:20 AM · core 200 min · REM 190 min · awake 10 min"
    assert items[3]["summary"] == ("asleep 5 h 30 min · in bed 7 h 0 min · 11:00 PM to 6:00 AM · core 120 min · deep 60 min · REM 150 min · "
                                   "resting HR 44 bpm · sleeping HR 48 bpm · HRV 80 ms · breathing 17.0/min")  # heart_rate: avg_val
    assert all(ItemSynced(ts=NOW, connector="health", **i) for i in items)
    assert all("Dominick" not in i["title"] + i["summary"] and "step" not in i["summary"] for i in items)


def test_in_bed_only_rows_are_nights_only_when_asked_for(tmp_path):
    options = a_week(tmp_path) | {"days": 10}
    assert "sleep:2026-10-04" not in [i["external_id"] for i in read_nights(options, NOW, env={})]
    [phone] = [i for i in read_nights(options | {"export_in_bed_nights": True}, NOW, env={}) if i["external_id"] == "sleep:2026-10-04"]
    assert phone["summary"] == "asleep 8 h 0 min · in bed 8 h 0 min · 11:00 PM to 7:00 AM"  # time in bed, no stages, no heart numbers


def test_session_rows_follow_the_stage_rules_and_never_displace_a_staged_night():
    row = lambda start, minutes, asleep_h: {"sleep_start": start, "sleep_end": start + timedelta(minutes=minutes), "total_sleep": asleep_h}  # noqa: E731
    [joined] = session_nights([row(at(9, "23:00"), 120, 1.9), row(at(10, "03:00"), 180, 2.9)], NY)  # exactly two hours apart
    assert (joined.day, joined.asleep_min, joined.in_bed_min) == (date(2026, 10, 10), 288, 420)
    assert len(session_nights([row(at(8, "23:00"), 120, 1.9), row(at(9, "03:01"), 180, 2.9)], NY)) == 1  # further apart: the longer one
    assert session_nights([row(at(9, "14:00"), 90, 1.4)], NY) == []  # a nap
    assert session_nights([row(at(9, "20:00"), 17 * 60, 15.0)], NY) == [] and SESSION_CAP == timedelta(hours=16)
    [capped] = session_nights([row(at(9, "23:00"), 360, 9.0)], NY)  # a row cannot have slept longer than it lasted
    assert capped.asleep_min == 360
    staged = nights(watch_stages([{"start_time": at(9, "23:00"), "end_time": at(10, "05:00"), "stage": "Core", "source": ""}], []), NY)
    same_day = session_nights([row(at(9, "22:00"), 600, 9.5)], NY)  # a longer row for the night the stages already give
    assert same_day[0].day == date(2026, 10, 10) and with_sessions(staged, same_day) == staged
    assert [n.day for n in with_sessions(staged, session_nights([row(at(7, "23:00"), 420, 6.5)], NY))] == [date(2026, 10, 8), date(2026, 10, 10)]


def test_the_database_comes_first_then_the_export_then_the_fixture(tmp_path, capsys):
    def down(url):
        raise OSError(f"could not connect to {url}")

    options = a_week(tmp_path)
    env = {"FREEREPS_DATABASE_URL": "postgresql://reader:hunter2@100.64.0.9/freereps"}
    from_export = read_nights(options, NOW, env={})
    assert read_nights(options | {"fixture": str(FIXTURE)}, NOW, env={}) == from_export  # the export wins over the fixture
    assert read_nights(options, NOW, env=env, connect=down) == from_export  # the database is tried first, then the export
    said = capsys.readouterr().err
    assert "OSError" in said and "hunter2" not in said and "100.64.0.9" not in said
    missing = {"export_dir": str(tmp_path / "nowhere"), "agent_config": AGENT}
    assert read_nights(missing, NOW, env={}) == []  # a folder that is not there: nothing, and no error
    assert len(read_nights(missing | {"fixture": str(FIXTURE)}, NOW, env={})) == 4  # ... or the fixture when one is set


def test_export_rebase_moves_the_newest_night_to_last_night_keeping_local_clock_times(tmp_path):
    options = a_week(tmp_path) | {"export_rebase": True}
    assert read_nights(options, NOW, env={}) == read_nights(a_week(tmp_path), NOW, env={})  # already ends today: nothing moves
    now = datetime(2026, 11, 3, 10, 0, tzinfo=NY)  # 24 days on, across the end of daylight saving time
    items = read_nights(options, now, env={})
    assert [i["external_id"] for i in items] == ["sleep:2026-10-30", "sleep:2026-10-31", "sleep:2026-11-02", "sleep:2026-11-03"]
    assert items[-1]["when"] == "2026-11-03T06:00:00-05:00" and "11:00 PM to 6:00 AM" in items[-1]["summary"]
    assert parse_summary(items[-1]["summary"]).resting_hr == 44  # the heart numbers moved with their night
    early = read_nights(options, datetime(2026, 11, 3, 3, 0, tzinfo=NY), env={})  # 3 AM: that night has not ended yet today,
    assert [i["external_id"] for i in early][-1] == "sleep:2026-11-02" and early[-1]["title"] == items[-1]["title"]  # so it is yesterday's
    assert read_nights(a_week(tmp_path), now, env={}) == []  # without the rebase the week is simply long past


def test_the_source_reads_only_what_was_asked_for_and_sees_a_new_file(tmp_path):
    options = a_week(tmp_path)
    source = ExportSource(tmp_path, NY)
    assert {r["stage"] for r in source.sleep_stages(at(1, "00:00"), at(12, "00:00"))} == {"In Bed", "Asleep Core", "Asleep REM", "Asleep Deep", "Awake"}
    assert [r["value"] for r in source.metrics(at(10, "00:00"), at(10, "06:00"), ["heart_rate"])] == [46.0, 50.0]
    assert source.newest() == at(10, "06:00") and source.moved(2).newest() == at(12, "06:00")
    assert len(read_nights(options, NOW, env={})) == 4
    export(tmp_path, [stage(at(9, "23:00"), 300, "Asleep Core")])  # a fresh pull replaces the files
    assert [i["title"] for i in read_nights(options, NOW, env={})] == ["Sleep 5 h 0 min"]


def test_the_committed_config_names_the_export_options_and_leaves_them_off():
    spec = next(s for s in load_specs(ROOT / "config" / "connectors.yaml") if s.name == "health")
    assert (spec.options["export_dir"], spec.options["export_rebase"], spec.options["export_in_bed_nights"]) == (None, False, False)


# --- the real export ---


@real
def test_the_real_export_gives_whole_nights_for_the_last_two_weeks_it_covers():
    options = {"export_dir": str(REAL), "agent_config": AGENT, "days": 14}
    items = read_nights(options, datetime(2026, 9, 29, 12, 0, tzinfo=NY), env={})
    assert [i["external_id"][-5:] for i in items] == ["09-17", "09-18", "09-19", "09-20", "09-21", "09-22", "09-23", "09-24",
                                                      "09-25", "09-26", "09-27", "09-28", "09-29"]  # not worn on the 16th
    readings = [parse_summary(i["summary"]) for i in items]
    assert all(4 * 60 <= r.asleep_min <= 10 * 60 and {"core", "deep", "rem"} <= set(r.stage_min) for r in readings)
    assert all(r.sleeping_hr and r.hrv_ms and r.resp for r in readings)  # resting HR is missing on two of the days
    # FreeReps' sleep_sessions has this night as one 28 hour row; the stages give the night itself
    assert (items[4]["title"], readings[4].in_bed_min) == ("Sleep 6 h 36 min", 438)
    assert all(ItemSynced(ts=NOW, connector="health", **i) for i in items)
    assert all("Dominick" not in i["title"] + i["summary"] for i in items)


@real
def test_the_real_export_rebased_ends_last_night_with_the_same_numbers():
    options = {"export_dir": str(REAL), "agent_config": AGENT, "days": 7}
    there = read_nights(options, datetime(2026, 9, 29, 12, 0, tzinfo=NY), env={})
    here = read_nights(options | {"export_rebase": True}, NOW, env={})
    assert [i["external_id"] for i in here][-1] == "sleep:2026-10-10" and len(here) == len(there) == 7
    assert [(i["title"], i["summary"]) for i in here] == [(i["title"], i["summary"]) for i in there]
