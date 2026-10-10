"""The health connector reading a FreeReps export directory (K9 follow-up): <table>.csv.gz, as fixtures/watch has them.

Every row in these tests is made up here, in files with the real headers. The one test over the committed real
export asserts only what any export would show (how many nights, plausible hours), never a value from it.
"""

import csv
import gzip
import os
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from contexts.connectors.domain.health import as_time, nights, parse_summary, watch_stages
from contexts.connectors.infrastructure import freereps
from contexts.connectors.infrastructure.freereps import METRICS_FILE, SLEEP_FILE, SLEEP_TYPE, ExportSource
from contexts.connectors.servers.health import read_nights
from contracts.connectors import ItemSynced

NY = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[3]
REAL = ROOT / "fixtures" / "watch"
FIXTURE = ROOT / "fixtures" / "health" / "freereps.yaml"
AGENT = str(ROOT / "config" / "agent.yaml")
NOW = datetime(2026, 10, 10, 12, 0, tzinfo=NY)

# the real files' headers (python -m watch pull writes them; user_id is dropped)
SAMPLE_COLUMNS = ["id", "type", "value", "value_label", "start_date", "end_date", "source"]
METRIC_COLUMNS = ["time", "metric_name", "source", "units", "qty", "min_val", "avg_val", "max_val", "systolic",
                  "diastolic", "source_uuid"]
WATCH = "Sam’s Apple Watch"  # as Apple Health writes a watch's name: curly apostrophe, no-break space
OLD_LABEL = "Sam Example"  # rows from before a device was renamed carry another label
VALUES = {"In Bed": 0, "Awake": 2, "Asleep Core": 3, "Asleep Deep": 4, "Asleep REM": 5}


def utc(ts: datetime) -> str:
    """A timestamp as Postgres prints it with TIME ZONE 'UTC': "2026-10-10 03:30:00+00"."""
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00")


def at(day: int, clock: str, month: int = 10) -> datetime:
    h, m = clock.split(":")
    return datetime(2026, month, day, int(h), int(m), tzinfo=NY)


def sleep(start: datetime, *parts, source: str = WATCH) -> list[dict]:
    """Back-to-back sleep rows from `start`: ("Asleep Core", 60), ("gap", 30), ("In Bed", 400), ..."""
    out, cur = [], start
    for label, minutes in parts:
        end = cur + timedelta(minutes=minutes)
        if label != "gap":
            out.append({"type": SLEEP_TYPE, "value": VALUES[label], "value_label": label, "start_date": utc(cur),
                        "end_date": utc(end), "source": source})
        cur = end
    return out


def metric(ts: datetime, name: str, qty=None, avg=None) -> dict:
    """A health_metrics row: no source, and heart_rate with min/avg/max in place of qty, as the real rows are."""
    return {"time": utc(ts), "metric_name": name, "source": "", "units": "count/min", "qty": "" if qty is None else qty,
            "min_val": "" if avg is None else avg - 3, "avg_val": "" if avg is None else avg,
            "max_val": "" if avg is None else avg + 6}


def export(directory: Path, samples: list[dict], metrics: list[dict] = ()) -> dict:
    """Write the two tables as the export has them, ordered by time; the options that read them."""
    directory.mkdir(exist_ok=True)
    samples = [{"id": f"00000000-0000-4000-8000-{i:012d}", **row} for i, row in enumerate(samples)]
    for name, columns, rows, key in ((SLEEP_FILE, SAMPLE_COLUMNS, samples, "start_date"),
                                     (METRICS_FILE, METRIC_COLUMNS, list(metrics), "time")):
        with gzip.open(directory / name, "wt", encoding="utf-8", newline="") as f:
            out = csv.DictWriter(f, columns, restval="")
            out.writeheader()
            out.writerows(sorted(rows, key=lambda r: r[key]))
    return {"export_dir": str(directory), "agent_config": AGENT}


def a_night(day: int, month: int = 10, source: str = WATCH) -> list[dict]:
    """11:30 PM the evening before to 6:42 AM: 7 h asleep, 12 minutes awake."""
    start = datetime(2026, month, day, tzinfo=NY) - timedelta(minutes=30)
    return sleep(start, ("Asleep Core", 200), ("Asleep Deep", 60), ("Awake", 12), ("Asleep REM", 160), source=source)


# --- 1. sleep comes from category_samples; sleep_sessions is not read ---


def test_nights_are_clustered_from_the_sleep_rows_of_category_samples(tmp_path):
    broken = sleep(at(8, "23:00"), ("Asleep Core", 180), ("Awake", 20), ("gap", 70), ("Asleep REM", 150))  # up for 90 minutes
    stand = [{"type": "HKCategoryTypeIdentifierAppleStandHour", "value": 1, "value_label": "Stood",
              "start_date": utc(at(9, "02:00")), "end_date": utc(at(9, "03:00")), "source": WATCH}]
    nap = sleep(at(9, "14:00"), ("Asleep Core", 50))
    options = export(tmp_path, a_night(10) + broken + stand + nap)
    # what FreeReps' own table would say of the broken night: its first stretch only. It is not there to be read.
    (tmp_path / "sleep_sessions.csv.gz").write_bytes(gzip.compress(b"date,total_sleep\n2026-10-09,3.0\n"))
    items = read_nights(options, NOW, env={})
    assert [(i["external_id"], i["title"], i["when"]) for i in items] == [
        ("sleep:2026-10-09", "Sleep 5 h 30 min", "2026-10-09T06:00:00-04:00"),
        ("sleep:2026-10-10", "Sleep 7 h 0 min", "2026-10-10T06:42:00-04:00")]
    assert items[0]["summary"] == "asleep 5 h 30 min · in bed 7 h 0 min · 11:00 PM to 6:00 AM · core 180 min · REM 150 min · awake 20 min"
    assert all(ItemSynced(ts=NOW, connector="health", **i) for i in items)


# --- 2. which rows are the watch's ---


def test_stage_rows_count_with_no_source_and_in_bed_rows_only_from_a_named_watch(tmp_path):
    unnamed = a_night(8, source="")  # most rows of the real tables name no source
    old_in_bed = sleep(at(8, "23:40"), ("In Bed", 420), source=OLD_LABEL)  # night ending the 9th: an older label, in bed only
    phone = sleep(at(9, "22:00"), ("In Bed", 600), source="Sam’s iPhone") + sleep(at(9, "22:00"), ("In Bed", 600), source="")
    ring = sleep(at(9, "23:00"), ("Asleep Core", 400), source="Some sleep ring")
    options = export(tmp_path, unnamed + old_in_bed + phone + ring)
    assert [i["title"] for i in read_nights(options, NOW, env={})] == ["Sleep 7 h 0 min"]  # default: ["Apple Watch"]
    named = read_nights(options | {"watch_sources": ["apple watch", OLD_LABEL]}, NOW, env={})
    assert [(i["external_id"], i["title"]) for i in named] == [
        ("sleep:2026-10-08", "Sleep 7 h 0 min"), ("sleep:2026-10-09", "In bed 7 h 0 min (no sleep stages)")]
    assert [i["title"] for i in read_nights(options | {"watch_sources": ["ring"]}, NOW, env={})] == [
        "Sleep 7 h 0 min", "Sleep 6 h 40 min"]  # the option still chooses between named devices
    assert read_nights(export(tmp_path / "watch", a_night(10)), NOW, env={})[0]["title"] == "Sleep 7 h 0 min"  # the no-break space


# --- 3. UTC in the files, the household's days and clock times in the items ---


def test_utc_timestamps_are_bucketed_in_the_households_time_zone(tmp_path):
    evening = sleep(at(8, "20:30"), ("Asleep Core", 200))  # 00:30 to 03:50 UTC on the 9th: the night that ends on the 9th
    usual = sleep(at(9, "23:30"), ("Asleep Core", 420))  # 03:30 to 10:30 UTC on the 10th
    resting = [metric(at(10, "00:01"), "resting_heart_rate", qty=52),  # 04:01 UTC on the 10th: the 10th here too
               metric(at(9, "23:59"), "resting_heart_rate", qty=70),  # 03:59 UTC on the 10th, but still the 9th here
               metric(at(9, "00:02"), "resting_heart_rate", qty=61)]
    options = export(tmp_path, evening + usual, resting)
    with gzip.open(tmp_path / SLEEP_FILE, "rt", encoding="utf-8") as f:
        assert all(row["start_date"].endswith("+00") and row["end_date"].endswith("+00") for row in csv.DictReader(f))
    first, second = read_nights(options, NOW, env={})
    assert (first["external_id"], first["when"]) == ("sleep:2026-10-09", "2026-10-08T23:50:00-04:00")
    assert "8:30 PM to 11:50 PM" in first["summary"] and first["summary"].endswith("resting HR 66 bpm")  # (61 + 70) / 2
    assert (second["external_id"], second["when"]) == ("sleep:2026-10-10", "2026-10-10T06:30:00-04:00")
    assert "11:30 PM to 6:30 AM" in second["summary"] and second["summary"].endswith("resting HR 52 bpm")
    winter = export(tmp_path / "winter", sleep(at(9, "23:30", month=12), ("Asleep Core", 420)))  # UTC-5 by then
    [item] = read_nights(winter, datetime(2026, 12, 10, 12, tzinfo=NY), env={})
    assert item["when"] == "2026-12-10T06:30:00-05:00" and "11:30 PM to 6:30 AM" in item["summary"]


def test_timestamps_parse_as_postgres_prints_them_on_python_3_10_too():
    """The Jetson's datetime.fromisoformat() takes neither "+00", "Z" nor a fraction that is not 3 or 6 digits."""
    want = datetime(2026, 1, 26, 5, 50, 25, tzinfo=timezone.utc)
    assert as_time("2026-01-26 05:50:25+00") == as_time("2026-01-26T05:50:25Z") == as_time("2026-01-26T00:50:25-05:00") == want
    assert as_time("2026-01-26 05:50:25.5+00") == want + timedelta(milliseconds=500)
    assert as_time("2026-01-26 05:50:25.1234567+00") == want + timedelta(microseconds=123456)
    with pytest.raises(ValueError):
        as_time("2026-01-26 05:50:25")


# --- 4. nights from before the watch recorded stages ---


def test_a_night_with_no_stages_is_an_in_bed_item_that_says_so_and_is_no_measurement(tmp_path):
    in_bed = sleep(at(8, "23:50"), ("In Bed", 190), ("gap", 25), ("In Bed", 235))  # night ending the 9th: in bed only
    twin = sleep(at(9, "23:30"), ("In Bed", 200), ("gap", 72), ("In Bed", 160))  # the 10th has stages: its In Bed rows add nothing
    heart = [metric(at(9, "02:00"), "heart_rate", avg=55)]
    options = export(tmp_path, in_bed + a_night(10) + twin, heart)
    unstaged, staged = read_nights(options, NOW, env={})
    assert unstaged == {
        "external_id": "sleep:2026-10-09", "kind": "health", "title": "In bed 7 h 5 min (no sleep stages)",
        "when": "2026-10-09T07:20:00-04:00",
        "summary": "in bed 7 h 5 min · 11:50 PM to 7:20 AM · no sleep stages: time in bed, not measured sleep · sleeping HR 55 bpm"}
    assert parse_summary(unstaged["summary"]) is None  # Habits reads no measured sleep out of it
    assert staged["title"] == "Sleep 7 h 0 min" and parse_summary(staged["summary"]).asleep_min == 420
    assert ItemSynced(ts=NOW, connector="health", **unstaged)


# --- 5. heart numbers, and only the days that are asked for ---


def test_heart_rate_comes_from_avg_val_and_all_day_totals_are_never_read(tmp_path):
    numbers = [metric(at(10, "01:00"), "heart_rate", avg=50), metric(at(10, "03:00"), "heart_rate", avg=54),
               metric(at(10, "13:00"), "heart_rate", avg=120),  # awake and moving: not the sleeping heart rate
               metric(at(10, "02:00"), "heart_rate_variability", qty=61.6), metric(at(10, "02:10"), "respiratory_rate", qty=14.25),
               metric(at(10, "00:01"), "resting_heart_rate", qty=53), metric(at(10, "09:00"), "step_count", qty=4321),
               metric(at(10, "02:30"), "heart_rate")]  # neither qty nor avg_val: no value
    options = export(tmp_path, a_night(10), numbers)
    [item] = read_nights(options, NOW, env={})
    assert item["summary"].endswith("resting HR 53 bpm · sleeping HR 52 bpm · HRV 62 ms · breathing 14.2/min")
    assert "4321" not in item["summary"] and "step" not in item["summary"]
    read = ExportSource(tmp_path, NY).metrics(at(9, "00:00"), at(11, "00:00"), ["heart_rate", "step_count"])
    assert [(r["metric_name"], r["value"]) for r in read] == [
        ("heart_rate", 50.0), ("heart_rate", None), ("heart_rate", 54.0), ("step_count", 4321.0), ("heart_rate", 120.0)]


def test_only_the_days_asked_for_are_parsed_and_the_file_is_read_again_only_when_it_changes(tmp_path, monkeypatch):
    years = [metric(datetime(2020, 1, 1, tzinfo=NY) + timedelta(days=d), "heart_rate", avg=60) for d in range(2400)]
    recent = [metric(at(10, "01:00"), "heart_rate", avg=50), metric(at(10, "03:00"), "heart_rate", avg=54)]
    options = export(tmp_path, a_night(10), years + recent)
    parsed, opened = [], []
    real_time, real_open = freereps.as_time, gzip.open
    monkeypatch.setattr(freereps, "as_time", lambda value: parsed.append(value) or real_time(value))
    monkeypatch.setattr(freereps.gzip, "open", lambda path, *a, **kw: opened.append(Path(path).name) or real_open(path, *a, **kw))
    [first] = read_nights(options, NOW, env={})
    assert "sleeping HR 52 bpm" in first["summary"]
    assert len([v for v in parsed if v.startswith("202") and "2026-10" not in v]) == 0  # six years of rows, none parsed
    assert len(parsed) < 40 and opened == [SLEEP_FILE, METRICS_FILE]
    assert read_nights(options, NOW + timedelta(minutes=30), env={}) == [first]
    assert opened == [SLEEP_FILE, METRICS_FILE, SLEEP_FILE]  # half an hour later: the big file is not opened again
    export(tmp_path, a_night(10), years + recent + [metric(at(10, "05:00"), "heart_rate", avg=70)])
    os.utime(tmp_path / METRICS_FILE, ns=(1, 1))  # a new pull: the file's time (and size) changed
    opened.clear()
    assert "sleeping HR 58 bpm" in read_nights(options, NOW + timedelta(minutes=60), env={})[0]["summary"]
    assert opened == [SLEEP_FILE, METRICS_FILE]


# --- reader order ---


class FakeCursor:
    def __init__(self, log: list) -> None:
        self.log, self.description = log, [("start_date",), ("end_date",), ("value_label",), ("source",)]

    def execute(self, sql: str, params: tuple = ()) -> None:
        self.log.append(sql)

    def fetchall(self) -> list[tuple]:
        return []


class FakeConnection:
    def __init__(self, log: list) -> None:
        self.log = log

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.log)

    def close(self) -> None:
        pass


def test_the_database_comes_first_then_the_export_then_the_fixture_then_nothing(tmp_path, capsys):
    both = export(tmp_path, a_night(10)) | {"fixture": str(FIXTURE)}
    env = {"FREEREPS_DATABASE_URL": "postgresql://reader:hunter2@100.64.0.9/freereps"}
    asked: list = []
    assert read_nights(both, NOW, env=env, connect=lambda url: FakeConnection(asked)) == []  # the (empty) database, nothing else
    assert len(asked) == 1 and "FROM category_samples" in asked[0]
    assert [i["title"] for i in read_nights(both, NOW, env={})] == ["Sleep 7 h 0 min"]  # the export, not the fixture's four nights
    assert len(read_nights({"fixture": str(FIXTURE), "agent_config": AGENT}, NOW, env={})) == 4
    assert len(read_nights(both | {"export_dir": None}, NOW, env={})) == 4 and read_nights({"agent_config": AGENT}, NOW, env={}) == []

    def down(url):
        raise OSError(f"could not connect to {url}")

    assert [i["title"] for i in read_nights(both, NOW, env=env, connect=down)] == ["Sleep 7 h 0 min"]  # database down: the export
    said = capsys.readouterr().err
    assert "OSError" in said and "hunter2" not in said and "100.64.0.9" not in said


def test_an_export_dir_without_the_files_is_a_failed_sync_that_names_the_file(tmp_path):
    with pytest.raises(RuntimeError, match="category_samples.csv.gz"):
        read_nights({"export_dir": str(tmp_path / "nowhere"), "agent_config": AGENT}, NOW, env={})


# --- export_rebase ---


def test_export_rebase_moves_the_exports_last_night_to_end_today_keeping_local_clock_times(tmp_path):
    """The export stops on the day it was pulled; a demo weeks later still shows "last night"."""
    worn = a_night(24, month=9) + a_night(27, month=9) + a_night(29, month=9)
    nap_after = sleep(at(29, "15:00", month=9), ("Asleep Core", 40))  # the export's last sleep row is not its last night
    heart = [metric(at(29, "02:00", month=9), "heart_rate", avg=51), metric(at(29, "00:01", month=9), "resting_heart_rate", qty=54),
             metric(at(27, "02:00", month=9), "heart_rate", avg=57)]
    options = export(tmp_path, worn + nap_after, heart)
    now = datetime(2026, 11, 3, 10, 0, tzinfo=NY)  # five weeks on, and across the end of daylight saving time
    assert read_nights(options, now, env={}) == []  # off by default: nothing in the last 7 days
    items = read_nights(options | {"export_rebase": True}, now, env={})
    assert [i["external_id"] for i in items] == ["sleep:2026-10-29", "sleep:2026-11-01", "sleep:2026-11-03"]
    assert items[-1]["when"] == "2026-11-03T06:42:00-05:00" and items[-1]["title"] == "Sleep 7 h 0 min"
    assert "11:30 PM to 6:42 AM" in items[-1]["summary"] and items[-1]["summary"].endswith("resting HR 54 bpm · sleeping HR 51 bpm")
    assert "sleeping HR 57 bpm" in items[1]["summary"]  # the heart numbers move with their nights
    assert read_nights(options | {"export_rebase": True}, now + timedelta(days=1), env={})[-1]["external_id"] == "sleep:2026-11-04"
    still = read_nights(options | {"export_rebase": True}, datetime(2026, 9, 29, 12, tzinfo=NY), env={})
    assert [i["external_id"] for i in still] == ["sleep:2026-09-24", "sleep:2026-09-27", "sleep:2026-09-29"]  # nothing to move


# --- the committed real export: shapes only ---


@pytest.mark.skipif(not (REAL / SLEEP_FILE).exists() or not (REAL / METRICS_FILE).exists(), reason="no export in fixtures/watch")
def test_the_real_export_reads_as_plausible_nights():
    source = ExportSource(REAL, NY)
    last = source.last_sleep_end()
    now = last + timedelta(hours=12)
    options = {"export_dir": str(REAL), "agent_config": AGENT, "days": 60}
    items = read_nights(options, now, env={})
    assert len(items) >= 20 and len({i["external_id"] for i in items}) == len(items)
    readings = [parse_summary(i["summary"]) for i in items]
    assert all(r is not None and 1 <= r.asleep_h <= 14 and r.asleep_min <= r.in_bed_min for r in readings)  # all staged by now
    assert all(ItemSynced(ts=now, connector="health", **i) for i in items)
    found = nights(watch_stages(source.sleep_samples(now - timedelta(days=61), now), ["Apple Watch"]), NY, time(20), time(8))
    assert found and all(n.wake > n.bed and n.wake - n.bed < timedelta(hours=16) for n in found)
    assert sum("HR" in i["summary"] for i in items) >= len(items) // 2  # heart numbers are found for most nights
    week = read_nights(options | {"days": 7, "export_rebase": True}, datetime.now(timezone.utc), env={})
    assert week and week[-1]["external_id"] == f"sleep:{datetime.now(NY).date()}"

    # before the watch recorded stages: In Bed rows under another label. Named, they are in-bed nights; unnamed, nothing.
    samples = source.sleep_samples(datetime(2017, 1, 1, tzinfo=NY), now)
    labels = {r["source"] for r in samples if r["source"] and "watch" not in r["source"].lower()}
    early = min(r["start_date"] for r in samples) + timedelta(days=60)
    assert read_nights(options, early, env={}) == []
    old = read_nights(options | {"watch_sources": ["Apple Watch", *labels]}, early, env={})
    assert len(old) >= 5 and all(i["title"].startswith("In bed ") and parse_summary(i["summary"]) is None for i in old)


@pytest.mark.skipif(not (REAL / METRICS_FILE).exists() or sys.version_info < (3, 11), reason="no export, or watch.load cannot run")
def test_the_export_reader_and_the_owners_loader_read_the_same_rows():
    """`watch.load` / `watch.analyze` are the data owner's reader. This one does not import them at run time
    (see freereps.py), so here they are held to the same rows: sleep samples, and COALESCE(qty, avg_val)."""
    from watch import analyze, load

    source = ExportSource(REAL, NY)
    theirs = [r for r in load.rows("category_samples", REAL) if r["type"] == SLEEP_TYPE]
    ours = source.sleep_samples(datetime(2017, 1, 1, tzinfo=NY), datetime(2100, 1, 1, tzinfo=NY))
    key = lambda start, end, label: (start, end, label)  # noqa: E731
    assert sorted(key(r["start_date"], r["end_date"], r["value_label"]) for r in theirs) == sorted(
        key(r["start_date"], r["end_date"], r["value_label"]) for r in ours)
    end = source.last_sleep_end() + timedelta(days=1)
    start = end - timedelta(days=30)
    want = [(r["time"], analyze.value(r)) for r in load.metric("heart_rate", REAL) if start <= r["time"] < end]
    assert want and [(r["time"], r["value"]) for r in source.metrics(start, end, ["heart_rate"])] == want
