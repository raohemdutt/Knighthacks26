import asyncio
import json
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from contexts.connectors.application.hub import ConnectorHub
from contexts.connectors.domain.health import (
    Stage, attach_metrics, clusters, night_day, nights, parse_summary, summary_line, to_item, watch_stages,
)
from contexts.connectors.domain.spec import load_specs
from contexts.connectors.infrastructure.freereps import METRICS_SQL, SESSION_OPTIONS, STAGES_SQL, FixtureSource, PostgresSource
from contexts.connectors.infrastructure.mcp_client import McpConnector
from contexts.connectors.servers.health import read_nights
from contracts.connectors import ItemSynced

NY = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "fixtures" / "health" / "freereps.yaml"
WATCH = "Dominick’s Apple Watch"  # the curly apostrophe is what Apple Health writes
PHONE = "Dominick’s iPhone"
NOW = datetime(2026, 10, 10, 12, 0, tzinfo=NY)
OPTIONS = {"fixture": str(FIXTURE), "agent_config": str(ROOT / "config" / "agent.yaml")}


def at(day: int, clock: str) -> datetime:
    h, m = clock.split(":")
    return datetime(2026, 10, day, int(h), int(m), tzinfo=NY)


def rows(start: datetime, *parts, source=WATCH) -> list[dict]:
    """Back-to-back stage rows from `start`: ("Core", 60), ("gap", 30), ..."""
    out, cur = [], start
    for stage, minutes in parts:
        end = cur + timedelta(minutes=minutes)
        if stage != "gap":
            out.append({"start_time": cur, "end_time": end, "stage": stage, "duration_hr": minutes / 60, "source": source})
        cur = end
    return out


def night_of(*stage_rows, sources=("Apple Watch",)):
    return nights(watch_stages([r for part in stage_rows for r in part], list(sources)), NY)


# --- the rules, on rows made here ---


def test_only_the_watchs_rows_count_and_in_bed_rows_never_do():
    phone = [{"start_time": at(9, "22:00"), "end_time": at(10, "09:00"), "stage": "In Bed", "source": PHONE}]
    watch_in_bed = [{"start_time": at(9, "22:00"), "end_time": at(10, "09:00"), "stage": "In Bed", "source": WATCH}]
    other = rows(at(9, "21:00"), ("Core", 600), source="Somebody's sleep ring")
    [night] = night_of(phone, watch_in_bed, other, rows(at(9, "23:00"), ("Core", 240), ("Awake", 30), ("REM", 180)))
    assert (night.day, night.bed, night.wake) == (date(2026, 10, 10), at(9, "23:00"), at(10, "06:30"))
    assert (night.asleep_min, night.in_bed_min, night.stage_min) == (420, 450, {"core": 240, "awake": 30, "rem": 180})
    assert night_of(phone, watch_in_bed) == []  # a night with only In Bed rows is a night with no data


def test_watch_sources_is_a_list_of_name_parts_and_the_apostrophe_does_not_matter():
    stages = rows(at(9, "23:00"), ("Core", 300))
    assert len(night_of(stages, sources=["Dominick's Apple Watch"])) == 1  # straight apostrophe in the config
    assert len(night_of(stages, sources=["apple watch"])) == 1
    assert night_of(stages, sources=["Garmin"]) == []
    assert len(night_of(rows(at(9, "23:00"), ("Core", 300), source="Maya's Garmin"), sources=["Apple Watch", "Garmin"])) == 1


def test_a_gap_of_up_to_two_hours_is_the_same_sleep_and_a_longer_one_starts_a_new_one():
    stages = watch_stages(rows(at(9, "23:00"), ("Core", 120), ("gap", 120), ("REM", 60), ("gap", 121), ("Core", 30)), ["Apple Watch"])
    assert [len(c) for c in clusters(stages)] == [2, 1]
    overlapping = [Stage(at(9, "23:00"), at(10, "03:00"), "core"), Stage(at(9, "23:30"), at(10, "00:00"), "deep"),
                   Stage(at(10, "04:30"), at(10, "06:00"), "rem")]  # 1.5 h after the long row ended, 4.5 h after the short one
    assert [len(c) for c in clusters(overlapping)] == [3]


def test_a_night_broken_by_time_awake_stays_one_night_and_awake_is_not_sleep():
    [night] = night_of(rows(at(9, "23:00"), ("Core", 180), ("Awake", 40), ("gap", 50), ("Core", 200)))
    assert (night.bed, night.wake) == (at(9, "23:00"), at(10, "06:50"))
    assert (night.asleep_min, night.in_bed_min) == (380, 470)  # in bed counts the time awake, asleep does not


def test_the_longest_sleep_is_the_night_and_a_nap_is_never_one():
    late = rows(at(10, "01:00"), ("Core", 300))  # 1 AM to 6 AM
    back_to_bed = rows(at(10, "08:30"), ("Core", 60))  # more than two hours later, the same day: the shorter one
    nap = rows(at(10, "14:00"), ("Core", 90))
    [night] = night_of(late, back_to_bed, nap)
    assert (night.day, night.asleep_min) == (date(2026, 10, 10), 300)
    assert night_of(nap) == []  # a day with only a nap has no night: no data, not ninety minutes of sleep
    assert night_of(rows(at(10, "09:00"), ("Core", 400))) == []  # asleep after wake time: no sleep window touched


def test_a_night_counts_for_the_day_it_ends_on_even_when_it_ends_before_midnight():
    [early] = night_of(rows(at(9, "20:30"), ("Core", 180)))  # asleep 8:30 PM, up at 11:30 PM
    assert early.day == date(2026, 10, 10)
    [usual] = night_of(rows(at(9, "23:30"), ("Core", 420)))
    assert usual.day == date(2026, 10, 10)
    [late] = night_of(rows(at(10, "04:00"), ("Core", 420)))  # 4 AM to 11 AM
    assert late.day == date(2026, 10, 10)


def test_duplicate_and_overlapping_rows_are_counted_once():
    twice = rows(at(9, "23:00"), ("Core", 300)) * 2 + rows(at(10, "03:00"), ("Deep", 120))  # deep overlaps the last hour of core
    [night] = night_of(twice)
    assert (night.asleep_min, night.in_bed_min) == (360, 360)


def test_heart_numbers_come_from_the_night_itself_and_resting_from_its_day():
    [night] = night_of(rows(at(9, "23:00"), ("Core", 420)))
    metric = lambda ts, name, value: {"time": ts, "metric_name": name, "value": value}  # noqa: E731
    attach_metrics(night, [
        metric(at(10, "00:00"), "resting_heart_rate", 53), metric(at(9, "00:00"), "resting_heart_rate", 60),
        metric(at(10, "01:00"), "heart_rate", 50), metric(at(10, "03:00"), "heart_rate", 54), metric(at(10, "13:00"), "heart_rate", 120),
        metric(at(10, "02:00"), "heart_rate_variability", 60), metric(at(10, "04:00"), "heart_rate_variability", 65),
        metric(at(10, "02:00"), "respiratory_rate", 14.25), metric(at(10, "02:30"), "heart_rate", None),
    ], NY)
    assert (night.resting_hr, night.sleeping_hr, night.hrv_ms, night.resp) == (53, 52, 62, 14.2)
    bare = attach_metrics(night_of(rows(at(9, "23:00"), ("Core", 420)))[0], [], NY)
    assert (bare.resting_hr, bare.sleeping_hr, bare.hrv_ms, bare.resp) == (None, None, None, None)
    assert "HR" not in summary_line(bare, NY) and "breathing" not in summary_line(bare, NY)


def test_the_summary_line_is_fixed_and_parses_back_to_the_same_numbers():
    [night] = night_of(rows(at(9, "23:48"), ("Awake", 10), ("Core", 240), ("Deep", 62), ("REM", 130), ("Awake", 18)))
    attach_metrics(night, [{"time": at(10, "00:00"), "metric_name": "resting_heart_rate", "value": 52},
                           {"time": at(10, "02:00"), "metric_name": "respiratory_rate", "value": 14.2}], NY)
    item = to_item(night, NY)
    assert item == {
        "external_id": "sleep:2026-10-10", "kind": "health", "title": "Sleep 7 h 12 min", "when": "2026-10-10T07:28:00-04:00",
        "summary": "asleep 7 h 12 min · in bed 7 h 40 min · 11:48 PM to 7:28 AM · core 240 min · deep 62 min · REM 130 min · "
                   "awake 28 min · resting HR 52 bpm · breathing 14.2/min"}
    reading = parse_summary(item["summary"])
    assert (reading.asleep_min, reading.in_bed_min, reading.asleep_h) == (432, 460, 7.2)
    assert reading.stage_min == {"core": 240, "deep": 62, "rem": 130, "awake": 28}
    assert (reading.resting_hr, reading.sleeping_hr, reading.hrv_ms, reading.resp) == (52, None, None, 14.2)
    assert night_day(item["external_id"]) == date(2026, 10, 10)
    assert parse_summary("until 5:00 PM · Library, 3rd floor") is None and parse_summary(None) is None
    assert night_day("outlook:study-session") is None and night_day("sleep:tonight") is None


# --- the fixture week, end to end through the connector's read ---


def test_the_fixture_week_gives_one_item_for_each_night_the_watch_was_worn():
    items = read_nights(OPTIONS, NOW, env={})
    assert [(i["external_id"], i["title"], i["when"]) for i in items] == [
        ("sleep:2026-10-05", "Sleep 5 h 5 min", "2026-10-05T05:11:00-04:00"),
        ("sleep:2026-10-07", "Sleep 6 h 50 min", "2026-10-07T07:40:00-04:00"),  # the nap that afternoon is not the night
        ("sleep:2026-10-08", "Sleep 6 h 35 min", "2026-10-08T07:05:00-04:00"),  # one night, across the 45 min with no stages
        ("sleep:2026-10-10", "Sleep 7 h 28 min", "2026-10-10T07:26:00-04:00"),
    ]  # Oct 4, 6 and 9: phone rows only (and a nap on the 9th), so no item at all
    assert all(i["kind"] == "health" and set(i) == {"external_id", "kind", "title", "when", "summary"} for i in items)
    last = items[-1]["summary"]
    assert last == ("asleep 7 h 28 min · in bed 7 h 46 min · 11:40 PM to 7:26 AM · core 250 min · deep 75 min · REM 123 min · "
                    "awake 18 min · resting HR 53 bpm · sleeping HR 57 bpm · HRV 67 ms · breathing 14.7/min")
    assert parse_summary(last).asleep_h == 7.47
    broken = parse_summary(items[2]["summary"])
    assert (broken.asleep_min, broken.in_bed_min, broken.stage_min["awake"]) == (395, 465, 25)
    assert all("step" not in i["summary"] and "Dominick" not in i["summary"] + i["title"] for i in items)
    assert all(ItemSynced(ts=NOW, connector="health", **i) for i in items)  # every item fits the contract


def test_the_fixture_has_the_quirks_the_real_tables_have():
    source = FixtureSource(FIXTURE)
    stages = source.sleep_stages(at(1, "00:00"), at(12, "00:00"))
    assert {r["source"] for r in stages} == {WATCH, PHONE}
    assert {r["stage"] for r in stages if r["source"] == PHONE} == {"In Bed"} and len([r for r in stages if r["source"] == PHONE]) == 7
    assert {r["stage"] for r in stages if r["source"] == WATCH} == {"Core", "Deep", "REM", "Awake"}
    raw = __import__("yaml").safe_load(FIXTURE.read_text(encoding="utf-8"))["health_metrics"]
    heart = [r for r in raw if r["metric_name"] == "heart_rate"]
    assert heart and all(r["qty"] is None and r["avg_val"] is not None for r in heart)
    assert any(r["metric_name"] == "step_count" for r in raw)
    read = source.metrics(at(1, "00:00"), at(12, "00:00"), ["heart_rate", "resting_heart_rate"])
    assert {r["metric_name"] for r in read} == {"heart_rate", "resting_heart_rate"} and all(r["value"] is not None for r in read)


def test_days_limits_how_far_back_it_reads():
    assert [i["external_id"] for i in read_nights(OPTIONS | {"days": 3}, NOW, env={})] == ["sleep:2026-10-08", "sleep:2026-10-10"]
    assert read_nights(OPTIONS, NOW + timedelta(days=30), env={}) == []


def test_fixture_rebase_moves_the_week_to_end_today_keeping_local_clock_times():
    now = datetime(2026, 11, 3, 10, 0, tzinfo=NY)  # the moved week crosses the end of daylight saving time
    items = read_nights(OPTIONS | {"fixture_rebase": True}, now, env={})
    assert [i["external_id"] for i in items] == ["sleep:2026-10-29", "sleep:2026-10-31", "sleep:2026-11-01", "sleep:2026-11-03"]
    assert items[-1]["when"] == "2026-11-03T07:26:00-05:00" and items[-1]["title"] == "Sleep 7 h 28 min"


def test_a_json_export_in_the_same_shapes_reads_the_same(tmp_path):
    export = tmp_path / "export.json"
    export.write_text(json.dumps({"sleep_stages": [
        {"start_time": "2026-10-09T23:00:00-04:00", "end_time": "2026-10-10T05:00:00-04:00", "stage": "Core", "source": WATCH}]}))
    [item] = read_nights({"fixture": str(export)}, NOW, env={})
    assert item["title"] == "Sleep 6 h 0 min"


# --- not connected, and the database ---


def test_with_no_database_url_and_no_fixture_it_does_nothing():
    def never(url):
        raise AssertionError("nothing to connect to")

    assert read_nights({"days": 7}, NOW, env={}, connect=never) == []
    assert read_nights({}, NOW, env={"FREEREPS_DATABASE_URL": ""}, connect=never) == []


class FakeCursor:
    """Answers the two SELECTs from the fixture, the way psycopg would: tuples plus a description."""

    def __init__(self, log: list) -> None:
        self.log, self.source, self.rows, self.description = log, FixtureSource(FIXTURE), [], []

    def execute(self, sql: str, params: tuple = ()) -> None:
        self.log.append((sql, params))
        if "FROM sleep_stages" in sql:
            found, columns = self.source.sleep_stages(*params), ["start_time", "end_time", "stage", "source"]
        else:
            names, start, end = params
            found, columns = self.source.metrics(start, end, names), ["time", "metric_name", "source", "value"]
        self.description = [(c,) for c in columns]
        self.rows = [tuple(r[c] for c in columns) for r in found]

    def fetchall(self) -> list[tuple]:
        return self.rows


class FakeConnection:
    def __init__(self, log: list) -> None:
        self.log, self.closed = log, False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.log)

    def close(self) -> None:
        self.closed = True


def test_the_postgres_reader_only_selects_and_reads_heart_rate_through_coalesce():
    log, opened = [], []

    def connect(url):
        opened.append((url, FakeConnection(log)))
        return opened[-1][1]

    env = {"FREEREPS_DATABASE_URL": "postgresql://reader@pi.test/freereps"}
    items = read_nights({"agent_config": OPTIONS["agent_config"]}, NOW, env=env, connect=connect)
    assert items == read_nights(OPTIONS, NOW, env={})  # the same week, whichever way it is read
    assert [url for url, _ in opened] == [env["FREEREPS_DATABASE_URL"]] * 2 and all(conn.closed for _, conn in opened)
    statements = [sql for sql, _ in log]
    assert statements == [STAGES_SQL, METRICS_SQL]
    for sql in statements:
        assert sql.startswith("SELECT ") and ";" not in sql
        assert not re.search(r"\b(insert|update|delete|drop|alter|create|truncate|grant|copy|set)\b", sql, re.I)
        assert "sleep_sessions" not in sql
    assert "COALESCE(qty, avg_val)" in METRICS_SQL
    (_, stage_params), (_, metric_params) = log
    assert stage_params[1] == NOW and stage_params[0] < NOW - timedelta(days=7)  # parameters, never formatted into the SQL
    assert set(metric_params[0]) == {"resting_heart_rate", "heart_rate", "heart_rate_variability", "respiratory_rate"}
    assert "default_transaction_read_only=on" in SESSION_OPTIONS and "statement_timeout=" in SESSION_OPTIONS


def test_the_real_driver_is_asked_for_a_read_only_session_with_timeouts(monkeypatch):
    psycopg = pytest.importorskip("psycopg")
    seen = {}
    monkeypatch.setattr(psycopg, "connect", lambda url, **kw: seen.update(url=url, **kw) or FakeConnection([]))
    PostgresSource("postgresql://reader@pi.test/freereps").sleep_stages(NOW - timedelta(days=1), NOW)
    assert seen["options"] == SESSION_OPTIONS and seen["connect_timeout"] == 5 and seen["autocommit"] is True


def test_an_unreachable_database_falls_back_to_the_fixture_or_fails_without_the_address(capsys):
    def down(url):
        raise OSError(f"could not connect to {url}")

    env = {"FREEREPS_DATABASE_URL": "postgresql://reader:hunter2@100.64.0.9/freereps"}
    assert len(read_nights(OPTIONS, NOW, env=env, connect=down)) == 4
    with pytest.raises(RuntimeError) as failed:
        read_nights({}, NOW, env=env, connect=down)
    said = str(failed.value) + capsys.readouterr().err
    assert "OSError" in said and "hunter2" not in said and "100.64.0.9" not in said  # a sync failure is an event: no credentials in it
    assert failed.value.__cause__ is None


# --- the committed config, and the server as the hub runs it ---


def test_the_committed_config_is_enabled_and_does_nothing_until_it_is_connected():
    spec = next(s for s in load_specs(ROOT / "config" / "connectors.yaml") if s.name == "health")
    assert (spec.enabled, spec.every_s, spec.env, spec.hosts) == (True, 1800, ["FREEREPS_DATABASE_URL"], [])
    assert spec.reads == {"list_nights": "health"} and not spec.writes and not spec.tools
    assert (spec.options["days"], spec.options["watch_sources"], spec.options["fixture"]) == (7, ["Apple Watch"], None)
    assert "postgresql://" not in (ROOT / "config" / "connectors.yaml").read_text()  # the address is a secret: .env only
    assert read_nights(spec.options, NOW, env={}) == []


def test_the_server_syncs_the_fixture_through_the_hub_and_is_quiet_when_unset(monkeypatch):
    monkeypatch.delenv("FREEREPS_DATABASE_URL", raising=False)
    monkeypatch.chdir(ROOT)
    [spec] = [s for s in load_specs(ROOT / "config" / "connectors.yaml") if s.name == "health"]
    now = datetime.now(timezone.utc)

    async def sync(options: dict) -> list:
        hub = ConnectorHub([spec.model_copy(update={"options": options})], factory=McpConnector)
        try:
            return await hub.sync("health", now) + await hub.sync("health", now)  # the second sync has nothing new
        finally:
            await hub.close()

    assert asyncio.run(sync(spec.options)) == []  # the committed config: no items and no connector.sync.failed
    events = asyncio.run(sync(spec.options | {"fixture": str(FIXTURE), "fixture_rebase": True}))
    assert [type(e) for e in events] == [ItemSynced] * 4
    assert {(e.connector, e.kind) for e in events} == {("health", "health")}
    today = now.astimezone(NY).date()
    assert events[-1].external_id == f"sleep:{today}"
    # The rebased fixture's last night ends at 7:26 AM "today". Before then that night is not over, so what the server
    # says about it is not the finished night: its wake time and total are checked only once 7:26 has passed.
    if now.astimezone(NY).time() >= time(7, 26):
        assert events[-1].when.astimezone(NY).time() == time(7, 26)
        assert parse_summary(events[-1].summary).asleep_min == 448
