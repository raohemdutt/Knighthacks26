"""Health import from a watch (K9) as an MCP connector: python -m contexts.connectors.servers.health

One read tool, `list_nights`: one item per night of the last `days` days the watch was worn, kind `health`:

    {"external_id": "sleep:2026-10-10", "kind": "health", "title": "Sleep 7 h 28 min",
     "when": "2026-10-10T07:26:00-04:00",
     "summary": "asleep 7 h 28 min · in bed 7 h 46 min · 11:40 PM to 7:26 AM · core 250 min · deep 75 min · "
                "REM 123 min · awake 18 min · resting HR 53 bpm · sleeping HR 57 bpm · HRV 67 ms · breathing 14.7/min"}

A night the watch recorded no stages for (only `In Bed` rows) says so, and gives no time asleep:

    {"external_id": "sleep:2025-01-14", "kind": "health", "title": "In bed 7 h 5 min (no sleep stages)",
     "summary": "in bed 7 h 5 min · 11:50 PM to 7:20 AM · no sleep stages: time in bed, not measured sleep · ..."}

The data is Apple Health, uploaded by the FreeReps iPhone app to a Postgres database on a Raspberry Pi and
read with a read-only role. How a night is put together from its rows: contexts/connectors/domain/health.py.
Steps and the other all-day metrics are not read.

Where it reads from, the first of these that is there:
- `FREEREPS_DATABASE_URL` (.env; a credential, so never in config): the database;
- `options.export_dir`: a directory of the tables as `<table>.csv.gz` (fixtures/watch is one person's real
  export, so each box's owner opts in);
- `options.fixture`: one YAML or JSON file in the tables' shapes (fixtures/health/freereps.yaml is synthetic);
- none of them: nothing, no items and no error, like a calendar feed whose URL is not set on this box.
When the database is set but cannot be reached, the next one that is configured is read instead.

Raw health data stays on the box: these items go to Memory and Habits and nowhere else.

options:
  days: 7                          # how many days back to read
  watch_sources: ["Apple Watch"]   # stage rows count when their source is empty or contains one of these;
                                   # In Bed rows only when it contains one (add an older device's label here)
  export_dir: null                 # a directory with category_samples.csv.gz and health_metrics.csv.gz
  export_rebase: false             # move the export's dates so its most recent night ends today (demos)
  fixture: null                    # a YAML or JSON file with category_samples and health_metrics rows
  fixture_rebase: false            # move the file's dates so its `as_of` day is today (demos)
  agent_config: config/agent.yaml  # the household's time zone and sleep window (`sleep`)
"""

import os
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from mcp.server.fastmcp import FastMCP

from contexts.connectors.domain.health import METRICS, Night, attach_metrics, nights, to_item, watch_stages
from contexts.connectors.infrastructure.freereps import ExportSource, FixtureSource, HealthSource, PostgresSource
from contexts.connectors.infrastructure.mcp_client import server_options

URL_ENV = "FREEREPS_DATABASE_URL"
DEFAULT_WATCH_SOURCES = ["Apple Watch"]
REBASE_LOOKBACK = timedelta(days=14)  # the export's last night is looked for this far before its last sleep row


def household(options: dict) -> tuple[ZoneInfo, time, time]:
    """(time zone, bedtime, wake) from config/agent.yaml `sleep`; the same defaults as Habits."""
    path = Path(options.get("agent_config", "config/agent.yaml"))
    raw = ((yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.exists() else {}).get("sleep") or {}
    return (ZoneInfo(raw.get("timezone", "America/New_York")), time.fromisoformat(raw.get("bedtime", "20:00")),
            time.fromisoformat(raw.get("wake", "08:00")))


def _found(source: HealthSource, options: dict, start: datetime, end: datetime) -> list[Night]:
    tz, bedtime, wake = household(options)
    stages = watch_stages(source.sleep_samples(start, end), options.get("watch_sources") or DEFAULT_WATCH_SOURCES)
    return nights(stages, tz, bedtime, wake)


def nights_from(source: HealthSource, options: dict, now: datetime) -> list[dict]:
    """The last `days` days of nights from one source, oldest first."""
    tz = household(options)[0]
    days = int(options.get("days", 7))
    today = now.astimezone(tz).date()
    start = datetime.combine(today - timedelta(days=days + 1), time(12), tz)  # room for the first night's evening
    found = [n for n in _found(source, options, start, now) if n.day > today - timedelta(days=days)]
    if not found:
        return []
    metrics = source.metrics(found[0].bed - timedelta(days=1), now, list(METRICS))
    return [to_item(attach_metrics(n, metrics, tz), tz) for n in found]


def export_source(options: dict, now: datetime) -> ExportSource:
    """The export directory; with `export_rebase`, moved by whole local days so its last night ends today."""
    tz = household(options)[0]
    source = ExportSource(options["export_dir"], tz)
    last = source.last_sleep_end() if options.get("export_rebase") else None
    if last:
        found = _found(source, options, last - REBASE_LOOKBACK, last + timedelta(seconds=1))
        if found:
            source.shift_days = (now.astimezone(tz).date() - found[-1].day).days
    return source


def read_nights(options: dict, now: datetime, env=os.environ, connect=None) -> list[dict]:
    """What `list_nights` returns. `connect` replaces the Postgres driver in tests."""
    url, export_dir, fixture = env.get(URL_ENV), options.get("export_dir"), options.get("fixture")
    if url:
        try:
            return nights_from(PostgresSource(url, connect), options, now)
        except Exception as exc:
            if not (export_dir or fixture):
                raise RuntimeError(f"the health database could not be read ({type(exc).__name__})") from None
            # the driver's own message can carry the host or the user name: only the kind of error is said
            print(f"health: database not reachable ({type(exc).__name__}); reading {export_dir or fixture}", file=sys.stderr)
    if export_dir:
        return nights_from(export_source(options, now), options, now)
    if not fixture:
        return []  # not connected on this box
    tz = household(options)[0]
    today = now.astimezone(tz).date() if options.get("fixture_rebase") else None
    return nights_from(FixtureSource(fixture, tz, today), options, now)


mcp = FastMCP("atlas-health")


@mcp.tool()
def list_nights() -> list[dict]:
    """One item per night the watch measured in the last few days: time asleep, stages, heart numbers."""
    return read_nights(server_options(), datetime.now(timezone.utc))


if __name__ == "__main__":
    mcp.run()
