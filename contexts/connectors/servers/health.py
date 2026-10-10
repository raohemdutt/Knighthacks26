"""Health import from a watch (K9) as an MCP connector: python -m contexts.connectors.servers.health

One read tool, `list_nights`: one item per night of the last `days` days the watch was worn, kind `health`:

    {"external_id": "sleep:2026-10-10", "kind": "health", "title": "Sleep 7 h 28 min",
     "when": "2026-10-10T07:26:00-04:00",
     "summary": "asleep 7 h 28 min · in bed 7 h 46 min · 11:40 PM to 7:26 AM · core 250 min · deep 75 min · "
                "REM 123 min · awake 18 min · resting HR 53 bpm · sleeping HR 57 bpm · HRV 67 ms · breathing 14.7/min"}

The data is Apple Health, uploaded by the FreeReps iPhone app to a Postgres database on a Raspberry Pi and
read with a read-only role, or an export of that database on disk. How a night is put together from its
rows: contexts/connectors/domain/health.py. Steps and the other all-day metrics are not read.

Where it reads from, the first of these that is set:
- `FREEREPS_DATABASE_URL` (.env; a credential, so never in config);
- `options.export_dir`: a folder of gzipped CSVs written by `python -m watch pull` (fixtures/watch), also
  read when that database cannot be reached. Nights come from the stage intervals where the export has
  them, else from its `sleep_sessions` rows;
- `options.fixture`: one YAML or JSON file in the tables' shapes (fixtures/health/freereps.yaml);
- with none of them, nothing: no items and no error, like a calendar feed whose URL is not set on this box.

Raw health data stays on the box: these items go to Memory and Habits and nowhere else.

options:
  days: 7                          # how many days back to read
  watch_sources: ["Apple Watch"]   # a stage row that names a source must contain one of these; [] takes any
  export_dir: null                 # a folder with category_samples, sleep_sessions and health_metrics .csv.gz
  export_rebase: false             # move the export's dates so its newest night is last night (demos)
  export_in_bed_nights: false      # also take sleep_sessions rows with no time asleep (the phone's In Bed)
  fixture: null                    # a YAML or JSON file with sleep_stages and health_metrics rows
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

from contexts.connectors.domain.health import (
    METRICS, Night, attach_metrics, nights, session_nights, to_item, watch_stages, with_sessions,
)
from contexts.connectors.infrastructure.freereps import FixtureSource, HealthSource, PostgresSource
from contexts.connectors.infrastructure.health_export import ExportSource
from contexts.connectors.infrastructure.mcp_client import server_options

URL_ENV = "FREEREPS_DATABASE_URL"
DEFAULT_WATCH_SOURCES = ["Apple Watch"]


def household(options: dict) -> tuple[ZoneInfo, time, time]:
    """(time zone, bedtime, wake) from config/agent.yaml `sleep`; the same defaults as Habits."""
    path = Path(options.get("agent_config", "config/agent.yaml"))
    raw = ((yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.exists() else {}).get("sleep") or {}
    return (ZoneInfo(raw.get("timezone", "America/New_York")), time.fromisoformat(raw.get("bedtime", "20:00")),
            time.fromisoformat(raw.get("wake", "08:00")))


def watch_names(options: dict) -> list[str]:
    names = options.get("watch_sources")
    return DEFAULT_WATCH_SOURCES if names is None else list(names)


def measured_nights(source: HealthSource, options: dict, start: datetime, end: datetime) -> list[Night]:
    """Every night between `start` and `end`, oldest first, without its heart numbers: the watch's stages,
    and for a source that has `sleep_sessions` (an export) those rows on the days with no stages."""
    tz, bedtime, wake = household(options)
    found = nights(watch_stages(source.sleep_stages(start, end), watch_names(options)), tz, bedtime, wake)
    sessions = getattr(source, "sleep_sessions", None)
    if sessions is not None:
        rows = sessions(start, end)
        found = with_sessions(found, session_nights(rows, tz, bedtime, wake, in_bed_only=bool(options.get("export_in_bed_nights"))))
    return found


def nights_from(source: HealthSource, options: dict, now: datetime) -> list[dict]:
    """The last `days` days of nights from one source, oldest first."""
    tz = household(options)[0]
    days = int(options.get("days", 7))
    today = now.astimezone(tz).date()
    start = datetime.combine(today - timedelta(days=days + 1), time(12), tz)  # room for the first night's evening
    found = [n for n in measured_nights(source, options, start, now) if n.day > today - timedelta(days=days)]
    if not found:
        return []
    metrics = source.metrics(found[0].bed - timedelta(days=1), now, list(METRICS))
    return [to_item(attach_metrics(n, metrics, tz), tz) for n in found]


def export_source(options: dict, now: datetime) -> ExportSource | None:
    """The export in `options.export_dir`, or None when the folder is not there.

    With `export_rebase` its dates are moved by whole local days so that the newest night in it is last
    night: the one that ended today, or yesterday when it is still earlier in the day than that night ended."""
    tz = household(options)[0]
    source = ExportSource(options["export_dir"], tz)
    if not source.exists():
        print(f"health: no export in {options['export_dir']}", file=sys.stderr)
        return None
    newest = source.newest() if options.get("export_rebase") else None
    if newest is None:
        return source
    found = measured_nights(source, options, newest - timedelta(days=30), newest + timedelta(seconds=1))
    if not found:
        return source
    days = (now.astimezone(tz).date() - found[-1].day).days
    if source.moved(days).shift(found[-1].wake) > now:
        days -= 1
    return source.moved(days)


def read_nights(options: dict, now: datetime, env=os.environ, connect=None) -> list[dict]:
    """What `list_nights` returns. `connect` replaces the Postgres driver in tests."""
    url, export, fixture = env.get(URL_ENV), options.get("export_dir"), options.get("fixture")
    if url:
        try:
            return nights_from(PostgresSource(url, connect), options, now)
        except Exception as exc:
            if not export and not fixture:
                raise RuntimeError(f"the health database could not be read ({type(exc).__name__})") from None
            # the driver's own message can carry the host or the user name: only the kind of error is said
            print(f"health: database not reachable ({type(exc).__name__}); reading {export or fixture}", file=sys.stderr)
    if export:
        source = export_source(options, now)
        if source is not None:
            return nights_from(source, options, now)
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
