"""Where the watch's data is read from (K9): the FreeReps database, or a file in the same shapes.

FreeReps (an iPhone app) uploads Apple Health data to a Postgres database on a Raspberry Pi. Two of its
tables are read here, and nothing is ever written:

    sleep_stages(start_time, end_time, stage, duration_hr, source)
    health_metrics(time, metric_name, source, units, qty, min_val, avg_val, max_val)

    source = PostgresSource(url)          # url from the environment, never from config
    source = FixtureSource(path)          # a YAML or JSON file: {sleep_stages: [...], health_metrics: [...]}
    source.sleep_stages(start, end)       # rows as dicts: start_time, end_time, stage, source
    source.metrics(start, end, names)     # rows as dicts: time, metric_name, source, value

`value` is COALESCE(qty, avg_val): heart_rate rows have no qty, only min/avg/max.

The Postgres session is read-only (`default_transaction_read_only=on`), every statement is a SELECT, and both
the connection and each statement have a timeout, so an unreachable Pi costs a few seconds, not a sync.
The driver (psycopg 3, the `neon` extra: pip install -e ".[neon]") is imported on first use.
"""

import json
from collections.abc import Callable
from datetime import date, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Protocol

import yaml

from contexts.connectors.domain.health import as_time

CONNECT_TIMEOUT_S = 5
STATEMENT_TIMEOUT_MS = 10_000
SESSION_OPTIONS = f"-c default_transaction_read_only=on -c statement_timeout={STATEMENT_TIMEOUT_MS}"

STAGES_SQL = (
    "SELECT start_time, end_time, stage, source FROM sleep_stages "
    "WHERE end_time >= %s AND start_time < %s ORDER BY start_time"
)
METRICS_SQL = (
    "SELECT time, metric_name, source, COALESCE(qty, avg_val) AS value FROM health_metrics "
    "WHERE metric_name = ANY(%s) AND time >= %s AND time < %s ORDER BY time"
)


class HealthSource(Protocol):
    def sleep_stages(self, start: datetime, end: datetime) -> list[dict]: ...

    def metrics(self, start: datetime, end: datetime, names: list[str]) -> list[dict]: ...


def _connect(url: str):
    try:
        import psycopg
    except ImportError as e:
        raise RuntimeError('the Postgres driver is not installed: pip install -e ".[neon]"') from e
    return psycopg.connect(url, connect_timeout=CONNECT_TIMEOUT_S, autocommit=True, options=SESSION_OPTIONS)


class PostgresSource:
    """One short read-only connection per read: the box polls every half hour."""

    def __init__(self, url: str, connect: Callable[[str], object] | None = None) -> None:
        self._url = url
        self._connect = connect or _connect

    def _select(self, sql: str, params: tuple) -> list[dict]:
        conn = self._connect(self._url)
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            columns = [d[0] for d in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]
        finally:
            conn.close()

    def sleep_stages(self, start: datetime, end: datetime) -> list[dict]:
        return self._select(STAGES_SQL, (start, end))

    def metrics(self, start: datetime, end: datetime, names: list[str]) -> list[dict]:
        return self._select(METRICS_SQL, (list(names), start, end))


class FixtureSource:
    """The same two tables from a file: a test fixture, or an export for a box that cannot reach the Pi.

    With `today` and an `as_of` date in the file, every timestamp is moved by whole local days so that
    `as_of` becomes `today` (for demos on the real clock); local clock times stay what they were."""

    def __init__(self, path: str | Path, tz: tzinfo | None = None, today: date | None = None) -> None:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        raw = (json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)) or {}
        as_of = raw.get("as_of")
        days = (today - date.fromisoformat(str(as_of))).days if today and as_of and tz else 0

        def moved(value) -> datetime:
            ts = as_time(value)
            if not days:
                return ts
            return (ts.astimezone(tz).replace(tzinfo=None) + timedelta(days=days)).replace(tzinfo=tz)

        self._stages = [row | {"start_time": moved(row["start_time"]), "end_time": moved(row["end_time"])}
                        for row in raw.get("sleep_stages") or []]
        self._metrics = [row | {"time": moved(row["time"])} for row in raw.get("health_metrics") or []]

    def sleep_stages(self, start: datetime, end: datetime) -> list[dict]:
        rows = [r for r in self._stages if r["end_time"] >= start and r["start_time"] < end]
        return sorted(rows, key=lambda r: r["start_time"])

    def metrics(self, start: datetime, end: datetime, names: list[str]) -> list[dict]:
        rows = [r for r in self._metrics if r["metric_name"] in names and start <= r["time"] < end]
        return [{"time": r["time"], "metric_name": r["metric_name"], "source": r.get("source"),
                 "value": r.get("qty") if r.get("qty") is not None else r.get("avg_val")}  # COALESCE(qty, avg_val)
                for r in sorted(rows, key=lambda r: r["time"])]
