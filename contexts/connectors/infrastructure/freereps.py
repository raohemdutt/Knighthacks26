"""Where the watch's data is read from (K9): the FreeReps database, an export of it, or a file in the same shapes.

FreeReps (an iPhone app) uploads Apple Health data to a Postgres database on a Raspberry Pi. Two of its
tables are read here, and nothing is ever written. Their names and columns are the real ones, as the export
in fixtures/watch shows them (it is a dump of those tables, minus `user_id`):

    category_samples(id, type, value, value_label, start_date, end_date, source)
    health_metrics(time, metric_name, source, units, qty, min_val, avg_val, max_val, systolic, diastolic, source_uuid)

    source = PostgresSource(url)          # url from the environment, never from config
    source = ExportSource(directory, tz)  # <table>.csv.gz files, as `python -m watch pull` writes them
    source = FixtureSource(path)          # a YAML or JSON file: {category_samples: [...], health_metrics: [...]}
    source.sleep_samples(start, end)      # rows as dicts: start_date, end_date, value_label, source
    source.metrics(start, end, names)     # rows as dicts: time, metric_name, source, value

Sleep is the `category_samples` rows of type HKCategoryTypeIdentifierSleepAnalysis; there is no table of
stages. `value` is COALESCE(qty, avg_val): heart_rate rows have no qty, only min/avg/max.

The Postgres session is read-only (`default_transaction_read_only=on`), every statement is a SELECT, and both
the connection and each statement have a timeout, so an unreachable Pi costs a few seconds, not a sync.
The driver (psycopg 3, the `neon` extra: pip install -e ".[neon]") is imported on first use.

The export is read with the standard library only, and not through `watch.load`: that module needs Python
3.11 to read Postgres' "+00" offsets (the Jetson runs 3.10), it is not an installed package, and it parses
every row of the 595,000 in health_metrics where this skips all but the last few days before parsing any.
contexts/connectors/tests/test_health_export.py checks that both read the same rows out of the real files.
"""

import csv
import gzip
import itertools
import json
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Protocol

import yaml

from contexts.connectors.domain.health import as_time

CONNECT_TIMEOUT_S = 5
STATEMENT_TIMEOUT_MS = 10_000
SESSION_OPTIONS = f"-c default_transaction_read_only=on -c statement_timeout={STATEMENT_TIMEOUT_MS}"

SLEEP_TYPE = "HKCategoryTypeIdentifierSleepAnalysis"
SLEEP_SQL = (
    "SELECT start_date, end_date, value_label, source FROM category_samples "
    "WHERE type = %s AND end_date >= %s AND start_date < %s ORDER BY start_date"
)
METRICS_SQL = (
    "SELECT time, metric_name, source, COALESCE(qty, avg_val) AS value FROM health_metrics "
    "WHERE metric_name = ANY(%s) AND time >= %s AND time < %s ORDER BY time"
)
SLEEP_FILE, METRICS_FILE = "category_samples.csv.gz", "health_metrics.csv.gz"


class HealthSource(Protocol):
    def sleep_samples(self, start: datetime, end: datetime) -> list[dict]: ...

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

    def sleep_samples(self, start: datetime, end: datetime) -> list[dict]:
        return self._select(SLEEP_SQL, (SLEEP_TYPE, start, end))

    def metrics(self, start: datetime, end: datetime, names: list[str]) -> list[dict]:
        return self._select(METRICS_SQL, (list(names), start, end))


def moved(ts: datetime, tz: tzinfo, days: int) -> datetime:
    """`ts` this many local days later, at the same local clock time (so a night keeps its bed and wake times)."""
    if not days:
        return ts
    return (ts.astimezone(tz).replace(tzinfo=None) + timedelta(days=days)).replace(tzinfo=tz)


def _value(row: dict):
    """COALESCE(qty, avg_val), from a file's row (NULL is an empty cell in a CSV, null in YAML)."""
    return next((row[key] for key in ("qty", "avg_val") if row.get(key) not in (None, "")), None)


class ExportSource:
    """The same two tables from a directory of `<table>.csv.gz` dumps (fixtures/watch): timestamps in UTC as
    Postgres prints them, empty cells for NULL.

    `shift_days` moves every timestamp that many local days later (see `export_rebase` in the server).
    health_metrics is large, so its lines are dropped on their date (the first ten characters, which sort as
    text because the whole file is UTC) before anything is parsed, and what one window needed is kept until
    the file changes (its mtime and size)."""

    _kept: dict[tuple, list[dict]] = {}  # (file, mtime, size, first day, last day, names) -> parsed rows

    def __init__(self, directory: str | Path, tz: tzinfo, shift_days: int = 0) -> None:
        self.directory, self.tz, self.shift_days = Path(directory), tz, shift_days
        self._sleep: list[dict] | None = None

    def _file(self, name: str) -> Path:
        path = self.directory / name
        if not path.is_file():
            raise RuntimeError(f"the health export has no {name} in {self.directory}")
        return path

    def _all_sleep(self) -> list[dict]:
        if self._sleep is None:
            with gzip.open(self._file(SLEEP_FILE), "rt", encoding="utf-8", newline="") as f:
                self._sleep = [{"start_date": as_time(r["start_date"]), "end_date": as_time(r["end_date"]),
                                "value_label": r["value_label"], "source": r["source"]}
                               for r in csv.DictReader(f)
                               if r["type"] == SLEEP_TYPE and r["start_date"] and r["end_date"]]
        return self._sleep

    def last_sleep_end(self) -> datetime | None:
        """When the newest sleep row in the export ends (not moved)."""
        return max((r["end_date"] for r in self._all_sleep()), default=None)

    def sleep_samples(self, start: datetime, end: datetime) -> list[dict]:
        rows = [r | {"start_date": moved(r["start_date"], self.tz, self.shift_days),
                     "end_date": moved(r["end_date"], self.tz, self.shift_days)} for r in self._all_sleep()]
        return sorted((r for r in rows if r["end_date"] >= start and r["start_date"] < end), key=lambda r: r["start_date"])

    def _metric_rows(self, first: date, last: date, names: tuple[str, ...]) -> list[dict]:
        """Rows of these metrics dated (UTC) from `first` to `last`, parsed; kept until the file changes."""
        path = self._file(METRICS_FILE)
        stat = path.stat()
        key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size, first, last, names)
        if key not in self._kept:
            lo, hi = first.isoformat(), last.isoformat()
            with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
                header = f.readline()
                lines = (line for line in f if lo <= line[:10] <= hi)
                found = [{"time": as_time(r["time"]), "metric_name": r["metric_name"], "source": r["source"],
                          "value": float(v) if (v := _value(r)) is not None else None}
                         for r in csv.DictReader(itertools.chain([header], lines)) if r["metric_name"] in names]
            self._kept.clear()  # one window at a time is all a connector asks for
            self._kept[key] = found
        return self._kept[key]

    def metrics(self, start: datetime, end: datetime, names: list[str]) -> list[dict]:
        back = timedelta(days=self.shift_days)
        first = (start - back).astimezone(timezone.utc).date() - timedelta(days=1)  # a day's room each side: moving
        last = (end - back).astimezone(timezone.utc).date() + timedelta(days=1)  # by local days can cross a clock change
        rows = [r | {"time": moved(r["time"], self.tz, self.shift_days)}
                for r in self._metric_rows(first, last, tuple(sorted(names)))]
        return sorted((r for r in rows if start <= r["time"] < end), key=lambda r: r["time"])


class FixtureSource:
    """The same two tables from one file: a test fixture in the tables' shapes (fixtures/health/freereps.yaml).

    With `today` and an `as_of` date in the file, every timestamp is moved by whole local days so that
    `as_of` becomes `today` (for demos on the real clock); local clock times stay what they were."""

    def __init__(self, path: str | Path, tz: tzinfo | None = None, today: date | None = None) -> None:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        raw = (json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)) or {}
        as_of = raw.get("as_of")
        days = (today - date.fromisoformat(str(as_of))).days if today and as_of and tz else 0
        self._sleep = [row | {"start_date": moved(as_time(row["start_date"]), tz, days),
                              "end_date": moved(as_time(row["end_date"]), tz, days)}
                       for row in raw.get("category_samples") or [] if row.get("type", SLEEP_TYPE) == SLEEP_TYPE]
        self._metrics = [row | {"time": moved(as_time(row["time"]), tz, days)} for row in raw.get("health_metrics") or []]

    def sleep_samples(self, start: datetime, end: datetime) -> list[dict]:
        rows = [r for r in self._sleep if r["end_date"] >= start and r["start_date"] < end]
        return sorted(rows, key=lambda r: r["start_date"])

    def metrics(self, start: datetime, end: datetime, names: list[str]) -> list[dict]:
        rows = [r for r in self._metrics if r["metric_name"] in names and start <= r["time"] < end]
        return [{"time": r["time"], "metric_name": r["metric_name"], "source": r.get("source"), "value": _value(r)}
                for r in sorted(rows, key=lambda r: r["time"])]
