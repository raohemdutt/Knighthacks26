"""The watch's data from an export on disk (K9): `python -m watch pull` writes FreeReps' tables as gzipped CSVs.

    source = ExportSource("fixtures/watch", tz)       # the folder with *.csv.gz
    source.sleep_stages(start, end)                    # category_samples' sleep rows: start_time, end_time, stage, source
    source.sleep_sessions(start, end)                  # sleep_sessions rows (hours), times parsed
    source.metrics(start, end, names)                  # health_metrics: time, metric_name, source, value
    source.workouts(start, end)                        # start_time, end_time, name, duration_sec
    source.newest()                                    # the end of the last sleep row in the files
    source.moved(days)                                 # the same data, every timestamp `days` local days later

Read with the standard library only, and nothing is ever written. The files' timestamps are UTC, printed the
way Postgres prints them ("2026-09-29 09:29:49+00"), which Python 3.10's `fromisoformat` does not read, so
they are parsed here (`watch/load.py` needs 3.11 for the same reason).

The export has no `sleep_stages` table. The stage intervals are the `HKCategoryTypeIdentifierSleepAnalysis`
rows of `category_samples`, whose labels are "Asleep Core", "Asleep Deep", "Asleep REM", "Awake" and "In Bed".

`health_metrics` is 600 thousand rows, so only the asked-for names and days are parsed, and each table is read
once per file version (the connector's server stays up between syncs), keeping only the columns used.
"""

import csv
import gzip
import re
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path

SLEEP_TYPE = "HKCategoryTypeIdentifierSleepAnalysis"
SESSION_COLUMNS = ("date", "total_sleep", "asleep", "core", "deep", "rem", "in_bed",
                   "sleep_start", "sleep_end", "in_bed_start", "in_bed_end")
_TS = re.compile(r"^(\d{4})-(\d\d)-(\d\d)[ T](\d\d):(\d\d):(\d\d)(?:\.(\d{1,6}))?(?:Z|\+00(?::?00)?)$")
_cache: dict[tuple, list] = {}

csv.field_size_limit(2**31 - 1)  # a workout's raw JSON is one very long cell


def parse_utc(text: str) -> datetime:
    """"2026-09-29 09:29:49+00" (also with fractions, "Z" or "+00:00") as an aware UTC datetime."""
    m = _TS.match(text)
    if not m:
        raise ValueError(f"not a UTC timestamp: {text!r}")
    micro = int((m[7] or "0").ljust(6, "0"))
    return datetime(*(int(g) for g in m.groups()[:6]), micro, tzinfo=timezone.utc)


def _key(ts: datetime) -> str:
    """A timestamp as the files print it, to the second: such strings sort in time order."""
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _table(path: Path, tag: str, keep, columns: tuple[str, ...]) -> list[dict]:
    """The kept rows of one file, only `columns` of them, read once per version of the file."""
    if not path.exists():
        return []
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size, tag)
    if key not in _cache:
        for old in [k for k in _cache if k[0] == key[0]]:
            del _cache[old]
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            _cache[key] = [{c: row.get(c, "") for c in columns} for row in csv.DictReader(f) if keep(row)]
    return _cache[key]


def _number(text: str | None) -> float | None:
    return float(text) if text not in (None, "") else None


class ExportSource:
    """FreeReps' tables from a folder of gzipped CSVs, in the shapes the database reader gives.

    `days` moves every timestamp that many whole local days later (for demos on the real clock): local clock
    times stay what they were, the same way `FixtureSource` moves its week."""

    def __init__(self, folder: str | Path, tz: tzinfo | None = None, days: int = 0) -> None:
        self._dir, self._tz, self._days = Path(folder), tz, days if tz else 0

    def exists(self) -> bool:
        return (self._dir / "category_samples.csv.gz").exists() or (self._dir / "sleep_sessions.csv.gz").exists()

    def moved(self, days: int) -> "ExportSource":
        return ExportSource(self._dir, self._tz, days)

    def shift(self, ts: datetime) -> datetime:
        if not self._days:
            return ts
        return (ts.astimezone(self._tz).replace(tzinfo=None) + timedelta(days=self._days)).replace(tzinfo=self._tz)

    # --- the tables ---

    def _stage_rows(self) -> list[dict]:
        return _table(self._dir / "category_samples.csv.gz", "sleep", lambda r: r["type"] == SLEEP_TYPE,
                      ("start_date", "end_date", "value_label", "source"))

    def _session_rows(self) -> list[dict]:
        return _table(self._dir / "sleep_sessions.csv.gz", "all", lambda r: True, SESSION_COLUMNS)

    def sleep_stages(self, start: datetime, end: datetime) -> list[dict]:
        out = []
        for r in self._stage_rows():
            a, b = self.shift(parse_utc(r["start_date"])), self.shift(parse_utc(r["end_date"]))
            if b >= start and a < end:
                out.append({"start_time": a, "end_time": b, "stage": r["value_label"], "source": r["source"]})
        return sorted(out, key=lambda r: r["start_time"])

    def sleep_sessions(self, start: datetime, end: datetime) -> list[dict]:
        out = []
        for r in self._session_rows():
            row: dict = {"date": date.fromisoformat(r["date"])}
            for k in ("total_sleep", "asleep", "core", "deep", "rem", "in_bed"):
                row[k] = _number(r.get(k))
            for k in ("sleep_start", "sleep_end", "in_bed_start", "in_bed_end"):
                row[k] = self.shift(parse_utc(r[k])) if r.get(k) else None
            a, b = row["sleep_start"] or row["in_bed_start"], row["sleep_end"] or row["in_bed_end"]
            if a and b and b >= start and a < end:
                out.append(row)
        return sorted(out, key=lambda r: r["sleep_start"] or r["in_bed_start"])

    def metrics(self, start: datetime, end: datetime, names: list[str]) -> list[dict]:
        wanted = frozenset(names)
        rows = _table(self._dir / "health_metrics.csv.gz", ",".join(sorted(wanted)), lambda r: r["metric_name"] in wanted,
                      ("time", "metric_name", "source", "qty", "avg_val"))
        # where the span sits in the files, a day wide either way for a clock change the move may cross
        moved = timedelta(days=self._days)
        low, high = _key(start - moved - timedelta(days=1)), _key(end - moved + timedelta(days=1))
        out = []
        for r in rows:
            if not low <= r["time"][:19] <= high:  # text first: parsing 140 thousand heart rates takes a second
                continue
            ts = self.shift(parse_utc(r["time"]))
            if start <= ts < end:
                qty = _number(r["qty"])
                out.append({"time": ts, "metric_name": r["metric_name"], "source": r["source"],
                            "value": qty if qty is not None else _number(r["avg_val"])})  # COALESCE(qty, avg_val)
        return sorted(out, key=lambda r: r["time"])

    def workouts(self, start: datetime, end: datetime) -> list[dict]:
        out = []
        for r in _table(self._dir / "workouts.csv.gz", "all", lambda r: True, ("start_time", "end_time", "name", "duration_sec")):
            a = self.shift(parse_utc(r["start_time"]))
            if start <= a < end:
                out.append({"start_time": a, "end_time": self.shift(parse_utc(r["end_time"])) if r["end_time"] else None,
                            "name": r["name"], "duration_sec": _number(r["duration_sec"])})
        return sorted(out, key=lambda r: r["start_time"])

    def newest(self) -> datetime | None:
        """The end of the last sleep row (a stage, or a session when the files have no stages), as moved."""
        ends = [r["end_date"] for r in self._stage_rows() if r["value_label"] != "In Bed"]
        if not ends:
            ends = [r["sleep_end"] or r["in_bed_end"] for r in self._session_rows()]
        ends = [e for e in ends if e]
        return self.shift(parse_utc(max(ends))) if ends else None
