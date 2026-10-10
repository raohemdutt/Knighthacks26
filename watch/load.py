"""Read fixtures/watch/*.csv.gz into plain dicts with typed values (stdlib only)."""

from __future__ import annotations

import csv
import gzip
import sys
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path

from watch.pull import DEFAULT_OUT as DATA_DIR

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))  # the ECG voltages are one huge JSON cell

TIME_COLS = {"time", "start_time", "end_time", "start_date", "end_date",
             "sleep_start", "sleep_end", "in_bed_start", "in_bed_end"}
DATE_COLS = {"date"}
TEXT_COLS = {"id", "workout_id", "source_uuid", "name", "metric_name", "source", "units", "type",
             "value_label", "location", "classification", "raw_json", "voltage_measurements",
             "active_energy_units", "total_energy_units", "distance_units"}
BOOL_COLS = {"is_indoor"}


def parse_ts(s: str) -> datetime:
    # Postgres prints "2017-11-09 10:00:00+00"; Python 3.11+ fromisoformat accepts that offset form
    return datetime.fromisoformat(s)


def _convert(col: str, raw: str):
    if col in TEXT_COLS:
        return raw
    if raw == "":
        return None
    if col in TIME_COLS:
        return parse_ts(raw)
    if col in DATE_COLS:
        return date.fromisoformat(raw)
    if col in BOOL_COLS:
        return raw == "t"
    return float(raw)


def parse_row(row: dict[str, str]) -> dict:
    return {k: _convert(k, v) for k, v in row.items()}


def rows(table: str, data_dir: Path = DATA_DIR) -> Iterator[dict]:
    with gzip.open(data_dir / f"{table}.csv.gz", "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            yield parse_row(row)


def metric(name: str, data_dir: Path = DATA_DIR) -> Iterator[dict]:
    """Rows of one health_metrics series. Cheaper than rows(): skips other metrics before parsing."""
    with gzip.open(data_dir / "health_metrics.csv.gz", "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["metric_name"] == name:
                yield parse_row(row)
