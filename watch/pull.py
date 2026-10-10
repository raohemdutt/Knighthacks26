"""Pull the watch data from FreeReps (TimescaleDB on atlas) into fixtures/watch/*.csv.gz.

Needs `ssh <host>` to work without a password and the FreeReps db container running there.
Credentials, tokens, sync state, users and import logs are never pulled; user_id is dropped.
The SQL goes in on stdin, so nothing here needs shell quoting on the far side.
"""

from __future__ import annotations

import argparse
import gzip
import subprocess
from pathlib import Path

DEFAULT_HOST = "domdd@atlas"
DEFAULT_CONTAINER = "freereps-db-1"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "fixtures" / "watch"

# file stem -> SELECT. Ordered so the files diff cleanly between pulls.
TABLES: dict[str, str] = {
    "health_metrics": """
        SELECT time, metric_name, source, units, qty, min_val, avg_val, max_val, systolic, diastolic, source_uuid
        FROM health_metrics ORDER BY time, metric_name, source_uuid""",
    "workouts": """
        SELECT id, name, start_time, end_time, duration_sec, location, is_indoor,
               active_energy_burned, active_energy_units, total_energy, total_energy_units,
               distance, distance_units, avg_heart_rate, max_heart_rate, min_heart_rate,
               elevation_up, elevation_down, source, raw_json
        FROM workouts ORDER BY start_time""",
    "workout_heart_rate": """
        SELECT time, workout_id, min_bpm, avg_bpm, max_bpm, source
        FROM workout_heart_rate ORDER BY time, workout_id""",
    "sleep_sessions": """
        SELECT date, total_sleep, asleep, core, deep, rem, in_bed,
               sleep_start, sleep_end, in_bed_start, in_bed_end
        FROM sleep_sessions ORDER BY date, sleep_start""",
    "category_samples": """
        SELECT id, type, value, value_label, start_date, end_date, source
        FROM category_samples ORDER BY start_date, id""",
    "activity_summaries": """
        SELECT date, active_energy, active_energy_goal, exercise_time, exercise_time_goal,
               stand_hours, stand_hours_goal
        FROM activity_summaries ORDER BY date""",
    "ecg_recordings": """
        SELECT id, classification, average_heart_rate, sampling_frequency, start_date, source,
               voltage_measurements
        FROM ecg_recordings ORDER BY start_date""",
}


def copy_sql(select: str) -> str:
    body = " ".join(select.split())
    return f"SET TIME ZONE 'UTC';\nCOPY ({body}) TO STDOUT WITH CSV HEADER;\n"


def pull_table(stem: str, host: str, container: str) -> bytes:
    remote = f'docker exec -i {container} sh -c \'psql -X -q -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"\''
    proc = subprocess.run(["ssh", host, remote], input=copy_sql(TABLES[stem]).encode(),
                          capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{stem}: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


def write_gz(path: Path, data: bytes) -> None:
    # mtime=0 so an unchanged table gives a byte-identical file
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(data)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m watch pull", description=__doc__.splitlines()[0])
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--container", default=DEFAULT_CONTAINER)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("tables", nargs="*", metavar="table", help=f"default: all of {', '.join(TABLES)}")
    args = ap.parse_args(argv)
    unknown = set(args.tables) - set(TABLES)
    if unknown:
        ap.error(f"unknown table(s): {', '.join(sorted(unknown))}")
    args.out.mkdir(parents=True, exist_ok=True)
    for stem in args.tables or TABLES:
        data = pull_table(stem, args.host, args.container)
        write_gz(args.out / f"{stem}.csv.gz", data)
        lines = data.count(b"\n")  # one per row plus the header (JSON cells hold no raw newlines)
        print(f"{stem}: {max(lines - 1, 0)} rows")
    return 0
