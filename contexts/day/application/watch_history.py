"""The watch's history for `GET /api/day/patterns`: read from the export on disk when the request comes in.

    export_patterns(config_dir, clock) -> [{"text": ..., "n": ..., "from": ..., "to": ..., "source": "watch"}, ...]

It reads the export the health connector is configured to read (config/connectors.yaml,
`health.options.export_dir`), and nothing when that is not set. The export's own dates are used, whatever
`export_rebase` says: this is history, not last night. Only nights with measured stages count (the phone's
In Bed rows of earlier years are not measurements). No event is published and nothing is stored or sent.
"""

from datetime import datetime, timedelta
from pathlib import Path

import yaml

from contexts.day.domain.watch_patterns import watch_patterns


def export_patterns(config_dir: str | Path, clock) -> list[dict]:
    path = Path(config_dir) / "connectors.yaml"
    health = ((yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.exists() else {}).get("health") or {}
    options = health.get("options") or {}
    if not options.get("export_dir") or health.get("enabled") is False:
        return []
    # the connector's own reader and its rules for what a night is: read-only, and written down in one place only
    from contexts.connectors.domain.health import nights, watch_stages
    from contexts.connectors.infrastructure.health_export import ExportSource
    from watch import analyze

    source = ExportSource(options["export_dir"], clock.tz)
    last = source.newest()
    if last is None:
        return []
    first, end = datetime(2000, 1, 1, tzinfo=clock.tz), last + timedelta(days=1)
    names = options.get("watch_sources")
    found = nights(watch_stages(source.sleep_stages(first, end), ["Apple Watch"] if names is None else names),
                   clock.tz, clock.bedtime, clock.wake)
    if not found:
        return []
    start = found[0].bed - timedelta(days=1)
    workout_days = {w["start_time"].astimezone(clock.tz).date() for w in source.workouts(start, end)}
    resting = analyze.daily(({"time": r["time"], "qty": r["value"]} for r in source.metrics(start, end, ["resting_heart_rate"])),
                            tz=clock.tz)
    return watch_patterns(found, workout_days, resting, clock.tz)
