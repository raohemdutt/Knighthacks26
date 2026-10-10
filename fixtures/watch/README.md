# Apple Watch / Apple Health export (Dominick)

Real exemplar data: Dominick's Apple Health history, as stored in FreeReps (TimescaleDB) on the
`atlas` Pi. Dominick has OK'd the team seeing it.
Pulled 2026-10-10; it covers 2017-11 to 2026-10.

| File | Rows | What it is |
| --- | --- | --- |
| `health_metrics.csv.gz` | 595,531 | Long format: one sample per row (`time`, `metric_name`, `units`, `qty` or `min_val`/`avg_val`/`max_val`). 34 metrics: heart_rate, resting_heart_rate, heart_rate_variability, respiratory_rate, step_count, walking_*, running_speed, active/basal energy, weight and body composition, audio exposure, and more. Run `python -m watch summary` for the full list. |
| `workouts.csv.gz` | 135 | One row per workout: type, start/end, duration, distance, energy, avg/min/max HR, and the raw export JSON. |
| `workout_heart_rate.csv.gz` | 5,772 | Per-minute min/avg/max bpm during workouts (`workout_id` → `workouts.id`). |
| `sleep_sessions.csv.gz` | 1,335 | One row per night: hours asleep, core/deep/REM, in-bed, start/end times. |
| `category_samples.csv.gz` | 16,363 | HealthKit category events: stand hours, sleep stage intervals, low/high HR alerts, mindful sessions, audio exposure events. |
| `activity_summaries.csv.gz` | 1,113 | Daily ring totals and goals. |
| `ecg_recordings.csv.gz` | 1 | One ECG (sinus rhythm), with the raw voltage samples as a JSON array. |

All timestamps are UTC. `user_id` is dropped. FreeReps' credentials, OAuth tokens, users, sync state
and import logs were never pulled.

## Quirks worth knowing
- `heart_rate` rows have `qty` NULL. Use `avg_val`; `watch.analyze.value()` does `COALESCE(qty, avg_val)`.
- The watch is mostly worn for **runs and sleep**. All-day totals (steps, calories, stand hours, rings)
  undercount, so treat them as biased.
- `source` is empty for most rows. Older watch rows are labelled with a previous device name.
- Sleep stages (core/deep/REM) exist only from 2026-01-26. Nights before that have only `in_bed`
  (use `analyze.sleep_nights(..., staged_only=True)` for staged nights). 56 rows are under 1 h (naps,
  fragments) and 27 are over 12 h (merged nights). Filter those out if you need clean nightly numbers.
- Days are bucketed in America/New_York by the analysis code.

## Using it

```
python -m watch summary                                   # every metric: rows, units, date span
python -m watch daily resting_heart_rate --days 180 --weekly
python -m watch daily step_count --agg sum --rolling 7
python -m watch sleep --days 90 --goal 8 [--nights]
python -m watch workouts --max-hr 195                    # per type, plus HR-zone minutes
python -m watch pull                                      # refresh from atlas (needs ssh domdd@atlas)
```

In Python:

```python
from watch import load, analyze
rhr = analyze.daily(load.metric("resting_heart_rate"))     # {date: bpm}
nights = analyze.sleep_nights(load.rows("sleep_sessions"))
analyze.sleep_summary(nights[-30:])
```

The code (stdlib only) is in `watch/`: `pull.py` (export), `load.py` (typed rows), `analyze.py`
(daily/weekly/rolling series, describe, trend, sleep, workouts, HR zones), `cli.py`.
