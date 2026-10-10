"""Patterns from a watch's own history (the K9 export): counted statements, nothing fitted, or nothing at all.

    watch_patterns(nights, workout_days, resting) ->
        [{"text": "On the 31 nights before a workout day, median sleep was 6 h 2 min, against 6 h 55 min on the
                   other 118 (watch, 2026-01-26 to 2026-09-29).", "n": 149, "from": "2026-01-26", "to": "2026-09-29",
          "source": "watch"}, ...]

`nights` are the measured nights (contexts/connectors/domain/health.py: each has the day it ended on, when it
began and its minutes asleep), `workout_days` the local days with a workout, `resting` the resting heart rate
per local day. Every statement says how many nights it rests on and the dates they span, and is left out when
either side of its comparison has fewer than MIN_N nights. These are one person's counts, not findings: with
the watch worn mostly for runs and sleep, they say what was recorded and nothing about why.

contexts/day/application/watch_history.py reads them from the export at request time. Nothing here is
published as an event, stored, or sent anywhere.
"""

from datetime import date, datetime, time, tzinfo

MIN_N = 8
SHORT_H, LONG_H = 6.0, 7.0
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _hm(minutes: float) -> str:
    minutes = round(minutes)
    return f"{minutes // 60} h {minutes % 60} min"


def _clock(hours_after_noon: float) -> str:
    minutes = round((hours_after_noon + 12) % 24 * 60) % (24 * 60)
    return datetime.combine(date(2000, 1, 1), time(minutes // 60, minutes % 60)).strftime("%I:%M %p").lstrip("0")


def watch_patterns(nights: list, workout_days: set[date], resting: dict[date, float], tz: tzinfo, min_n: int = MIN_N) -> list[dict]:
    from watch import analyze  # the team's stdlib helpers for this data; not a dependency unless a watch export is read

    if len(nights) < 2 * min_n:
        return []
    median = lambda xs: analyze.describe(xs)["median"]  # noqa: E731
    days = [n.day for n in nights]
    span = {"n": len(nights), "from": min(days).isoformat(), "to": max(days).isoformat(), "source": "watch"}
    tail = f"(watch, {span['from']} to {span['to']})"
    out: list[dict] = []

    # sleep on the night before a day with a workout, against the other nights
    before = [n.asleep_min for n in nights if n.day in workout_days]
    other = [n.asleep_min for n in nights if n.day not in workout_days]
    if len(before) >= min_n and len(other) >= min_n:
        out.append(span | {"text": f"On the {len(before)} nights before a workout day, median sleep was {_hm(median(before))}, "
                                   f"against {_hm(median(other))} on the other {len(other)} {tail}."})

    # resting heart rate on the day a short night ended, against the day a long one did
    short = [resting[n.day] for n in nights if n.day in resting and n.asleep_min < SHORT_H * 60]
    long = [resting[n.day] for n in nights if n.day in resting and n.asleep_min > LONG_H * 60]
    if len(short) >= min_n and len(long) >= min_n:
        out.append(span | {"n": len(short) + len(long),
                           "text": f"Resting heart rate after the {len(short)} nights under {SHORT_H:g} h had a median of "
                                   f"{median(short):.0f} bpm, against {median(long):.0f} bpm after the {len(long)} nights over "
                                   f"{LONG_H:g} h {tail}."})

    # bedtime by the weekday the night ran into: the latest and the earliest, of the weekdays with enough nights
    by_weekday: dict[int, list[float]] = {}
    for n in nights:
        bed = n.bed.astimezone(tz)
        by_weekday.setdefault(n.day.weekday(), []).append((bed.hour + bed.minute / 60 - 12) % 24)  # hours after noon
    medians = {w: median(v) for w, v in by_weekday.items() if len(v) >= min_n}
    if len(medians) >= 2:
        late, early = max(medians, key=medians.get), min(medians, key=medians.get)
        if late != early:
            out.append(span | {"n": sum(len(by_weekday[w]) for w in medians),
                               "text": f"Bedtime was latest on nights into {WEEKDAYS[late]} (median {_clock(medians[late])}, "
                                       f"{len(by_weekday[late])} nights) and earliest on nights into {WEEKDAYS[early]} "
                                       f"({_clock(medians[early])}, {len(by_weekday[early])} nights): "
                                       f"{_hm((medians[late] - medians[early]) * 60)} apart {tail}."})
    return out

