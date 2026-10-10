"""Energy over request windows: the integral, its edges, the split between overlapping requests, and the clock offset.

Times are seconds on one clock (the caller moves every time onto it first); power is watts.

    integrate(trace, start, end)          J under the trace over [start, end], or None when the trace does not
                                          bracket the window
    attribute(trace, windows)             each window's share of the node's energy, split by output tokens
    clock_offset(t1, t2, t3, t4)          (offset, round trip, error) from one four-timestamp exchange

The trace. A node's power is the straight line between consecutive samples, every sample kept: repeated identical
readings are points of the line like any other, never dropped (a deduplicated trace can leave a window inside a
steady stretch with no sample at all). The integral over a window uses the last sample at or before its start and
the first at or after its end, and clips the line to the window by linear interpolation, so the partial stretch at
each edge is counted, not dropped or counted whole.

The split. Between consecutive cut points (every sample time and every window edge) the line is one straight
piece, and the requests running over that piece are the same throughout. Each piece's energy goes to the requests
running in it in proportion to their output tokens (equally when none of them reports any), so a request alone on
the node gets exactly its window's integral and the shares of any set of requests never sum past the node's energy
over the stretch they cover. A window marked `fixed` was already reported: a piece it ran in was given out then,
so the others running in that piece get nothing more from it.
"""

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    key: str
    start: float
    end: float
    tokens: int | None = None
    fixed: bool = False  # already reported: its pieces are spent


def brackets(times: Sequence[float], start: float, end: float) -> tuple[int, int] | None:
    """Indices of the last sample at or before `start` and the first at or after `end`; None when either is missing."""
    lo = bisect_right(times, start) - 1
    hi = bisect_left(times, end)
    if lo < 0 or hi >= len(times):
        return None
    return lo, hi


def _pieces(trace: Sequence[tuple[float, float]], start: float, end: float, cuts: Sequence[float] = ()):
    """(a, b, joules) for each straight piece of the trace inside [start, end], cut also at `cuts`."""
    inner = sorted({c for c in cuts if start < c < end})
    for (t0, w0), (t1, w1) in zip(trace, trace[1:]):
        if t1 <= t0 or t1 <= start or t0 >= end:
            continue  # a repeated timestamp is a point, not a stretch; or outside the window
        a, b = max(start, t0), min(end, t1)
        slope = (w1 - w0) / (t1 - t0)
        points = [a, *(c for c in inner[bisect_right(inner, a):bisect_left(inner, b)]), b]
        for x, y in zip(points, points[1:]):
            wx, wy = w0 + slope * (x - t0), w0 + slope * (y - t0)
            yield x, y, (wx + wy) / 2 * (y - x)


def integrate(trace: Sequence[tuple[float, float]], start: float, end: float) -> float | None:
    """Joules under the trace over [start, end], the edges clipped by interpolation; None unless it is bracketed."""
    if end < start:
        raise ValueError("the window ends before it starts")
    if brackets([t for t, _ in trace], start, end) is None:
        return None
    return sum(j for _, _, j in _pieces(trace, start, end))


def attribute(trace: Sequence[tuple[float, float]], windows: Sequence[Window]) -> dict[str, float | None]:
    """Each window's joules: its token-proportional share of every piece of the trace it ran in.

    A window the trace does not bracket gets None and takes no share."""
    times = [t for t, _ in trace]
    out: dict[str, float | None] = {}
    live = []
    for w in windows:
        if w.end < w.start:
            raise ValueError(f"window {w.key} ends before it starts")
        if brackets(times, w.start, w.end) is None:
            out[w.key] = None
        else:
            out[w.key] = 0.0
            live.append(w)
    if not live:
        return out
    start, end = min(w.start for w in live), max(w.end for w in live)
    cuts = [x for w in live for x in (w.start, w.end)]
    for a, b, joules in _pieces(trace, start, end, cuts):
        running = [w for w in live if w.start <= a and w.end >= b]
        if not running or any(w.fixed for w in running):
            continue
        weights = [max(w.tokens or 0, 0) for w in running]
        total = sum(weights)
        for w, weight in zip(running, weights):
            out[w.key] += joules * (weight / total if total else 1 / len(running))
    return out


def clock_offset(t1: float, t2: float, t3: float, t4: float) -> tuple[float, float, float]:
    """(offset, round trip, error) of a remote clock from one exchange, as NTP computes them (RFC 5905, section 8).

    t1 request sent and t4 reply received are on the local clock; t2 request received and t3 reply sent on the
    remote one. offset = ((t2 - t1) + (t3 - t4)) / 2 is remote minus local, round trip = (t4 - t1) - (t3 - t2),
    and the true offset lies within offset +- round trip / 2 whatever the split of the delay between the two ways."""
    if t4 < t1 or t3 < t2:
        raise ValueError("an exchange ends before it starts")
    offset = ((t2 - t1) + (t3 - t4)) / 2
    round_trip = (t4 - t1) - (t3 - t2)
    if round_trip < 0:
        raise ValueError("the remote held the request longer than the round trip took")
    return offset, round_trip, round_trip / 2
