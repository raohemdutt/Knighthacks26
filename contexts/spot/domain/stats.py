"""The small amount of statistics the inspection analysis needs, in pure Python.

- `wilson(k, n)`: a 95% interval for a proportion that behaves at small n and at 0% or 100%.
- `compare(k1, n1, k2, n2)`: the difference of two proportions with Newcombe's interval (built from the two Wilson
  intervals) and a two-sided p-value: Fisher's exact test when any expected cell is under 5, else the pooled z-test.
- `theil_sen(points)`: a slope that one bad reading cannot drag (the median of all pairwise slopes).
- `mann_whitney(a, b)`: are the values in one group generally larger than in the other? (normal approximation, tie-corrected)
- `median`, `mad`: a robust middle and spread.

`MIN_N` is the house rule: under that many rows a rate is shown with its count, but no conclusion is drawn from it.
"""

import math
from dataclasses import dataclass

Z95 = 1.959963984540054
MIN_N = 8  # fewer rows than this: "not enough data"
MIN_TREND_N = 5  # fewer good readings than this: no trend


def rate(k: int, n: int) -> float | None:
    return k / n if n else None


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """The Wilson score interval for k successes out of n. None when n is 0."""
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def fisher_exact(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher's exact p-value for the 2x2 table [[k1, n1-k1], [k2, n2-k2]]."""
    total_k, total = k1 + k2, n1 + n2
    if total == 0:
        return 1.0

    def prob(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, total_k - x) / math.comb(total, total_k)

    observed = prob(k1)
    lo, hi = max(0, total_k - n2), min(n1, total_k)
    return min(1.0, sum(p for p in (prob(x) for x in range(lo, hi + 1)) if p <= observed * (1 + 1e-9)))


@dataclass(frozen=True)
class Comparison:
    k1: int
    n1: int
    k2: int
    n2: int
    p1: float | None
    p2: float | None
    diff: float | None  # p1 - p2
    interval: tuple[float, float] | None  # 95% for the difference
    p_value: float | None
    test: str  # "fisher" | "z" | "none"
    enough: bool  # both groups have at least MIN_N rows

    @property
    def clear(self) -> bool:
        """The interval for the difference leaves out zero, on enough rows."""
        return bool(self.enough and self.interval and (self.interval[0] > 0 or self.interval[1] < 0))


def compare(k1: int, n1: int, k2: int, n2: int, min_n: int = MIN_N) -> Comparison:
    """Proportion 1 minus proportion 2, with an interval and a p-value."""
    if n1 <= 0 or n2 <= 0:
        return Comparison(k1, n1, k2, n2, rate(k1, n1), rate(k2, n2), None, None, None, "none", False)
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)  # type: ignore[misc]
    l2, u2 = wilson(k2, n2)  # type: ignore[misc]
    diff = p1 - p2
    interval = (diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))
    pooled = (k1 + k2) / (n1 + n2)
    smallest_expected = min(n1 * pooled, n1 * (1 - pooled), n2 * pooled, n2 * (1 - pooled))
    if smallest_expected < 5:
        p_value, test = fisher_exact(k1, n1, k2, n2), "fisher"
    else:
        se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
        p_value, test = (2 * (1 - _phi(abs(diff) / se)) if se else 1.0), "z"
    return Comparison(k1, n1, k2, n2, p1, p2, diff, interval, p_value, test, n1 >= min_n and n2 >= min_n)


def median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def mad(values: list[float]) -> float | None:
    """Median absolute deviation, scaled so that it reads like a standard deviation for bell-shaped data."""
    m = median(values)
    if m is None:
        return None
    return 1.4826 * median([abs(v - m) for v in values])  # type: ignore[operator]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def theil_sen(points: list[tuple[float, float]]) -> float | None:
    """The median slope over every pair of points with different x. None with fewer than two such points."""
    slopes = [(y2 - y1) / (x2 - x1) for i, (x1, y1) in enumerate(points) for x2, y2 in points[i + 1:] if x2 != x1]
    return median(slopes)


def mann_whitney(a: list[float], b: list[float]) -> float | None:
    """Two-sided p-value that values in `a` and `b` come from the same distribution. None when a group is empty."""
    n1, n2 = len(a), len(b)
    if not n1 or not n2:
        return None
    ranked = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = [0.0] * len(ranked)
    ties = 0.0
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        t = j - i + 1
        ties += t**3 - t
        i = j + 1
    r1 = sum(r for r, (_, g) in zip(ranks, ranked) if g == 0)
    u = r1 - n1 * (n1 + 1) / 2
    n = n1 + n2
    var = n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0:
        return 1.0
    z = (abs(u - n1 * n2 / 2) - 0.5) / math.sqrt(var)  # continuity correction
    return min(1.0, 2 * (1 - _phi(max(z, 0.0))))


def pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def pct_range(interval: tuple[float, float] | None) -> str:
    return "n/a" if interval is None else f"{100 * interval[0]:.0f}% to {100 * interval[1]:.0f}%"
