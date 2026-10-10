"""Verdict math: is B ahead of A, is A ahead of B, is there no difference, or is it too early to say.

An arm is one of two things. A rate is `Counts(x, n)`, x successes in n trials, read with the pooled
two-proportion z test. A measurement such as tokens per second or joules per token is `Samples(n, mean, sd)`
(or just the list of run values), read with Welch's t test, which does not assume the two arms share a spread.
`verdict(a, b)` takes two arms of the same kind, A the old or the control and B the new, and returns a `Verdict`:

    label           too_early, no_difference, b_ahead or a_ahead
    statistic, p    z or t of B against A (positive when B is higher) and the two-sided p-value
    needed_per_arm  trials or runs an arm needs for a gap as big as the observed one to stand out from noise
    more_a, more_b  how many more each arm still needs before the reading can be called

    verdict(Counts(3, 60), Counts(12, 60)).label                       # "b_ahead", z = 2.484, p = 0.013
    verdict(old_tok_s_runs, new_tok_s_runs)                            # lists of run values, higher is better
    verdict(old_j_per_token, new_j_per_token, higher_is_better=False)  # b_ahead means B burns less
    verdict(...).as_dict()                                             # plain values for a JSON ledger row

Reading rules.
- An arm under the floor (50 trials for a rate, 5 runs for a measurement; `min_per_arm` changes it) makes the
  label too_early whatever the statistic says, and no rate or mean is a finding yet.
- At or over the floor, p under alpha (0.05, two-sided) names the arm that is ahead; otherwise no_difference.
  Ahead means better: the higher value, or the lower one when `higher_is_better` is false (joules per token).
- A statistic that cannot be worked out is too_early as well: an arm with no trials or fewer than two runs, or
  two arms whose runs each repeat one value but differ, which leaves the run-to-run noise unknown.
- needed_per_arm is sized on the observed gap at the given alpha and power, so at small n it is a rough guide.
  It is None once the label is decided, and when the observed values are equal, because there is no gap to size
  on. more_a and more_b are None only when more runs of the same kind cannot help (the repeated-value case).
- Each call is one comparison. Reading several arms against one control with no correction makes it likelier
  that one of them looks ahead by luck.

To plan an experiment before any data exists, `trials_per_arm(p1, p2)` and `runs_per_arm(sd, delta)` (or
`runs_per_arm_relative(sd, mean, rel_diff)` for a gap quoted as a fraction of the mean) give the minimum n.

Everything is standard library, pure functions, no I/O. The normal CDF is math.erfc. The t CDF is the
regularized incomplete beta function by continued fraction, exact to rounding for any real df above 0,
fractional df from Welch's test included, which is well inside three decimals from 2 to 60 df.
"""

import dataclasses
import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass

ALPHA = 0.05
POWER = 0.80
#: Trials an arm needs before a rate is read, and runs an arm needs before a mean is read.
MIN_TRIALS_PER_ARM = 50
MIN_RUNS_PER_ARM = 5

TOO_EARLY = "too_early"
NO_DIFFERENCE = "no_difference"
B_AHEAD = "b_ahead"
A_AHEAD = "a_ahead"
LABELS = (TOO_EARLY, NO_DIFFERENCE, B_AHEAD, A_AHEAD)

TEST_TWO_PROPORTION = "two_proportion"
TEST_WELCH = "welch"

__all__ = [
    "ALPHA", "POWER", "MIN_TRIALS_PER_ARM", "MIN_RUNS_PER_ARM",
    "TOO_EARLY", "NO_DIFFERENCE", "B_AHEAD", "A_AHEAD", "LABELS", "TEST_TWO_PROPORTION", "TEST_WELCH",
    "norm_cdf", "norm_ppf", "t_two_sided_p", "t_cdf", "t_ppf",
    "two_proportion", "welch", "trials_per_arm", "runs_per_arm", "runs_per_arm_relative",
    "Counts", "Samples", "Verdict", "verdict",
]

_SQRT2 = math.sqrt(2.0)
_NORMAL = statistics.NormalDist()
#: Past this many runs an arm the t correction to a sample size is under a tenth of a percent, so the normal
#: quantiles are used (and the bisection for a t quantile at a huge df is never run).
_T_CORRECTION_LIMIT = 2000
_BETA_EPS = 1e-15
_BETA_MAX_ITER = 20000
_TINY = 1e-300


# --- the distributions ---------------------------------------------------------------------------------------

def norm_cdf(z: float) -> float:
    """P(Z <= z) for a standard normal, by erfc so the far lower tail keeps its digits."""
    return 0.5 * math.erfc(-z / _SQRT2)


def norm_ppf(p: float) -> float:
    """The z with P(Z <= z) = p, for p strictly between 0 and 1."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"a quantile needs a probability strictly between 0 and 1, not {p!r}")
    return _NORMAL.inv_cdf(p)


def _beta_cf(a: float, b: float, x: float) -> float:
    """The continued fraction of the incomplete beta function, by the modified Lentz method."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d
    for m in range(1, _BETA_MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _BETA_EPS:
            return h
    raise ArithmeticError(f"the incomplete beta fraction did not converge for a={a}, b={b}, x={x}")


def _beta_inc(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta function I_x(a, b) for a, b > 0 and x in [0, 1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _beta_cf(a, b, x) / a
    return 1.0 - math.exp(log_front) * _beta_cf(b, a, 1.0 - x) / b


def _check_df(df: float) -> None:
    if not df > 0.0:
        raise ValueError(f"degrees of freedom must be above 0, not {df!r}")


def t_two_sided_p(t: float, df: float) -> float:
    """P(|T| >= |t|) for Student's t with df degrees of freedom (any real df above 0, infinity included).

    Read straight off the incomplete beta function, so a tiny p keeps its relative accuracy."""
    _check_df(df)
    if t != t:
        raise ValueError("t is not a number")
    if math.isinf(t):
        return 0.0
    if t == 0.0:
        return 1.0
    if math.isinf(df):
        return math.erfc(abs(t) / _SQRT2)
    return _beta_inc(df / 2.0, 0.5, df / (df + t * t))


def t_cdf(t: float, df: float) -> float:
    """P(T <= t) for Student's t with df degrees of freedom."""
    p = t_two_sided_p(t, df)
    if t == 0.0:
        return 0.5
    return 1.0 - 0.5 * p if t > 0.0 else 0.5 * p


def t_ppf(p: float, df: float) -> float:
    """The t with P(T <= t) = p, for p strictly between 0 and 1, by bisection on the two-sided tail."""
    _check_df(df)
    if not 0.0 < p < 1.0:
        raise ValueError(f"a quantile needs a probability strictly between 0 and 1, not {p!r}")
    if p == 0.5:
        return 0.0
    two_sided = 2.0 * min(p, 1.0 - p)
    lo, hi = 0.0, 1.0
    while t_two_sided_p(hi, df) > two_sided and hi < 1e300:
        lo, hi = hi, hi * 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_two_sided_p(mid, df) > two_sided:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * hi:
            break
    t = 0.5 * (lo + hi)
    return t if p > 0.5 else -t


# --- two proportions -----------------------------------------------------------------------------------------

def _check_counts(x: int, n: int) -> None:
    if not isinstance(x, int) or not isinstance(n, int):
        raise TypeError(f"counts are whole numbers, not {x!r} of {n!r}")
    if n < 0 or not 0 <= x <= n:
        raise ValueError(f"{x} successes in {n} trials cannot happen")


def _check_level(value: float, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1, not {value!r}")


def two_proportion(x1: int, n1: int, x2: int, n2: int) -> tuple[float | None, float | None]:
    """(z, p) of the pooled two-proportion z test of x2/n2 against x1/n1, two-sided.

    z is positive when the second rate is higher. (None, None) when either arm has no trials; (0.0, 1.0) when
    the pooled rate is 0 or 1, because then there is nothing to tell apart."""
    _check_counts(x1, n1)
    _check_counts(x2, n2)
    if n1 == 0 or n2 == 0:
        return None, None
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))
    if se == 0.0:
        return 0.0, 1.0
    z = (x2 / n2 - x1 / n1) / se
    return z, math.erfc(abs(z) / _SQRT2)


def trials_per_arm(p1: float, p2: float, *, alpha: float = ALPHA, power: float = POWER) -> int:
    """The trials each arm needs to tell rate p2 from rate p1 at two-sided alpha with the given power, by the
    pooled normal approximation. 5% against 10% at the defaults is 435 an arm."""
    _check_level(alpha, "alpha")
    _check_level(power, "power")
    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0) or p1 == p2:
        raise ValueError(f"two different rates between 0 and 1, not {p1!r} and {p2!r}")
    pooled = (p1 + p2) / 2.0
    spread_null = norm_ppf(1.0 - alpha / 2.0) * math.sqrt(2.0 * pooled * (1.0 - pooled))
    spread_alt = norm_ppf(power) * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    return math.ceil((spread_null + spread_alt) ** 2 / (p2 - p1) ** 2)


# --- two sets of runs ----------------------------------------------------------------------------------------

def runs_per_arm(sd: float, delta: float, *, alpha: float = ALPHA, power: float = POWER) -> int:
    """The runs each arm needs for a two-sided Welch test at alpha to find a true gap of `delta` between the
    means with the given power, when a single run has standard deviation `sd` in both arms.

    This is the smallest n with n >= 2 ((t(1 - alpha/2, 2n - 2) + t(power, 2n - 2)) sd / delta)^2: the normal
    formula with each quantile taken from the t distribution. At alpha 0.05 it equals the exact noncentral-t
    answer for every gap checked from 0.3 to 3 spreads (3 spreads takes 4 runs an arm, 2 take 6, 1 takes 17,
    half takes 64), where the plain normal formula is one or two short; at a stricter alpha such as 0.01 it can
    still come out one short. Beyond 2000 runs an arm the t correction is under a tenth of a percent and the
    normal formula is returned. Never fewer than 2, the least a t test can use."""
    _check_level(alpha, "alpha")
    _check_level(power, "power")
    if not (math.isfinite(sd) and sd >= 0.0):
        raise ValueError(f"the spread must be a finite number that is not negative, not {sd!r}")
    if not (math.isfinite(delta) and delta != 0.0):
        raise ValueError(f"the difference to find must be a finite number that is not zero, not {delta!r}")
    if sd == 0.0:
        return 2
    delta = abs(delta)
    z_sum = norm_ppf(1.0 - alpha / 2.0) + norm_ppf(power)
    n = max(2, math.ceil(2.0 * (z_sum * sd / delta) ** 2))
    if n > _T_CORRECTION_LIMIT:
        return n
    while True:
        df = 2 * n - 2
        t_sum = t_ppf(1.0 - alpha / 2.0, df) + t_ppf(power, df)
        if n >= 2.0 * (t_sum * sd / delta) ** 2:
            return n
        n += 1


def runs_per_arm_relative(sd: float, mean: float, rel_diff: float, *, alpha: float = ALPHA,
                          power: float = POWER) -> int:
    """`runs_per_arm` for a gap quoted as a fraction of the mean: sd 5 on a mean of 100, to see 5%, is a gap
    of 5 and takes 17 runs an arm."""
    if not (math.isfinite(mean) and mean != 0.0):
        raise ValueError(f"a relative difference needs a finite mean that is not zero, not {mean!r}")
    if not (math.isfinite(rel_diff) and rel_diff != 0.0):
        raise ValueError(f"the relative difference to find must be finite and not zero, not {rel_diff!r}")
    return runs_per_arm(sd, abs(mean) * abs(rel_diff), alpha=alpha, power=power)


@dataclass(frozen=True)
class Counts:
    """A rate: x successes in n trials."""

    x: int
    n: int

    def __post_init__(self) -> None:
        _check_counts(self.x, self.n)


@dataclass(frozen=True)
class Samples:
    """A measurement: n runs with their mean and sample standard deviation (the n - 1 divisor, not the
    standard error and not the range). No runs have no mean, and fewer than two have no spread: None."""

    n: int
    mean: float | None
    sd: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.n, int) or self.n < 0:
            raise ValueError(f"the number of runs must be a whole number that is not negative, not {self.n!r}")
        if (self.n == 0) != (self.mean is None):
            raise ValueError("a mean goes with at least one run, and no runs have none")
        if (self.n < 2) != (self.sd is None):
            raise ValueError("a spread goes with at least two runs, and fewer than two have none")
        if self.mean is not None and not math.isfinite(self.mean):
            raise ValueError(f"the mean must be a finite number, not {self.mean!r}")
        if self.sd is not None and not (math.isfinite(self.sd) and self.sd >= 0.0):
            raise ValueError(f"the spread must be a finite number that is not negative, not {self.sd!r}")

    @classmethod
    def from_values(cls, values: Iterable[float]) -> "Samples":
        """The summary of a list of run values."""
        xs = [float(v) for v in values]
        if not all(math.isfinite(x) for x in xs):
            raise ValueError("run values must be finite numbers")
        if not xs:
            return cls(0, None, None)
        return cls(len(xs), statistics.fmean(xs), statistics.stdev(xs) if len(xs) >= 2 else None)


def welch(a: Samples, b: Samples) -> tuple[float | None, float | None, float | None]:
    """(t, p, df) of Welch's two-sided t test of b's mean against a's, with the Welch-Satterthwaite df.

    t is positive when b's mean is higher. (None, None, None) when an arm has fewer than two runs, and when
    both arms repeat a single value yet differ, since then the noise is unknown. Two arms that repeat the same
    value give (0.0, 1.0, None)."""
    if a.n < 2 or b.n < 2:
        return None, None, None
    var_a, var_b = a.sd ** 2 / a.n, b.sd ** 2 / b.n
    var_sum = var_a + var_b
    diff = b.mean - a.mean
    if var_sum == 0.0:
        return (0.0, 1.0, None) if diff == 0.0 else (None, None, None)
    t = diff / math.sqrt(var_sum)
    w_a, w_b = var_a / var_sum, var_b / var_sum
    df = 1.0 / (w_a * w_a / (a.n - 1) + w_b * w_b / (b.n - 1))
    return t, t_two_sided_p(t, df), df


# --- the verdict ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Verdict:
    """What `verdict` found. `statistic` is z or t of B against A; `diff` is B minus A (a rate difference as a
    fraction, or a difference of means); `min_per_arm` is the floor that applied. See the module docstring for
    needed_per_arm, more_a and more_b."""

    label: str
    test: str
    statistic: float | None
    p: float | None
    df: float | None
    diff: float | None
    n_a: int
    n_b: int
    min_per_arm: int
    needed_per_arm: int | None
    more_a: int | None
    more_b: int | None
    note: str

    def as_dict(self) -> dict:
        """Plain values for JSON: every field, in order."""
        return dataclasses.asdict(self)


def _as_arm(arm: object) -> Counts | Samples:
    if isinstance(arm, (Counts, Samples)):
        return arm
    if isinstance(arm, (str, bytes)) or not isinstance(arm, Iterable):
        raise TypeError(f"an arm is Counts, Samples or a list of run values, not {type(arm).__name__}")
    return Samples.from_values(arm)


def _floor(min_per_arm: int | None, default: int, lowest: int) -> int:
    floor = default if min_per_arm is None else min_per_arm
    if not isinstance(floor, int) or floor < lowest:
        raise ValueError(f"min_per_arm must be a whole number of at least {lowest}, not {floor!r}")
    return floor


def _label(p: float | None, diff: float | None, n_min: int, floor: int, alpha: float,
           higher_is_better: bool) -> str:
    if p is None or n_min < floor:
        return TOO_EARLY
    if p < alpha:
        return B_AHEAD if (diff > 0.0) == higher_is_better else A_AHEAD
    return NO_DIFFERENCE


def _remaining(label: str, p: float | None, n_a: int, n_b: int, floor: int,
               needed: int | None) -> tuple[int | None, int | None, int | None]:
    """(needed_per_arm, more_a, more_b) for a label."""
    if label in (B_AHEAD, A_AHEAD):
        return None, 0, 0
    if p is None and min(n_a, n_b) >= floor:
        return None, None, None
    target = max(floor, needed or 0)
    return needed, max(0, target - n_a), max(0, target - n_b)


def _more_phrase(more_a: int, more_b: int) -> str:
    if more_a == 0 and more_b == 0:
        return "no more needed"
    if more_a == more_b:
        return f"{more_a} more each"
    return f"{more_a} more for A, {more_b} more for B"


def _fmt_p(p: float) -> str:
    return "p < 0.0001" if p < 0.0001 else f"p = {round(p, 4):g}"


def _stat_text(name: str, statistic: float, df: float | None, p: float) -> str:
    parts = [f"{name} = {statistic:.3f}"]
    if df is not None:
        parts.append(f"df = {df:.1f}")
    parts.append(_fmt_p(p))
    return ", ".join(parts)


def _ahead_note(label: str, value_a: str, value_b: str, stats: str, higher_is_better: bool) -> str:
    winner, loser = ("B", "A") if label == B_AHEAD else ("A", "B")
    value_w, value_l = (value_b, value_a) if label == B_AHEAD else (value_a, value_b)
    tail = "" if higher_is_better else "; lower is better"
    return f"{winner} ahead of {loser}: {value_w} against {value_l} ({stats}{tail})"


def _sizing_tail(needed: int | None, more_a: int, more_b: int, unit: str, power: float) -> str:
    if needed is None:
        return "the observed values are equal, so no sample size follows from them"
    return (f"a difference this size needs {needed} {unit} per arm at {100 * power:g}% power, "
            f"{_more_phrase(more_a, more_b)}")


def _pct(rate: float) -> str:
    return f"{100.0 * rate:.1f}%"


def _read_rates(a: Counts, b: Counts, higher_is_better: bool, alpha: float, power: float,
                min_per_arm: int | None) -> Verdict:
    floor = _floor(min_per_arm, MIN_TRIALS_PER_ARM, 1)
    z, p = two_proportion(a.x, a.n, b.x, b.n)
    rate_a = a.x / a.n if a.n else None
    rate_b = b.x / b.n if b.n else None
    diff = None if rate_a is None or rate_b is None else rate_b - rate_a
    label = _label(p, diff, min(a.n, b.n), floor, alpha, higher_is_better)
    sized = trials_per_arm(rate_a, rate_b, alpha=alpha, power=power) if diff else None
    needed, more_a, more_b = _remaining(label, p, a.n, b.n, floor, sized)
    if label in (B_AHEAD, A_AHEAD):
        note = _ahead_note(label, _pct(rate_a), _pct(rate_b), _stat_text("z", z, None, p), higher_is_better)
    elif label == NO_DIFFERENCE:
        note = (f"no difference shown: B {_pct(rate_b)} against A {_pct(rate_a)} ({_stat_text('z', z, None, p)}); "
                + _sizing_tail(needed, more_a, more_b, "trials", power))
    else:
        trial = "trial" if a.n == 1 else "trials"
        success = "success" if a.x == 1 else "successes"
        note = (f"too early to call: A has {a.n} {trial} ({a.x} {success}), B has {b.n} ({b.x}); "
                f"{max(floor, needed or 0)} each needed, {_more_phrase(more_a, more_b)}")
    return Verdict(label, TEST_TWO_PROPORTION, z, p, None, diff, a.n, b.n, floor, needed, more_a, more_b, note)


def _read_runs(a: Samples, b: Samples, higher_is_better: bool, alpha: float, power: float,
               min_per_arm: int | None) -> Verdict:
    floor = _floor(min_per_arm, MIN_RUNS_PER_ARM, 2)
    t, p, df = welch(a, b)
    diff = None if a.mean is None or b.mean is None else b.mean - a.mean
    label = _label(p, diff, min(a.n, b.n), floor, alpha, higher_is_better)
    sized = None
    if diff and a.sd is not None and b.sd is not None and (a.sd or b.sd):
        spread = math.sqrt((a.sd ** 2 + b.sd ** 2) / 2.0)
        sized = runs_per_arm(spread, abs(diff), alpha=alpha, power=power)
    needed, more_a, more_b = _remaining(label, p, a.n, b.n, floor, sized)
    if label in (B_AHEAD, A_AHEAD):
        note = _ahead_note(label, f"mean {a.mean:.4g}", f"mean {b.mean:.4g}", _stat_text("t", t, df, p),
                           higher_is_better)
    elif label == NO_DIFFERENCE:
        note = (f"no difference shown: B mean {b.mean:.4g} against A mean {a.mean:.4g} "
                f"({_stat_text('t', t, df, p)}); " + _sizing_tail(needed, more_a, more_b, "runs", power))
    elif more_a is None:
        note = (f"too early to call: both arms repeat one value ({a.mean:.4g} and {b.mean:.4g}), so the "
                "run-to-run noise cannot be estimated")
    else:
        run = "run" if a.n == 1 else "runs"
        mean_a = "" if a.mean is None else f" (mean {a.mean:.4g})"
        mean_b = "" if b.mean is None else f" (mean {b.mean:.4g})"
        note = (f"too early to call: A has {a.n} {run}{mean_a}, B has {b.n}{mean_b}; "
                f"{max(floor, needed or 0)} each needed, {_more_phrase(more_a, more_b)}")
    return Verdict(label, TEST_WELCH, t, p, df, diff, a.n, b.n, floor, needed, more_a, more_b, note)


def verdict(a: Counts | Samples | Iterable[float], b: Counts | Samples | Iterable[float], *,
            higher_is_better: bool = True, alpha: float = ALPHA, power: float = POWER,
            min_per_arm: int | None = None) -> Verdict:
    """Read arm B against arm A. Both are `Counts` (a rate) or both are measurements (`Samples`, or a plain list
    of run values). `alpha` is the two-sided level; `power` sizes needed_per_arm; `min_per_arm` replaces the
    floor of 50 trials (rates) or 5 runs (measurements). See the module docstring for the rules."""
    _check_level(alpha, "alpha")
    _check_level(power, "power")
    arm_a, arm_b = _as_arm(a), _as_arm(b)
    if isinstance(arm_a, Counts) and isinstance(arm_b, Counts):
        return _read_rates(arm_a, arm_b, higher_is_better, alpha, power, min_per_arm)
    if isinstance(arm_a, Samples) and isinstance(arm_b, Samples):
        return _read_runs(arm_a, arm_b, higher_is_better, alpha, power, min_per_arm)
    raise TypeError("both arms must be rates (Counts) or both must be measurements (Samples or a list of runs)")
