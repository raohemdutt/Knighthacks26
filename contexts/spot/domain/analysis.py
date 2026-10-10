"""What the inspection data says, as pure functions over a `Kit`.

Every function returns a `Finding`: the number, the n behind it, a 95% interval where one applies, one plain
sentence, a caveat when the data is thin or two explanations cannot be told apart, the breakdown rows, and the
ids of the rows that are its evidence. No finding without its evidence; under `MIN_N` rows the answer is
"not enough data", with the count still shown.

Rates are failure rates unless a label says otherwise (k = failed, n = attempted). Intervals are Wilson score
intervals; differences use Newcombe's interval and Fisher's exact test or a z-test (see domain/stats.py).
"Now" is the newest timestamp in the data, never the wall clock.
"""

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable

from pydantic import BaseModel

from contexts.spot.domain.models import CRITICALITIES, SEVERITIES, Action, Kit
from contexts.spot.domain.stats import (MIN_N, MIN_TREND_N, Comparison, compare, mad, mann_whitney, mean, median, pct,
                                        pct_range, theil_sen, wilson)

BATTERY_BINS = ((0, 20), (20, 40), (40, 60), (60, 80), (80, 101))
HIGHER_IS_BETTER = {"VISUAL_SCORE", "LOCALIZATION_CONFIDENCE_PCT"}
SEV_RANK = {s: i for i, s in enumerate(SEVERITIES)}
CRIT_RANK = {c: i for i, c in enumerate(CRITICALITIES)}


class Evidence(BaseModel):
    """Row ids behind a number. Every list holds real ids from the tables."""

    missions: list[str] = []
    actions: list[str] = []
    readings: list[str] = []
    anomalies: list[str] = []
    assets: list[str] = []
    locations: list[str] = []

    @property
    def count(self) -> int:
        return sum(len(v) for v in (self.missions, self.actions, self.readings, self.anomalies, self.assets, self.locations))

    def ids(self) -> list[str]:
        return [*self.missions, *self.actions, *self.anomalies, *self.readings, *self.assets, *self.locations]


class Bar(BaseModel):
    """One row of a breakdown: k of n, with its interval. `enough` is False under MIN_N rows."""

    key: str
    label: str
    k: int
    n: int
    rate: float | None = None
    lo: float | None = None
    hi: float | None = None
    enough: bool = True
    note: str | None = None
    evidence: Evidence = Evidence()
    extra: dict = {}


class Finding(BaseModel):
    id: str
    title: str
    question: str  # which of the brief's questions this answers
    sentence: str
    value: float | None = None
    unit: str = ""
    n: int = 0
    interval: tuple[float, float] | None = None
    p_value: float | None = None
    enough: bool = True
    caveat: str | None = None
    rows: list[Bar] = []
    evidence: Evidence = Evidence()
    extra: dict = {}


def _bar(key: str, label: str, k: int, n: int, evidence: Evidence | None = None, **extra) -> Bar:
    ci = wilson(k, n)
    return Bar(key=key, label=label, k=k, n=n, rate=k / n if n else None, lo=ci[0] if ci else None, hi=ci[1] if ci else None,
               enough=n >= MIN_N, note=None if n >= MIN_N else f"not enough data (n={n}, need {MIN_N})",
               evidence=evidence or Evidence(), extra=extra)


def _failure_bars(actions: Iterable[Action], key: Callable[[Action], str], label: Callable[[str], str] = str) -> list[Bar]:
    groups: dict[str, list[Action]] = defaultdict(list)
    for a in actions:
        groups[key(a)].append(a)
    bars = [_bar(k, label(k), sum(a.failed for a in g), len(g), Evidence(actions=[a.action_id for a in g if a.failed]))
            for k, g in groups.items()]
    # worst first, by the bottom of the interval: a 3-of-4 fluke does not outrank a steady 8-of-22
    return sorted(bars, key=lambda b: (-(b.lo or 0) if b.enough else 1, -(b.rate or 0), b.key))


def _words(s: str) -> str:
    return s.replace("_", " ").lower()


def _p(p: float | None) -> str:
    if p is None:
        return "p n/a"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}" if p < 0.1 else f"p={p:.2f}"


def _pts(c: Comparison) -> str:
    if c.diff is None or c.interval is None:
        return "n/a"
    return f"{100 * c.diff:+.1f} points (95% interval {100 * c.interval[0]:+.1f} to {100 * c.interval[1]:+.1f}, {_p(c.p_value)})"


def _groups(kit: Kit) -> list[str]:
    return sorted({m.configuration_version for m in kit.missions})


def _group_label(kit: Kit, version: str) -> str:
    scenarios = sorted({m.scenario for m in kit.missions if m.configuration_version == version})
    return f"{version} ({'/'.join(scenarios)})" if scenarios else version


def _loc_label(kit: Kit, location_id: str) -> str:
    loc = kit.location.get(location_id)
    return f"{location_id} {loc.location_name}" if loc and loc.location_name else location_id


def _asset_label(kit: Kit, asset_id: str) -> str:
    a = kit.asset.get(asset_id)
    return f"{asset_id} {a.asset_name}" if a and a.asset_name else asset_id


# --- headline numbers ---


def kpis(kit: Kit) -> list[dict]:
    """The numbers for the top strip, each with its count and interval."""
    completed = [m for m in kit.missions if m.mission_status == "COMPLETED"]
    ok = [a for a in kit.actions if not a.failed]
    hidden = [m for m in completed if kit.failed_in(m.mission_id)]
    review = [x for x in kit.anomalies if x.severity in ("HIGH", "CRITICAL")]

    def k(key: str, label: str, num: int, den: int, good: str, note: str = "") -> dict:
        ci = wilson(num, den)
        return {"key": key, "label": label, "k": num, "n": den, "rate": num / den if den else None,
                "lo": ci[0] if ci else None, "hi": ci[1] if ci else None, "good": good, "note": note}

    return [
        k("mission_completion", "Missions marked COMPLETED", len(completed), len(kit.missions), "high"),
        k("action_success", "Actions that succeeded", len(ok), len(kit.actions), "high"),
        k("hidden_failures", "COMPLETED missions with a failed action inside", len(hidden), len(completed), "low",
          "what the mission status hides"),
        k("anomalies_for_review", "Anomalies HIGH or CRITICAL", len(review), len(kit.anomalies), "low"),
    ]


# --- mission status against what happened inside ---


def hidden_failures(kit: Kit) -> Finding:
    completed = [m for m in kit.missions if m.mission_status == "COMPLETED"]
    rows = []
    for m in completed:
        failed = kit.failed_in(m.mission_id)
        if not failed:
            continue
        anomalies = kit.anomalies_in(m.mission_id)
        worst = max((x.severity for x in anomalies), key=SEV_RANK.get, default=None)
        rows.append(_bar(m.mission_id, f"{m.mission_id} ({m.configuration_version})", len(failed), len(kit.actions_of[m.mission_id]),
                         Evidence(missions=[m.mission_id], actions=[a.action_id for a in failed], anomalies=[x.anomaly_id for x in anomalies]),
                         worst_severity=worst, anomalies=len(anomalies), failure_types=[a.failure_type for a in failed],
                         version=m.configuration_version, start=m.start_time.isoformat()))
    rows.sort(key=lambda b: (-b.k, -SEV_RANK.get(b.extra["worst_severity"] or "", -1), b.key))
    failed_all = [a for a in kit.actions if a.failed]
    inside = [a for a in failed_all if a.mission_id in {m.mission_id for m in completed}]
    not_completed = [m for m in kit.missions if m.mission_status != "COMPLETED"]
    ci = wilson(len(rows), len(completed))
    caveat = None
    if not_completed:
        same = [b.key for b in rows if b.k >= max(len(kit.failed_in(m.mission_id)) for m in not_completed)]
        m = not_completed[0]
        caveat = (f"{len(not_completed)} mission(s) not COMPLETED ({', '.join(x.mission_id for x in not_completed)}). {m.mission_id} had "
                  f"{len(kit.failed_in(m.mission_id))} failed action(s); {len(same)} COMPLETED mission(s) had as many or more "
                  f"({', '.join(same[:6])}{'...' if len(same) > 6 else ''}), so the failed-action count does not explain the mission status. "
                  "The tables do not say what does.")
    sentence = (f"{len(rows)} of {len(completed)} missions marked COMPLETED ({pct(len(rows) / len(completed)) if completed else 'n/a'}) "
                f"contain at least one failed action; {len(inside)} of all {len(failed_all)} failed actions sit inside a COMPLETED mission. "
                "Mission status alone would show almost none of them.") if completed else "No COMPLETED missions in the data."
    return Finding(id="hidden_failures", title="Completed missions that hide failed actions",
                   question="Can a mission be COMPLETED even when actions inside it fail?", sentence=sentence,
                   value=len(rows) / len(completed) if completed else None, unit="of COMPLETED missions", n=len(completed), interval=ci,
                   enough=len(completed) >= MIN_N, caveat=caveat, rows=rows,
                   evidence=Evidence(missions=[b.key for b in rows], actions=[a.action_id for a in inside]),
                   extra={"failed_inside_completed": len(inside), "failed_total": len(failed_all)})


# --- reliability by one dimension ---


def _versus_rest(bars: list[Bar]) -> tuple[Bar, Comparison] | None:
    usable = [b for b in bars if b.enough]
    if len(usable) < 2:
        return None
    worst = usable[0]
    rest_k, rest_n = sum(b.k for b in bars) - worst.k, sum(b.n for b in bars) - worst.n
    return worst, compare(worst.k, worst.n, rest_k, rest_n)


def _reliability(kit: Kit, fid: str, title: str, question: str, what: str, key: Callable[[Action], str],
                 label: Callable[[str], str] = str, actions: list[Action] | None = None) -> Finding:
    acts = kit.actions if actions is None else actions
    bars = _failure_bars(acts, key, label)
    found = _versus_rest(bars)
    failed = [a.action_id for a in acts if a.failed]
    if not found:
        return Finding(id=fid, title=title, question=question, n=len(acts), rows=bars, enough=False,
                       sentence=f"Not enough data to compare {what}s: fewer than two have {MIN_N} or more actions.",
                       evidence=Evidence(actions=failed))
    worst, cmp = found
    best = min((b for b in bars if b.enough), key=lambda b: (b.hi or 1, b.key))
    sentence = (f"Least reliable {what}: {worst.label}, {worst.k} of {worst.n} actions failed ({pct(worst.rate)}, 95% interval "
                f"{pct_range((worst.lo, worst.hi))}), against {pct(cmp.p2)} ({cmp.k2}/{cmp.n2}) for all the others: {_pts(cmp)}. "
                f"Most reliable: {best.label}, {best.k} of {best.n} ({pct(best.rate)}).")
    thin = [b.label for b in bars if not b.enough]
    caveats = []
    if not cmp.clear:
        caveats.append("The interval for the difference includes zero: this ranking could be chance at this n.")
    if thin:
        caveats.append(f"Not enough data for {', '.join(thin)} (under {MIN_N} actions each).")
    return Finding(id=fid, title=title, question=question, sentence=sentence, value=worst.rate, unit="failure rate", n=worst.n,
                   interval=(worst.lo, worst.hi) if worst.lo is not None else None, p_value=cmp.p_value, enough=True,
                   caveat=" ".join(caveats) or None, rows=bars, evidence=Evidence(actions=worst.evidence.actions),
                   extra={"worst": worst.key, "best": best.key, "difference": cmp.diff, "difference_interval": cmp.interval,
                          "clear": cmp.clear, "overall": {"k": len(failed), "n": len(acts)}})


def reliability_by_type(kit: Kit) -> Finding:
    return _reliability(kit, "reliability_by_type", "Reliability by inspection type", "Which inspection types appear least reliable?",
                        "action type", lambda a: a.action_type, _words)


def reliability_by_location(kit: Kit) -> Finding:
    f = _reliability(kit, "reliability_by_location", "Reliability by waypoint", "Are failures concentrated at particular locations?",
                     "waypoint", lambda a: a.location_id, lambda k: _loc_label(kit, k))
    for b in f.rows:
        loc = kit.location.get(b.key)
        b.extra = {"floor": loc.floor if loc else None, "zone_type": loc.zone_type if loc else "",
                   "environment": loc.environment if loc else ""}
    return f


def reliability_by_asset(kit: Kit) -> Finding:
    f = _reliability(kit, "reliability_by_asset", "Reliability by asset", "Are failures concentrated on particular assets?",
                     "asset", lambda a: a.asset_id, lambda k: _asset_label(kit, k))
    for b in f.rows:
        a = kit.asset.get(b.key)
        b.extra = {"criticality": a.criticality if a else "", "asset_type": a.asset_type if a else ""}
    return f


def reliability_by_floor(kit: Kit) -> Finding:
    return _reliability(kit, "reliability_by_floor", "Reliability by floor", "Are failures concentrated on one floor?", "floor",
                        lambda a: str(kit.location[a.location_id].floor) if a.location_id in kit.location else "?",
                        lambda k: f"Floor {k}")


def reliability_by_environment(kit: Kit) -> Finding:
    return _reliability(kit, "reliability_by_environment", "Reliability by operating environment",
                        "Does the environment at a waypoint go with failures?", "environment",
                        lambda a: kit.location[a.location_id].environment if a.location_id in kit.location else "?", _words)


def failure_types(kit: Kit) -> Finding:
    """Which failure types there are and where each one concentrates (action type and waypoint)."""
    failed = [a for a in kit.actions if a.failed]
    groups: dict[str, list[Action]] = defaultdict(list)
    for a in failed:
        groups[a.failure_type or "(none given)"].append(a)
    rows = []
    for ftype, acts in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        by_type, by_loc = Counter(a.action_type for a in acts), Counter(a.location_id for a in acts)
        top_type, top_loc = by_type.most_common(1)[0], by_loc.most_common(1)[0]
        b = _bar(ftype, _words(ftype), len(acts), len(failed), Evidence(actions=[a.action_id for a in acts]),
                 by_action_type=dict(by_type), by_location=dict(by_loc), top_action_type=top_type[0], top_location=top_loc[0])
        b.enough, b.note = True, f"mostly {_words(top_type[0])} ({top_type[1]}/{len(acts)}); most at {top_loc[0]} ({top_loc[1]}/{len(acts)})"
        rows.append(b)
    if not rows:
        return Finding(id="failure_types", title="Failure types", question="What kinds of failure are there, and where?",
                       sentence="No failed actions in the data.", n=0, enough=False)
    top = rows[0]
    return Finding(id="failure_types", title="Failure types and where they concentrate",
                   question="What kinds of failure are there, and where?", n=len(failed), value=top.rate, unit="of failed actions",
                   sentence=f"{len(failed)} failed actions in {len(rows)} failure types. The most common is {top.label} "
                            f"({top.k} of {len(failed)}), {top.note}.",
                   caveat="Shares of a small total: a type with two or three failures tells little about where it concentrates.",
                   rows=rows, evidence=Evidence(actions=[a.action_id for a in failed]))


def navigation_hotspot(kit: Kit) -> Finding:
    """Is there a waypoint where navigation fails clearly more often than everywhere else?"""
    nav = [a for a in kit.actions if a.action_type == "NAVIGATION"]
    bars = _failure_bars(nav, lambda a: a.location_id, lambda k: _loc_label(kit, k))
    usable = [b for b in bars if b.enough]
    title, q = "Navigation failures by waypoint", "Are navigation failures concentrated at one waypoint?"
    if not usable:
        return Finding(id="navigation_hotspot", title=title, question=q, n=len(nav), rows=bars, enough=False,
                       sentence=f"Not enough data: no waypoint has {MIN_N} or more navigation actions.")
    worst = max(usable, key=lambda b: (b.rate or 0, b.n))
    cmp = compare(worst.k, worst.n, sum(a.failed for a in nav) - worst.k, len(nav) - worst.n)
    first = [a for a in kit.actions if a.sequence == 1]
    first_here = sum(1 for a in first if a.location_id == worst.key)
    verdict = "a clear difference" if cmp.clear else "suggestive, not established at this n"
    caveat = (f"{worst.key} is the first stop of {first_here} of {len(first)} missions, so 'at {worst.key}' and 'first action of a "
              "mission' cannot be told apart here." if first and first_here / len(first) > 0.5 else None)
    thin = sum(1 for b in bars if not b.enough)
    if thin:
        caveat = ((caveat + " ") if caveat else "") + f"{thin} other waypoint(s) have under {MIN_N} navigation actions each; they are pooled as 'elsewhere'."
    return Finding(id="navigation_hotspot", title=title, question=q, value=worst.rate, unit="navigation failure rate", n=worst.n,
                   interval=(worst.lo, worst.hi), p_value=cmp.p_value, enough=True, caveat=caveat, rows=bars,
                   sentence=f"Navigation to {worst.label} failed {worst.k} of {worst.n} times ({pct(worst.rate)}, 95% interval "
                            f"{pct_range((worst.lo, worst.hi))}) against {cmp.k2} of {cmp.n2} ({pct(cmp.p2)}) at every other waypoint: "
                            f"{_pts(cmp)}; {verdict}.",
                   evidence=Evidence(actions=worst.evidence.actions, locations=[worst.key]),
                   extra={"location_id": worst.key, "clear": cmp.clear, "difference": cmp.diff, "difference_interval": cmp.interval,
                          "elsewhere": {"k": cmp.k2, "n": cmp.n2}})


# --- V1 against V2 ---


def _rate_row(key: str, label: str, a: tuple[int, int], b: tuple[int, int], good: str, ev_a: list[str], ev_b: list[str]) -> dict:
    c = compare(b[0], b[1], a[0], a[1])  # second group minus first: positive means the newer one is higher
    ca, cb = wilson(*a), wilson(*b)
    verdict = "not enough data" if not c.enough else "changed" if c.clear else "no clear change"
    return {"key": key, "label": label, "kind": "rate", "good": good, "verdict": verdict,
            "a": {"k": a[0], "n": a[1], "value": c.p2, "lo": ca[0] if ca else None, "hi": ca[1] if ca else None},
            "b": {"k": b[0], "n": b[1], "value": c.p1, "lo": cb[0] if cb else None, "hi": cb[1] if cb else None},
            "diff": c.diff, "diff_lo": c.interval[0] if c.interval else None, "diff_hi": c.interval[1] if c.interval else None,
            "p_value": c.p_value, "test": c.test, "evidence": {"a": ev_a, "b": ev_b}}


def _mean_row(key: str, label: str, unit: str, a: list[float], b: list[float], good: str, ev_a: list[str], ev_b: list[str]) -> dict:
    p = mann_whitney(a, b)
    enough = len(a) >= MIN_N and len(b) >= MIN_N
    verdict = "not enough data" if not enough else "changed" if p is not None and p < 0.05 else "no clear change"
    side = lambda v: {"n": len(v), "value": mean(v), "lo": min(v) if v else None, "hi": max(v) if v else None, "median": median(v)}  # noqa: E731
    return {"key": key, "label": label, "kind": "mean", "unit": unit, "good": good, "verdict": verdict, "a": side(a), "b": side(b),
            "diff": (mean(b) - mean(a)) if a and b else None, "diff_lo": None, "diff_hi": None, "p_value": p, "test": "mann-whitney",
            "evidence": {"a": ev_a, "b": ev_b}}


def version_comparison(kit: Kit) -> Finding:
    """What differs in outcomes between the two configuration versions, what does not, and what cannot be told apart."""
    title, q = "V1 against V2: what changed", "Does performance differ between configuration versions?"
    groups = _groups(kit)
    if len(groups) != 2:
        return Finding(id="version_comparison", title=title, question=q, enough=False, n=len(kit.missions),
                       sentence=f"The data has {len(groups)} configuration version(s) ({', '.join(groups) or 'none'}); a comparison needs exactly two.")
    va, vb = groups
    ma = [m for m in kit.missions if m.configuration_version == va]
    mb = [m for m in kit.missions if m.configuration_version == vb]
    ida, idb = {m.mission_id for m in ma}, {m.mission_id for m in mb}
    aa = [a for a in kit.actions if a.mission_id in ida]
    ab = [a for a in kit.actions if a.mission_id in idb]
    ok = lambda acts: (sum(not a.failed for a in acts), len(acts))  # noqa: E731
    failed_ids = lambda acts: [a.action_id for a in acts if a.failed]  # noqa: E731
    rows = [
        _rate_row("mission_completion", "Missions COMPLETED", (sum(m.mission_status == "COMPLETED" for m in ma), len(ma)),
                  (sum(m.mission_status == "COMPLETED" for m in mb), len(mb)), "high",
                  [m.mission_id for m in ma if m.mission_status != "COMPLETED"], [m.mission_id for m in mb if m.mission_status != "COMPLETED"]),
        _rate_row("action_success", "Actions succeeded", ok(aa), ok(ab), "high", failed_ids(aa), failed_ids(ab)),
    ]
    for t in sorted({a.action_type for a in kit.actions}):
        ta, tb = [a for a in aa if a.action_type == t], [a for a in ab if a.action_type == t]
        rows.append(_rate_row(f"success_{t}", f"{_words(t).capitalize()} succeeded", ok(ta), ok(tb), "high", failed_ids(ta), failed_ids(tb)))
    clean = lambda ms: (sum(1 for m in ms if not kit.failed_in(m.mission_id)), len(ms))  # noqa: E731
    rows.append(_rate_row("clean_missions", "Missions with no failed action", clean(ma), clean(mb), "high",
                          [m.mission_id for m in ma if kit.failed_in(m.mission_id)], [m.mission_id for m in mb if kit.failed_in(m.mission_id)]))
    per = lambda ms, pred: [float(sum(1 for x in kit.anomalies_in(m.mission_id) if pred(x))) for m in ms]  # noqa: E731
    an_ids = lambda ids, pred: [x.anomaly_id for x in kit.anomalies if x.mission_id in ids and pred(x)]  # noqa: E731
    high = lambda x: x.severity in ("HIGH", "CRITICAL")  # noqa: E731
    rows += [
        _mean_row("anomalies_per_mission", "Anomalies per mission", "", per(ma, lambda x: True), per(mb, lambda x: True), "low",
                  an_ids(ida, lambda x: True), an_ids(idb, lambda x: True)),
        _mean_row("high_anomalies_per_mission", "HIGH or CRITICAL anomalies per mission", "", per(ma, high), per(mb, high), "low",
                  an_ids(ida, high), an_ids(idb, high)),
        _mean_row("mission_duration", "Mission duration", "min", [m.duration_min for m in ma], [m.duration_min for m in mb], "none",
                  sorted(ida), sorted(idb)),
        _mean_row("action_duration", "Action duration", "s", [a.duration_sec for a in aa], [a.duration_sec for a in ab], "none", [], []),
        _mean_row("start_battery", "Battery at mission start", "%", [m.start_battery_pct for m in ma], [m.start_battery_pct for m in mb],
                  "none", sorted(ida), sorted(idb)),
        _mean_row("end_battery", "Battery at mission end", "%", [m.end_battery_pct for m in ma], [m.end_battery_pct for m in mb], "none",
                  sorted(ida), sorted(idb)),
        _mean_row("battery_used", "Battery used per mission", "points", [m.start_battery_pct - m.end_battery_pct for m in ma],
                  [m.start_battery_pct - m.end_battery_pct for m in mb], "none", sorted(ida), sorted(idb)),
    ]
    # the battery range both versions actually ran in
    floor = max(min((a.battery_pct for a in aa), default=0), min((a.battery_pct for a in ab), default=0))
    ceil = min(max((a.battery_pct for a in aa), default=100), max((a.battery_pct for a in ab), default=100))
    band_a = [a for a in aa if floor <= a.battery_pct <= ceil]
    band_b = [a for a in ab if floor <= a.battery_pct <= ceil]
    matched = _rate_row("action_success_matched_battery", f"Actions succeeded, battery {floor:.0f}% to {ceil:.0f}% only (the range both ran in)",
                        ok(band_a), ok(band_b), "high", failed_ids(band_a), failed_ids(band_b))
    rows.append(matched)
    below = [a for a in aa if a.battery_pct < floor]
    by = {r["key"]: r for r in rows}
    head, comp = by["action_success"], by["mission_completion"]
    la, lb = _group_label(kit, va), _group_label(kit, vb)
    changed = [r["label"].lower() for r in rows if r["verdict"] == "changed" and r["key"] not in ("action_success", "action_success_matched_battery")]
    same = [r["label"].lower() for r in rows if r["verdict"] == "no clear change"]
    thin = [r["label"].lower() for r in rows if r["verdict"] == "not enough data"]

    def pts(r: dict) -> str:
        return (f"{100 * r['diff']:+.1f} points, 95% interval {100 * r['diff_lo']:+.1f} to {100 * r['diff_hi']:+.1f}, {_p(r['p_value'])}"
                if r["diff"] is not None else "n/a")

    sentence = (f"Action success is {pct(head['b']['value'])} ({head['b']['k']}/{head['b']['n']}) under {lb} against "
                f"{pct(head['a']['value'])} ({head['a']['k']}/{head['a']['n']}) under {la}: {pts(head)}. "
                f"Mission completion is {pct(comp['b']['value'])} ({comp['b']['k']}/{comp['b']['n']}) against {pct(comp['a']['value'])} "
                f"({comp['a']['k']}/{comp['a']['n']}): {comp['verdict']}.")
    confound = None
    if below:
        kb = sum(a.failed for a in below)
        confound = (f"{vb} never ran below {floor:.0f}% battery; {va} did for {len(below)} actions and {kb} of them failed ({pct(kb / len(below))}). "
                    f"Inside the battery range both ran in, the gap is {pct(matched['b']['value'])} ({matched['b']['k']}/{matched['b']['n']}) "
                    f"against {pct(matched['a']['value'])} ({matched['a']['k']}/{matched['a']['n']}): {pts(matched)}, {matched['verdict']}. "
                    f"So the data cannot separate 'the {vb} configuration is more reliable' from '{vb} missions end with more battery'. "
                    "The kit does not say what changed; neither do we.")
    return Finding(id="version_comparison", title=title, question=q, sentence=sentence, value=head["diff"], unit="points of action success",
                   n=head["a"]["n"] + head["b"]["n"], interval=(head["diff_lo"], head["diff_hi"]) if head["diff_lo"] is not None else None,
                   p_value=head["p_value"], enough=head["verdict"] != "not enough data", caveat=confound,
                   evidence=Evidence(missions=sorted(ida | idb), actions=failed_ids(aa) + failed_ids(ab)),
                   extra={"a": {"version": va, "label": la, "missions": len(ma), "actions": len(aa)},
                          "b": {"version": vb, "label": lb, "missions": len(mb), "actions": len(ab)}, "metrics": rows,
                          "changed": changed, "unchanged": same, "not_enough": thin, "battery_band": [floor, ceil],
                          "below_band": {"k": sum(a.failed for a in below), "n": len(below), "actions": failed_ids(below)}})


# --- battery, sequence, time of day, durations ---


def battery_vs_reliability(kit: Kit) -> Finding:
    def bin_of(a: Action) -> str:
        return next((f"{lo}-{min(hi, 100)}" for lo, hi in BATTERY_BINS if lo <= a.battery_pct < hi), "?")

    order = [f"{lo}-{min(hi, 100)}" for lo, hi in BATTERY_BINS]
    bars = sorted(_failure_bars(kit.actions, bin_of, lambda k: f"{k}%"), key=lambda b: order.index(b.key) if b.key in order else 99)
    for b in bars:
        acts = [a for a in kit.actions if bin_of(a) == b.key]
        b.extra = {"by_version": {v: {"k": sum(a.failed for a in acts if a.configuration_version == v),
                                      "n": sum(1 for a in acts if a.configuration_version == v)} for v in _groups(kit)}}
    low = [a for a in kit.actions if a.battery_pct < 40]
    rest = [a for a in kit.actions if a.battery_pct >= 40]
    cmp = compare(sum(a.failed for a in low), len(low), sum(a.failed for a in rest), len(rest))
    title, q = "Battery against reliability", "Does battery state appear related to reliability?"
    if not cmp.enough:
        return Finding(id="battery", title=title, question=q, rows=bars, n=len(kit.actions), enough=False,
                       sentence=f"Not enough data: {len(low)} action(s) below 40% battery and {len(rest)} at 40% or above.")
    versions_low = Counter(a.configuration_version for a in low)
    lowest = bars[0] if bars else None
    extra_low = (f" Under 20% it is {lowest.k} of {lowest.n} ({pct(lowest.rate)}, interval {pct_range((lowest.lo, lowest.hi))})."
                 if lowest and lowest.key == "0-20" and lowest.n else "")
    seq = Counter(a.sequence for a in low)
    late = sum(n for s, n in seq.items() if s >= 6)
    caveat = (f"Battery falls through every mission, so low battery and late-in-the-mission are the same actions here "
              f"({late} of {len(low)} low-battery actions are step 6 or later). All low-battery actions are "
              f"{', '.join(f'{v} ({n})' for v, n in sorted(versions_low.items()))}. The data shows an association; it cannot show that "
              "battery is the cause.")
    return Finding(id="battery", title=title, question=q, value=cmp.p1, unit="failure rate below 40% battery", n=len(low),
                   interval=wilson(cmp.k1, cmp.n1), p_value=cmp.p_value, enough=True, caveat=caveat, rows=bars,
                   sentence=f"Below 40% battery {cmp.k1} of {cmp.n1} actions failed ({pct(cmp.p1)}); at 40% or above, {cmp.k2} of {cmp.n2} "
                            f"({pct(cmp.p2)}): {_pts(cmp)}.{extra_low}",
                   evidence=Evidence(actions=[a.action_id for a in low if a.failed]),
                   extra={"clear": cmp.clear, "difference": cmp.diff, "difference_interval": cmp.interval, "threshold": 40})


def sequence_effect(kit: Kit) -> Finding:
    bars = sorted(_failure_bars(kit.actions, lambda a: str(a.sequence), lambda k: f"step {k}"), key=lambda b: int(b.key))
    for b in bars:
        acts = [a for a in kit.actions if str(a.sequence) == b.key]
        b.extra = {"by_version": {v: {"k": sum(a.failed for a in acts if a.configuration_version == v),
                                      "n": sum(1 for a in acts if a.configuration_version == v)} for v in _groups(kit)},
                   "mean_battery": mean([a.battery_pct for a in acts])}
    last = max((a.sequence for a in kit.actions), default=0)
    tail = [a for a in kit.actions if a.sequence >= last - 1]
    head = [a for a in kit.actions if a.sequence < last - 1]
    cmp = compare(sum(a.failed for a in tail), len(tail), sum(a.failed for a in head), len(head))
    title, q = "Position in the mission", "Do failures come at a particular point in a mission?"
    if not cmp.enough:
        return Finding(id="sequence", title=title, question=q, rows=bars, n=len(kit.actions), enough=False, sentence="Not enough data.")
    by_v = {v: (sum(a.failed for a in tail if a.configuration_version == v), sum(1 for a in tail if a.configuration_version == v))
            for v in _groups(kit)}
    return Finding(id="sequence", title=title, question=q, value=cmp.p1, unit="failure rate in the last two steps", n=len(tail),
                   interval=wilson(cmp.k1, cmp.n1), p_value=cmp.p_value, rows=bars,
                   sentence=f"The last two steps of a mission (steps {last - 1} and {last}) failed {cmp.k1} of {cmp.n1} times ({pct(cmp.p1)}) "
                            f"against {cmp.k2} of {cmp.n2} ({pct(cmp.p2)}) for the earlier steps: {_pts(cmp)}. By version: "
                            + ", ".join(f"{v} {k}/{n}" for v, (k, n) in by_v.items()) + ".",
                   caveat="Late steps are also the low-battery steps; see the battery finding. Step 1 is nearly always navigation to the same waypoint.",
                   evidence=Evidence(actions=[a.action_id for a in tail if a.failed]), extra={"clear": cmp.clear, "by_version": by_v})


def time_of_day(kit: Kit) -> Finding:
    bars = sorted(_failure_bars(kit.actions, lambda a: f"{kit.mission[a.mission_id].start_time.hour:02d}:00" if a.mission_id in kit.mission else "?"),
                  key=lambda b: b.key)
    for b in bars:
        b.label = f"missions starting {b.key}"
    found = _versus_rest(sorted(bars, key=lambda b: -(b.rate or 0)))
    title, q = "Time of day", "Does the time a mission starts go with failures?"
    if not found:
        return Finding(id="time_of_day", title=title, question=q, rows=bars, n=len(kit.actions), enough=False, sentence="Not enough data.")
    worst, cmp = found
    verdict = "a clear difference" if cmp.clear else "no clear effect: the interval includes zero"
    return Finding(id="time_of_day", title=title, question=q, value=worst.rate, unit="failure rate", n=worst.n, p_value=cmp.p_value,
                   interval=(worst.lo, worst.hi), rows=bars,
                   sentence=f"Missions start at {len(bars)} times of day. The highest failure rate is for {worst.label}: {worst.k} of {worst.n} "
                            f"({pct(worst.rate)}) against {pct(cmp.p2)} ({cmp.k2}/{cmp.n2}) otherwise: {_pts(cmp)}; {verdict}.",
                   caveat="Timestamps have no time zone; hours are as written in the files.",
                   evidence=Evidence(actions=worst.evidence.actions), extra={"clear": cmp.clear})


def durations(kit: Kit) -> Finding:
    failed = [a.duration_sec for a in kit.actions if a.failed]
    ok = [a.duration_sec for a in kit.actions if not a.failed]
    p = mann_whitney(failed, ok)
    title, q = "Action durations", "Do failed actions take longer (a sign of retries or timeouts)?"
    if len(failed) < MIN_N or len(ok) < MIN_N:
        return Finding(id="durations", title=title, question=q, n=len(kit.actions), enough=False, sentence="Not enough data.")
    rows = []
    for ftype in sorted({a.failure_type for a in kit.actions if a.failed}):
        v = [a.duration_sec for a in kit.actions if a.failure_type == ftype]
        b = _bar(ftype, _words(ftype), len(v), len(failed), Evidence(actions=[a.action_id for a in kit.actions if a.failure_type == ftype]),
                 mean_sec=mean(v), median_sec=median(v))
        b.enough, b.note = len(v) >= MIN_N, None if len(v) >= MIN_N else f"not enough data (n={len(v)}, need {MIN_N})"
        rows.append(b)
    differs = p is not None and p < 0.05
    return Finding(id="durations", title=title, question=q, value=mean(failed), unit="s mean for failed actions", n=len(failed), p_value=p,
                   rows=rows,
                   sentence=f"Failed actions took {mean(failed):.0f} s on average (median {median(failed):.0f}, n={len(failed)}); successful ones "
                            f"{mean(ok):.0f} s (median {median(ok):.0f}, n={len(ok)}): {'a clear difference' if differs else 'no clear difference'} "
                            f"({_p(p)}).",
                   caveat="The kit has no retry count or timeout field, so retries cannot be measured; duration is the only proxy and it "
                          + ("differs." if differs else "does not separate failures from successes."),
                   evidence=Evidence(actions=[a.action_id for a in kit.actions if a.failed]),
                   extra={"failed": {"n": len(failed), "mean": mean(failed), "median": median(failed)},
                          "ok": {"n": len(ok), "mean": mean(ok), "median": median(ok)}})


def performance_over_time(kit: Kit) -> Finding:
    """Action success per mission in time order: improving or degrading?"""
    points = []
    for m in kit.missions:
        acts = kit.actions_of.get(m.mission_id, [])
        if acts:
            points.append({"mission_id": m.mission_id, "start": m.start_time.isoformat(), "version": m.configuration_version,
                           "k": sum(a.failed for a in acts), "n": len(acts), "status": m.mission_status,
                           "anomalies": len(kit.anomalies_in(m.mission_id)), "end_battery": m.end_battery_pct})
    title, q = "Performance over time", "Is performance improving or degrading?"
    groups = _groups(kit)
    latest = groups[-1] if groups else None
    recent = [p for p in points if p["version"] == latest]
    if len(points) < MIN_N:
        return Finding(id="over_time", title=title, question=q, n=len(points), enough=False, sentence="Not enough missions for a trend.",
                       extra={"points": points})
    slope_all = theil_sen([(i, p["k"] / p["n"]) for i, p in enumerate(points)])
    slope_recent = theil_sen([(i, p["k"] / p["n"]) for i, p in enumerate(recent)]) if len(recent) >= MIN_TREND_N else None
    if len(groups) == 2:
        first = [p for p in points if p["version"] == groups[0]]
        text = (f"Failed actions per mission fell from {mean([p['k'] for p in first]):.1f} of {first[0]['n']} ({groups[0]}, {len(first)} missions) "
                f"to {mean([p['k'] for p in recent]):.1f} ({latest}, {len(recent)} missions): improving, with the step at the change of version. ")
    else:
        text = f"Across {len(points)} missions the failure rate moves {100 * (slope_all or 0):+.2f} points per mission. "
    if slope_recent is None:
        text += f"Inside {latest} there are {len(recent)} missions: not enough for a trend."
    else:
        direction = "flat" if abs(slope_recent) < 0.005 else "rising" if slope_recent > 0 else "falling"
        text += (f"Inside {latest} the failure rate is {direction} ({100 * slope_recent:+.1f} points per mission, median slope over "
                 f"{len(recent)} missions): no sign of it degrading again." if direction != "rising" else
                 f"Inside {latest} the failure rate is rising ({100 * slope_recent:+.1f} points per mission over {len(recent)} missions): watch it.")
    return Finding(id="over_time", title=title, question=q, sentence=text, value=slope_recent, unit="failure-rate change per mission",
                   n=len(points), caveat=f"{len(recent)} missions in the latest version is a short run; one bad mission would change the slope.",
                   evidence=Evidence(missions=[p["mission_id"] for p in points]), extra={"points": points, "slope_all": slope_all})


# --- anomalies ---


def _confirmed(kit: Kit, anomaly) -> bool:
    """Raised by an action that succeeded: the inspection behind the anomaly can be trusted."""
    a = kit.action.get(anomaly.action_id)
    return a is not None and not a.failed


def anomaly_matrix(kit: Kit) -> Finding:
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    for x in kit.anomalies:
        crit = kit.asset[x.asset_id].criticality if x.asset_id in kit.asset else "LOW"
        cells[(x.severity, crit)].append(x.anomaly_id)
    matrix = [{"severity": s, "criticality": c, "n": len(cells.get((s, c), [])), "anomalies": cells.get((s, c), [])}
              for s in reversed(SEVERITIES) for c in reversed(CRITICALITIES)]
    hot = [x for x in kit.anomalies if x.severity in ("HIGH", "CRITICAL") and x.asset_id in kit.asset
           and kit.asset[x.asset_id].criticality in ("HIGH", "CRITICAL")]
    rows = [_bar(s, s, sum(1 for x in kit.anomalies if x.severity == s), len(kit.anomalies),
                 Evidence(anomalies=[x.anomaly_id for x in kit.anomalies if x.severity == s])) for s in reversed(SEVERITIES)]
    return Finding(id="anomaly_matrix", title="Anomaly severity against asset criticality", question="Why does it matter?",
                   value=len(hot), unit="HIGH or CRITICAL anomalies on HIGH or CRITICAL assets", n=len(kit.anomalies), rows=rows,
                   enough=len(kit.anomalies) >= MIN_N,
                   sentence=f"{len(hot)} of {len(kit.anomalies)} anomalies are HIGH or CRITICAL on an asset that is itself HIGH or CRITICAL: "
                            f"{', '.join(x.anomaly_id for x in hot) or 'none'}. Severity alone would rank "
                            f"{sum(1 for x in kit.anomalies if x.severity in ('HIGH', 'CRITICAL'))} anomalies the same.",
                   caveat="Criticality comes from assets.csv for the asset the anomaly names, even when the action happened at another waypoint.",
                   evidence=Evidence(anomalies=[x.anomaly_id for x in hot], assets=sorted({x.asset_id for x in hot})),
                   extra={"matrix": matrix})


def anomalies_in_successful_missions(kit: Kit) -> Finding:
    completed = {m.mission_id for m in kit.missions if m.mission_status == "COMPLETED"}
    inside = [x for x in kit.anomalies if x.mission_id in completed]
    serious = [x for x in inside if x.severity in ("HIGH", "CRITICAL")]
    clean = {m for m in completed if not kit.failed_in(m)}
    in_clean = [x for x in kit.anomalies if x.mission_id in clean]
    confirmed = [x for x in kit.anomalies if _confirmed(kit, x)]
    unverified = [x for x in kit.anomalies if not _confirmed(kit, x)]
    rows = []
    for mid in sorted({x.mission_id for x in serious}):
        xs = [x for x in serious if x.mission_id == mid]
        b = _bar(mid, mid, len(xs), len(kit.actions_of.get(mid, [])), Evidence(missions=[mid], anomalies=[x.anomaly_id for x in xs]),
                 severities=[x.severity for x in xs], failed_actions=len(kit.failed_in(mid)))
        b.enough, b.note = True, None
        rows.append(b)
    rows.sort(key=lambda b: (-max(SEV_RANK[s] for s in b.extra["severities"]), -b.k, b.key))
    return Finding(id="anomalies_hidden", title="Anomalies inside missions that look fine",
                   question="Are anomalies hidden inside otherwise successful missions?",
                   value=len(inside) / len(kit.anomalies) if kit.anomalies else None, unit="of anomalies are in COMPLETED missions",
                   n=len(kit.anomalies), enough=len(kit.anomalies) >= MIN_N, rows=rows,
                   sentence=f"{len(inside)} of {len(kit.anomalies)} anomalies were raised inside missions marked COMPLETED, {len(serious)} of them "
                            f"HIGH or CRITICAL (in {len(rows)} missions). {len(in_clean)} of the {len(inside)} sit(s) in a mission with no failed action at all "
                            f"({', '.join(x.anomaly_id for x in in_clean) or 'none'}). {len(confirmed)} anomalies come from an action that "
                            f"succeeded; the other {len(unverified)} come from an action that itself failed, so the inspection behind them "
                            "is questionable and the asset needs a second look before anyone is sent.",
                   caveat="The kit does not say whether an anomaly on a failed action is a real finding or a by-product of the failure. "
                          "We treat it as unverified, not as false.",
                   evidence=Evidence(anomalies=[x.anomaly_id for x in serious], missions=[b.key for b in rows]),
                   extra={"confirmed": [x.anomaly_id for x in confirmed], "unverified": [x.anomaly_id for x in unverified],
                          "in_clean_missions": [x.anomaly_id for x in in_clean], "in_completed": len(inside)})


def repeat_anomalies(kit: Kit) -> Finding:
    by_asset: dict[str, list] = defaultdict(list)
    for x in kit.anomalies:
        by_asset[x.asset_id].append(x)
    rows = []
    for asset_id, xs in by_asset.items():
        if len(xs) < 2:
            continue
        types = Counter(x.anomaly_type for x in xs)
        same_type = {t: n for t, n in types.items() if n >= 2}
        crit = kit.asset[asset_id].criticality if asset_id in kit.asset else ""
        b = _bar(asset_id, _asset_label(kit, asset_id), len(xs), len(kit.anomalies),
                 Evidence(assets=[asset_id], anomalies=[x.anomaly_id for x in xs]), criticality=crit, types=dict(types),
                 same_type=same_type, worst=max((x.severity for x in xs), key=SEV_RANK.get), first=xs[0].timestamp.isoformat(),
                 last=xs[-1].timestamp.isoformat(), missions=sorted({x.mission_id for x in xs}))
        b.enough = True
        b.note = ("same type twice or more: " + ", ".join(f"{_words(t)} x{n}" for t, n in same_type.items())) if same_type else "different types each time"
        rows.append(b)
    rows.sort(key=lambda b: (-b.k, -SEV_RANK[b.extra["worst"]], b.key))
    same = [b for b in rows if b.extra["same_type"]]
    if not rows:
        return Finding(id="anomaly_repeats", title="Repeat anomalies on one asset", question="What keeps coming back?", n=len(kit.anomalies),
                       sentence="No asset has more than one anomaly.", enough=False)
    top = rows[0]
    return Finding(id="anomaly_repeats", title="Repeat anomalies on one asset", question="What keeps coming back?", value=len(rows),
                   unit="assets with two or more anomalies", n=len(by_asset), rows=rows,
                   sentence=f"{len(rows)} of {len(by_asset)} assets with any anomaly have two or more. Most: {top.label} with {top.k} "
                            f"(worst {top.extra['worst']}, across missions {', '.join(top.extra['missions'])}). {len(same)} asset(s) saw the "
                            f"same anomaly type more than once ({', '.join(b.key for b in same) or 'none'}).",
                   caveat="Two anomalies of different types days apart on one asset may be unrelated; the same type repeating is the stronger sign.",
                   evidence=Evidence(assets=[b.key for b in rows], anomalies=[i for b in rows for i in b.evidence.anomalies]))


def reading_trends(kit: Kit) -> Finding:
    """Per asset and measurement: the points, a robust slope on the GOOD ones, and a reference band from the fleet."""
    bands: dict[str, dict] = {}
    for mtype in sorted({r.measurement_type for r in kit.readings}):
        good = [r.value for r in kit.readings if r.measurement_type == mtype and r.good]
        mid, spread = median(good), mad(good)
        if mid is None or spread is None or len(good) < MIN_N:
            continue
        unit = next((r.unit for r in kit.readings if r.measurement_type == mtype), "")
        bands[mtype] = {"median": mid, "lo": mid - 3 * spread, "hi": mid + 3 * spread, "n": len(good), "unit": unit,
                        "higher_is_better": mtype in HIGHER_IS_BETTER}
    series_of: dict[tuple[str, str], list] = defaultdict(list)
    for r in kit.readings:
        series_of[(r.asset_id, r.measurement_type)].append(r)
    start = min((r.timestamp for r in kit.readings), default=None)
    series = []
    for (asset_id, mtype), rs in sorted(series_of.items()):
        good = [r for r in rs if r.good]
        band = bands.get(mtype)
        item = {"asset_id": asset_id, "measurement_type": mtype, "unit": rs[0].unit, "n": len(rs), "n_good": len(good),
                "points": [{"reading_id": r.reading_id, "t": r.timestamp.isoformat(), "value": r.value, "good": r.good,
                            "action_id": r.action_id, "mission_id": r.mission_id} for r in rs],
                "slope_per_day": None, "enough": len(good) >= MIN_TREND_N, "outside_band": [], "toward_edge": False, "days_to_edge": None,
                "note": None}
        if band:
            item["outside_band"] = [r.reading_id for r in good if not band["lo"] <= r.value <= band["hi"]]
        if len(good) >= MIN_TREND_N and start is not None:
            slope = theil_sen([((r.timestamp - start).total_seconds() / 86400, r.value) for r in good])
            item["slope_per_day"] = slope
            if band and slope:
                last = good[-1].value
                edge = band["lo"] if slope < 0 else band["hi"]
                bad_edge = not (band["higher_is_better"] and slope > 0)
                days = (edge - last) / slope if slope else None
                if bad_edge and days is not None and 0 <= days <= 7:
                    item["toward_edge"], item["days_to_edge"] = True, days
        else:
            item["note"] = f"not enough data ({len(good)} good reading(s), need {MIN_TREND_N})"
        series.append(item)
    usable = [s for s in series if s["enough"]]
    flagged = [s for s in usable if s["toward_edge"] or s["outside_band"]]
    series.sort(key=lambda s: (not (s["toward_edge"] or bool(s["outside_band"])), not s["enough"], -s["n_good"], s["asset_id"]))
    outside = [(s["asset_id"], s["measurement_type"], i) for s in series for i in s["outside_band"]]
    text = (f"{len(series)} asset-and-measurement series from {len(kit.readings)} readings; {len(usable)} have {MIN_TREND_N} or more good "
            f"readings, enough for a trend. ")
    if flagged:
        text += "Worth a look: " + "; ".join(
            f"{s['asset_id']} {_words(s['measurement_type'])}"
            + (f" heading for the edge of the fleet's band in about {s['days_to_edge']:.0f} day(s)" if s["toward_edge"] else "")
            + (f" with {len(s['outside_band'])} reading(s) outside the band" if s["outside_band"] else "") for s in flagged[:4]) + "."
    else:
        text += "None of them is heading for the edge of the fleet's band within a week, and no good reading on them sits outside it."
    if outside and not flagged:
        text += f" {len(outside)} good reading(s) on shorter series sit outside the band."
    return Finding(id="reading_trends", title="Reading trends per asset", question="Is anything drifting toward a limit?", n=len(kit.readings),
                   value=len(flagged), unit="series worth a look", enough=bool(usable), sentence=text,
                   caveat="The kit gives no engineering thresholds. The band is ours: the fleet median of GOOD readings of that measurement "
                          "plus or minus three robust standard deviations (MAD). It says 'unusual for this fleet', not 'unsafe'. "
                          "QUESTIONABLE readings are drawn but left out of trends. Anomaly rows are not explained by reading values in this "
                          "kit (a visual defect is flagged on scores of 96 and 98), so values cannot stand in for anomalies.",
                   evidence=Evidence(readings=[i for s in flagged for i in (s["outside_band"] or [p["reading_id"] for p in s["points"]])],
                                     assets=sorted({s["asset_id"] for s in flagged})),
                   extra={"bands": bands, "series": series})


# --- everything, and the cross-check ---

FINDERS: tuple[Callable[[Kit], Finding], ...] = (
    hidden_failures, version_comparison, battery_vs_reliability, reliability_by_type, reliability_by_location, navigation_hotspot,
    reliability_by_asset, reliability_by_floor, reliability_by_environment, failure_types, sequence_effect, time_of_day, durations,
    performance_over_time, anomalies_in_successful_missions, anomaly_matrix, repeat_anomalies, reading_trends,
)
HEADLINES = ("hidden_failures", "version_comparison", "battery", "reliability_by_type", "navigation_hotspot", "anomalies_hidden")


def all_findings(kit: Kit) -> list[Finding]:
    return [f(kit) for f in FINDERS]


def validation(kit: Kit) -> dict:
    """Our numbers next to the sponsor's validation_results.json: agree, differ, or nothing to compare with."""
    theirs = kit.validation or {}
    rows = []

    def row(group: str, name: str, ours, how: str) -> None:
        expected = (theirs.get(group) or {}).get(name)
        if expected is None:
            status = "not in validation file"
        elif isinstance(ours, float) or isinstance(expected, float):
            status = "agree" if ours is not None and abs(float(ours) - float(expected)) < 0.051 else "differ"
        else:
            status = "agree" if ours == expected else "differ"
        rows.append({"group": group, "name": name, "ours": ours, "theirs": expected, "status": status, "how": how})

    for name, ours in (("missions", len(kit.missions)), ("actions", len(kit.actions)), ("readings", len(kit.readings)),
                       ("anomalies", len(kit.anomalies)), ("locations", len(kit.locations)), ("assets", len(kit.assets)),
                       ("replay_events", len(kit.replay))):
        row("counts", name, ours, "lines read from replay.ndjson" if name == "replay_events" else f"rows read from {name}.csv")

    def share(k: int, n: int) -> float | None:
        return round(100 * k / n, 1) if n else None

    for scenario in sorted({m.scenario for m in kit.missions}):
        ms = [m for m in kit.missions if m.scenario == scenario]
        acts = [a for a in kit.actions if a.scenario == scenario]
        row("metrics", f"{scenario.lower()}_mission_completion_pct", share(sum(m.mission_status == "COMPLETED" for m in ms), len(ms)),
            f"{sum(m.mission_status == 'COMPLETED' for m in ms)} of {len(ms)} {scenario} missions have mission_status COMPLETED")
        row("metrics", f"{scenario.lower()}_action_success_pct", share(sum(not a.failed for a in acts), len(acts)),
            f"{sum(not a.failed for a in acts)} of {len(acts)} {scenario} actions have status SUCCESS")
    for name in (theirs.get("metrics") or {}):
        parts = name.split("_")
        if len(parts) >= 4 and parts[0].upper() in kit.location and name.endswith("_failure_pct"):
            loc, atype = parts[0].upper(), "_".join(parts[1:-2]).upper()
            acts = [a for a in kit.actions if a.location_id == loc and a.action_type == atype]
            row("metrics", name, share(sum(a.failed for a in acts), len(acts)),
                f"{sum(a.failed for a in acts)} of {len(acts)} {atype} actions at {loc} have status FAILED")
    known = {r["name"] for r in rows}
    for group in ("counts", "metrics"):
        for name, value in (theirs.get(group) or {}).items():
            if name not in known:
                rows.append({"group": group, "name": name, "ours": None, "theirs": value, "status": "not computed", "how": "no matching calculation"})
    differ = [r for r in rows if r["status"] == "differ"]
    compared = [r for r in rows if r["status"] in ("agree", "differ")]
    return {"available": kit.validation is not None, "their_status": theirs.get("validation_status"),
            "their_problems": theirs.get("validation_problems", []), "generated_at": theirs.get("generated_at"),
            "agree": len(compared) - len(differ), "compared": len(compared), "differences": differ, "rows": rows,
            "summary": ("No validation_results.json in the kit folder." if kit.validation is None else
                        f"{len(compared) - len(differ)} of {len(compared)} numbers agree with the kit's validation_results.json."
                        + (f" Different: {', '.join(r['name'] for r in differ)}." if differ else "")),
            "issues": [i.model_dump() for i in kit.issues]}


def assumptions(kit: Kit) -> list[dict]:
    """Every assumption the analysis makes, in plain words, with why it matters. The brief asks for exactly this list."""
    now = kit.now
    out = [
        ("now", "'Now' is the newest timestamp in the data"
         + (f" ({now.isoformat(sep=' ', timespec='minutes')})" if now else "") + ", not the wall clock.",
         "Recency in the triage score and 'hours since last good inspection' are measured back from there."),
        ("status", "An action counts as failed when its status is anything other than SUCCESS. Mission status is read separately and never "
         "derived from actions.", "The kit says a mission may be COMPLETED with failed actions inside."),
        ("one_comparison", "scenario (BASELINE/IMPROVED) and configuration_version (V1/V2) are treated as one comparison.",
         "They split the missions identically, so they cannot be analysed as two factors."),
        ("no_cause", "No cause is claimed for the V1 to V2 difference.",
         "The kit does not say what changed, and V2 missions also start with more battery, end with far more and are shorter, so the "
         "configuration and the battery cannot be separated."),
        ("intervals", f"Rates carry 95% Wilson intervals; differences use Newcombe's interval with Fisher's exact test or a z-test. "
         f"Under {MIN_N} rows we say 'not enough data'; under {MIN_TREND_N} good readings there is no trend.",
         "The tables are small (hundreds of rows); a rate without its n is misleading."),
        ("no_correction", "Intervals are not corrected for the number of comparisons made.",
         "With fifteen waypoints and twenty assets, one of them will look bad by chance; treat single-waypoint findings as leads."),
        ("location", "A waypoint's numbers use the location on the action row. An asset is drawn on the map at its assets.csv location.",
         "About a fifth of actions name an asset at a different waypoint than assets.csv gives it."),
        ("inspection", "An asset counts as inspected when a non-navigation action on it succeeded, with or without a reading row. "
         "A failed action, or a QUESTIONABLE reading, means the asset was effectively not inspected that time.",
         "Reading quality is QUESTIONABLE exactly when the action failed, so quality adds nothing to status."),
        ("unverified", "An anomaly raised by a failed action is 'unverified': re-inspect before dispatching. One raised by a successful action "
         "is 'confirmed'.", "Most anomaly rows sit on failed actions; the kit does not say whether those findings are real."),
        ("no_thresholds", "There are no engineering limits in the kit. The band on reading charts is the fleet median of GOOD readings plus "
         "or minus three robust standard deviations.", "It marks 'unusual for this fleet', not 'unsafe'."),
        ("triage", "The triage score is a transparent sum we chose: anomaly severity weighted by how recent it is, asset criticality, repeat "
         "anomalies, whether the latest inspection attempt failed, and days since the last good inspection. The weights are judgement, not "
         "fitted to anything.", "Every item shows its parts so an operator can disagree with a weight."),
        ("review", "operator_review_required is not used on its own.", "It is True exactly for HIGH and CRITICAL anomalies."),
        ("replay", "The replay is played in event-time order. Readings and anomalies are attached when their action ends in the stream. "
         "Missions the stream never reaches are rebuilt from missions.csv and actions.csv and marked as rebuilt. Idle time between "
         "missions is skipped.", "replay.ndjson has mission and action events only, and stops before the last missions."),
        ("map", "Coordinates are drawn as given, x to the right and y downward, each floor on its own panel, both scaled 0 to 100.",
         "The kit's floor-plan image is not in this copy; the layout is a schematic, not a plan."),
        ("one_robot", "One robot is assumed: missions never overlap in time.", "The robot's position in the replay is the waypoint of the "
         "latest action."),
        ("timezone", "Timestamps have no zone and are used as written.", "Only differences between them matter to the analysis."),
        ("synthetic", "All of this is synthetic challenge data from the sponsor's kit.", "Nothing here describes a real facility or robot."),
    ]
    return [{"key": k, "assumption": a, "why": w} for k, a, w in out]
