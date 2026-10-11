"""Shift Run: plan SPOT's next round, see how it would probably go, and compare with a plan built by the same model.

    model = build_model(kit, triage, uninspected)      # every number measured from the kit, once
    result = simulate(model, ["HX-202", "HX-204"], turn_back_at=40)
    result["score"], result["stars"], result["steps"], result["reasons"]
    best = recommend(model)                            # greedy, then swaps and drops, on the same score

**A toy model, said plainly.** It is built from the kit's 270 recorded actions and nothing else: no physics, no map
distances, no knowledge of the robot. It is a training aid for thinking about order and battery, not a prediction.

How a check's chance of failing is put together (pure, deterministic):

- Start from the site-wide failure rate (49 of 270).
- Multiply the odds by one odds ratio for the kind of check, one for the waypoint, and one for "late and low":
  the larger of the battery-band ratio and the position-in-the-round ratio. Battery and position are the same
  actions in the kit (battery falls through every round), so multiplying both would count one effect twice.
- A group with fewer than `MIN_N` actions contributes nothing (ratio 1). Every ratio is capped to
  [`OR_MIN`, `OR_MAX`] and the final probability to [`P_MIN`, `P_MAX`], so no thin cell can dominate.

Battery and time per step are the kit's own averages: the battery drop and the minutes between consecutive actions,
taken separately for a step that stays on a floor and one that changes floor. The kit has no distances.

What the Spot User Guide (Release 1.1, the kit's Part Two) adds, and what it does not:
- p. 43-44: when an obstacle blocks a waypoint the operator takes control, walks SPOT past it and uses "skip part
  of the mission". Hard mode's blocked stop is that: the stop is skipped, the walk still costs battery and time.
- p. 44: a weak battery is a listed cause of SPOT losing its connection and stopping; the fix is to recharge. That
  is the reason for a turn-back level at all. The guide pages we were given state no battery life, runtime, walking
  speed or automatic return-to-dock, so none is assumed here: the turn-back level is the operator's own rule.

The score is an expected value (no luck): the value of each asset times the chance it is checked successfully, minus
a cost per expected failed check, minus a cost for every check run on a low battery (more under the danger band) and
for ending there. An asset is worth a little less for every stop it waits (`WAIT_DISCOUNT`): what is urgent is worth
knowing sooner. The values and costs are judgement, like the triage weights, and are returned part by part. The
"play" view draws one outcome per check from a seeded generator, for the animation only; it never changes the score.
"""

import random
from collections import defaultdict
from statistics import mean, median

from contexts.spot.domain.models import Kit
from contexts.spot.domain.stats import MIN_N

OR_MIN, OR_MAX = 0.5, 3.0  # no single factor may halve or more than triple the odds
P_MIN, P_MAX = 0.03, 0.85
BATTERY_BANDS = ((0, 20), (20, 40), (40, 60), (60, 80), (80, 101))
LOW, DANGER, EMPTY = 40.0, 20.0, 5.0  # below LOW checks fail more often; below DANGER most do; at EMPTY SPOT stops where it is
MAX_STOPS = 12
TURN_BACK_CHOICES = (25, 30, 35, 40, 45, 50, 55, 60)
LATE_FROM = 9  # the kit's rounds have ten actions; "late" is the last two

# value of one successful check of an asset
BASE_VALUE = 10.0
CRITICALITY_VALUE = {"CRITICAL": 15.0, "HIGH": 10.0, "MEDIUM": 4.0, "LOW": 0.0}
STALE_VALUE_PER_DAY, STALE_VALUE_CAP = 10.0, 20.0
WAIT_DISCOUNT = 0.02  # share of an asset's value lost per stop it waits: urgent things are worth knowing sooner
# costs
FAILED_CHECK_COST = 6.0
LOW_CHECK_COST = 8.0  # each check run under LOW battery: its result is the kind that fails and leaves an alarm nobody can confirm
DANGER_CHECK_COST = 30.0  # each check run under DANGER battery
DANGER_END_COST = 40.0  # the round ends (after the walk back) under DANGER
STRANDED_COST = 60.0  # the battery reached EMPTY out on the floor
STARS = ((0.9, 3), (0.7, 2), (0.4, 1))  # share of the recommended plan's score


def _odds(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return p / (1 - p)


def _ratio(k: int, n: int, base: float) -> float:
    """Odds ratio of a group against the site-wide rate; 1 when the group is too small to say."""
    if n < MIN_N:
        return 1.0
    rate = (k + 0.5) / (n + 1)  # a half-count keeps a 0-of-n group finite
    return min(OR_MAX, max(OR_MIN, _odds(rate) / _odds(base)))


def _band(battery: float) -> tuple[int, int]:
    return next((b for b in BATTERY_BANDS if b[0] <= battery < b[1]), BATTERY_BANDS[0])


def build_model(kit: Kit, triage: list[dict], uninspected: list[dict]) -> dict:
    """Everything the simulation needs, measured from the kit: rates, drains, times, and what each asset is worth."""
    acts = kit.actions
    base = sum(a.failed for a in acts) / len(acts)

    def ratios(key) -> dict:
        groups: dict = defaultdict(lambda: [0, 0])
        for a in acts:
            g = groups[key(a)]
            g[0] += int(a.failed)
            g[1] += 1
        return {k: {"k": v[0], "n": v[1], "ratio": _ratio(v[0], v[1], base)} for k, v in groups.items()}

    floors = {x.location_id: x.floor for x in kit.locations}
    drops: dict[bool, list[float]] = {True: [], False: []}
    gaps: dict[bool, list[float]] = {True: [], False: []}
    first_drop, first_gap = [], []
    for m in kit.missions:
        steps = kit.actions_of.get(m.mission_id, [])
        if steps:
            first_drop.append(m.start_battery_pct - steps[0].battery_pct)
            first_gap.append((steps[0].timestamp - m.start_time).total_seconds())
        for a, b in zip(steps, steps[1:]):
            changed = floors.get(a.location_id) != floors.get(b.location_id)
            drops[changed].append(a.battery_pct - b.battery_pct)
            gaps[changed].append((b.timestamp - a.timestamp).total_seconds())
    every_drop, every_gap = drops[True] + drops[False], gaps[True] + gaps[False]

    def avg(rows: list[float], fallback: float) -> float:
        return round(mean(rows), 2) if len(rows) >= MIN_N else fallback

    drop_all, gap_all = round(mean(every_drop), 2), round(mean(every_gap), 1)
    rank = {i["id"]: i for i in triage if i["kind"] == "asset"}
    risk = {r["asset_id"]: r for r in uninspected}
    assets = {}
    for a in kit.assets:
        mine = [x for x in acts if x.asset_id == a.asset_id and x.is_inspection]
        kinds = [x.action_type for x in mine]
        kind = max(sorted(set(kinds)), key=kinds.count) if kinds else "VISUAL_INSPECTION"
        item, r = rank.get(a.asset_id), risk.get(a.asset_id, {})
        hours = r.get("hours_since_good")
        stale = STALE_VALUE_CAP if hours is None else min(STALE_VALUE_CAP, STALE_VALUE_PER_DAY * hours / 24)
        parts = {"base": BASE_VALUE, "triage": float(item["score"]) if item else 0.0, "importance": CRITICALITY_VALUE.get(a.criticality, 0.0),
                 "stale": round(stale, 1)}
        assets[a.asset_id] = {"asset_id": a.asset_id, "location_id": a.location_id, "floor": floors.get(a.location_id, 1), "check": kind,
                              "criticality": a.criticality, "value": round(sum(parts.values()), 1), "value_parts": parts,
                              "triage_rank": item["rank"] if item else None, "hours_since_good": hours}
    last = kit.missions[-1] if kit.missions else None
    by_location = ratios(lambda a: a.location_id)
    return {"base": base, "base_k": sum(a.failed for a in acts), "base_n": len(acts), "rounds": len(kit.missions),
            "by_type": ratios(lambda a: a.action_type), "by_location": by_location,
            "by_band": ratios(lambda a: _band(a.battery_pct)), "late": ratios(lambda a: a.sequence >= LATE_FROM),
            "drop": {"same": avg(drops[False], drop_all), "other": avg(drops[True], drop_all), "first": round(mean(first_drop), 2)},
            "seconds": {"same": avg(gaps[False], gap_all), "other": avg(gaps[True], gap_all), "first": round(mean(first_gap), 1)},
            "start_battery": float(median(m.start_battery_pct for m in kit.missions)), "dock_floor": 1,
            "assets": assets, "floors": floors,
            "worst_location": max((k for k, v in by_location.items() if v["n"] >= MIN_N), key=lambda k: (by_location[k]["ratio"], k), default=None),
            "last_route": list(dict.fromkeys(a.asset_id for a in kit.actions_of.get(last.mission_id, []) if a.asset_id in assets)) if last else []}


def fail_probability(model: dict, check: str, location_id: str, battery: float, position: int) -> dict:
    """The chance one check fails, with the factors that made it (for the "why" on screen)."""
    one = lambda table, key: table.get(key, {"ratio": 1.0})["ratio"]  # noqa: E731
    kind, place = one(model["by_type"], check), one(model["by_location"], location_id)
    low, late = one(model["by_band"], _band(battery)), (one(model["late"], True) if position >= LATE_FROM else one(model["late"], False))
    late_low = max(low, late)  # one effect seen two ways: never multiplied together
    p = _odds(model["base"]) * kind * place * late_low
    p = p / (1 + p)
    return {"p": min(P_MAX, max(P_MIN, p)), "kind": kind, "place": place, "late_low": late_low}


def _walk(model: dict, state: dict, floor: int, first: bool = False) -> None:
    key = "first" if first else "same" if state["floor"] == floor else "other"
    state["battery"] -= model["drop"][key]
    state["seconds"] += model["seconds"][key]
    state["floor"] = floor


def simulate(model: dict, route: list[str], turn_back_at: float, seed: int = 0, blocked: str | None = None) -> dict:
    """Run a plan through the model. `route` is asset ids in visiting order; SPOT turns back when its battery is at or
    under `turn_back_at` before a stop. `blocked` is a waypoint SPOT cannot reach (hard mode)."""
    route = [a for a in route if a in model["assets"]][:MAX_STOPS]
    turn_back_at = float(min(max(turn_back_at, EMPTY), 95))
    rng = random.Random(f"{seed}|{'>'.join(route)}|{turn_back_at}|{blocked}")
    state = {"battery": model["start_battery"], "seconds": 0.0, "floor": model["dock_floor"]}
    steps: list[dict] = []
    not_yet: dict[str, float] = {}  # asset -> chance it has NOT been checked successfully so far
    score = 0.0
    stopped = None
    for i, asset_id in enumerate(route, 1):
        a = model["assets"][asset_id]
        row = {"n": i, "asset_id": asset_id, "location_id": a["location_id"], "check": a["check"], "value": a["value"]}
        if state["battery"] <= turn_back_at:
            stopped = stopped or "turn_back"
            steps.append({**row, "outcome": "not_reached", "battery_pct": round(state["battery"], 1), "p_fail": None, "gain": 0.0, "played": "not_reached"})
            continue
        _walk(model, state, a["floor"], first=i == 1)
        if state["battery"] <= EMPTY:
            stopped = "stranded"
            steps.append({**row, "outcome": "stranded", "battery_pct": round(max(0.0, state["battery"]), 1), "p_fail": None, "gain": 0.0, "played": "stranded"})
            for j, rest in enumerate(route[i:], i + 1):
                b = model["assets"][rest]
                steps.append({"n": j, "asset_id": rest, "location_id": b["location_id"], "check": b["check"], "value": b["value"],
                              "outcome": "not_reached", "battery_pct": round(max(0.0, state["battery"]), 1), "p_fail": None, "gain": 0.0, "played": "not_reached"})
            break
        if blocked and a["location_id"] == blocked:
            steps.append({**row, "outcome": "blocked", "battery_pct": round(state["battery"], 1), "p_fail": None, "gain": 0.0, "played": "blocked"})
            continue
        f = fail_probability(model, a["check"], a["location_id"], state["battery"], i)
        before = not_yet.get(asset_id, 1.0)
        worth = a["value"] * max(0.0, 1 - WAIT_DISCOUNT * (i - 1))
        gain = worth * before * (1 - f["p"])  # a second look only counts for what the first one missed
        not_yet[asset_id] = before * f["p"]
        danger, low = state["battery"] < DANGER, state["battery"] < LOW
        score += gain - FAILED_CHECK_COST * f["p"] - (DANGER_CHECK_COST if danger else LOW_CHECK_COST if low else 0.0)
        steps.append({**row, "outcome": "checked", "battery_pct": round(state["battery"], 1), "p_fail": round(f["p"], 3), "gain": round(gain, 1),
                      "factors": {k: round(v, 2) for k, v in f.items() if k != "p"}, "danger": danger, "low": low,
                      "played": "failed" if rng.random() < f["p"] else "ok", "minutes": round(state["seconds"] / 60, 1)})
    if stopped != "stranded":
        _walk(model, state, model["dock_floor"])  # the walk back to where it started
        if state["battery"] <= EMPTY:
            stopped = "stranded"
    end = max(0.0, state["battery"])
    if stopped == "stranded":
        score -= STRANDED_COST
    elif end < DANGER:
        score -= DANGER_END_COST
    checked = [s for s in steps if s["outcome"] == "checked"]
    return {"route": route, "turn_back_at": turn_back_at, "blocked": blocked, "seed": seed, "score": round(score, 1),
            "expected_ok": round(sum(1 - s["p_fail"] for s in checked), 2), "expected_failed": round(sum(s["p_fail"] for s in checked), 2),
            "checked": len(checked), "not_reached": sum(s["outcome"] == "not_reached" for s in steps), "stopped": stopped,
            "start_battery_pct": model["start_battery"], "end_battery_pct": round(end, 1), "minutes": round(state["seconds"] / 60),
            "steps": steps}


def expected_ok_by_step(result: dict) -> list[float]:
    return [round(1 - s["p_fail"], 4) if s["outcome"] == "checked" else 0.0 for s in result["steps"]]


def recommend(model: dict, blocked: str | None = None, pool: list[str] | None = None) -> dict:
    """A good plan by the same score: add the stop that helps most until none helps, then try swaps and drops, for each
    turn-back level. Deterministic; "good", not proven best."""
    ids = sorted(pool if pool is not None else model["assets"], key=lambda a: (-model["assets"][a]["value"], a))
    best: dict | None = None
    for level in TURN_BACK_CHOICES:
        route: list[str] = []
        score = simulate(model, route, level, blocked=blocked)["score"]
        while len(route) < min(MAX_STOPS, len(ids)):
            tries = [(simulate(model, route[:at] + [c] + route[at:], level, blocked=blocked)["score"], -at, c)
                     for c in ids if c not in route for at in range(len(route) + 1)]
            if not tries:
                break
            top = max(tries, key=lambda t: (t[0], t[1], t[2]))
            if top[0] <= score + 1e-9:
                break
            route.insert(-top[1], top[2])
            score = top[0]
        improved = True
        while improved:  # swaps and drops until nothing helps
            improved = False
            for i in range(len(route)):
                for j in range(i + 1, len(route)):
                    trial = route[:]
                    trial[i], trial[j] = trial[j], trial[i]
                    s = simulate(model, trial, level, blocked=blocked)["score"]
                    if s > score + 1e-9:
                        route, score, improved = trial, s, True
            for i in range(len(route)):
                trial = route[:i] + route[i + 1:]
                s = simulate(model, trial, level, blocked=blocked)["score"]
                if s > score + 1e-9:
                    route, score, improved = trial, s, True
                    break
        if best is None or score > best["score"] + 1e-9:
            best = simulate(model, route, level, blocked=blocked)
    return best or simulate(model, [], TURN_BACK_CHOICES[0], blocked=blocked)


def stars(score: float, best: float) -> int:
    if best <= 0:
        return 3 if score >= best else 0
    share = score / best
    return next((n for cut, n in STARS if share >= cut), 0)
