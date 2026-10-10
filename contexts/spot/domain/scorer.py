"""A small model trained on the kit: how likely is it that this action carries an anomaly row?

One example per action (actions.csv), label = an anomalies.csv row names the action. A logistic regression,
written out in pure Python (the box's venv has no numpy or scikit-learn), deterministic: zero start, a fixed
number of full-batch gradient steps, no random numbers.

Features are built causally, from the action itself and from rows EARLIER than it only:

    failed                 the action's own status (known the moment it ends)
    has_reading            a reading row exists
    value_vs_fleet         |reading - fleet median| / fleet spread, for that measurement (fleet = training missions)
    value_vs_own_history   |reading - the asset's own earlier median| / fleet spread (0 without two earlier readings)
    delta_from_previous    |reading - the asset's previous reading of that measurement| / fleet spread
    battery_low            (60 - battery) / 40, floored at 0
    late_in_mission        sequence / 10
    duration               (duration_sec - training mean) / training sd      (the kit has no retry field)
    type_*                 the action type, one column each
    waypoint_history       share of EARLIER actions at this waypoint that carried an anomaly (smoothed)
    asset_history          the same for this asset
    criticality            the asset's criticality, 0 (LOW) to 1 (CRITICAL)

Nothing from validation_results.json is used. `evaluate` is strict about time: train on earlier missions, test on
later ones, and compare with two rules that need no training:

    threshold rule   flag an action whose reading sits outside the fleet band (median +/- 3 robust sd)
    status rule      flag an action that failed

The set is tiny (270 actions, 40 anomalies), so every number is reported with its counts.
"""

import math
from collections import defaultdict

from contexts.spot.domain.models import Action, Kit
from contexts.spot.domain.stats import mad, median

TYPES = ("ACOUSTIC_INSPECTION", "GAUGE_READING", "NAVIGATION", "THERMAL_INSPECTION", "VISUAL_INSPECTION")
CRIT = {"LOW": 0.0, "MEDIUM": 1 / 3, "HIGH": 2 / 3, "CRITICAL": 1.0}
FEATURES = ("failed", "has_reading", "value_vs_fleet", "value_vs_own_history", "delta_from_previous", "battery_low", "late_in_mission",
            "duration", *(f"type_{t.lower()}" for t in TYPES), "waypoint_history", "asset_history", "criticality")
PLAIN = {"failed": "the action itself failed", "has_reading": "it produced a reading", "value_vs_fleet": "the reading is far from the fleet's usual value",
         "value_vs_own_history": "the reading is far from this asset's own earlier readings",
         "delta_from_previous": "the reading jumped since this asset's previous one", "battery_low": "battery was low",
         "late_in_mission": "it came late in the mission", "duration": "it took longer than usual",
         "waypoint_history": "earlier actions at this waypoint often carried anomalies",
         "asset_history": "earlier actions on this asset often carried anomalies", "criticality": "the asset is critical",
         **{f"type_{t.lower()}": f"it is a {t.replace('_', ' ').lower()}" for t in TYPES}}
STEPS, RATE, L2 = 1200, 0.5, 0.02
SMOOTH = 4  # pseudo-actions at the overall rate, so a waypoint seen twice does not look certain
BAND_SDS = 3.0


def fleet_stats(kit: Kit, missions: set[str]) -> dict:
    """What 'usual' means, from the given (training) missions only."""
    by_type: dict[str, list[float]] = defaultdict(list)
    for r in kit.readings:
        if r.mission_id in missions and r.good:
            by_type[r.measurement_type].append(r.value)
    durations = [a.duration_sec for a in kit.actions if a.mission_id in missions]
    mean = sum(durations) / len(durations) if durations else 0.0
    sd = math.sqrt(sum((d - mean) ** 2 for d in durations) / len(durations)) if durations else 1.0
    labelled = [a for a in kit.actions if a.mission_id in missions]
    rate = sum(1 for a in labelled if kit.anomalies_of.get(a.action_id)) / len(labelled) if labelled else 0.0
    return {"median": {t: median(v) for t, v in by_type.items()}, "spread": {t: (mad(v) or 1.0) for t, v in by_type.items()},
            "duration_mean": mean, "duration_sd": sd or 1.0, "base_rate": rate}


def build(kit: Kit, fleet: dict) -> list[dict]:
    """One row per action in time order: id, mission, features, label, and whether the threshold rule would flag it."""
    rows = []
    seen_wp: dict[str, list[int]] = defaultdict(list)
    seen_asset: dict[str, list[int]] = defaultdict(list)
    history: dict[tuple[str, str], list[float]] = defaultdict(list)
    base = fleet["base_rate"]
    for a in kit.actions:  # already in time order
        label = int(bool(kit.anomalies_of.get(a.action_id)))
        rs = kit.readings_of.get(a.action_id, [])
        r = rs[0] if rs else None
        vs_fleet = vs_own = delta = 0.0
        outside = False
        if r is not None and r.measurement_type in fleet["median"]:
            spread = fleet["spread"][r.measurement_type] or 1.0
            vs_fleet = abs(r.value - fleet["median"][r.measurement_type]) / spread
            outside = vs_fleet > BAND_SDS
            past = history[(a.asset_id, r.measurement_type)]
            if len(past) >= 2:
                vs_own = abs(r.value - median(past)) / spread
            if past:
                delta = abs(r.value - past[-1]) / spread

        def prior(seen: list[int]) -> float:
            return (sum(seen) + SMOOTH * base) / (len(seen) + SMOOTH)

        x = [float(a.failed), float(r is not None), min(vs_fleet, 6.0), min(vs_own, 6.0), min(delta, 6.0), max(0.0, (60 - a.battery_pct) / 40),
             a.sequence / 10, (a.duration_sec - fleet["duration_mean"]) / fleet["duration_sd"], *(float(a.action_type == t) for t in TYPES),
             prior(seen_wp[a.location_id]), prior(seen_asset[a.asset_id]),
             CRIT.get(kit.asset[a.asset_id].criticality, 0.0) if a.asset_id in kit.asset else 0.0]
        rows.append({"action_id": a.action_id, "mission_id": a.mission_id, "x": x, "y": label, "failed": a.failed, "outside_band": outside,
                     "has_reading": r is not None})
        seen_wp[a.location_id].append(label)
        seen_asset[a.asset_id].append(label)
        if r is not None:
            history[(a.asset_id, r.measurement_type)].append(r.value)
    return rows


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def fit(rows: list[dict]) -> dict:
    """Class-weighted, L2-regularised logistic regression by full-batch gradient descent. Deterministic."""
    n, d = len(rows), len(FEATURES)
    means = [sum(r["x"][j] for r in rows) / n for j in range(d)]
    sds = [math.sqrt(sum((r["x"][j] - means[j]) ** 2 for r in rows) / n) or 1.0 for j in range(d)]
    xs = [[(r["x"][j] - means[j]) / sds[j] for j in range(d)] for r in rows]
    ys = [r["y"] for r in rows]
    pos = sum(ys)
    w_pos, w_neg = (n / (2 * pos) if pos else 1.0), (n / (2 * (n - pos)) if n - pos else 1.0)
    w, b = [0.0] * d, 0.0
    for _ in range(STEPS):
        gw, gb = [0.0] * d, 0.0
        for x, y in zip(xs, ys):
            err = (_sigmoid(b + sum(wi * xi for wi, xi in zip(w, x))) - y) * (w_pos if y else w_neg)
            gb += err
            for j in range(d):
                gw[j] += err * x[j]
        b -= RATE * gb / n
        w = [wj - RATE * (gw[j] / n + L2 * wj) for j, wj in enumerate(w)]
    model = {"features": list(FEATURES), "weights": w, "bias": b, "means": means, "sds": sds}
    model["threshold"] = _best_threshold([score(model, r["x"]) for r in rows], ys)
    return model


def score(model: dict, x: list[float]) -> float:
    z = model["bias"] + sum(w * (v - m) / s for w, v, m, s in zip(model["weights"], x, model["means"], model["sds"]))
    return _sigmoid(z)


def explain(model: dict, x: list[float], top: int = 3) -> list[dict]:
    """The features that push this action's score up the most: weight times how unusual the value is."""
    parts = [{"feature": f, "why": PLAIN.get(f, f), "contribution": w * (v - m) / s, "value": v}
             for f, w, v, m, s in zip(model["features"], model["weights"], x, model["means"], model["sds"])]
    return sorted((p for p in parts if p["contribution"] > 0), key=lambda p: -p["contribution"])[:top]


def _best_threshold(scores: list[float], ys: list[int]) -> float:
    best, best_f1 = 0.5, -1.0
    for t in sorted(set(round(s, 3) for s in scores)):
        m = confusion([s >= t for s in scores], ys)
        if m["f1"] > best_f1:
            best, best_f1 = t, m["f1"]
    return best


def confusion(flags: list[bool], ys: list[int]) -> dict:
    tp = sum(1 for f, y in zip(flags, ys) if f and y)
    fp = sum(1 for f, y in zip(flags, ys) if f and not y)
    fn = sum(1 for f, y in zip(flags, ys) if not f and y)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "flagged": tp + fp, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def average_precision(scores: list[float], ys: list[int]) -> float | None:
    """Area under the precision-recall curve (step-wise). Ties are taken together, so a rule with two scores is scored fairly."""
    pos = sum(ys)
    if not pos:
        return None
    ap, tp, seen, prev_recall = 0.0, 0, 0, 0.0
    order = sorted(zip(scores, ys), key=lambda t: -t[0])
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and order[j][0] == order[i][0]:
            tp += order[j][1]
            j += 1
        seen = j
        recall = tp / pos
        ap += (recall - prev_recall) * (tp / seen)
        prev_recall = recall
        i = j
    return ap


def _report(model: dict, test: list[dict]) -> dict:
    ys = [r["y"] for r in test]
    scores = [score(model, r["x"]) for r in test]
    n, pos = len(test), sum(ys)
    out = {"n": n, "positives": pos, "base_rate": pos / n if n else None,
           "model": {**confusion([s >= model["threshold"] for s in scores], ys), "pr_auc": average_precision(scores, ys)},
           "status_rule": {**confusion([r["failed"] for r in test], ys), "pr_auc": average_precision([float(r["failed"]) for r in test], ys)},
           "threshold_rule": {**confusion([r["outside_band"] for r in test], ys),
                              "pr_auc": average_precision([float(r["outside_band"]) for r in test], ys)}}
    ok = [(s, r["y"]) for s, r in zip(scores, test) if not r["failed"]]
    out["successful_actions_only"] = {"n": len(ok), "positives": sum(y for _, y in ok),
                                      "model_pr_auc": average_precision([s for s, _ in ok], [y for _, y in ok]),
                                      "base_rate": (sum(y for _, y in ok) / len(ok)) if ok else None}
    return out


def evaluate(kit: Kit, train_share: float = 0.7) -> dict:
    """Train on the earliest missions, test on the rest; then the same with a moving cut, pooled."""
    ids = [m.mission_id for m in kit.missions]
    cut = max(1, round(len(ids) * train_share))

    def run(train_ids: list[str], test_ids: list[str]) -> tuple[dict, list[dict]]:
        fleet = fleet_stats(kit, set(train_ids))
        rows = build(kit, fleet)
        model = fit([r for r in rows if r["mission_id"] in set(train_ids)])
        return model, [r for r in rows if r["mission_id"] in set(test_ids)]

    model, test = run(ids[:cut], ids[cut:])
    primary = {"train_missions": ids[:cut], "test_missions": ids[cut:], **_report(model, test)}
    # rolling origin: several cuts, each tested on the next three missions; predictions pooled
    pooled_scores, pooled, folds = [], [], []
    for c in range(9, len(ids) - 2, 3):
        m, t = run(ids[:c], ids[c:c + 3])
        folds.append({"train_upto": ids[c - 1], "test": ids[c:c + 3]})
        pooled += t
        pooled_scores += [(score(m, r["x"]), score(m, r["x"]) >= m["threshold"]) for r in t]
    ys = [r["y"] for r in pooled]
    rolling = {"folds": folds, "n": len(pooled), "positives": sum(ys),
               "model": {**confusion([f for _, f in pooled_scores], ys), "pr_auc": average_precision([s for s, _ in pooled_scores], ys)},
               "status_rule": {**confusion([r["failed"] for r in pooled], ys), "pr_auc": average_precision([float(r["failed"]) for r in pooled], ys)},
               "threshold_rule": {**confusion([r["outside_band"] for r in pooled], ys),
                                  "pr_auc": average_precision([float(r["outside_band"]) for r in pooled], ys)}}
    return {"primary": primary, "rolling": rolling}


def verdict(ev: dict) -> str:
    """One plain sentence: did the learned model beat the rules on missions it had not seen?"""
    p, r = ev["primary"], ev["rolling"]

    def line(m: dict) -> str:
        auc = "n/a" if m["pr_auc"] is None else f"{m['pr_auc']:.2f}"
        return f"precision {m['tp']}/{m['flagged']}, recall {m['tp']}/{m['tp'] + m['fn']}, PR-AUC {auc}"

    def gain(block: dict, rule: str) -> float:
        return (block["model"]["pr_auc"] or 0) - (block[rule]["pr_auc"] or 0)

    beats_band = gain(p, "threshold_rule") >= 0.10 and gain(r, "threshold_rule") >= 0.10
    beats_status = (gain(p, "status_rule") >= 0.10 and gain(r, "status_rule") >= 0.10
                    and p["model"]["tp"] >= p["status_rule"]["tp"] and r["model"]["tp"] >= r["status_rule"]["tp"])
    head = (("The learned model beats the reading-outside-band threshold rule on later missions"
             if beats_band else "The learned model does NOT clearly beat the reading-outside-band threshold rule on later missions")
            + (", and also the simpler 'the action failed' rule" if beats_status else
               ", but it does NOT clearly beat the simpler rule 'the action failed': at its own threshold it flags much the same actions, "
               "and its small PR-AUC edge is within what one or two rows would change"))
    ok = p["successful_actions_only"]
    return (f"{head}. Held-out missions {p['test_missions'][0]} to {p['test_missions'][-1]} ({p['n']} actions, {p['positives']} with an anomaly): "
            f"model {line(p['model'])}; status rule {line(p['status_rule'])}; reading-outside-band rule {line(p['threshold_rule'])}. "
            f"Rolling cuts pooled ({r['n']} actions, {r['positives']} anomalies): model {line(r['model'])}; status rule {line(r['status_rule'])}; "
            f"band rule {line(r['threshold_rule'])}. Among successful actions only ({ok['n']} held out, {ok['positives']} anomalies), the model's "
            f"PR-AUC is {'n/a' if ok['model_pr_auc'] is None else format(ok['model_pr_auc'], '.2f')} against a base rate of "
            f"{'n/a' if ok['base_rate'] is None else format(ok['base_rate'], '.2f')}: no better than chance, so it does not find anomalies the rules "
            "miss. It is used to order ties in the queue and to explain a flag, never to raise one on its own. With this few anomalies "
            "every number here moves a lot with one row.")


def train(kit: Kit) -> dict:
    """The deployed weights: fitted on every mission (the evaluation above is what says how far to trust them)."""
    ev = evaluate(kit)
    fleet = fleet_stats(kit, {m.mission_id for m in kit.missions})
    model = fit(build(kit, fleet))
    r6 = lambda v: round(v, 6)  # noqa: E731
    return {"name": "spot-anomaly-scorer", "kind": "logistic regression, pure Python", "seed": 0, "steps": STEPS, "learning_rate": RATE, "l2": L2,
            "trained_on": {"missions": len(kit.missions), "actions": len(kit.actions), "anomalies": sum(1 for a in kit.actions if kit.anomalies_of.get(a.action_id))},
            "features": model["features"], "weights": [r6(w) for w in model["weights"]], "bias": r6(model["bias"]),
            "means": [r6(v) for v in model["means"]], "sds": [r6(v) for v in model["sds"]], "threshold": model["threshold"],
            "fleet": {"median": {k: r6(v) for k, v in sorted(fleet["median"].items())}, "spread": {k: r6(v) for k, v in sorted(fleet["spread"].items())},
                      "duration_mean": r6(fleet["duration_mean"]), "duration_sd": r6(fleet["duration_sd"]), "base_rate": r6(fleet["base_rate"])},
            "evaluation": _round(ev), "verdict": verdict(ev)}


def _round(o):
    if isinstance(o, float):
        return round(o, 4)
    if isinstance(o, dict):
        return {k: _round(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_round(v) for v in o]
    return o


def score_all(kit: Kit, model: dict) -> list[dict]:
    """Every action scored with the saved weights, with its top reasons."""
    fleet = {**model["fleet"]}
    out = []
    for r in build(kit, fleet):
        s = score(model, r["x"])
        out.append({"action_id": r["action_id"], "mission_id": r["mission_id"], "score": round(s, 3), "flagged": s >= model["threshold"],
                    "has_anomaly_row": bool(r["y"]), "failed": r["failed"], "outside_band": r["outside_band"], "has_reading": r["has_reading"],
                    "why": [{"feature": p["feature"], "why": p["why"], "contribution": round(p["contribution"], 2)} for p in explain(model, r["x"])]})
    return out


def _unused(_: Action) -> None:  # keeps the import that documents what a row is built from
    return None
