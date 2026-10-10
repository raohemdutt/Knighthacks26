"""What needs attention now: one ranked list with a score an operator can read and disagree with.

    score = severity (worst anomaly, weighted by how recent it is)
          + criticality of the asset
          + repeats (other anomalies on the same asset)
          + not inspected (the latest inspection attempt failed)
          + stale (days since the last good inspection)
          - seen since (a later inspection of the asset succeeded and raised nothing)

The weights below are judgement, not fitted to anything; every item carries its parts. Also here: the
"uninspected risk" list (important assets whose last good inspection is old) and the top three things to
investigate next. "Now" is the newest timestamp in the data.
"""

from collections import defaultdict
from datetime import datetime

from contexts.spot.domain import analysis
from contexts.spot.domain.analysis import CRIT_RANK, SEV_RANK, Evidence, _asset_label, _loc_label, _words
from contexts.spot.domain.models import Action, Anomaly, Kit
from contexts.spot.domain.stats import MIN_N, pct, wilson

SEVERITY_POINTS = {"CRITICAL": 40, "HIGH": 25, "MEDIUM": 10, "LOW": 3}
CRITICALITY_POINTS = {"CRITICAL": 25, "HIGH": 15, "MEDIUM": 6, "LOW": 0}
RECENCY = ((1, 1.0), (3, 0.75), (7, 0.5))  # (within this many days, weight); older counts 0.3
OLD_WEIGHT = 0.3
REPEAT_POINTS, REPEAT_CAP = 4, 12
NOT_INSPECTED_POINTS = 15
STALE_PER_DAY, STALE_CAP = 2, 10
SEEN_SINCE_CAP = 10  # taken off when a later inspection of the asset succeeded and raised nothing
ROUTE_ANOMALIES = {"NAVIGATION_OBSTRUCTION", "LOCALIZATION_ISSUE", "SIGNAL_DEGRADATION"}  # about the way there, not the asset
STALE_HOURS = 24  # three missions at the kit's one-every-eight-hours pace


def _days(now: datetime, then: datetime) -> float:
    return max(0.0, (now - then).total_seconds() / 86400)


def recency_weight(days: float) -> float:
    return next((w for limit, w in RECENCY if days <= limit), OLD_WEIGHT)


def _age(days: float) -> str:
    hours = days * 24
    if hours < 1:
        return "in the latest mission"
    return f"{hours:.0f} h ago" if hours < 48 else f"{days:.0f} days ago"


def inspection_state(kit: Kit, asset_id: str) -> dict:
    """When this asset was last inspected successfully, and what failed since."""
    now = kit.now
    acts = [a for a in kit.actions if a.asset_id == asset_id]
    inspections = [a for a in acts if a.is_inspection]
    good = [a for a in inspections if not a.failed]
    last_good = good[-1] if good else None
    since = [a for a in acts if a.failed and (last_good is None or a.timestamp > last_good.timestamp)]
    latest = inspections[-1] if inspections else None
    good_readings = [r for r in kit.readings if r.asset_id == asset_id and r.good and kit.action.get(r.action_id) and kit.action[r.action_id].is_inspection]
    hours = (now - last_good.timestamp).total_seconds() / 3600 if last_good and now else None
    return {"last_good_action": last_good.action_id if last_good else None,
            "last_good_at": last_good.timestamp.isoformat() if last_good else None,
            "last_good_mission": last_good.mission_id if last_good else None,
            "hours_since_good": hours,
            "last_good_reading": good_readings[-1].reading_id if good_readings else None,
            "last_good_reading_at": good_readings[-1].timestamp.isoformat() if good_readings else None,
            "failed_since": [a.action_id for a in since],
            "latest_attempt": latest.action_id if latest else None,
            "latest_attempt_failed": bool(latest and latest.failed),
            "latest_attempt_at": latest.timestamp.isoformat() if latest else None,
            "inspections": len(inspections), "inspections_ok": len(good)}


def _asset_item(kit: Kit, asset_id: str, anomalies: list[Anomaly]) -> dict | None:
    now = kit.now
    asset = kit.asset.get(asset_id)
    state = inspection_state(kit, asset_id)
    if not anomalies and not state["latest_attempt_failed"]:
        return None
    crit = asset.criticality if asset else "LOW"
    parts = []
    worst = None
    if anomalies:
        scored = sorted(((SEVERITY_POINTS.get(x.severity, 0) * recency_weight(_days(now, x.timestamp)), SEV_RANK.get(x.severity, 0), x.timestamp, x)
                         for x in anomalies), key=lambda t: (t[0], t[1], t[2]), reverse=True)
        points, _, _, worst = scored[0]
        days = _days(now, worst.timestamp)
        parts.append({"label": "severity", "points": round(points, 1),
                      "detail": f"{worst.anomaly_id} {_words(worst.anomaly_type)}, {worst.severity}, {_age(days)} "
                                f"({SEVERITY_POINTS.get(worst.severity, 0)} x {recency_weight(days):g} for recency)"})
    parts.append({"label": "criticality", "points": CRITICALITY_POINTS.get(crit, 0), "detail": f"asset criticality {crit}"})
    if len(anomalies) > 1:
        parts.append({"label": "repeats", "points": min(REPEAT_CAP, REPEAT_POINTS * (len(anomalies) - 1)),
                      "detail": f"{len(anomalies) - 1} other anomaly row(s) on this asset: "
                                + ", ".join(x.anomaly_id for x in anomalies if x is not worst)})
    if state["latest_attempt_failed"]:
        latest = kit.action[state["latest_attempt"]]
        parts.append({"label": "not inspected", "points": NOT_INSPECTED_POINTS,
                      "detail": f"the latest inspection attempt {latest.action_id} ({_words(latest.action_type)}, {latest.mission_id}) failed: "
                                f"{_words(latest.failure_type) or 'no failure type'}"})
    cleared = bool(worst and state["last_good_at"] and state["last_good_at"] > worst.timestamp.isoformat())
    if cleared:
        parts.append({"label": "seen since", "points": -round(min(SEEN_SINCE_CAP, parts[0]["points"] / 2), 1),
                      "detail": f"inspected successfully since then ({state['last_good_action']}, {state['last_good_mission']}) with no new anomaly"})
    if state["hours_since_good"] is None:
        parts.append({"label": "stale", "points": STALE_CAP, "detail": "no successful inspection of this asset anywhere in the data"})
    elif state["hours_since_good"] >= 12:
        days = state["hours_since_good"] / 24
        parts.append({"label": "stale", "points": round(min(STALE_CAP, STALE_PER_DAY * days), 1),
                      "detail": f"last good inspection {state['last_good_action']} was {_age(days)}"})
    score = round(sum(p["points"] for p in parts), 1)

    worst_action = kit.action.get(worst.action_id) if worst else None
    confirmed = bool(worst_action and not worst_action.failed)
    name = _asset_label(kit, asset_id)
    if worst and worst.anomaly_type in ROUTE_ANOMALIES:
        step = {"action": "check_path", "text": f"Check the path at {_loc_label(kit, worst.location_id)}: {worst.anomaly_id} is about getting "
                                                f"there ({_words(worst.anomaly_type)}), then re-inspect {asset_id}."}
    elif worst and confirmed and worst.severity in ("HIGH", "CRITICAL"):
        step = {"action": "dispatch", "text": f"Send a technician to {name}: {worst.anomaly_id} came from an inspection that succeeded, "
                                              "so the finding stands on a good reading."}
    elif worst and not confirmed:
        step = {"action": "reinspect", "text": f"Re-inspect {name} before sending anyone: {worst.anomaly_id} was raised by an action that itself "
                                               f"failed ({_words(worst_action.failure_type) if worst_action else 'unknown'}), so the reading "
                                               "behind it is questionable."}
    elif worst:
        step = {"action": "reinspect", "text": f"Re-inspect {name} on the next mission to see whether {worst.anomaly_id} "
                                               f"({_words(worst.anomaly_type)}) is still there."}
    else:
        step = {"action": "reinspect", "text": f"Re-inspect {name}: its latest inspection attempt failed, so it has not been looked at since "
                                               f"{state['last_good_at'] or 'the start of the data'}."}
    if worst:
        reason = (f"{worst.severity} {_words(worst.anomaly_type)} on a {crit} asset, {_age(_days(now, worst.timestamp))}"
                  + (", confirmed by a successful inspection" if confirmed else ", raised by a failed action (unverified)")
                  + (f"; {len(anomalies) - 1} more anomaly row(s) on it" if len(anomalies) > 1 else "")
                  + ("; latest inspection attempt failed" if state["latest_attempt_failed"] else "")
                  + ("; a later inspection succeeded and raised nothing" if cleared else "") + ".")
    else:
        reason = f"{crit} asset whose latest inspection attempt failed; no anomaly on record."
    related = [a.action_id for a in kit.actions if a.asset_id == asset_id and a.failed]
    return {"kind": "asset", "id": asset_id, "title": name, "score": score, "parts": parts, "reason": reason, "next_step": step,
            "asset_id": asset_id, "location_id": asset.location_id if asset else None, "criticality": crit,
            "severity": worst.severity if worst else None, "anomaly_id": worst.anomaly_id if worst else None, "confirmed": confirmed,
            "inspection": state,
            "evidence": Evidence(assets=[asset_id], anomalies=[x.anomaly_id for x in anomalies], actions=related,
                                 missions=sorted({x.mission_id for x in anomalies})).model_dump()}


def _waypoint_items(kit: Kit) -> list[dict]:
    """Waypoints that fail clearly more than the rest: a lead for 'check the waypoint or the path'."""
    acts = kit.actions
    if not acts:
        return []
    overall = sum(a.failed for a in acts) / len(acts)
    groups: dict[str, list[Action]] = defaultdict(list)
    for a in acts:
        groups[a.location_id].append(a)
    latest_version = analysis._groups(kit)[-1] if analysis._groups(kit) else None
    out = []
    for loc, g in groups.items():
        k, n = sum(a.failed for a in g), len(g)
        ci = wilson(k, n)
        if n < MIN_N or ci is None or ci[0] <= overall:
            continue
        recent = [a for a in g if a.configuration_version == latest_version]
        rk = sum(a.failed for a in recent)
        weight = 1.0 if rk else 0.5
        parts = [{"label": "failure rate", "points": round(60 * (k / n) * weight, 1),
                  "detail": f"{k} of {n} actions failed ({pct(k / n)}, 95% interval {pct(ci[0], 0)} to {pct(ci[1], 0)}); the rest of the site "
                            f"is at {pct(overall)}" + ("" if rk else f"; halved because none failed under {latest_version}")}]
        top = max({a.failure_type for a in g if a.failed}, key=lambda t: sum(1 for a in g if a.failure_type == t))
        recent_text = (f"{rk} of {len(recent)} failed under {latest_version}"
                       + (f" (under {MIN_N} actions: not enough to call it fixed)" if len(recent) < MIN_N else ""))
        out.append({"kind": "waypoint", "id": loc, "title": _loc_label(kit, loc), "score": round(sum(p["points"] for p in parts), 1),
                    "parts": parts, "reason": f"Fails more than the rest of the site: {k} of {n} actions ({pct(k / n)}); most common failure "
                                              f"{_words(top)}. {recent_text}.",
                    "next_step": {"action": "check_path", "text": f"Walk {_loc_label(kit, loc)}: check the path, lighting and noise there, "
                                                                  "and watch the next mission through it."},
                    "asset_id": None, "location_id": loc, "criticality": None, "severity": None, "anomaly_id": None, "confirmed": False,
                    "inspection": None,
                    "evidence": Evidence(locations=[loc], actions=[a.action_id for a in g if a.failed]).model_dump()})
    return out


def triage(kit: Kit) -> list[dict]:
    """Ranked: highest score first. Ties go to the more critical asset, then the id."""
    by_asset: dict[str, list[Anomaly]] = defaultdict(list)
    for x in kit.anomalies:
        by_asset[x.asset_id].append(x)
    items = [i for i in (_asset_item(kit, a.asset_id, by_asset.get(a.asset_id, [])) for a in kit.assets) if i]
    items += _waypoint_items(kit)
    items.sort(key=lambda i: (-i["score"], -CRIT_RANK.get(i["criticality"] or "", -1), i["id"]))
    for rank, item in enumerate(items, 1):
        item["rank"] = rank
    return items


def uninspected_risk(kit: Kit) -> list[dict]:
    """Assets by how long ago they were last inspected successfully, most critical first."""
    out = []
    for a in kit.assets:
        s = inspection_state(kit, a.asset_id)
        hours = s["hours_since_good"]
        important = a.criticality in ("HIGH", "CRITICAL")
        stale = hours is None or hours >= STALE_HOURS
        at_risk = important and (stale or bool(s["failed_since"]))
        if hours is None:
            text = f"{a.asset_id} ({a.criticality}) has no successful inspection anywhere in the data."
        else:
            text = (f"{a.asset_id} ({a.criticality}) was last inspected successfully {hours:.0f} h ago ({s['last_good_action']}, {s['last_good_mission']})"
                    + (f"; {len(s['failed_since'])} action(s) on it have failed since ({', '.join(s['failed_since'])})" if s["failed_since"] else "")
                    + ".")
        out.append({"asset_id": a.asset_id, "title": _asset_label(kit, a.asset_id), "criticality": a.criticality,
                    "location_id": a.location_id, "at_risk": at_risk, "stale": stale, "text": text, **s,
                    "evidence": Evidence(assets=[a.asset_id], actions=[x for x in [s["last_good_action"], *s["failed_since"]] if x]).model_dump()})
    out.sort(key=lambda r: (not r["at_risk"], -CRIT_RANK.get(r["criticality"], 0), -(r["hours_since_good"] if r["hours_since_good"] is not None else 1e9),
                            r["asset_id"]))
    return out


def investigate_next(kit: Kit, findings: dict[str, analysis.Finding] | None = None) -> list[dict]:
    """The top three, each one sentence with its evidence: an asset, a pattern, and a gap in coverage."""
    f = findings or {x.id: x for x in analysis.all_findings(kit)}
    out: list[dict] = []
    ranked = triage(kit)
    top = next((i for i in ranked if i["kind"] == "asset"), None)
    if top:
        verb = {"reinspect": "Re-inspect", "dispatch": "Send a technician to", "check_path": "Check the path to"}[top["next_step"]["action"]]
        what = (f"{top['severity']} anomaly, {'confirmed' if top['confirmed'] else 'unverified'}" if top["severity"] else "latest inspection failed")
        out.append({"key": "asset", "title": f"{verb} {top['id']} ({top['criticality']} asset): {what}", "sentence": f"{top['next_step']['text']} Why: {top['reason']}",
                    "target": {"kind": "asset", "id": top["id"]}, "finding_id": None, "score": top["score"], "evidence": top["evidence"]})
    # the strongest pattern: the clear comparison with the smallest p-value among the systemic findings
    systemic = [f[k] for k in ("battery", "sequence", "reliability_by_type", "reliability_by_location", "navigation_hotspot") if k in f]
    clear = sorted((x for x in systemic if x.enough and x.extra.get("clear") and x.p_value is not None), key=lambda x: x.p_value)
    if clear:
        p = clear[0]
        target = ({"kind": "location", "id": p.extra["worst"]} if p.id == "reliability_by_location"
                  else {"kind": "location", "id": p.extra["location_id"]} if p.id == "navigation_hotspot" else {"kind": "finding", "id": p.id})
        lead = {"battery": "Low battery goes with failures: watch the next mission that drops below 40%",
                "sequence": "The last steps of a mission fail most: watch them",
                "reliability_by_type": f"{_words(p.extra.get('worst', '')).capitalize()} is the least reliable inspection type",
                "reliability_by_location": f"Walk {p.extra.get('worst', '')}: it fails more than the rest of the site",
                "navigation_hotspot": f"Check the path to {p.extra.get('location_id', '')}"}[p.id]
        out.append({"key": "pattern", "title": lead, "sentence": p.sentence,
                    "target": target, "finding_id": p.id, "score": None, "evidence": p.evidence.model_dump()})
    gap = next((r for r in uninspected_risk(kit) if r["at_risk"] and (not top or r["asset_id"] != top["id"])), None)
    if gap:
        hours = gap["hours_since_good"]
        out.append({"key": "coverage", "title": f"Get eyes on {gap['asset_id']} ({gap['criticality']} asset): "
                                                + (f"no good inspection for {hours:.0f} h" if hours is not None else "never inspected successfully"), "sentence": gap["text"] + " Put it on the next mission.",
                    "target": {"kind": "asset", "id": gap["asset_id"]}, "finding_id": None, "score": None, "evidence": gap["evidence"]})
    for item in ranked:  # fewer than three so far: fill from the triage list
        if len(out) >= 3:
            break
        if all(o["target"]["id"] != item["id"] for o in out):
            out.append({"key": item["kind"], "title": item["title"], "sentence": f"{item['next_step']['text']} Why: {item['reason']}",
                        "target": {"kind": "location" if item["kind"] == "waypoint" else "asset", "id": item["id"]}, "finding_id": None,
                        "score": item["score"], "evidence": item["evidence"]})
    return out[:3]
