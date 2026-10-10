"""The operator's words: triage items, findings and asset states as short sentences a person on shift can act on.

    view = shift(spot)                 # GET /api/spot/shift: status line, three cards, every asset for the floor plan
    view["do_first"][0]["sentence"]    # "SPOT flagged strong vibration in the last round, but that check failed ..."
    round_view(spot, "M027")           # GET /api/spot/shift/rounds/M027: one round as steps, each a sentence

Every item is {id, headline, sentence, level, why_ref, action}: `level` is act | watch | ok, `why_ref` names the
evidence panel that backs the sentence (an asset, a waypoint, a finding or a mission), `action` is the button to offer
(reinspect | work_order) or None. Nothing here analyses anything: the numbers come from `Spot` (triage, findings,
inspection state) and are only reworded, with a glossary for the kit's enums. No model is called; the same data
gives the same words.

Words used: a "round" is a mission, a "check" is an action, an "alarm" is an anomaly row. An alarm raised by a
check that itself failed is "not confirmed"; one raised by a check that worked "stands".
"""

from datetime import datetime

from contexts.spot.application.service import Spot, not_configured
from contexts.spot.domain import triage as triage_mod
from contexts.spot.domain.analysis import CRIT_RANK, Finding
from contexts.spot.domain.models import Action, Anomaly, Kit
from contexts.spot.domain.triage import RECENCY, ROUTE_ANOMALIES, STALE_HOURS

LEVEL_WORDS = {"act": "Needs you", "watch": "Keep an eye", "ok": "Looks fine"}
LEVEL_RULE = {
    "act": "An alarm that stands on a good check, a serious alarm nobody has re-checked, or important equipment with no "
           f"good check for {STALE_HOURS} hours.",
    "watch": f"An alarm still waiting for a clean re-check, a serious alarm in the last {RECENCY[1][0]} days, a failed last "
             f"check, or no good check for {STALE_HOURS} hours.",
    "ok": "The last check worked and nothing above applies.",
}
WATCH_DAYS = RECENCY[1][0]  # "recent" as the triage score counts it (its second recency step)

# --- glossary: the kit's enums in everyday words (anything not listed falls back to lower case with spaces) ---

CHECKS = {  # action type -> (one, many, icon key)
    "THERMAL_INSPECTION": ("heat check", "heat checks", "heat"),
    "ACOUSTIC_INSPECTION": ("listening check", "listening checks", "sound"),
    "VISUAL_INSPECTION": ("camera check", "camera checks", "camera"),
    "GAUGE_READING": ("gauge reading", "gauge readings", "gauge"),
    "NAVIGATION": ("walk", "walks", "walk"),
}
CHECK_ALSO = {"THERMAL_INSPECTION": "thermal", "ACOUSTIC_INSPECTION": "acoustic", "VISUAL_INSPECTION": "visual"}
FAILURES = {
    "READING_UNRELIABLE": "the reading was not reliable",
    "SENSOR_TIMEOUT": "the sensor did not answer in time",
    "TARGET_NOT_DETECTED": "SPOT could not find the equipment",
    "LOCALIZATION_FAILURE": "SPOT lost track of where it was",
    "COMMUNICATION_LOSS": "SPOT lost its connection",
    "SIGNAL_NOISE": "there was too much noise",
    "OBSTACLE_DETECTION": "something was in the way",
    "PATH_BLOCKED": "the path was blocked",
}
ALARMS = {
    "ABNORMAL_VIBRATION": "strong vibration",
    "ABNORMAL_TEMPERATURE": "an unusual temperature",
    "VISUAL_DEFECT": "visible damage",
    "GAUGE_OUT_OF_RANGE": "a gauge outside its range",
    "SIGNAL_DEGRADATION": "a weak signal",
    "NAVIGATION_OBSTRUCTION": "a blocked route",
    "LOCALIZATION_ISSUE": "trouble knowing where it was",
}
SEVERITIES = {"CRITICAL": "most serious", "HIGH": "serious", "MEDIUM": "moderate", "LOW": "minor"}
IMPORTANCE = {"CRITICAL": "critical equipment", "HIGH": "important equipment", "MEDIUM": "standard equipment", "LOW": "minor equipment"}
ASSET_KINDS = {
    "Heat Exchanger": "passes heat from one fluid to another",
    "Pump": "moves fluid through the pipes",
    "Compressor": "squeezes gas to a higher pressure",
    "Electrical Panel": "holds switches and breakers",
    "Pressure Gauge": "shows the pressure in a line",
    "Motor": "drives a machine",
    "Valve": "opens and closes a pipe",
}
MEASURES = {  # measurement type -> (name, unit as written after the number)
    "TEMPERATURE_C": ("Temperature", " °C"),
    "SOUND_DB": ("Sound level", " dB"),
    "PRESSURE_PSI": ("Pressure", " psi"),
    "VISUAL_SCORE": ("Picture score", " out of 100"),
    "LOCALIZATION_CONFIDENCE_PCT": ("SPOT's confidence in where it was", "%"),
}
ENVIRONMENTS = {"HIGH_NOISE": "noisy areas", "LOW_LIGHT": "dim areas", "INDOOR_DUSTY": "dusty rooms", "INDOOR_DRY": "dry rooms",
                "NORMAL": "ordinary areas"}
GLOSSARY = [
    {"term": "round", "means": "One trip SPOT makes around the plant (the data calls it a mission)."},
    {"term": "check", "means": "One stop on a round: a photo, a heat picture, a listen or a gauge reading (the data calls it an action)."},
    {"term": "alarm", "means": "SPOT flagged something unusual about a piece of equipment (the data calls it an anomaly)."},
    {"term": "not confirmed", "means": "The check that raised the alarm failed, so the alarm may be wrong. Send SPOT back before a person."},
    {"term": "usual range", "means": "What SPOT normally reads across this site. The kit gives no plant limits, so this band is ours, not a safety limit."},
]
ASK = ["What should I check first?", "Is anything getting worse?", "What happened at {place}?", "Can I trust the last round?"]
PICKED = ("battery", "reliability_by_type", "reliability_by_location", "sequence", "reliability_by_floor", "navigation_hotspot")  # "Watch out" order
BUTTON = {"reinspect": "Send SPOT back", "work_order": "Send a technician"}
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def soft(enum: str) -> str:
    """An enum nobody listed, still not shouted: READING_UNRELIABLE -> reading unreliable."""
    return (enum or "").replace("_", " ").lower()


def check_word(action_type: str, many: bool = False) -> str:
    return CHECKS[action_type][1 if many else 0] if action_type in CHECKS else soft(action_type) + (" checks" if many else " check")


def failure_word(failure_type: str) -> str:
    return FAILURES.get(failure_type) or (soft(failure_type) if failure_type else "no reason was recorded")


def alarm_word(anomaly_type: str) -> str:
    return ALARMS.get(anomaly_type) or f"a problem ({soft(anomaly_type)})"


# --- names ---


def tidy(name: str) -> str:
    """'Heat Exchanger 02' -> 'Heat exchanger 2'."""
    out = []
    for i, w in enumerate(name.split()):
        out.append(str(int(w)) if w.isdigit() else w if i == 0 else w.lower())
    return " ".join(out)


def article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def asset_name(kit: Kit, asset_id: str) -> str:
    a = kit.asset.get(asset_id)
    return tidy(a.asset_name) if a and a.asset_name else (a.asset_type if a and a.asset_type else "Equipment")


def asset_label(kit: Kit, asset_id: str) -> str:
    """A name first, the tag in brackets: never the tag alone."""
    return f"{asset_name(kit, asset_id)} ({asset_id})"


def place_name(kit: Kit, location_id: str) -> str:
    loc = kit.location.get(location_id)
    return loc.location_name if loc and loc.location_name else "Stop"


def place_label(kit: Kit, location_id: str) -> str:
    return f"{place_name(kit, location_id)} ({location_id})"


# --- time, as people say it ---


def clock(t: datetime) -> str:
    return f"{t.hour % 12 or 12}:{t.minute:02d} {'AM' if t.hour < 12 else 'PM'}"


def day(now: datetime, t: datetime) -> str:
    gap = (now.date() - t.date()).days
    return "today" if gap == 0 else "yesterday" if gap == 1 else f"{DAYS[t.weekday()]} {t.day} {MONTHS[t.month - 1]}"


def when(now: datetime, t: datetime) -> str:
    return f"{day(now, t)} {clock(t)}"


def span(hours: float) -> str:
    """A length of time: 40 minutes, 5 hours, 2 days."""
    if hours < 1:
        n = max(1, round(hours * 60))
        return f"{n} minute{'s' if n != 1 else ''}"
    if hours < 48:
        n = round(hours)
        return f"{n} hour{'s' if n != 1 else ''}"
    return f"{round(hours / 24)} days"


def ago(now: datetime, t: datetime) -> str:
    return f"{span(max(0.0, (now - t).total_seconds() / 3600))} ago"


def one_in(k: int, n: int) -> str:
    """13 of 43 -> 'about 1 in 3'."""
    if k <= 0:
        return "none"
    if 2 * k > n:
        return "more than half"
    return f"about 1 in {round(n / k)}"


def times(a: tuple[int, int], b: tuple[int, int]) -> str | None:
    """How many times as often a fails as b: (25, 71) against (24, 199) -> 'about 3 times'."""
    if not (a[1] and b[1] and b[0] and a[0]):
        return None
    ratio = (a[0] / a[1]) / (b[0] / b[1])
    if ratio < 1.5:
        return None
    return "about twice" if ratio < 2.5 else f"about {round(ratio)} times"


def bars(a: tuple[str, int, int], b: tuple[str, int, int]) -> dict:
    """The tiny picture next to a pattern: two shares side by side."""
    return {"kind": "compare", "unit": "checks failed", "bars": [{"label": x[0], "k": x[1], "n": x[2]} for x in (a, b)]}


def item(id: str, headline: str, sentence: str, level: str, why: tuple[str, str], *, action: dict | None = None, **more) -> dict:
    return {"id": id, "headline": headline, "sentence": sentence, "level": level, "level_words": LEVEL_WORDS[level],
            "why_ref": {"kind": why[0], "id": why[1]}, "action": action, **more}


# --- one asset: what SPOT last saw, in words ---


def _alarm_sentence(kit: Kit, x: Anomaly, cleared: bool) -> str:
    now = kit.now
    act = kit.action.get(x.action_id)
    confirmed = bool(act and not act.failed)
    at = "in the last round" if kit.missions and x.mission_id == kit.missions[-1].mission_id else ago(now, x.timestamp)
    if x.anomaly_type in ROUTE_ANOMALIES:
        text = f"SPOT reported {alarm_word(x.anomaly_type)} on its way here {at}"
        text += ", and that step worked, so it stands." if confirmed else ", but that step failed, so it is not confirmed."
    elif confirmed:
        text = f"SPOT flagged {alarm_word(x.anomaly_type)} {at}, and that check worked, so the alarm stands."
    else:
        why = f" ({failure_word(act.failure_type)})" if act and act.failure_type else ""
        text = f"SPOT flagged {alarm_word(x.anomaly_type)} {at}, but that check failed{why}, so the alarm is not confirmed."
    return text + (" A later check worked and found nothing new." if cleared else "")


def _coverage_sentence(kit: Kit, state: dict) -> str:
    """How long this asset has gone without a check that worked."""
    now = kit.now
    good = datetime.fromisoformat(state["last_good_at"]) if state["last_good_at"] else None
    if good is None:
        return "SPOT has never had a check here that worked."
    if state["latest_attempt_failed"]:
        latest = kit.action[state["latest_attempt"]]
        return (f"SPOT's last check here failed ({failure_word(latest.failure_type)}). "
                f"The last one that worked was {ago(now, good)}.")
    return f"No check here has worked for {span((now - good).total_seconds() / 3600)}."


def _reading(spot: Spot, kit: Kit, state: dict) -> dict:
    """The last good reading in words, with its number and unit, against the site's usual band (ours, not a plant limit)."""
    now = kit.now
    rid = state.get("last_good_reading")
    r = next((x for x in kit.readings if x.reading_id == rid), None) if rid else None
    if r is None:
        good = datetime.fromisoformat(state["last_good_at"]) if state["last_good_at"] else None
        text = f"No good reading in {span((now - good).total_seconds() / 3600)}." if good else "No good reading on record."
        return {"sentence": text, "value": None, "unit": None, "measure": None, "in_range": None, "at": None, "range": None}
    name, unit = MEASURES.get(r.measurement_type, (soft(r.measurement_type).capitalize(), f" {r.unit}" if r.unit else ""))
    band = (spot.findings["reading_trends"].extra.get("bands") or {}).get(r.measurement_type) if "reading_trends" in spot.findings else None
    top = 100.0 if r.unit in ("pct", "score") else None
    lo, hi = (max(0.0, band["lo"]), min(band["hi"], top) if top else band["hi"]) if band else (None, None)
    inside = None if band is None else bool(lo <= r.value <= hi)
    verdict = ("" if inside is None else ", in its usual range" if inside else ", outside its usual range")
    stale = (now - r.timestamp).total_seconds() / 3600
    text = f"{name} {r.value:g}{unit}{verdict}." + (
        f" That reading is from {ago(now, r.timestamp)}; SPOT has not brought back a newer good one." if stale >= STALE_HOURS else "")
    return {"sentence": text, "value": r.value, "unit": unit.strip(), "measure": name, "in_range": inside, "at": r.timestamp.isoformat(),
            "range": None if band is None else {"lo": round(lo, 1), "hi": round(hi, 1),
                                                "words": f"Usual range here: {lo:.0f} to {hi:.0f}{unit}. It is our own band from this data; "
                                                         "the kit gives no plant limits."}}


def _last_check(kit: Kit, state: dict) -> dict:
    now = kit.now
    latest = kit.action.get(state["latest_attempt"]) if state["latest_attempt"] else None
    if latest is None:
        return {"sentence": "SPOT has not checked this yet.", "at": None, "worked": None, "check": None}
    what = check_word(latest.action_type)
    if not latest.failed:
        text = f"SPOT checked it {ago(now, latest.timestamp)} ({what}, {when(now, latest.timestamp)}) and the check worked."
    else:
        good = datetime.fromisoformat(state["last_good_at"]) if state["last_good_at"] else None
        text = (f"SPOT tried {ago(now, latest.timestamp)} ({what}, {when(now, latest.timestamp)}) but the check failed: "
                f"{failure_word(latest.failure_type)}. " + (f"The last check that worked was {ago(now, good)}." if good else "No check here has worked yet."))
    return {"sentence": text, "at": latest.timestamp.isoformat(), "worked": not latest.failed, "check": CHECKS.get(latest.action_type, ("", "", "camera"))[2]}


def asset_state(spot: Spot, asset_id: str) -> dict:
    """One asset for the floor plan and its panel: a level, a sentence, the last check, the last reading, the alarm, the button."""
    kit = spot.kit
    now = kit.now
    a = kit.asset[asset_id]
    it = next((i for i in spot.triage if i["kind"] == "asset" and i["id"] == asset_id), None)
    rk = next((r for r in spot.uninspected if r["asset_id"] == asset_id), None)
    state = triage_mod.inspection_state(kit, asset_id)
    alarms = [x for x in kit.anomalies if x.asset_id == asset_id]
    worst = kit.anomaly.get(it["anomaly_id"]) if it and it["anomaly_id"] else None
    confirmed = bool(it and it["confirmed"])
    cleared = bool(worst and state["last_good_at"] and state["last_good_at"] > worst.timestamp.isoformat())
    serious = bool(worst and worst.severity in ("HIGH", "CRITICAL"))
    days = (now - worst.timestamp).total_seconds() / 86400 if worst else None
    step = it["next_step"]["action"] if it else None
    stale = bool(rk and rk["stale"])
    at_risk = bool(rk and rk["at_risk"])
    important = a.criticality in ("HIGH", "CRITICAL")
    label = asset_label(kit, asset_id)
    home = place_label(kit, a.location_id)

    reasons: list[str] = []
    if step == "dispatch":
        reasons.append("an alarm stands on a check that worked")
    if worst and not cleared and serious:
        reasons.append("a serious alarm has not been re-checked")
    if at_risk:
        reasons.append(f"{IMPORTANCE[a.criticality]} without a good check" + (f" for {span(state['hours_since_good'])}" if state["hours_since_good"] is not None else ""))
    level = "act" if reasons else "ok"
    if level == "ok":
        if worst and not cleared:
            reasons.append("an alarm is waiting for a clean re-check")
        if serious and days is not None and days <= WATCH_DAYS:
            reasons.append(f"a serious alarm in the last {WATCH_DAYS} days")
        if state["latest_attempt_failed"]:
            reasons.append("the last check failed")
        if stale and not state["latest_attempt_failed"]:
            reasons.append(f"no good check for {STALE_HOURS} hours")
        level = "watch" if reasons else "ok"

    alarm_text = _alarm_sentence(kit, worst, cleared) if worst else None
    lead_with_alarm = bool(worst) and (step == "dispatch" or not cleared)
    kind = "work_order" if step == "dispatch" else "reinspect"
    if level == "act":
        weight = f" This is {IMPORTANCE[a.criticality]}." if important else ""
        if step == "dispatch":
            headline, sentence = f"Send a technician to {label}", f"{alarm_text}{weight} Send a technician to look."
        elif lead_with_alarm and worst.anomaly_type in ROUTE_ANOMALIES:
            headline = f"Walk the route to {label}"
            sentence = f"{alarm_text}{weight} Walk the route at {place_label(kit, worst.location_id)} first, then send SPOT back."
        elif lead_with_alarm:
            headline, sentence = f"Send SPOT back to {label}", f"{alarm_text}{weight} Send SPOT back before sending a person."
        else:
            older = (f" An older alarm ({alarm_word(worst.anomaly_type)}, {ago(now, worst.timestamp)}) was never confirmed."
                     if worst and not confirmed else "")
            headline, sentence = f"Send SPOT back to {label}", f"{_coverage_sentence(kit, state)}{weight}{older} Put it on SPOT's next round."
    elif level == "watch":
        headline = f"Keep an eye on {label}"
        sentence = (alarm_text if alarm_text else _coverage_sentence(kit, state)) + " Nothing to do yet: look again after the next round."
    else:
        good = datetime.fromisoformat(state["last_good_at"]) if state["last_good_at"] else None
        headline = f"{label} looks fine"
        sentence = (f"Checked {ago(now, good)} and the check worked." if good else "No check has failed here.") + (
            " An earlier alarm was followed by a check that found nothing new." if alarms else " No alarm on record.")

    latest = kit.action.get(state["latest_attempt"]) if state["latest_attempt"] else None
    sample = spot.sample_image(latest.action_type) if latest else None
    alarm = None
    if worst:
        alarm = {"sentence": alarm_text, "what": alarm_word(worst.anomaly_type), "severity": SEVERITIES.get(worst.severity, soft(worst.severity)),
                 "confirmed": confirmed, "cleared": cleared, "at": worst.timestamp.isoformat(), "count": len(alarms),
                 "count_words": "" if len(alarms) < 2 else f"SPOT has flagged this equipment {len(alarms)} times in all."}
    kind_words = ASSET_KINDS.get(a.asset_type)
    return item(asset_id, headline, sentence, level, ("asset", asset_id),
                action={"kind": kind, "label": BUTTON[kind], "asset_id": asset_id, "anomaly_id": worst.anomaly_id if kind == "work_order" and worst else None},
                asset_id=asset_id, name=asset_name(kit, asset_id), label=label, location_id=a.location_id, place=home,
                floor=kit.location[a.location_id].floor if a.location_id in kit.location else None,
                what=(f"{article(a.asset_type).capitalize()} {a.asset_type.lower()} at {place_name(kit, a.location_id)}" if a.asset_type else f"At {place_name(kit, a.location_id)}")
                     + (f": it {kind_words}." if kind_words else ".") + f" {IMPORTANCE.get(a.criticality, 'equipment').capitalize()}.",
                importance=IMPORTANCE.get(a.criticality, soft(a.criticality)), reasons=reasons,
                rank=it["rank"] if it else None, last_check=_last_check(kit, state), reading=_reading(spot, kit, state),
                alarm=alarm or {"sentence": "No alarm on record.", "what": None, "severity": None, "confirmed": None, "cleared": None, "at": None,
                                "count": 0, "count_words": ""},
                sample={"url": sample["url"], "label": f"Sample picture from the kit of a {check_word(latest.action_type)}. It is not a picture of this equipment."} if sample else None)


def triage_item(spot: Spot, it: dict) -> dict:
    """Any row of the triage queue as a sentence: an asset (see asset_state) or a waypoint SPOT struggles at."""
    kit = spot.kit
    if it["kind"] == "asset":
        return asset_state(spot, it["id"])
    loc = it["id"]
    acts = [a for a in kit.actions if a.location_id == loc]
    k, n = sum(a.failed for a in acts), len(acts)
    kinds = [a.failure_type for a in acts if a.failed]
    top = max(set(kinds), key=kinds.count) if kinds else ""
    label = place_label(kit, loc)
    return item(loc, f"Walk the route at {label}",
                f"SPOT's checks fail here more than anywhere else: {k} of {n}. Most often {failure_word(top)}. "
                "Check the path, the lighting and the noise there, and watch SPOT's next round through it.",
                "watch", ("location", loc), rank=it["rank"], location_id=loc, label=label, facts={"k": k, "n": n})


# --- findings as sentences ---


def _row(f: Finding, key: str):
    return next((r for r in f.rows if r.key == key), None)


def _rest(f: Finding, key: str) -> tuple[int, int]:
    others = [r for r in f.rows if r.key != key]
    return sum(r.k for r in others), sum(r.n for r in others)


def _hedge(clear: bool) -> str:
    return "" if clear else " The gap is small enough to be chance, so treat it as a hint."


def _hidden(spot: Spot, f: Finding) -> dict:
    k, n = len(f.rows), f.n
    return {"headline": "A 'completed' round can still hide failed checks",
            "sentence": f"{k} of {n} completed rounds still had a failed check inside. We show you the checks, not just the round.",
            "level": "watch", "facts": {"k": k, "n": n}}


def _version(spot: Spot, f: Finding) -> dict:
    head = next((m for m in f.extra.get("metrics", []) if m["key"] == "action_success"), None)
    if not f.enough or head is None:
        return {"headline": "SPOT's setup was updated", "sentence": "There are too few rounds on each setup to compare them.", "level": "ok", "facts": {}}
    a, b = head["a"], head["b"]
    better = b["value"] > a["value"]
    return {"headline": "More checks work since SPOT's setup was updated" if better else "SPOT's setup was updated",
            "sentence": (f"Since SPOT's setup was updated, {b['k']} of {b['n']} checks worked; before, {a['k']} of {a['n']}. "
                         "The newer rounds also ended with more battery left, so we cannot say the update is the reason."),
            "level": "ok", "facts": {"k_new": b["k"], "n_new": b["n"], "k_old": a["k"], "n_old": a["n"]},
            "visual": {"kind": "compare", "unit": "checks worked", "bars": [{"label": "Newer setup", "k": b["k"], "n": b["n"]},
                                                                             {"label": "Older setup", "k": a["k"], "n": a["n"]}]}}


def _battery(spot: Spot, f: Finding) -> dict:
    cut = f.extra.get("threshold", 40)
    low = [r for r in f.rows if int(r.key.split("-")[1]) <= cut]
    high = [r for r in f.rows if r not in low]
    lo, hi = (sum(r.k for r in low), sum(r.n for r in low)), (sum(r.k for r in high), sum(r.n for r in high))
    more = times(lo, hi)
    clear = bool(f.extra.get("clear")) and more is not None
    lead = (f"SPOT's checks fail {more} as often when its battery is under {cut}%" if clear
            else f"SPOT's checks fail a little more often when its battery is under {cut}%")
    return {"headline": "Low battery, more failed checks",
            "sentence": f"{lead}: {lo[0]} of {lo[1]} failed, against {hi[0]} of {hi[1]} above. " + ("Bring it back to charge earlier." if clear else "Treat it as a hint."),
            "note": "A pattern, not proof: low battery also means late in the round.",
            "level": "watch" if clear else "ok", "facts": {"k": lo[0], "n": lo[1], "k_rest": hi[0], "n_rest": hi[1], "threshold": cut},
            "visual": bars((f"Battery under {cut}%", *lo), (f"{cut}% or more", *hi))}


def _by_type(spot: Spot, f: Finding) -> dict:
    worst, best = _row(f, f.extra.get("worst", "")), _row(f, f.extra.get("best", ""))
    if worst is None:
        return {"headline": "Checks by kind", "sentence": "There are too few checks of each kind to compare them.", "level": "ok", "facts": {}}
    rest = _rest(f, worst.key)
    many = check_word(worst.key, many=True)
    also = f" ({CHECK_ALSO[worst.key]})" if worst.key in CHECK_ALSO else ""
    clear = bool(f.extra.get("clear"))
    best_text = f" {check_word(best.key, many=True).capitalize()} are the most dependable." if best and best.key != worst.key else ""
    return {"headline": f"{many.capitalize()} fail most",
            "sentence": f"{many.capitalize()}{also} miss {one_in(worst.k, worst.n)}: {worst.k} of {worst.n} failed.{best_text}" + _hedge(clear)
                        + (" Do not rely on one alone." if clear else ""),
            "level": "watch" if clear else "ok", "facts": {"k": worst.k, "n": worst.n},
            "visual": bars((many.capitalize(), worst.k, worst.n), ("All other checks", *rest))}


def _by_location(spot: Spot, f: Finding) -> dict:
    worst = _row(f, f.extra.get("worst", ""))
    if worst is None:
        return {"headline": "Checks by place", "sentence": "There are too few checks at each stop to compare them.", "level": "ok", "facts": {}}
    rest = _rest(f, worst.key)
    clear = bool(f.extra.get("clear"))
    name = place_name(spot.kit, worst.key)
    return {"headline": f"{name} trips SPOT up",
            "sentence": f"SPOT has the most trouble at {place_label(spot.kit, worst.key)}: {worst.k} of {worst.n} checks there failed, against "
                        f"{rest[0]} of {rest[1]} everywhere else." + (" Walk the route there." if clear else _hedge(clear)),
            "level": "watch" if clear else "ok", "facts": {"k": worst.k, "n": worst.n, "k_rest": rest[0], "n_rest": rest[1]},
            "why_ref": {"kind": "location", "id": worst.key}, "visual": bars((name, worst.k, worst.n), ("Everywhere else", *rest))}


def _navigation(spot: Spot, f: Finding) -> dict:
    loc = f.extra.get("location_id")
    worst = _row(f, loc) if loc else None
    if worst is None:
        return {"headline": "Getting around", "sentence": "SPOT has walked to each stop too few times to compare them.", "level": "ok", "facts": {}}
    other = f.extra.get("elsewhere") or {}
    clear = bool(f.extra.get("clear"))
    return {"headline": f"The walk to {place_name(spot.kit, loc)}",
            "sentence": f"Getting to {place_label(spot.kit, loc)} failed {worst.k} of {worst.n} times, against {other.get('k', 0)} of "
                        f"{other.get('n', 0)} for every other stop." + ("" if clear else " That is a hint, not a firm pattern: it is also SPOT's first stop on most rounds."),
            "level": "watch" if clear else "ok", "facts": {"k": worst.k, "n": worst.n, "k_rest": other.get("k", 0), "n_rest": other.get("n", 0)},
            "visual": bars((place_name(spot.kit, loc), worst.k, worst.n), ("Every other stop", other.get("k", 0), other.get("n", 0)))}


def _by_asset(spot: Spot, f: Finding) -> dict:
    worst = _row(f, f.extra.get("worst", ""))
    if worst is None:
        return {"headline": "Checks by equipment", "sentence": "There are too few checks on each piece of equipment to compare them.", "level": "ok", "facts": {}}
    clear = bool(f.extra.get("clear"))
    return {"headline": "Equipment SPOT struggles to check",
            "sentence": f"Checks on {asset_label(spot.kit, worst.key)} fail most: {worst.k} of {worst.n}." + _hedge(clear),
            "level": "watch" if clear else "ok", "facts": {"k": worst.k, "n": worst.n}, "why_ref": {"kind": "asset", "id": worst.key}}


def _by_floor(spot: Spot, f: Finding) -> dict:
    worst, best = _row(f, f.extra.get("worst", "")), _row(f, f.extra.get("best", ""))
    if worst is None or best is None or worst.key == best.key:
        return {"headline": "Checks by floor", "sentence": "There is only one floor to look at.", "level": "ok", "facts": {}}
    clear = bool(f.extra.get("clear"))
    return {"headline": f"Floor {worst.key} is harder for SPOT",
            "sentence": f"On floor {worst.key}, {worst.k} of {worst.n} checks failed; on floor {best.key}, {best.k} of {best.n}." + _hedge(clear),
            "level": "watch" if clear else "ok", "facts": {"k": worst.k, "n": worst.n, "k_rest": best.k, "n_rest": best.n},
            "visual": bars((f"Floor {worst.key}", worst.k, worst.n), (f"Floor {best.key}", best.k, best.n))}


def _by_environment(spot: Spot, f: Finding) -> dict:
    worst = _row(f, f.extra.get("worst", ""))
    if worst is None:
        return {"headline": "Checks by surroundings", "sentence": "There are too few checks in each kind of area to compare them.", "level": "ok", "facts": {}}
    clear = bool(f.extra.get("clear"))
    where = ENVIRONMENTS.get(worst.key, f"{soft(worst.key)} areas")
    return {"headline": f"{where.capitalize()} are harder for SPOT",
            "sentence": f"Checks in {where} fail most often: {worst.k} of {worst.n}." + _hedge(clear),
            "level": "watch" if clear else "ok", "facts": {"k": worst.k, "n": worst.n}}


def _failure_types(spot: Spot, f: Finding) -> dict:
    top = f.rows[0] if f.rows else None
    if top is None:
        return {"headline": "Why checks fail", "sentence": "No check has failed.", "level": "ok", "facts": {}}
    return {"headline": "Why checks fail",
            "sentence": f"When a check fails, the most common reason is that {failure_word(top.key)}: {top.k} of {top.n} failed checks.",
            "level": "ok", "facts": {"k": top.k, "n": top.n}}


def _sequence(spot: Spot, f: Finding) -> dict:
    if len(f.rows) < 3:
        return {"headline": "Where in a round checks fail", "sentence": "The rounds are too short to compare their stops.", "level": "ok", "facts": {}}
    tail, early = f.rows[-2:], f.rows[:-2]
    a, b = (sum(r.k for r in tail), sum(r.n for r in tail)), (sum(r.k for r in early), sum(r.n for r in early))
    clear = bool(f.extra.get("clear"))
    return {"headline": "The end of a round is the weak part",
            "sentence": f"The last two stops of a round fail most: {a[0]} of {a[1]}, against {b[0]} of {b[1]} for the earlier stops. "
                        "That is also when the battery is lowest." + _hedge(clear),
            "level": "watch" if clear else "ok", "facts": {"k": a[0], "n": a[1], "k_rest": b[0], "n_rest": b[1]},
            "visual": bars(("Last two stops", *a), ("Earlier stops", *b))}


def _time_of_day(spot: Spot, f: Finding) -> dict:
    clear = bool(f.extra.get("clear"))
    worst = f.rows and max(f.rows, key=lambda r: r.rate or 0)
    if clear and worst:
        return {"headline": "Time of day matters", "sentence": f"Rounds that start at {worst.label} fail most: {worst.k} of {worst.n} checks.",
                "level": "watch", "facts": {"k": worst.k, "n": worst.n}}
    return {"headline": "Time of day does not matter", "sentence": "The time a round starts makes no clear difference to how many checks fail.",
            "level": "ok", "facts": {}}


def _durations(spot: Spot, f: Finding) -> dict:
    bad, good = (f.extra.get("failed") or {}).get("mean"), (f.extra.get("ok") or {}).get("mean")
    if bad is None or good is None:
        return {"headline": "How long checks take", "sentence": "There are too few checks to compare how long they take.", "level": "ok", "facts": {}}
    return {"headline": "A slow check is not a warning sign",
            "sentence": f"Failed checks take about as long as ones that work: {bad:.0f} seconds against {good:.0f}.",
            "level": "ok", "facts": {"failed_s": round(bad), "ok_s": round(good)}}


def _over_time(spot: Spot, f: Finding) -> dict:
    points = f.extra.get("points") or []
    versions = sorted({p["version"] for p in points})
    if len(versions) < 2:
        mean = sum(p["k"] for p in points) / len(points) if points else 0
        return {"headline": "Is SPOT getting better or worse?", "sentence": f"About {mean:.1f} checks fail per round, with no change of setup to compare.",
                "level": "ok", "facts": {"per_round": round(mean, 1)}}
    old, new = ([p["k"] for p in points if p["version"] == v] for v in (versions[0], versions[-1]))
    a, b = sum(old) / len(old), sum(new) / len(new)
    better = b < a
    return {"headline": "SPOT is getting better, not worse" if better else "SPOT is not getting better",
            "sentence": f"Failed checks per round went from {a:.1f} to {b:.1f} after SPOT's setup was updated"
                        + (", and have stayed level since." if abs(f.value or 0) < 0.005 else ", and are still changing.")
                        + f" That is {len(new)} rounds: a short run.",
            "level": "ok" if better else "watch", "facts": {"before": round(a, 1), "after": round(b, 1), "rounds": len(new)}}


def _alarms_hidden(spot: Spot, f: Finding) -> dict:
    inside, total = f.extra.get("in_completed", 0), f.n
    stands, unsure = len(f.extra.get("confirmed", [])), len(f.extra.get("unverified", []))
    return {"headline": "Most alarms are not confirmed",
            "sentence": f"{unsure} of {total} alarms came from a check that itself failed, so they are not confirmed; {stands} came from a check "
                        f"that worked. {inside} of the {total} were raised in rounds marked completed.",
            "level": "watch", "facts": {"unsure": unsure, "n": total, "stands": stands, "inside": inside}}


def _alarm_matrix(spot: Spot, f: Finding) -> dict:
    k = int(f.value or 0)
    return {"headline": "Serious alarms on important equipment",
            "sentence": f"{k} of {f.n} alarms are serious ones on important equipment. Those come first in the list.",
            "level": "watch" if k else "ok", "facts": {"k": k, "n": f.n}}


def _repeats(spot: Spot, f: Finding) -> dict:
    k, n = int(f.value or 0), f.n
    top = f.rows[0] if f.rows else None
    if not k or top is None:
        return {"headline": "Repeat alarms", "sentence": "No piece of equipment has been flagged more than once.", "level": "ok", "facts": {}}
    return {"headline": "Some equipment keeps being flagged",
            "sentence": f"{k} of the {n} pieces of equipment with an alarm have been flagged more than once. "
                        f"{asset_label(spot.kit, top.key)} most: {top.k} times.",
            "level": "watch", "facts": {"k": k, "n": n, "most": top.k}, "why_ref": {"kind": "asset", "id": top.key}}


def _trends(spot: Spot, f: Finding) -> dict:
    series = f.extra.get("series") or []
    enough = sum(1 for s in series if s.get("enough"))
    drifting = int(f.value or 0)
    lead = ("No reading is drifting toward the edge of its usual range." if not drifting
            else f"{drifting} reading{'s are' if drifting != 1 else ' is'} drifting toward the edge of the usual range.")
    return {"headline": "Readings are steady" if not drifting else "A reading is drifting",
            "sentence": f"{lead} Only {enough} of {len(series)} reading histories are long enough to judge.",
            "note": "Usual range is our own band from this data. The kit gives no plant limits.",
            "level": "watch" if drifting else "ok", "facts": {**({"drifting": drifting} if drifting else {}), "enough": enough, "series": len(series)}}


FINDING_WORDS = {"hidden_failures": _hidden, "version_comparison": _version, "battery": _battery, "reliability_by_type": _by_type,
                 "reliability_by_location": _by_location, "navigation_hotspot": _navigation, "reliability_by_asset": _by_asset,
                 "reliability_by_floor": _by_floor, "reliability_by_environment": _by_environment, "failure_types": _failure_types,
                 "sequence": _sequence, "time_of_day": _time_of_day, "durations": _durations, "over_time": _over_time,
                 "anomalies_hidden": _alarms_hidden, "anomaly_matrix": _alarm_matrix, "anomaly_repeats": _repeats, "reading_trends": _trends}


def finding_item(spot: Spot, f: Finding) -> dict:
    """One finding as a sentence. `facts` holds every number the sentence states, straight from the finding."""
    make = FINDING_WORDS.get(f.id)
    if make is None or not f.enough:
        words = {"headline": "Not enough data yet" if not f.enough else "We also looked at this",
                 "sentence": "There is not enough data to say anything about this yet." if not f.enough else "Open it to see what was found.",
                 "level": "ok", "facts": {}}
    else:
        words = make(spot, f)
    why = words.pop("why_ref", None) or {"kind": "finding", "id": f.id}
    out = item(f.id, words.pop("headline"), words.pop("sentence"), words.pop("level"), (why["kind"], why["id"]), finding_id=f.id, **words)
    out.setdefault("note", None)
    out.setdefault("visual", None)
    return out


# --- rounds ---


def _number(kit: Kit, mission_id: str) -> int:
    digits = "".join(c for c in mission_id if c.isdigit())
    return int(digits) if digits else [m.mission_id for m in kit.missions].index(mission_id) + 1


def round_label(spot: Spot, mission_id: str) -> dict:
    """'Round 27 · today 10:00 AM · 1 check failed'."""
    kit = spot.kit
    m = kit.mission[mission_id]
    failed = len(kit.failed_in(mission_id))
    alarms = len(kit.anomalies_in(mission_id))
    finished = m.mission_status == "COMPLETED"
    result = ("did not finish" if not finished else "all checks worked" if not failed else f"{failed} check{'s' if failed != 1 else ''} failed")
    if not finished and failed:
        result += f", {failed} check{'s' if failed != 1 else ''} failed"
    n = _number(kit, mission_id)
    return {"mission_id": mission_id, "number": n, "label": f"Round {n} · {when(kit.now, m.start_time)} · {result}", "name": f"Round {n}",
            "started": m.start_time.isoformat(), "when": when(kit.now, m.start_time), "result": result, "failed": failed, "alarms": alarms,
            "checks": len(kit.actions_of.get(mission_id, [])), "level": "act" if not finished else "watch" if failed or alarms else "ok",
            "start_battery_pct": m.start_battery_pct, "end_battery_pct": m.end_battery_pct}


def _value(r) -> str:
    name, unit = MEASURES.get(r.measurement_type, (soft(r.measurement_type), f" {r.unit}" if r.unit else ""))
    if r.measurement_type == "LOCALIZATION_CONFIDENCE_PCT":
        return f"{r.value:g}% sure of its position"
    if r.measurement_type == "VISUAL_SCORE":
        return f"picture score {r.value:g} out of 100"
    return f"{r.value:g}{unit}"


def step(spot: Spot, a: Action) -> dict:
    """One check as a sentence: '10:12 AM · Listening check at Heat exchanger 2 · failed: the reading was not reliable'."""
    kit = spot.kit
    walk = not a.is_inspection
    title = (f"Walk to {place_name(kit, a.location_id)}" if walk else f"{check_word(a.action_type).capitalize()} at {asset_name(kit, a.asset_id)}")
    readings = kit.readings_of.get(a.action_id, [])
    alarms = kit.anomalies_of.get(a.action_id, [])
    parts = [clock(a.timestamp), title, f"failed: {failure_word(a.failure_type)}" if a.failed else "worked"]
    if readings and not a.failed:
        parts.append(_value(readings[0]))
    for x in alarms:
        parts.append(f"alarm: {alarm_word(x.anomaly_type)}" + (" (not confirmed)" if a.failed else ""))
    return {"action_id": a.action_id, "n": a.sequence, "time": clock(a.timestamp), "at": a.timestamp.isoformat(), "check": CHECKS.get(a.action_type, ("", "", "camera"))[2],
            "title": title, "worked": not a.failed, "alarm": bool(alarms), "result": "alarm" if alarms else "failed" if a.failed else "ok",
            "sentence": " · ".join(parts), "battery_pct": a.battery_pct, "location_id": a.location_id, "place": place_label(kit, a.location_id),
            "asset_id": a.asset_id, "asset": asset_label(kit, a.asset_id), "why_ref": {"kind": "asset", "id": a.asset_id}}


def round_view(spot: Spot, mission_id: str) -> dict | None:
    """One round, replayed: its label, a one-line summary and every check as a sentence."""
    kit = spot.kit
    if kit is None or mission_id not in kit.mission:
        return None
    head = round_label(spot, mission_id)
    steps = [step(spot, a) for a in kit.actions_of.get(mission_id, [])]
    m = kit.mission[mission_id]
    worked = sum(1 for s in steps if s["worked"])
    alarms = head["alarms"]
    summary = (f"{head['name']} ran {when(kit.now, m.start_time)} to {clock(m.end_time)}. {worked} of {len(steps)} checks worked"
               + (f", and SPOT raised {alarms} alarm{'s' if alarms != 1 else ''}" if alarms else "") + ". "
               + ("The round is marked completed even though a check failed inside it. " if head["failed"] and m.mission_status == "COMPLETED"
                  else "" if m.mission_status == "COMPLETED" else "The round is marked as not finished. ")
               + f"Battery went from {m.start_battery_pct:.0f}% to {m.end_battery_pct:.0f}%.")
    return {**head, "summary": summary, "steps": steps, "ended": m.end_time.isoformat(), "why_ref": {"kind": "mission", "id": mission_id}}


# --- the shift ---


def shift(spot: Spot) -> dict:
    """Everything the first screen needs, in words: status, trust, three cards, every asset, the rounds, the map."""
    kit = spot.kit
    if kit is None:
        return {**not_configured(), "do_first": [], "watch": [], "assets": [], "rounds": [], "triage": [], "findings": []}
    now = kit.now
    assets = [asset_state(spot, a.asset_id) for a in kit.assets]
    order = {"act": 0, "watch": 1, "ok": 2}
    assets.sort(key=lambda s: (order[s["level"]], s["rank"] if s["rank"] is not None else 10_000,
                               -CRIT_RANK.get(kit.asset[s["asset_id"]].criticality, 0), s["asset_id"]))
    act = [s for s in assets if s["level"] == "act"]
    watch_assets = [s for s in assets if s["level"] == "watch"]
    fine = [s for s in assets if s["level"] == "ok"]

    findings = [finding_item(spot, f) for f in spot.findings.values()]
    words = {f["id"]: f for f in findings}
    watch = [words[k] for k in PICKED if k in words and words[k]["level"] == "watch"][:3]
    hidden = words.get("hidden_failures")
    trust = None
    if hidden:
        trust = {"question": "How much can you trust a 'completed' round?", "sentence": hidden["sentence"], "facts": hidden["facts"],
                 "why_ref": hidden["why_ref"], "level": hidden["level"]}

    n_act, n_watch, n_fine = len(act), len(watch_assets), len(fine)
    need = ("Nothing needs you right now." if not n_act else f"{n_act} thing{'s' if n_act != 1 else ''} need{'s' if n_act == 1 else ''} you.")
    rest = (f" {n_watch} to keep an eye on." if n_watch else "") + (
        " Everything else looks normal." if n_fine and (n_act or n_watch) else " Everything looks normal." if n_fine else "")
    rounds = [round_label(spot, m.mission_id) for m in kit.missions]
    last = rounds[-1] if rounds else None
    status = {"sentence": f"SPOT has run {len(rounds)} round{'s' if len(rounds) != 1 else ''}. {need}{rest}",
              "rounds": len(rounds), "act": n_act, "watch": n_watch, "ok": n_fine, "assets": len(assets),
              "as_of": f"Data ends {DAYS[now.weekday()]} {now.day} {MONTHS[now.month - 1]}, {clock(now)}",
              "last_round": f"Last round: {last['label']}" if last else None,
              "note": "Practice data from the challenge kit. Times are the data's own, not the clock on the wall."}
    looks_fine = {"headline": f"{n_fine} of {len(assets)} look fine",
                  "sentence": (f"{n_fine} of {len(assets)} pieces of equipment had a check that worked in the last {STALE_HOURS} hours and have no open alarm."
                               if n_fine else f"None of the {len(assets)} pieces of equipment is clear right now."),
                  "level": "ok", "count": n_fine, "total": len(assets), "names": [s["label"] for s in fine[:3]],
                  "why_ref": {"kind": "finding", "id": "reading_trends"}}
    top_place = place_name(kit, act[0]["location_id"]) if act else place_name(kit, kit.locations[0].location_id) if kit.locations else "the plant"

    route = []
    if kit.missions:
        for a in kit.actions_of.get(kit.missions[-1].mission_id, []):
            if not route or route[-1] != a.location_id:
                route.append(a.location_id)
    return {"configured": True, "synthetic": True, "now": now.isoformat(), "status": status, "trust": trust,
            "do_first": act[:3], "do_first_total": n_act, "watch": watch, "fine": looks_fine, "assets": assets,
            "triage": [triage_item(spot, i) for i in spot.triage], "findings": findings, "rounds": rounds,
            "latest_round": last["mission_id"] if last else None,
            "map": {"plan": spot.plan, "route": route,
                    "places": [{"location_id": x.location_id, "name": place_name(kit, x.location_id), "floor": x.floor, "x": x.x_coord, "y": x.y_coord}
                               for x in kit.locations]},
            "levels": [{"level": k, "words": LEVEL_WORDS[k], "rule": LEVEL_RULE[k]} for k in ("act", "watch", "ok")],
            "glossary": GLOSSARY, "ask": [q.format(place=f"the {top_place}") for q in ASK]}


# --- Ask, in the same words ---


def answer(spot: Spot, draft: dict) -> str | None:
    """The assistant's answer in operator words, for the intents the first screen asks about. None: keep the exact wording."""
    kit = spot.kit
    if kit is None:
        return None
    intent, target = draft.get("intent"), draft.get("target") or {}
    view = shift(spot)
    if intent in ("investigate_next", "triage", "coverage"):
        if not view["do_first"]:
            return "Nothing needs you right now. " + view["fine"]["sentence"]
        more = view["do_first_total"] - len(view["do_first"])
        return ("Start here, in this order.\n" + "\n".join(f"{n}. {i['headline']}. {i['sentence']}" for n, i in enumerate(view["do_first"], 1))
                + (f"\nAfter that, {more} more need{'s' if more == 1 else ''} you: see the list." if more > 0 else ""))
    if intent == "mission" and target.get("id") in kit.mission:
        r = round_view(spot, target["id"])
        bad = [s for s in r["steps"] if not s["worked"]]
        text = r["summary"]
        if bad:
            text += " What failed: " + "; ".join(f"{s['title']} ({failure_word(kit.action[s['action_id']].failure_type)})" for s in bad) + "."
            text += " Trust the checks that worked. For the ones that failed, send SPOT back before acting on them."
        else:
            text += " Every check worked, so this round can be trusted as it stands."
        return text
    if intent == "waypoint" and target.get("id") in kit.location:
        loc = target["id"]
        acts = [a for a in kit.actions if a.location_id == loc]
        k = sum(a.failed for a in acts)
        here = [s for s in view["assets"] if s["location_id"] == loc]
        text = f"At {place_label(kit, loc)}, {k} of {len(acts)} checks have failed."
        if here:
            text += " Equipment here: " + "; ".join(f"{s['label']}: {s['level_words'].lower()}" for s in here) + "."
        if acts:
            text += f" SPOT was last here {ago(kit.now, acts[-1].timestamp)}: {step(spot, acts[-1])['sentence']}."
        return text
    if intent == "asset" and target.get("id") in kit.asset:
        s = next(x for x in view["assets"] if x["asset_id"] == target["id"])
        return f"{s['label']}: {s['level_words'].lower()}. {s['sentence']} {s['last_check']['sentence']} {s['reading']['sentence']}"
    words = {f["id"]: f for f in view["findings"]}
    if intent == "version_comparison":
        parts = [words[k]["sentence"] for k in ("over_time", "version_comparison", "reading_trends") if k in words]
        return " ".join(parts) or None
    if intent == "where" and "reliability_by_location" in words:
        return " ".join(words[k]["sentence"] for k in ("reliability_by_location", "reliability_by_floor") if k in words)
    if intent == "anomalies" and "anomalies_hidden" in words:
        return " ".join(words[k]["sentence"] for k in ("anomalies_hidden", "anomaly_repeats") if k in words)
    if intent in words:
        f = words[intent]
        return f["sentence"] + (f" {f['note']}" if f.get("note") else "")
    return None
