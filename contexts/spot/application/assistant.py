"""Ask, with evidence.

    answer = await Assistant(spot, runtime).ask("what should I check first this shift?")
    answer["text"], answer["citations"], answer["intent"], answer["rephrased"]

The answer is always built by code from the analysis (a deterministic intent layer; no model decides a fact),
and by default it is returned at once: `ask(text)` makes no model call, so an operator never waits on the
model for the facts. `ask(text, rephrase=True)` additionally asks the on-device model to REPHRASE that grounded
draft, at low priority, and only when the draft is short enough to rewrite inside a small budget
(`REPHRASE_MAX_DRAFT_CHARS`, `REPHRASE_TIMEOUT_S`); long, number-dense answers (the version comparison) are
never sent. Its wording is used only if every number and every id in it also appears in the draft (`grounded`);
on any mismatch, a timeout, an error, or with the fake runtime, the draft itself is returned.
"""

import asyncio
import re

from contexts.spot.application.service import Spot
from contexts.spot.domain.analysis import Evidence, _words
from contexts.spot.domain.stats import MIN_N, MIN_TREND_N, pct, pct_range
from contracts.types import ChatMessage

# The rephrase budget. The box writes about 14 tokens a second, one request at a time: a 450-character draft is
# roughly 110 tokens, so about 8 s. Anything longer is shown as the code built it.
REPHRASE_MAX_DRAFT_CHARS = 450
REPHRASE_TIMEOUT_S = 8
NEVER_REPHRASED = ("not_configured", "handoff", "version_comparison")  # nothing to reword, or too many numbers to risk
ID = re.compile(r"\b(?:AN\d{3}|M\d{3}|A\d{4}|R\d{4}|WP\d{2}|E\d{5}|[A-Z]{2,6}-\d{3}|V\d)\b")
NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)*")
SYSTEM = ("You rewrite a briefing for a plant operator. Use plain, direct sentences. Keep every number and every id exactly as "
          "written. Do not add facts, numbers, ids, causes or advice that are not in the briefing. No preamble, no sign-off.")
SUGGESTED = ["What should I check first this shift?", "What needs attention now?", "Show hidden failures", "Is V2 better than V1?",
             "Which inspection types are least reliable?", "What happened at WP05?", "Does battery affect reliability?",
             "Status of HX-202", "Summarize the last mission", "Are anomalies hidden in successful missions?"]


def _numbers(text: str) -> set[str]:
    """Numbers in a text, normalised, with ids taken out first so 'WP05' does not count as 5."""
    out = set()
    for raw in NUMBER.findall(ID.sub(" ", text)):
        s = raw.replace(",", "").lstrip("+").rstrip(".")
        try:
            out.add(f"{abs(float(s)):g}")
        except ValueError:
            continue
    return out


def grounded(candidate: str, draft: str) -> tuple[bool, str]:
    """True when `candidate` states no number and no id that the draft does not."""
    if not candidate or not candidate.strip():
        return False, "empty"
    if len(candidate) > 1.6 * len(draft) + 200:
        return False, "much longer than the draft"
    new_ids = set(ID.findall(candidate)) - set(ID.findall(draft))
    if new_ids:
        return False, f"ids not in the draft: {', '.join(sorted(new_ids))}"
    new_numbers = _numbers(candidate) - _numbers(draft)
    if new_numbers:
        return False, f"numbers not in the draft: {', '.join(sorted(new_numbers))}"
    return True, "ok"


class Assistant:
    def __init__(self, spot: Spot, runtime=None, rephrase: bool | None = None) -> None:
        self.spot = spot
        self.runtime = runtime
        # rephrase only with a real model: the fake runtime echoes its input
        self.rephrase = rephrase if rephrase is not None else bool(runtime is not None and runtime.model_info().base != "fake")

    # --- the grounded draft ---

    def draft(self, text: str) -> dict:
        spot = self.spot
        if not spot.configured:
            return {"intent": "not_configured", "text": "No SPOT kit is loaded on this box, so there is nothing to answer from yet.",
                    "citations": Evidence().model_dump(), "target": None}
        q = text.strip()
        low = q.lower()
        kit = spot.kit
        waypoint = next((w.upper() for w in re.findall(r"\bwp\s?(\d{1,2})\b", low)), None)
        if waypoint:
            waypoint = f"WP{int(waypoint):02d}"
        else:  # or by its name: "what happened at the pipe gallery?"
            waypoint = next((x.location_id for x in kit.locations if x.location_name and x.location_name.lower() in low), None)
        asset = next((a.asset_id for a in kit.assets if a.asset_id.lower() in low or (a.asset_name and a.asset_name.lower() in low)), None)
        mission = next((m.upper() for m in re.findall(r"\bm\d{3}\b", low)), None)
        has = lambda *words: any(w in low for w in words)  # noqa: E731

        pair = re.search(r"between (?:mission )?m?0*(\d{1,3}) and (?:mission )?m?0*(\d{1,3})", low)
        if pair and has("worse", "changed", "new", "differ"):
            return self._worse(f"M{int(pair.group(1)):03d}", f"M{int(pair.group(2)):03d}")
        if mission and mission in kit.mission:
            return self._mission(mission)
        if asset and has("normal", "in range", "usual", "okay"):
            return self._reading_normal(asset, low)
        if asset:
            return self._asset(asset)
        if waypoint and waypoint in kit.location:
            return self._waypoint(waypoint)
        if has("handoff", "hand off", "hand-off", "hand to", "shift summary", "summarize the shift", "summarise the shift", "next operator"):
            h = spot.handoff()
            ev = Evidence()
            for line in h["lines"]:
                for k, v in line["evidence"].items():
                    setattr(ev, k, sorted(set(getattr(ev, k)) | set(v)))
            return {"intent": "handoff", "text": h["text"], "citations": ev.model_dump(), "target": None}
        if has("last mission", "latest mission", "most recent mission", "last round", "latest round"):
            return self._mission(kit.missions[-1].mission_id)
        if has("assumption", "method", "how do you", "how did you"):
            items = spot.assumptions
            return {"intent": "assumptions", "text": "The analysis rests on these assumptions: " + " ".join(f"({i}) {a['assumption']}" for i, a in enumerate(items[:6], 1))
                    + f" There are {len(items)} in all; the Assumptions and method panel lists every one.", "citations": Evidence().model_dump(), "target": {"kind": "assumptions", "id": "assumptions"}}
        if has("check first", "inspect first", "investigate", "this shift", "what should i", "what next", "look at first", "where do i start"):
            return self._next()
        if has("attention", "triage", "urgent", "priority", "priorit", "most important", "biggest issue", "worst"):
            return self._triage()
        if has("anomal"):
            return self._anomalies()
        if has("hidden", "completed") and has("fail", "hidden") or has("mission status"):
            return self._finding("hidden_failures", extra_rows=5)
        if has("uninspected", "not inspected", "not been inspected", "coverage", "overdue", "last good"):
            return self._coverage()
        if has("battery", "charge"):
            return self._finding("battery")
        if has("v1", "v2", "version", "baseline", "improved", "configuration", "improving", "degrading", "getting better", "getting worse", "trend"):
            return self._version(over_time=has("improving", "degrading", "getting better", "getting worse", "trend"))
        if has("least reliable", "inspection type", "action type", "which type", "types", "kind of inspection", "weakest"):
            return self._finding("reliability_by_type", extra_rows=5)
        if has("where", "location", "concentrat", "waypoint", "floor"):
            return self._where()
        if has("failure type", "why do", "kinds of failure", "fail"):
            return self._finding("failure_types", extra_rows=4)
        if has("time of day", "night", "hour"):
            return self._finding("time_of_day")
        if has("duration", "retr", "long"):
            return self._finding("durations")
        if has("reading", "drift", "threshold", "temperature", "pressure"):
            return self._finding("reading_trends")
        if has("replay", "stream", "event"):
            return self._replay()
        if has("validation", "data quality", "surprise", "issue"):
            v = spot.validation
            return {"intent": "validation", "text": v["summary"] + " " + " ".join(i["message"] for i in v["issues"] if i["level"] != "info"),
                    "citations": Evidence().model_dump(), "target": {"kind": "validation", "id": "validation"}}
        return self._overview()

    def _pack(self, intent: str, text: str, evidence: Evidence | dict, target: dict | None = None) -> dict:
        return {"intent": intent, "text": text, "citations": evidence.model_dump() if isinstance(evidence, Evidence) else evidence, "target": target}

    def _finding(self, fid: str, extra_rows: int = 0) -> dict:
        f = self.spot.findings[fid]
        text = f.sentence
        if extra_rows and f.rows:
            text += " In order: " + "; ".join(f"{b.label} {b.k}/{b.n}" + (f" ({pct(b.rate)})" if b.rate is not None and fid != "failure_types" else "")
                                              + ("" if b.enough else " (not enough data)") for b in f.rows[:extra_rows]) + "."
        if f.caveat:
            text += f" Caveat: {f.caveat}"
        return self._pack(fid, text, f.evidence, {"kind": "finding", "id": fid})

    def _next(self) -> dict:
        items = self.spot.next_up
        ev = Evidence()
        for it in items:
            for k, v in it["evidence"].items():
                setattr(ev, k, sorted(set(getattr(ev, k)) | set(v)))
        text = "Check these first, in this order. " + " ".join(f"({i}) {it['title']}. {it['sentence']}" for i, it in enumerate(items, 1))
        return self._pack("investigate_next", text, ev, items[0]["target"] if items else None)

    def _triage(self) -> dict:
        items = self.spot.triage[:5]
        ev = Evidence(assets=[i["id"] for i in items if i["kind"] == "asset"], locations=[i["id"] for i in items if i["kind"] == "waypoint"],
                      anomalies=[i["anomaly_id"] for i in items if i["anomaly_id"]])
        text = (f"{len(self.spot.triage)} items need attention; the top {len(items)} by score: "
                + " ".join(f"({i['rank']}) {i['id']}, score {i['score']:g}: {i['reason']} Next: {i['next_step']['text']}" for i in items)
                + " The score adds severity (weighted by recency), asset criticality, repeat anomalies, a failed latest inspection and "
                  "days since the last good one.")
        return self._pack("triage", text, ev, {"kind": "asset", "id": items[0]["id"]} if items and items[0]["kind"] == "asset" else None)

    def _coverage(self) -> dict:
        risky = [r for r in self.spot.uninspected if r["at_risk"]]
        if not risky:
            return self._pack("coverage", "No HIGH or CRITICAL asset is overdue: each has a successful inspection in the last 24 hours of data "
                                          "and no failed attempt since.", Evidence())
        ev = Evidence(assets=[r["asset_id"] for r in risky], actions=[a for r in risky for a in r["evidence"]["actions"]])
        return self._pack("coverage", f"{len(risky)} HIGH or CRITICAL asset(s) have not been inspected successfully for 24 hours or have a "
                                      "failed attempt since their last good inspection: " + " ".join(r["text"] for r in risky),
                          ev, {"kind": "asset", "id": risky[0]["asset_id"]})

    def _version(self, over_time: bool = False) -> dict:
        f = self.spot.findings["version_comparison"]
        text = f.sentence
        if f.extra.get("changed"):
            text += " Also changed: " + ", ".join(f.extra["changed"]) + "."
        if f.extra.get("unchanged"):
            text += " No clear change in: " + ", ".join(f.extra["unchanged"]) + "."
        if f.extra.get("not_enough"):
            text += " Not enough data for: " + ", ".join(f.extra["not_enough"]) + "."
        if f.caveat:
            text += f" Caveat: {f.caveat}"
        if over_time:
            text = self.spot.findings["over_time"].sentence + " " + text
        return self._pack("version_comparison", text, f.evidence, {"kind": "finding", "id": f.id})

    def _anomalies(self) -> dict:
        f = self.spot.findings
        a, m, r = f["anomalies_hidden"], f["anomaly_matrix"], f["anomaly_repeats"]
        ev = Evidence(anomalies=sorted(set(a.evidence.anomalies) | set(m.evidence.anomalies)), missions=a.evidence.missions, assets=m.evidence.assets)
        return self._pack("anomalies", f"{a.sentence} {m.sentence} {r.sentence} Caveat: {a.caveat}", ev, {"kind": "finding", "id": a.id})

    def _where(self) -> dict:
        f = self.spot.findings
        loc, nav, floor = f["reliability_by_location"], f["navigation_hotspot"], f["reliability_by_floor"]
        ev = Evidence(actions=sorted(set(loc.evidence.actions) | set(nav.evidence.actions)), locations=sorted({loc.extra.get("worst", ""), nav.extra.get("location_id", "")} - {""}))
        text = f"{loc.sentence} {nav.sentence} {floor.sentence}"
        caveats = " ".join(c for c in (loc.caveat, nav.caveat) if c)
        return self._pack("where", text + (f" Caveat: {caveats}" if caveats else ""), ev, {"kind": "location", "id": loc.extra.get("worst")})

    def _waypoint(self, location_id: str) -> dict:
        loc = next(x for x in self.spot.locations()["locations"] if x["location_id"] == location_id)
        failed, anomalies = loc["failed_actions"], loc["anomalies"]
        text = (f"{location_id} {loc['location_name']} (floor {loc['floor']}, {_words(loc['zone_type'])}, {_words(loc['environment'])}): "
                f"{loc['k']} of {loc['n']} actions there failed")
        text += (f" ({pct(loc['rate'])}, 95% interval {pct_range((loc['lo'], loc['hi']))})." if loc["enough"] else f": not enough data (under {MIN_N} actions).")
        if loc["by_type"]:
            text += " By type: " + ", ".join(f"{_words(t)} {v['k']}/{v['n']}" for t, v in sorted(loc["by_type"].items(), key=lambda kv: -kv[1]["n"])) + "."
        if failed:
            text += " Failed: " + ", ".join(f"{a['action_id']} ({a['mission_id']}, {_words(a['action_type'])}, {_words(a['failure_type'])})" for a in failed[:8])
            text += f" and {len(failed) - 8} more." if len(failed) > 8 else "."
        if anomalies:
            text += f" {len(anomalies)} anomaly row(s) here: " + ", ".join(f"{x['anomaly_id']} ({x['severity']}, {x['asset_id']})" for x in anomalies[:8]) + "."
        if loc["first_stop_of"]:
            text += f" It is the first stop of {loc['first_stop_of']} missions."
        nav = self.spot.findings["navigation_hotspot"]
        if nav.extra.get("location_id") == location_id:
            text += f" {nav.sentence}"
        ev = Evidence(locations=[location_id], actions=[a["action_id"] for a in failed], anomalies=[x["anomaly_id"] for x in anomalies],
                      missions=sorted({a["mission_id"] for a in failed}))
        return self._pack("waypoint", text, ev, {"kind": "location", "id": location_id})

    def _asset(self, asset_id: str) -> dict:
        a = self.spot.asset(asset_id)
        s = a["inspection"]
        text = (f"{asset_id} {a['asset_name']} ({a['criticality']} criticality, {a['asset_type']}, at {a['location_id']}): {a['k']} of {a['n']} "
                f"actions on it failed" + (f" ({pct(a['rate'])})." if a["enough"] else f": not enough data (under {MIN_N} actions)."))
        if s["last_good_action"]:
            text += f" Last good inspection: {s['last_good_action']} in {s['last_good_mission']}, {s['hours_since_good']:.0f} h before the end of the data."
        else:
            text += " It has no successful inspection in the data."
        if s["failed_since"]:
            text += f" Failed since then: {', '.join(s['failed_since'])}."
        if a["anomalies_list"]:
            text += " Anomalies: " + ", ".join(f"{x['anomaly_id']} ({x['severity']} {_words(x['anomaly_type'])}, {x['mission_id']}, "
                                               f"{'confirmed' if x['confirmed'] else 'unverified'})" for x in a["anomalies_list"]) + "."
        else:
            text += " No anomaly on record."
        if a["triage"]:
            text += f" Triage rank {a['triage']['rank']}, score {a['triage']['score']:g}. Next: {a['triage']['next_step']['text']}"
        ev = Evidence(assets=[asset_id], anomalies=[x["anomaly_id"] for x in a["anomalies_list"]],
                      actions=[x for x in [s["last_good_action"], *s["failed_since"]] if x])
        return self._pack("asset", text, ev, {"kind": "asset", "id": asset_id})

    def _mission(self, mission_id: str) -> dict:
        m = self.spot.mission(mission_id)
        failed = [a for a in m["timeline"] if a["status"] != "SUCCESS"]
        anomalies = [(a, x) for a in m["timeline"] for x in a["anomalies"]]
        text = (f"{mission_id} ({m['configuration_version']}, {m['scenario']}) ran {m['start_time'].replace('T', ' ')[:16]} for {m['duration_min']:.0f} min, "
                f"battery {m['start_battery_pct']:.0f}% to {m['end_battery_pct']:.0f}%, and ended {m['mission_status']}. "
                f"{len(failed)} of {m['actions']} actions failed")
        text += (": " + ", ".join(f"{a['action_id']} ({_words(a['action_type'])} of {a['asset_id']} at {a['location_id']}, {_words(a['failure_type'])})" for a in failed) + "."
                 if failed else ".")
        if m["hidden_failure"]:
            text += " The mission status hides those failures."
        text += (f" {len(anomalies)} anomaly row(s): " + ", ".join(f"{x['anomaly_id']} ({x['severity']} {_words(x['anomaly_type'])} on {a['asset_id']}, "
                                                                  f"{'confirmed' if x['confirmed'] else 'unverified'})" for a, x in anomalies) + "."
                 if anomalies else " No anomalies.")
        ev = Evidence(missions=[mission_id], actions=[a["action_id"] for a in failed], anomalies=[x["anomaly_id"] for _, x in anomalies])
        return self._pack("mission", text, ev, {"kind": "mission", "id": mission_id})

    def _worse(self, before: str, after: str) -> dict:
        """Assets that were fine in one mission and failed or raised an anomaly in a later one."""
        kit = self.spot.kit
        if before not in kit.mission or after not in kit.mission:
            return self._pack("worse", f"I do not have both {before} and {after} in the data.", Evidence())

        def state(mid: str) -> dict[str, dict]:
            out: dict[str, dict] = {}
            for a in kit.actions_of.get(mid, []):
                st = out.setdefault(a.asset_id, {"failed": [], "anomalies": []})
                if a.failed:
                    st["failed"].append(a.action_id)
                st["anomalies"] += [x.anomaly_id for x in kit.anomalies_of.get(a.action_id, [])]
            return out

        a, b = state(before), state(after)
        worse = [(asset, st) for asset, st in sorted(b.items()) if (st["failed"] or st["anomalies"])
                 and not (a.get(asset, {}).get("failed") or a.get(asset, {}).get("anomalies"))]
        if not worse:
            return self._pack("worse", f"No asset got worse between {before} and {after}: nothing failed or raised an anomaly in {after} that was "
                                       f"clean or unvisited in {before}.", Evidence(missions=[before, after]), {"kind": "mission", "id": after})
        parts = []
        for asset, st in worse:
            was = "was clean" if asset in a else "was not visited"
            what = " and ".join(x for x in ((f"failed {', '.join(st['failed'])}" if st["failed"] else ""),
                                            (f"raised {', '.join(st['anomalies'])}" if st["anomalies"] else "")) if x)
            parts.append(f"{asset} ({kit.asset[asset].criticality if asset in kit.asset else '?'}) {was} in {before} and {what} in {after}")
        ev = Evidence(missions=[before, after], assets=[x for x, _ in worse], actions=[i for _, st in worse for i in st["failed"]],
                      anomalies=[i for _, st in worse for i in st["anomalies"]])
        return self._pack("worse", f"{len(worse)} asset(s) got worse between {before} and {after}: " + "; ".join(parts) + ".", ev,
                          {"kind": "mission", "id": after})

    def _reading_normal(self, asset_id: str, low: str) -> dict:
        """Is this asset's latest reading of a measurement inside the fleet's band? The band is ours, not an engineering limit."""
        trends = self.spot.findings["reading_trends"].extra
        series = [s for s in trends["series"] if s["asset_id"] == asset_id]
        hint = {"TEMPERATURE_C": ("temp", "thermal"), "SOUND_DB": ("sound", "acoustic", "vibration"), "PRESSURE_PSI": ("pressure", "gauge"),
                "VISUAL_SCORE": ("visual",), "LOCALIZATION_CONFIDENCE_PCT": ("localization", "localisation")}
        wanted = next((s for s in series if any(w in low for w in hint.get(s["measurement_type"], ()))), None)
        wanted = wanted or (max(series, key=lambda s: (s["n"], s["measurement_type"])) if series else None)
        if wanted is None:
            return self._pack("reading_normal", f"{asset_id} has no readings in the data, so I cannot say whether it is normal.", Evidence(assets=[asset_id]),
                              {"kind": "asset", "id": asset_id})
        band = trends["bands"].get(wanted["measurement_type"])
        last = wanted["points"][-1]
        name = _words(wanted["measurement_type"])
        text = f"{asset_id} {name}: the latest reading is {last['value']} {wanted['unit']} ({last['reading_id']}, {last['mission_id']}), "
        text += "marked GOOD" if last["good"] else "marked QUESTIONABLE because the action failed, so it should not be trusted"
        if band:
            inside = band["lo"] <= last["value"] <= band["hi"]
            text += (f". The fleet's usual range for {name} is {band['lo']:.1f} to {band['hi']:.1f} {wanted['unit']} (median {band['median']:.1f}, "
                     f"from {band['n']} good readings), and this one is {'inside' if inside else 'outside'} it.")
        else:
            text += f". There are too few good {name} readings in the fleet to say what is usual."
        text += f" {wanted['n_good']} good of {wanted['n']} readings on this asset" + (
            f"; trend {wanted['slope_per_day']:+.2f} {wanted['unit']} per day." if wanted["enough"] and wanted["slope_per_day"] is not None
            else f": not enough for a trend (need {MIN_TREND_N} good).")
        flagged = self.spot.kit.anomalies_of.get(last["action_id"], [])
        if flagged:
            text += (" Even so, " + ", ".join(f"{x.anomaly_id} ({x.severity} {_words(x.anomaly_type)})" for x in flagged)
                     + " was raised on that same action: in this kit an anomaly row is not explained by the reading's value.")
        text += " The kit gives no engineering limits; 'usual for this fleet' is not the same as 'safe'."
        return self._pack("reading_normal", text, Evidence(assets=[asset_id], readings=[p["reading_id"] for p in wanted["points"]]),
                          {"kind": "asset", "id": asset_id})

    def _replay(self) -> dict:
        kit = self.spot.kit
        issues = [i for i in kit.issues if i.code.startswith("replay") or i.code == "missions_not_in_replay"]
        text = (f"The replay stream has {len(kit.replay)} events (mission started and completed, action started, completed and failed). "
                "Played in order it gives a live picture the tables do not: which mission is running, where the robot is, each failure "
                "and anomaly at the moment it happens, and a mission ending COMPLETED with failures inside. "
                + " ".join(i.message for i in issues))
        return self._pack("replay", text, Evidence(missions=[x for i in issues if i.code == "missions_not_in_replay" for x in i.ids]),
                          {"kind": "validation", "id": "validation"})

    def _overview(self) -> dict:
        kpis = {k["key"]: k for k in self.spot.overview()["kpis"]}
        a, h = kpis["action_success"], kpis["hidden_failures"]
        top = self.spot.next_up[0] if self.spot.next_up else None
        text = (f"I answer from the inspection tables on this box. In short: action success is {pct(a['rate'])} (n={a['n']}); {h['k']} of {h['n']} "
                f"COMPLETED missions contain a failed action. " + (f"First thing to check: {top['title']}. " if top else "")
                + "Try: what needs attention, is V2 better, what happened at WP05, status of HX-202, summarize the last mission, "
                  "does battery matter, show hidden failures.")
        return self._pack("overview", text, Evidence(), None)

    # --- rephrasing ---

    def can_rephrase(self, d: dict) -> bool:
        """True when a model is configured and this draft is short enough to rewrite inside the budget."""
        return bool(self.rephrase and self.runtime is not None and d["intent"] not in NEVER_REPHRASED
                    and len(d["text"]) <= REPHRASE_MAX_DRAFT_CHARS)

    async def ask(self, text: str, rephrase: bool = False) -> dict:
        """The code-built answer, at once. With `rephrase=True` a short draft is also offered to the model (bounded)."""
        d = self.draft(text)
        can = self.can_rephrase(d)
        # `matched` is False when no intent recognised the question (the overview stands in): the screen then asks the box instead
        out = {**d, "matched": d["intent"] != "overview", "question": text, "draft": d["text"], "rephrased": False,
               "model": "none (built from the tables by code)", "check": None, "can_rephrase": can, "suggested": SUGGESTED}
        if not rephrase or not can:
            return out
        try:
            result = await asyncio.wait_for(asyncio.to_thread(
                self.runtime.chat, [ChatMessage(role="system", content=SYSTEM), ChatMessage(role="user", content=d["text"])],
                None, "summary"), REPHRASE_TIMEOUT_S)
        except Exception as exc:  # timeout, server down, anything: the draft stands
            out["check"] = f"model not used: {type(exc).__name__}"
            return out
        candidate = (result.content or "").strip()
        ok, why = grounded(candidate, d["text"])
        out["check"] = why
        if ok:
            out.update(text=candidate, rephrased=True, model=result.model_version)
        return out
