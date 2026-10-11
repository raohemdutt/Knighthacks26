"""Read side of the inspection context: the kit, its analysis (computed once), and the shapes the screen asks for.

    spot = Spot(load_kit())        # Spot(None) when there is no kit: every method still answers, `configured` False
    spot.overview(); spot.findings; spot.triage; spot.mission("M017"); spot.asset("HX-202"); spot.locations(); spot.handoff()

Nothing here publishes events or calls a model.
"""

import json
import math
from functools import cached_property
from pathlib import Path

from contexts.spot.domain import analysis, scorer, triage as triage_mod
from contexts.spot.domain.analysis import HEADLINES, Evidence, Finding, _words
from contexts.spot.domain.models import Kit
from contexts.spot.domain.stats import MIN_N, pct, wilson
from contexts.spot.infrastructure.kit import HOW_TO

# From the kit's legend image (04_ASSETS/waypoint_reference_legend.png), typed out by us: the one thing it adds to locations.csv.
ZONE_TEXT = {"ACCESS": "Entrances, corridors, general access areas", "ELECTRICAL": "Panels, batteries, electrical equipment",
             "PROCESS": "Piping, valves, process systems", "MECHANICAL": "Pumps, compressors, turbines, mechanical systems"}
# Which of the kit's sample pictures illustrates which kind of action. They are examples, not tied to any action id.
SAMPLE_IMAGE = {"GAUGE_READING": "gauge_pressure_01.jpg", "THERMAL_INSPECTION": "thermal_pump_bearing_01.jpg",
                "ACOUSTIC_INSPECTION": "acoustic_valve_01.jpg", "VISUAL_INSPECTION": "visual_pump_motor_01.jpg",
                "NAVIGATION": "pipe_gallery_01.jpg"}
PLAN_NAMES = ("fictional_floor_plan.png",)
ASSET_URL = "/api/spot/kit/assets/"


def not_configured() -> dict:
    return {"configured": False, "note": HOW_TO}


def affine_fit(pairs: list[tuple[tuple[float, float], tuple[float, float]]]) -> dict | None:
    """Least-squares affine map from (x, y) to (u, v) and how badly it misses. None with fewer than four pairs."""
    if len(pairs) < 4:
        return None

    def solve(target: int) -> list[float] | None:
        # normal equations for u = a*x + b*y + c
        m = [[0.0] * 4 for _ in range(3)]
        for (x, y), uv in pairs:
            row = (x, y, 1.0)
            for i in range(3):
                for j in range(3):
                    m[i][j] += row[i] * row[j]
                m[i][3] += row[i] * uv[target]
        for i in range(3):  # Gauss-Jordan
            pivot = max(range(i, 3), key=lambda r: abs(m[r][i]))
            if abs(m[pivot][i]) < 1e-12:
                return None
            m[i], m[pivot] = m[pivot], m[i]
            m[i] = [v / m[i][i] for v in m[i]]
            for r in range(3):
                if r != i:
                    m[r] = [a - m[r][i] * b for a, b in zip(m[r], m[i])]
        return [m[0][3], m[1][3], m[2][3]]

    cu, cv = solve(0), solve(1)
    if cu is None or cv is None:
        return None
    errors = [math.hypot(cu[0] * x + cu[1] * y + cu[2] - u, cv[0] * x + cv[1] * y + cv[2] - v) for (x, y), (u, v) in pairs]
    return {"n": len(pairs), "rms": math.sqrt(sum(e * e for e in errors) / len(errors)), "max": max(errors), "u": cu, "v": cv}


class Spot:
    def __init__(self, kit: Kit | None) -> None:
        self.kit = kit

    @property
    def configured(self) -> bool:
        return self.kit is not None

    # --- computed once ---

    @cached_property
    def findings(self) -> dict[str, Finding]:
        return {f.id: f for f in analysis.all_findings(self.kit)} if self.kit else {}

    @cached_property
    def model(self) -> dict | None:
        """The anomaly scorer's saved weights (fixtures/spot/anomaly_scorer.json, written by `python -m contexts.spot.train`)."""
        if self.kit is None:
            return None
        path = Path(self.kit.path).parent / "anomaly_scorer.json"
        try:
            model = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        except ValueError:
            return None
        return model if model and model.get("features") == list(scorer.FEATURES) else None

    @cached_property
    def scores(self) -> dict[str, dict]:
        return {r["action_id"]: r for r in scorer.score_all(self.kit, self.model)} if self.model else {}

    @cached_property
    def triage(self) -> list[dict]:
        """The ranked queue. The model never raises or lowers a score: it only orders items whose scores are equal, and
        each asset item shows the model's read of the latest action on that asset."""
        if not self.kit:
            return []
        items = triage_mod.triage(self.kit)
        for item in items:
            item["model"] = None
            if item["kind"] == "asset" and self.scores:
                last = next((a for a in reversed(self.kit.actions) if a.asset_id == item["id"]), None)
                if last:
                    row = self.scores[last.action_id]
                    item["model"] = {"action_id": last.action_id, "score": row["score"], "flagged": row["flagged"], "why": row["why"]}
        items.sort(key=lambda i: (-i["score"], -(i["model"]["score"] if i["model"] else 0), i["rank"]))
        for rank, item in enumerate(items, 1):
            item["rank"] = rank
        return items

    def model_view(self) -> dict:
        kit, model = self.kit, self.model
        if kit is None:
            return {**not_configured(), "available": False}
        if model is None:
            return {"configured": True, "available": False,
                    "note": "No trained weights next to the kit. Run: python -m contexts.spot.train"}
        quiet = []  # flagged by the model although the reading-threshold rule calls the action normal
        for row in self.scores.values():
            if row["flagged"] and not row["outside_band"]:
                a = kit.action[row["action_id"]]
                quiet.append({**row, "asset_id": a.asset_id, "location_id": a.location_id, "action_type": a.action_type,
                              "failure_type": a.failure_type, "timestamp": a.timestamp.isoformat()})
        quiet.sort(key=lambda r: (r["timestamp"], r["score"]), reverse=True)
        weights = sorted(({"feature": f, "weight": w, "why": scorer.PLAIN.get(f, f)} for f, w in zip(model["features"], model["weights"])),
                         key=lambda x: -abs(x["weight"]))
        return {"configured": True, "available": True, "name": model["name"], "kind": model["kind"], "verdict": model["verdict"],
                "evaluation": model["evaluation"], "threshold": model["threshold"], "trained_on": model["trained_on"], "weights": weights,
                "quiet_flags": quiet[:15], "quiet_total": len(quiet), "quiet_without_anomaly_row": sum(1 for r in quiet if not r["has_anomaly_row"]),
                "use": "Orders ties in the triage queue and explains a flag with its top features. It never raises an alert by itself.",
                "reproduce": "python -m contexts.spot.train   (deterministic; --check compares with the committed weights)"}

    @cached_property
    def uninspected(self) -> list[dict]:
        return triage_mod.uninspected_risk(self.kit) if self.kit else []

    @cached_property
    def next_up(self) -> list[dict]:
        return triage_mod.investigate_next(self.kit, self.findings) if self.kit else []

    @cached_property
    def validation(self) -> dict:
        return analysis.validation(self.kit) if self.kit else {}

    @cached_property
    def assumptions(self) -> list[dict]:
        return analysis.assumptions(self.kit) if self.kit else []

    @cached_property
    def plan(self) -> dict | None:
        """The kit's floor-plan image with where each waypoint's pin is on it, when both are there."""
        kit = self.kit
        if kit is None or not kit.assets_dir:
            return None
        name = next((n for n in PLAN_NAMES if n in kit.asset_files), None)
        pixels_file = Path(kit.path).parent / "waypoint_pixels.json"
        if name is None or not pixels_file.is_file():
            return None
        try:
            px = json.loads(pixels_file.read_text(encoding="utf-8"))
        except ValueError:
            return None
        if px.get("image") != name:
            return None
        points = {k: v for k, v in (px.get("waypoints") or {}).items() if k in kit.location}
        small = Path(kit.path).parent / "floor_plan.jpg"
        fits = {}
        for label, locs in (("all", kit.locations), *((f"floor {f}", [x for x in kit.locations if x.floor == f])
                                                      for f in sorted({x.floor for x in kit.locations}))):
            fit = affine_fit([((x.x_coord, x.y_coord), (points[x.location_id]["x"] * px["width"], points[x.location_id]["y"] * px["height"]))
                              for x in locs if x.location_id in points])
            if fit:
                fits[label] = {"n": fit["n"], "rms_px": round(fit["rms"], 1), "max_px": round(fit["max"], 1)}
        worst = max((f["rms_px"] for f in fits.values()), default=None)
        return {"image": name, "image_url": "/api/spot/plan/image" if small.is_file() else ASSET_URL + name,
                "width": px.get("width"), "height": px.get("height"), "panels": px.get("panels", {}), "waypoints": points,
                "fit": {"by_group": fits, "follows_csv": bool(worst is not None and worst < 0.03 * (px.get("width") or 1)),
                        "note": ("Pin positions were read off the image by eye (about 6 px). A least-squares affine map from locations.csv "
                                 f"x/y to the pins misses by {fits.get('all', {}).get('rms_px', 'n/a')} px RMS on a {px.get('width')} px wide "
                                 "image, so the CSV coordinates do not describe this picture; markers on the plan use the pin positions, "
                                 "and the schematic view uses the CSV coordinates as given.")},
                "source": "04_ASSETS/fictional_floor_plan.png (the kit); pin positions in fixtures/spot/waypoint_pixels.json (ours)"}

    @cached_property
    def walkable(self):
        """The walkable map traced from the plan picture (fixtures/spot/floor_walkable.json), when the plan and the trace are both there."""
        from contexts.spot.domain.walkable import Walkable

        plan = self.plan
        trace_file = Path(self.kit.path).parent / "floor_walkable.json" if self.kit is not None else None
        if plan is None or trace_file is None or not trace_file.is_file():
            return None
        try:
            trace = json.loads(trace_file.read_text(encoding="utf-8"))
            if trace.get("image") != plan["image"]:
                return None
            return Walkable.of(trace, {k: {"floor": self.kit.location[k].floor, "x": v["x"], "y": v["y"]} for k, v in plan["waypoints"].items()})
        except (ValueError, KeyError, TypeError):
            return None

    def sample_image(self, action_type: str) -> dict | None:
        name = SAMPLE_IMAGE.get(action_type)
        path = f"sample_inspection_images/{name}"
        if self.kit is None or name is None or path not in self.kit.asset_files:
            return None
        return {"url": ASSET_URL + path, "label": f"Sample image from the kit ({_words(action_type)}); not from this action", "name": name}

    # --- shapes ---

    def overview(self) -> dict:
        kit = self.kit
        if kit is None:
            return not_configured()
        first, last = kit.missions[0], kit.missions[-1]
        headlines = [{"id": f.id, "title": f.title, "question": f.question, "sentence": f.sentence, "caveat": f.caveat, "n": f.n,
                      "enough": f.enough, "evidence_count": f.evidence.count} for f in (self.findings[i] for i in HEADLINES if i in self.findings)]
        return {"configured": True, "kit_path": kit.path, "synthetic": True, "now": kit.now.isoformat() if kit.now else None,
                "period": {"from": first.start_time.isoformat(), "to": last.end_time.isoformat()},
                "counts": {"missions": len(kit.missions), "actions": len(kit.actions), "readings": len(kit.readings),
                           "anomalies": len(kit.anomalies), "locations": len(kit.locations), "assets": len(kit.assets),
                           "replay_events": len(kit.replay)},
                "kpis": analysis.kpis(kit), "next": self.next_up, "headlines": headlines,
                "latest_mission": self.mission_summary(last.mission_id),
                "validation": {"summary": self.validation["summary"], "agree": self.validation["agree"], "compared": self.validation["compared"]},
                "issues": {"warn": sum(i.level == "warn" for i in kit.issues), "error": sum(i.level == "error" for i in kit.issues),
                           "info": sum(i.level == "info" for i in kit.issues)},
                "assets": {"files": kit.asset_files, "floor_plan": bool(self.plan)}, "min_n": MIN_N,
                # so an id cited anywhere on the screen can open the right panel
                "index": {"action_mission": {x.action_id: x.mission_id for x in kit.actions},
                          "anomaly_asset": {x.anomaly_id: x.asset_id for x in kit.anomalies}},
                "samples": {t: self.sample_image(t) for t in SAMPLE_IMAGE}}

    def mission_summary(self, mission_id: str) -> dict:
        kit = self.kit
        m = kit.mission[mission_id]
        acts = kit.actions_of.get(mission_id, [])
        failed = [a for a in acts if a.failed]
        anomalies = kit.anomalies_in(mission_id)
        hidden = m.mission_status == "COMPLETED" and bool(failed)
        badge = (f"{m.mission_status.lower()}, {len(failed)} action{'s' if len(failed) != 1 else ''} failed" if failed else m.mission_status.lower())
        return {"mission_id": m.mission_id, "mission_name": m.mission_name, "scenario": m.scenario, "configuration_version": m.configuration_version,
                "start_time": m.start_time.isoformat(), "end_time": m.end_time.isoformat(), "duration_min": m.duration_min,
                "start_battery_pct": m.start_battery_pct, "end_battery_pct": m.end_battery_pct, "mission_status": m.mission_status,
                "actions": len(acts), "failed": len(failed), "anomalies": len(anomalies),
                "worst_severity": max((x.severity for x in anomalies), key=analysis.SEV_RANK.get, default=None),
                "hidden_failure": hidden, "badge": badge}

    def missions(self) -> list[dict]:
        return [self.mission_summary(m.mission_id) for m in self.kit.missions] if self.kit else []

    def _action_row(self, a) -> dict:
        kit = self.kit
        return {"action_id": a.action_id, "mission_id": a.mission_id, "sequence": a.sequence, "timestamp": a.timestamp.isoformat(),
                "location_id": a.location_id, "asset_id": a.asset_id, "action_type": a.action_type, "battery_pct": a.battery_pct,
                "status": a.status, "failure_type": a.failure_type, "duration_sec": a.duration_sec,
                "readings": [{"reading_id": r.reading_id, "measurement_type": r.measurement_type, "value": r.value, "unit": r.unit,
                              "quality": r.quality} for r in kit.readings_of.get(a.action_id, [])],
                "anomalies": [{"anomaly_id": x.anomaly_id, "anomaly_type": x.anomaly_type, "severity": x.severity,
                               "description": x.description, "confirmed": not a.failed} for x in kit.anomalies_of.get(a.action_id, [])]}

    def mission(self, mission_id: str) -> dict | None:
        if self.kit is None or mission_id not in self.kit.mission:
            return None
        return {**self.mission_summary(mission_id), "timeline": [self._action_row(a) for a in self.kit.actions_of.get(mission_id, [])]}

    def locations(self) -> dict:
        kit = self.kit
        if kit is None:
            return {**not_configured(), "locations": [], "plan": None}
        overall = (sum(a.failed for a in kit.actions), len(kit.actions))
        out = []
        for loc in kit.locations:
            acts = [a for a in kit.actions if a.location_id == loc.location_id]
            k, n = sum(a.failed for a in acts), len(acts)
            ci = wilson(k, n)
            by_type = {}
            for a in acts:
                t = by_type.setdefault(a.action_type, {"k": 0, "n": 0})
                t["n"] += 1
                t["k"] += int(a.failed)
            anomalies = [x for x in kit.anomalies if x.location_id == loc.location_id]
            out.append({"location_id": loc.location_id, "location_name": loc.location_name, "zone_type": loc.zone_type,
                        "zone_text": ZONE_TEXT.get(loc.zone_type), "floor": loc.floor, "x": loc.x_coord, "y": loc.y_coord,
                        "environment": loc.environment, "k": k, "n": n, "rate": k / n if n else None, "lo": ci[0] if ci else None,
                        "hi": ci[1] if ci else None, "enough": n >= MIN_N, "by_type": by_type,
                        "health": ("unknown" if n < MIN_N else "bad" if ci and ci[0] > overall[0] / overall[1] else
                                   "watch" if k / n > overall[0] / overall[1] else "ok"),
                        "failed_actions": [self._action_row(a) for a in acts if a.failed],
                        "anomalies": [{"anomaly_id": x.anomaly_id, "severity": x.severity, "anomaly_type": x.anomaly_type, "asset_id": x.asset_id,
                                       "mission_id": x.mission_id, "timestamp": x.timestamp.isoformat(),
                                       "confirmed": analysis._confirmed(kit, x)} for x in anomalies],
                        "assets": [a.asset_id for a in kit.assets if a.location_id == loc.location_id],
                        "first_stop_of": sum(1 for a in acts if a.sequence == 1)})
        return {"configured": True, "locations": out, "plan": self.plan, "overall": {"k": overall[0], "n": overall[1]},
                "health_rule": f"bad: the bottom of the 95% interval is above the site-wide failure rate ({pct(overall[0] / overall[1])}); "
                               f"watch: the rate is above it; unknown: under {MIN_N} actions.",
                "zone_text_source": "the kit's legend image (04_ASSETS/waypoint_reference_legend.png)"}

    def assets(self) -> list[dict]:
        kit = self.kit
        if kit is None:
            return []
        ranks = {i["id"]: i for i in self.triage if i["kind"] == "asset"}
        risk = {r["asset_id"]: r for r in self.uninspected}
        out = []
        for a in kit.assets:
            acts = [x for x in kit.actions if x.asset_id == a.asset_id]
            anomalies = [x for x in kit.anomalies if x.asset_id == a.asset_id]
            k, n = sum(x.failed for x in acts), len(acts)
            ci = wilson(k, n)
            out.append({"asset_id": a.asset_id, "asset_name": a.asset_name, "asset_type": a.asset_type, "location_id": a.location_id,
                        "criticality": a.criticality, "k": k, "n": n, "rate": k / n if n else None, "lo": ci[0] if ci else None,
                        "hi": ci[1] if ci else None, "enough": n >= MIN_N, "anomalies": len(anomalies),
                        "worst_severity": max((x.severity for x in anomalies), key=analysis.SEV_RANK.get, default=None),
                        "triage_rank": ranks[a.asset_id]["rank"] if a.asset_id in ranks else None,
                        "triage_score": ranks[a.asset_id]["score"] if a.asset_id in ranks else None,
                        "at_risk": risk[a.asset_id]["at_risk"], "hours_since_good": risk[a.asset_id]["hours_since_good"]})
        return out

    def asset(self, asset_id: str) -> dict | None:
        kit = self.kit
        if kit is None or asset_id not in kit.asset:
            return None
        base = next(a for a in self.assets() if a["asset_id"] == asset_id)
        acts = [x for x in kit.actions if x.asset_id == asset_id]
        trends = self.findings["reading_trends"].extra
        series = [s for s in trends["series"] if s["asset_id"] == asset_id]
        types = sorted({a.action_type for a in acts if a.is_inspection}, key=lambda t: -sum(1 for a in acts if a.action_type == t))
        home = kit.location.get(base["location_id"])
        return {**base, "location": {"location_id": home.location_id, "location_name": home.location_name, "floor": home.floor} if home else None,
                "inspection": triage_mod.inspection_state(kit, asset_id),
                "triage": next((i for i in self.triage if i["id"] == asset_id), None),
                "series": [{**s, "band": trends["bands"].get(s["measurement_type"])} for s in series],
                "anomalies_list": [{"anomaly_id": x.anomaly_id, "severity": x.severity, "anomaly_type": x.anomaly_type, "mission_id": x.mission_id,
                                    "action_id": x.action_id, "location_id": x.location_id, "timestamp": x.timestamp.isoformat(),
                                    "description": x.description, "confirmed": analysis._confirmed(kit, x)}
                                   for x in kit.anomalies if x.asset_id == asset_id],
                "actions": [self._action_row(a) for a in acts],
                "seen_at": sorted({a.location_id for a in acts}),
                "samples": [s for s in (self.sample_image(t) for t in types[:2]) if s]}

    def handoff(self) -> dict:
        """Five to eight lines for the next shift, each with the rows behind it."""
        kit = self.kit
        if kit is None:
            return {**not_configured(), "lines": [], "text": ""}
        f = self.findings
        last = self.mission_summary(kit.missions[-1].mission_id)
        lines: list[dict] = []

        def add(text: str, evidence: dict | Evidence, target: dict | None = None) -> None:
            ev = evidence.model_dump() if isinstance(evidence, Evidence) else evidence
            lines.append({"text": text, "evidence": ev, "target": target})

        failed_last = [a.action_id for a in kit.failed_in(last["mission_id"])]
        add(f"Latest mission {last['mission_id']} ({last['configuration_version']}) ended {last['mission_status']} with {last['failed']} of "
            f"{last['actions']} actions failed and {last['anomalies']} anomaly row(s); battery {last['start_battery_pct']:.0f}% to "
            f"{last['end_battery_pct']:.0f}%.",
            Evidence(missions=[last["mission_id"]], actions=failed_last, anomalies=[x.anomaly_id for x in kit.anomalies_in(last["mission_id"])]),
            {"kind": "mission", "id": last["mission_id"]})
        for item in [i for i in self.triage if i["kind"] == "asset"][:2]:
            add(f"{item['id']} (score {item['score']:g}): {item['next_step']['text']}", item["evidence"], {"kind": "asset", "id": item["id"]})
        gap = next((r for r in self.uninspected if r["at_risk"] and all(r["asset_id"] != ln["target"]["id"] for ln in lines if ln["target"])), None)
        if gap:
            add(f"Coverage gap: {gap['text']}", gap["evidence"], {"kind": "asset", "id": gap["asset_id"]})
        hf = f["hidden_failures"]
        add(f"Do not trust mission status alone: {hf.sentence.split(';')[0]}.", hf.evidence, {"kind": "finding", "id": hf.id})
        vc = f["version_comparison"]
        add(vc.sentence.split(". Mission completion")[0] + ". Cause not established: battery and configuration changed together.",
            Evidence(missions=vc.evidence.missions), {"kind": "finding", "id": vc.id})
        bt = f["battery"]
        if bt.enough:
            add(f"Watch battery: {bt.sentence.split(':')[0]}.", bt.evidence, {"kind": "finding", "id": bt.id})
        warn = [i for i in kit.issues if i.level in ("warn", "error")]
        add(f"Data: {self.validation['summary']}" + (f" {len(warn)} thing(s) to know about the files: " + " ".join(i.message for i in warn[:2]) if warn else ""),
            Evidence(missions=[x for i in warn if i.code == "missions_not_in_replay" for x in i.ids]), {"kind": "validation", "id": "validation"})
        lines = lines[:8]
        stamp = kit.now.isoformat(sep=" ", timespec="minutes") if kit.now else ""
        text = f"SPOT shift handoff, data to {stamp}\n" + "\n".join(f"{i}. {ln['text']}" for i, ln in enumerate(lines, 1))
        return {"configured": True, "as_of": kit.now.isoformat() if kit.now else None, "lines": lines, "text": text}
