"""Build a fine-tune dataset for the language model from the kit: operator questions with grounded, cited answers.

    python -m contexts.spot.dataset --out <dir>            # writes <dir>/spot-qa-train/ and <dir>/spot-qa-heldout/
    python -m contexts.spot.dataset --out <dir> --sample 20   # also writes <dir>/spot-qa-sample/ (what fixtures/spot/dataset-sample is)

Format: the box's export layout (contracts/learning.py), the one lab/dataset.py reads and lab/fixture_dataset.py
writes for the text-only tool-calling set. Each folder holds

    manifest.json    DatasetManifest. task "spot-qa"; instruction = the system prompt; output_schema = the label's JSON
                     schema; action_id "fixture:spot-kit" (nobody approved an export: this is built from the sponsor's
                     synthetic kit, not from anything the box recorded)
    examples.jsonl   one DatasetExample per line: image "" (text-only), frame_ref "spot:group:<group>" (every wording of
                     one question shares a group), ts = the newest data time the answer may use,
                     context = the question plus the evidence a tool would hand the model (JSON),
                     target = the label: {"answer": ..., "citations": [row ids]}
    expected.jsonl   per example: question, intent, group, anchor mission, split, the label, the cited ids
    images/          empty

Unlike the tool-calling fixture, `target` is filled: the answers are written by code from the tables (the same
intent layer the app's Ask uses, contexts/spot/application/assistant.py), so no teacher model is needed. A teacher
may still rewrite them; expected.jsonl keeps the reference.

Every example is "as of" an anchor mission: the kit is cut off after that mission and the question is answered from
what was known then. The split is by anchor mission in time order: train anchors come first, held-out anchors last,
with one mission left unused between them. No held-out anchor mission, and no row newer than the last train anchor,
is ever in the train set. (Held-out answers do draw on earlier missions as history; that is what "as of" means.)

Deterministic: nothing reads the clock or a random number; wordings are picked by position. This command only
writes files. It does not train anything and does not contact the lab.
"""

import argparse
import json
import sys
from datetime import timezone
from pathlib import Path

from contexts.spot.application.assistant import ID, Assistant
from contexts.spot.application.service import Spot
from contexts.spot.domain.models import Kit
from contexts.spot.infrastructure.kit import kit_path, load_kit
from contracts.learning import DatasetExample, DatasetManifest, TaskSpec

TASK = "spot-qa"
FIRST_ANCHOR = 6  # fewer missions than this and most rates are "not enough data"
HELDOUT_ANCHORS = 6  # the last six missions
GAP = 1
WORDINGS = 2  # wordings written per question
INSTRUCTION = ("You are Atlas, the on-site assistant for a plant operator who supervises a SPOT inspection robot. Answer from the evidence "
               "given and nothing else. Keep every number with its count (\"11 of 43\"), name the rows you rely on by id, say \"not enough "
               "data\" when the evidence says so, and never claim a cause the evidence does not show. Reply as JSON: "
               "{\"answer\": \"...\", \"citations\": [\"<row id>\", ...]}.")
SCHEMA = {"type": "object", "properties": {"answer": {"type": "string", "minLength": 1},
                                           "citations": {"type": "array", "items": {"type": "string"}}},
          "required": ["answer", "citations"], "additionalProperties": False}

Q = {
    "mission": ["Summarize mission {m}.", "What happened in {m}?", "Give me the rundown on {m}.", "How did {m} go?"],
    "waypoint": ["What happened at {wp}?", "What is wrong at {wp}?", "How reliable is {wp}?", "Anything I should know about {wp}?"],
    "asset": ["Status of {asset}", "How is {asset} doing?", "What do we know about {asset}?", "Should I worry about {asset}?"],
    "reading_normal": ["Is the reading at {asset} normal?", "Is {asset} in its usual range?", "Is the latest {kind} reading on {asset} normal?"],
    "worse": ["Which assets got worse between mission {a} and {b}?", "What changed for the worse between {ma} and {mb}?",
              "What is new and bad between mission {a} and {b}?"],
    "triage": ["What needs attention now?", "What is the most urgent thing right now?", "Give me the triage list.", "What are the top priorities?"],
    "investigate_next": ["What should I check first this shift?", "What should the next shift re-inspect first and why?", "Where do I start?",
                         "What should I investigate next?"],
    "hidden_failures": ["Show hidden failures.", "Can a mission be COMPLETED with failed actions inside?", "Which completed missions hide failures?"],
    "reliability_by_type": ["Which inspection types are least reliable?", "Which action type fails most?", "What kind of inspection is the weakest?"],
    "battery": ["Does battery affect reliability?", "Is low battery a problem?", "Do actions fail more when the battery is low?"],
    "where": ["Where are failures concentrated?", "Which waypoint or floor fails most?", "Are failures tied to a location?"],
    "anomalies": ["Are anomalies hidden in successful missions?", "Tell me about the anomalies.", "Which anomalies matter most?"],
    "coverage": ["Which assets are overdue for inspection?", "What has not been inspected lately?", "Where is our coverage gap?"],
    "handoff": ["Write the shift handoff.", "Summarize the shift for the next operator.", "What do I hand to the next shift?"],
    "version_comparison": ["Is V2 better than V1?", "Did the new configuration improve things?", "Does performance differ between versions?"],
}


def truncate(kit: Kit, count: int) -> Kit:
    """The kit as it stood after its first `count` missions."""
    keep = {m.mission_id for m in kit.missions[:count]}
    return Kit(missions=[m for m in kit.missions if m.mission_id in keep], actions=[a for a in kit.actions if a.mission_id in keep],
               readings=[r for r in kit.readings if r.mission_id in keep], anomalies=[a for a in kit.anomalies if a.mission_id in keep],
               locations=kit.locations, assets=kit.assets, replay=[], path=kit.path)


def _evidence(spot: Spot, draft: dict) -> dict:
    """What a tool would hand the model for this question: the rows and counts the answer is written from, compact."""
    kit, intent, target = spot.kit, draft["intent"], draft.get("target") or {}
    if intent == "mission":
        m = spot.mission(target["id"])
        return {"mission": {k: m[k] for k in ("mission_id", "scenario", "configuration_version", "start_time", "duration_min", "start_battery_pct",
                                              "end_battery_pct", "mission_status")},
                "actions": [{"id": a["action_id"], "type": a["action_type"], "asset": a["asset_id"], "at": a["location_id"], "status": a["status"],
                             "failure": a["failure_type"], "anomalies": [f"{x['anomaly_id']} {x['severity']} {x['anomaly_type']}" for x in a["anomalies"]]}
                            for a in m["timeline"]]}
    if intent == "waypoint":
        loc = next(x for x in spot.locations()["locations"] if x["location_id"] == target["id"])
        return {"waypoint": {k: loc[k] for k in ("location_id", "location_name", "floor", "zone_type", "environment", "k", "n", "enough", "by_type",
                                                 "first_stop_of")},
                "interval": [round(loc["lo"], 3), round(loc["hi"], 3)] if loc["lo"] is not None else None,
                "failed": [{"id": a["action_id"], "mission": a["mission_id"], "type": a["action_type"], "failure": a["failure_type"]} for a in loc["failed_actions"]],
                "anomalies": [{"id": x["anomaly_id"], "severity": x["severity"], "asset": x["asset_id"]} for x in loc["anomalies"]]}
    if intent in ("asset", "reading_normal"):
        a = spot.asset(target["id"])
        out = {"asset": {k: a[k] for k in ("asset_id", "asset_name", "asset_type", "location_id", "criticality", "k", "n", "enough")},
               "inspection": {k: a["inspection"][k] for k in ("last_good_action", "last_good_mission", "hours_since_good", "failed_since")},
               "anomalies": [{"id": x["anomaly_id"], "severity": x["severity"], "type": x["anomaly_type"], "mission": x["mission_id"],
                              "confirmed": x["confirmed"]} for x in a["anomalies_list"]],
               "triage": {"rank": a["triage"]["rank"], "score": a["triage"]["score"], "next": a["triage"]["next_step"]["action"]} if a["triage"] else None}
        if intent == "reading_normal":
            cited = set(draft["citations"]["readings"])
            out["readings"] = [{"measurement": s["measurement_type"], "unit": s["unit"], "band": {k: round(v, 1) for k, v in s["band"].items() if k in ("lo", "hi", "median")} if s["band"] else None,
                                "points": [{"id": p["reading_id"], "value": p["value"], "good": p["good"], "mission": p["mission_id"]} for p in s["points"]]}
                               for s in a["series"] if cited & {p["reading_id"] for p in s["points"]}]
        return out
    if intent == "worse":
        return {"missions": {mid: [{"id": a.action_id, "asset": a.asset_id, "status": a.status,
                                    "anomalies": [x.anomaly_id for x in kit.anomalies_of.get(a.action_id, [])]} for a in kit.actions_of.get(mid, [])]
                             for mid in draft["citations"]["missions"]}}
    if intent in ("triage", "investigate_next", "coverage", "handoff"):
        out = {"triage": [{"rank": i["rank"], "id": i["id"], "score": i["score"], "criticality": i["criticality"], "severity": i["severity"],
                           "anomaly": i["anomaly_id"], "confirmed": i["confirmed"], "next": i["next_step"]["action"],
                           "parts": {p["label"]: p["points"] for p in i["parts"]}} for i in spot.triage[:5]],
               "uninspected": [{"asset": r["asset_id"], "criticality": r["criticality"], "hours_since_good": r["hours_since_good"],
                                "last_good": r["last_good_action"], "failed_since": r["failed_since"]} for r in spot.uninspected if r["at_risk"]]}
        if intent in ("investigate_next", "handoff"):
            b = spot.findings["battery"]
            out["patterns"] = {f.id: f.sentence for f in (b, spot.findings["hidden_failures"]) if f.enough}
        if intent == "handoff":
            out["latest_mission"] = spot.mission_summary(kit.missions[-1].mission_id)
        return out
    ids = {"where": ("reliability_by_location", "navigation_hotspot", "reliability_by_floor"), "anomalies": ("anomalies_hidden", "anomaly_matrix", "anomaly_repeats")}.get(intent, (intent,))
    out = {}
    for fid in ids:
        f = spot.findings.get(fid)
        if f is None:
            continue
        item = {"n": f.n, "enough": f.enough, "p_value": round(f.p_value, 4) if f.p_value is not None else None,
                "rows": [{"label": b.label, "k": b.k, "n": b.n, "enough": b.enough} for b in f.rows[:12]], "caveat": f.caveat}
        if fid == "version_comparison":
            item["metrics"] = [{"label": m["label"], "verdict": m["verdict"], "a": {k: m["a"].get(k) for k in ("k", "n", "value")},
                                "b": {k: m["b"].get(k) for k in ("k", "n", "value")}} for m in f.extra.get("metrics", [])]
        out[fid] = item
    return out


def _questions(kit: Kit, spot: Spot, anchor: int) -> list[tuple[str, str, list[str]]]:
    """(intent, subject, wordings) for one anchor mission. The subject keeps every wording of a question in one group."""
    m = kit.missions[anchor - 1]
    prev = kit.missions[anchor - 2]
    acts = kit.actions_of[m.mission_id]
    by_weight = sorted(acts, key=lambda a: (-(a.failed + len(kit.anomalies_of.get(a.action_id, []))), a.sequence))
    wps = list(dict.fromkeys(a.location_id for a in by_weight))[:3]
    assets = list(dict.fromkeys(a.asset_id for a in by_weight))[:3]
    readings = [(r.asset_id, r.measurement_type) for a in acts for r in kit.readings_of.get(a.action_id, []) if a.is_inspection][:2]
    kind = {"TEMPERATURE_C": "temperature", "SOUND_DB": "sound", "PRESSURE_PSI": "pressure", "VISUAL_SCORE": "visual"}
    num = lambda x: str(int(x.mission_id[1:]))  # noqa: E731
    out = [("mission", m.mission_id, [q.format(m=m.mission_id) for q in Q["mission"]])]
    out += [("waypoint", wp, [q.format(wp=wp) for q in Q["waypoint"]]) for wp in wps]
    out += [("asset", a, [q.format(asset=a) for q in Q["asset"]]) for a in assets]
    out += [("reading_normal", f"{a}-{t}", [Q["reading_normal"][2].format(asset=a, kind=kind.get(t, "sensor")), f"Is the {kind.get(t, 'sensor')} at {a} normal?",
                                            f"Is {a} in its usual {kind.get(t, 'sensor')} range?"]) for a, t in readings]
    out.append(("worse", f"{prev.mission_id}-{m.mission_id}", [q.format(a=num(prev), b=num(m), ma=prev.mission_id, mb=m.mission_id) for q in Q["worse"]]))
    for intent in ("triage", "investigate_next", "hidden_failures", "reliability_by_type", "battery", "where", "anomalies", "coverage", "handoff"):
        out.append((intent, "site", Q[intent]))
    if len({x.configuration_version for x in kit.missions}) == 2:
        out.append(("version_comparison", "site", Q["version_comparison"]))
    return out


def build(full: Kit) -> dict[str, list[dict]]:
    """{"train": [...], "heldout": [...]}: each item has the DatasetExample fields plus what expected.jsonl needs."""
    total = len(full.missions)
    held = list(range(total - HELDOUT_ANCHORS + 1, total + 1))
    train = list(range(FIRST_ANCHOR, held[0] - GAP))
    out: dict[str, list[dict]] = {"train": [], "heldout": []}
    for split, anchors in (("train", train), ("heldout", held)):
        for anchor in anchors:
            kit = truncate(full, anchor)
            spot = Spot(kit)
            assistant = Assistant(spot, None, rephrase=False)
            mission = kit.missions[-1]
            for n, (intent, subject, wordings) in enumerate(_questions(kit, spot, anchor)):
                group = f"{intent}-{mission.mission_id}-{subject}"
                for w in range(WORDINGS):
                    question = wordings[(anchor + n + w) % len(wordings)]
                    draft = assistant.draft(question)
                    if draft["intent"] != intent:  # the wording did not reach the intended answer: a bug in the templates, not data
                        raise ValueError(f"{question!r} was answered as {draft['intent']}, expected {intent}")
                    cited = sorted({i for v in draft["citations"].values() for i in v} | set(ID.findall(draft["text"])) - {"V1", "V2"})
                    label = {"answer": draft["text"], "citations": cited}
                    context = (f"Data to {kit.now.isoformat(sep=' ', timespec='minutes')} (through mission {mission.mission_id}).\n"
                               f"Question: {question}\nEvidence: {json.dumps(_evidence(spot, draft), separators=(',', ':'), default=str)}")
                    out[split].append({"image": "", "frame_ref": f"spot:group:{group}", "ts": kit.now.replace(tzinfo=timezone.utc), "context": context,
                                       "target": json.dumps(label, ensure_ascii=False), "question": question, "intent": intent, "group": group,
                                       "anchor_mission": mission.mission_id, "split": split, "label": label})
    for split, items in out.items():
        for n, item in enumerate(items, 1):
            item["id"] = f"spot-{split}-{n:04d}"
    return out


def write(folder: Path, dataset_id: str, items: list[dict], created) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "images").mkdir(exist_ok=True)
    (folder / "images" / ".gitkeep").write_text("")
    manifest = DatasetManifest(dataset_id=dataset_id, spec=TaskSpec(task=TASK, instruction=INSTRUCTION, output_schema=SCHEMA), created=created,
                               examples=len(items), action_id="fixture:spot-kit")
    (folder / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    with (folder / "examples.jsonl").open("w", encoding="utf-8") as f:
        for it in items:
            f.write(DatasetExample(id=it["id"], image="", frame_ref=it["frame_ref"], ts=it["ts"], context=it["context"], target=it["target"]).model_dump_json() + "\n")
    with (folder / "expected.jsonl").open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({k: it[k] for k in ("id", "question", "intent", "group", "anchor_mission", "split", "label")}, ensure_ascii=False) + "\n")
    return {"dataset_id": dataset_id, "folder": str(folder), "examples": len(items), "intents": len({i["intent"] for i in items}),
            "anchors": sorted({i["anchor_mission"] for i in items})}


def sample(data: dict[str, list[dict]], count: int) -> list[dict]:
    """A small spread for a fixture: one example per intent first (from the held-out side), then more from train."""
    picked, seen = [], set()
    for it in data["heldout"] + data["train"]:
        if it["intent"] not in seen:
            seen.add(it["intent"])
            picked.append(it)
    step = max(1, len(data["train"]) // max(1, count))
    for it in data["train"][::step]:
        if len(picked) >= count:
            break
        if it not in picked:
            picked.append(it)
    return picked[:count]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="folder to write spot-qa-train/ and spot-qa-heldout/ into")
    ap.add_argument("--kit", default=None)
    ap.add_argument("--sample", type=int, default=0, help="also write spot-qa-sample/ with this many examples")
    args = ap.parse_args(argv)
    kit = load_kit(args.kit)
    if kit is None:
        print(f"no SPOT kit at {kit_path(args.kit)}", file=sys.stderr)
        return 2
    data = build(kit)
    created = kit.now.replace(tzinfo=timezone.utc)
    out = Path(args.out)
    report = [write(out / "spot-qa-train", "spot-qa-train", data["train"], created), write(out / "spot-qa-heldout", "spot-qa-heldout", data["heldout"], created)]
    if args.sample:
        report.append(write(out / "spot-qa-sample", "spot-qa-sample", sample(data, args.sample), created))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
