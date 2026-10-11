"""The under-the-hood report: one HTML page and a JSON data pack that show the lab's optimization work.

    python -m lab report                      # writes lab/refine/report/under-the-hood.html and .json
    python -m lab report --out DIR            # somewhere else
    python -m lab report --root DIR           # read the committed files of another checkout (tests use a fixture)

Built only from committed files, never from a pod, the hub or the network:

- lab/refine/metrics.jsonl, the lab's measured rows (each number keeps its line);
- the chip sheet lab/refine/research/kernels/data/chipsheet-gfx1151-<date>.json (the newest measured one);
- the decode profile decode-profile-gfx1151-<date>.json in the same folder, when one is committed;
- the public prior: lab/public_prior.py fitted on public-strix-halo-benchmarks.csv, always labelled a prior;
- any lab ledger copy (ledger*.jsonl) in the same folder, for the experiments that were tried and dropped.

The page has six sections, each a chart drawn here as inline SVG and one sentence a non-expert can say aloud:
(1) the ceiling the chip allows against what each build achieves at batch 1, (2) the batching frontier,
joules per token against concurrent streams, (3) where a decode step's time goes, (4) engines side by side,
(5) the fine-tune run, (6) what was tried and dropped. A section with no rows says "not measured" and names
the run that would fill it; nothing is invented or interpolated.

Every number in the pack is a dict: {id, label, value, unit, kind, source: {file, line, ref}, gpu, chip,
conditions, ts, low_confidence, reason, other_chip}. `kind` is "measured" (a row's own value), "derived"
(worked out here from other numbers, with `formula` and `inputs`), "stated" (an estimate a committed file states,
such as the teacher's weight bytes), "prior" (the public prior's prediction, never a measurement) or
"not measured" (value None, with `why`). `gpu` is the row's conditions.gpu, `chip` its conditions.chip; either may be None.
Rows from another chip (the R9700s, by lab/slide_numbers.py's own rule, which the Learning lab's bench reader
uses too) are never drawn with Strix Halo rows: they sit in each section's `other_chip` list and in a panel of
their own on the page.

The data pack's shape is documented in lab/README.md, section "Under the hood report".
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
from pathlib import Path
from typing import Any

FORMAT = "atlas.lab_report/1"
REPO = Path(__file__).resolve().parents[1]
METRICS = "lab/refine/metrics.jsonl"
DATA_DIR = "lab/refine/research/kernels/data"
OUT_DIR = "lab/refine/report"
HTML_NAME = "under-the-hood.html"
PACK_NAME = "under-the-hood.json"
CHIP = "gfx1151"
STUDENT_NODE = "jetson"
LAB_MOVED_AT = "2026-10-10T20:25:00Z"
NOT_MEASURED = "not measured"
TEACHER_TARGET = "teacher-27b-int4"
PRIOR_BACKENDS = ("vllm", "llama.cpp-rocm", "llama.cpp-vulkan")
STEP_BUCKETS = (("w4a16_matmul", "quantized matmul"), ("attention", "attention"), ("other_gpu", "other GPU work"),
                ("sampling", "sampling"), ("copies", "copies"), ("host_side", "host side (GPU idle)"))

# The runs whose rows a later build picks up, with what each adds. Listed in the pack under `awaited`.
AWAITED = (
    {"task": "R52", "what": "request-side settings sweep of the teacher (batch 1, 4, 8, 16)",
     "where": f"{DATA_DIR}/sweep-request-gfx1151-<date>.json, rows in {METRICS}", "sections": ["frontier"],
     "glob": "sweep-request-gfx1151-*.json"},
    {"task": "R58", "what": "llama.cpp Vulkan decode on the Strix Halo pod",
     "where": f"rows in {METRICS} whose conditions.engine names llama.cpp", "sections": ["ceiling", "engines"],
     "glob": None},
    {"task": "R60", "what": "vLLM main with the RDNA hybrid W4A16 decode kernel on the Strix Halo pod",
     "where": f"rows in {METRICS} with a new conditions.engine build", "sections": ["ceiling", "engines"],
     "glob": None},
    {"task": "R53", "what": "vLLM b1388b1 with the hybrid W4A16 kernel backported (lab/vllm-gfx1151.diff)",
     "where": f"rows in {METRICS} whose conditions.engine names the gfx1151 diff", "sections": ["ceiling", "engines"],
     "glob": None},
    {"task": "R47", "what": "decode profile: where one decode step's time goes",
     "where": f"{DATA_DIR}/decode-profile-gfx1151-<date>.json", "sections": ["step"],
     "glob": "decode-profile-gfx1151-*.json"},
    {"task": "L16", "what": "the first real fine-tune's step telemetry (loss and step time per step)",
     "where": f"rows in {METRICS} named telemetry.train.* or trainer.* with conditions.step", "sections": ["finetune"],
     "glob": None},
)


# --- reading the committed files -----------------------------------------------------------------------------------

def _ts(text: Any) -> str | None:
    return text if isinstance(text, str) and re.match(r"\d{4}-\d\d-\d\dT", text) else None


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def other_chip(row: dict[str, Any]) -> bool:
    """True for a pod row from the R9700s: tagged gfx1201 or R9700, or untagged and measured before the move.

    The rule is lab/slide_numbers.py's `_old_chip` (the one the Learning lab's bench reader imports); the Jetson
    is not one of the lab's cards, so its rows never are."""
    if row.get("node") == STUDENT_NODE:
        return False
    try:
        from lab.slide_numbers import _old_chip
    except ImportError:  # httpx missing: the same rule, restated
        c = row.get("conditions") or {}
        gpu = str(c.get("gpu", ""))
        if "gfx1151" in gpu or "Strix" in gpu:
            return False
        if "gfx1201" in gpu or "R9700" in gpu:
            return True
        return bool(_ts(row.get("ts"))) and str(row["ts"]) < LAB_MOVED_AT
    return bool(_old_chip(row))


def read_metrics(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The metrics file's rows, each with `_line` (from 1, blank lines counted), and the lines that are not rows."""
    if not path.is_file():
        return [], []
    rows, skipped = [], []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            skipped.append({"line": n, "problem": "not JSON"})
            continue
        if not isinstance(row, dict) or not isinstance(row.get("metric"), str) or _num(row.get("value")) is None:
            skipped.append({"line": n, "problem": "no metric name or no numeric value"})
            continue
        if not isinstance(row.get("conditions"), dict):
            row["conditions"] = {}
        row["_line"] = n
        rows.append(row)
    return rows, skipped


def line_of(path: Path, needle: str, start: int = 1) -> int | None:
    """The first line (from 1) at or after `start` that contains `needle`, or None."""
    if not path.is_file():
        return None
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if n >= start and needle in line:
            return n
    return None


def newest(folder: Path, pattern: str) -> Path | None:
    """The newest file by its dated name (names sort by date), or None."""
    found = sorted(p for p in folder.glob(pattern) if p.is_file())
    return found[-1] if found else None


def _rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


# --- numbers ------------------------------------------------------------------------------------------------------

def batch_of(row: dict[str, Any]) -> float | None:
    c = row["conditions"]
    b = _num(c.get("batch_size"))
    return b if b is not None else _num(c.get("batch"))


def build_of(c: dict[str, Any]) -> str:
    engine = c.get("engine") if isinstance(c.get("engine"), str) else None
    commit = c.get("engine_commit") if isinstance(c.get("engine_commit"), str) else None
    if engine and commit and re.fullmatch(r"[0-9a-f]{7,40}", commit):
        engine = f"{engine}@{commit[:7]}"
    quant = c.get("quantization") if isinstance(c.get("quantization"), str) else None
    return " ".join(x for x in (engine, quant) if x) or "engine not recorded"


def engine_family(c: dict[str, Any]) -> str:
    """vllm, llama.cpp, sglang or the engine's first word; "not recorded" when the row names none."""
    e = str(c.get("engine") or "").lower()
    for name in ("vllm", "llama.cpp", "sglang"):
        if name in e:
            return name
    return e.split("@")[0].split()[0] if e.strip() else "not recorded"


def from_row(row: dict[str, Any], label: str) -> dict[str, Any]:
    c = row["conditions"]
    low = bool(row.get("low_confidence"))
    return {
        "id": f"metrics:{row['_line']}", "label": label, "value": row["value"], "unit": row.get("unit"),
        "kind": "measured", "source": {"file": METRICS, "line": row["_line"], "ref": row.get("source")},
        "gpu": c.get("gpu"), "chip": c.get("chip"), "node": row.get("node"), "metric": row["metric"],
        "conditions": c, "ts": row.get("ts"), "low_confidence": low,
        "reason": c.get("low_confidence_reason") if low else None, "other_chip": other_chip(row),
    }


def derived(nid: str, label: str, value: float | None, unit: str, formula: str, inputs: list[dict[str, Any]],
            gpu: str | None = None, chip: str | None = None) -> dict[str, Any]:
    first = inputs[0]["source"] if inputs else {"file": None, "line": None, "ref": None}
    return {"id": nid, "label": label, "value": value, "unit": unit, "kind": "derived", "formula": formula,
            "inputs": [i["id"] for i in inputs], "source": dict(first), "gpu": gpu, "chip": chip,
            "conditions": {}, "ts": None, "low_confidence": any(i.get("low_confidence") for i in inputs),
            "reason": None, "other_chip": any(i.get("other_chip") for i in inputs)}


def missing(nid: str, label: str, why: str, task: str | None = None) -> dict[str, Any]:
    return {"id": nid, "label": label, "value": None, "unit": None, "kind": NOT_MEASURED, "why": why, "task": task,
            "source": {"file": None, "line": None, "ref": None}, "gpu": None, "chip": None, "conditions": {},
            "ts": None, "low_confidence": False, "reason": None, "other_chip": False}


# --- section 1: the ceiling against what each build achieves ------------------------------------------------------

def chip_bandwidth(root: Path) -> dict[str, Any]:
    """The chip sheet's device-to-device copy over the largest buffer, the figure a decode's weight stream meets."""
    folder = root / DATA_DIR
    sheet_path = newest(folder, f"chipsheet-{CHIP}-*.json")
    if sheet_path is None:
        return missing("chipsheet:copy", "memory bandwidth (copy, largest buffer)",
                       f"no {DATA_DIR}/chipsheet-{CHIP}-<date>.json is committed", "R33")
    sheet = json.loads(sheet_path.read_text(encoding="utf-8"))
    from lab.kernels.chipsheet import chipsheet_format as sheet_format
    best = sheet_format.copy_largest(sheet) if sheet.get("mode") != "self-check" else None
    if best is None:
        return missing("chipsheet:copy", "memory bandwidth (copy, largest buffer)",
                       f"{_rel(root, sheet_path)} carries no measured device-to-device copy", "R33")
    rel = _rel(root, sheet_path)
    line = line_of(sheet_path, f'"id": "{best["id"]}"')
    gpu = ((sheet.get("device") or {}).get("rocminfo") or {}).get("marketing_name")
    return {"id": f"chipsheet:{best['id']}", "label": "memory bandwidth (copy, largest buffer)",
            "value": round(float(best["value"]), 1), "unit": "GB/s", "kind": "measured",
            "source": {"file": rel, "line": line, "ref": f"{best['id']}: {best.get('what')}; {best.get('op')}"},
            "gpu": f"{gpu}, {sheet.get('gfx')}" if gpu else sheet.get("gfx"), "chip": sheet.get("gfx"),
            "node": sheet.get("node"), "conditions": {"node": sheet.get("node"), "mode": sheet.get("mode"),
                                                       "bytes": (best.get("params") or {}).get("bytes"),
                                                       "repeats": best.get("repeats"), **(sheet.get("versions") or {})},
            "ts": sheet.get("ended"), "low_confidence": False, "reason": None, "other_chip": sheet.get("gfx") != CHIP}


def bytes_per_token(root: Path) -> dict[str, Any]:
    """The teacher's bytes read per decoded token, as the public prior's target table states it (an estimate)."""
    path = root / "lab" / "public_prior.py"
    line = line_of(path, f'"{TEACHER_TARGET}":')
    if line is None:
        return missing("prior:bytes", "weights read per decoded token",
                       f"lab/public_prior.py states no {TEACHER_TARGET} byte budget")
    match = re.search(r":\s*([0-9.]+)", path.read_text(encoding="utf-8").splitlines()[line - 1])
    return {"id": "public_prior:teacher_gb", "label": "weights read per decoded token (teacher, 27B INT4)",
            "value": float(match.group(1)) if match else None, "unit": "GB", "kind": "stated",
            "formula": "the byte budget of the teacher's weights, as lab/public_prior.py TARGETS states it",
            "inputs": [], "source": {"file": "lab/public_prior.py", "line": line,
                                     "ref": "TARGETS; the note public_strix_halo_benchmarks.md gives the same figure"},
            "gpu": None, "chip": None, "conditions": {}, "ts": None, "low_confidence": True,
            "reason": "an estimate of the weight bytes (KV-cache reads not counted), not a measurement",
            "other_chip": False}


def prior_numbers(root: Path, gb: float | None) -> tuple[list[dict[str, Any]], str | None]:
    """The public prior's batch-1 decode rate for the teacher, per backend; never a measurement."""
    csv_path = root / DATA_DIR / "public-strix-halo-benchmarks.csv"
    if gb is None or not csv_path.is_file():
        return [], "the public prior's CSV or the teacher's byte budget is not committed"
    from lab import public_prior
    rows, _ = public_prior.load_rows(csv_path)
    fit = public_prior.fit(rows)
    out = []
    for backend in PRIOR_BACKENDS:
        if backend not in fit.a:
            continue
        p = public_prior.predict(fit, gb, backend, 1)
        out.append({"id": f"prior:{backend}", "label": f"public prior, {backend}", "value": p["per_request_tok_s"],
                    "unit": "tok/s", "kind": "prior", "backend": backend,
                    "formula": "lab/public_prior.py fit on the public CSV, predict(gb, backend, batch 1)",
                    "source": {"file": _rel(root, csv_path), "line": None,
                               "ref": f"fit over {fit.rows} rows from {fit.sources} sources, R2 {fit.r2:.2f}"},
                    "gpu": "Radeon 8060S (other people's machines, published)", "chip": CHIP, "conditions": {"gb_per_token": gb},
                    "ts": None, "low_confidence": True, "reason": "a prior fitted to published benchmarks, not ours",
                    "other_chip": False})
    return out, None


def decode_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Teacher batch-1 decode rows (per stream), oldest first."""
    found = [r for r in rows if r["metric"].endswith(".decode_tok_s") and r.get("node") != STUDENT_NODE
             and batch_of(r) == 1]
    return sorted(found, key=lambda r: (str(r.get("ts")), r["_line"]))


def run_label(row: dict[str, Any]) -> str:
    """A short name for one batch-1 decode row: engine family and build, what was asked, and its line."""
    c = row["conditions"]
    engine = str(c.get("engine") or "engine not recorded")
    build = engine_family(c) + (" " + engine.split("@")[1][:7] if "@" in engine else "")
    if "gfx1201.diff" in engine or "gfx1151.diff" in engine:
        build += " + " + ("gfx1151" if "gfx1151.diff" in engine else "gfx1201") + " patch"
    asked = c.get("prompt") or (c.get("prompt_set") or {}).get("id") if isinstance(c.get("prompt_set"), dict) \
        else c.get("prompt")
    return f"{build}, {asked or 'prompt not recorded'} (line {row['_line']})"


def section_ceiling(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    bw, gb = chip_bandwidth(root), bytes_per_token(root)
    ceiling = None
    if bw["value"] is not None and gb["value"]:
        ceiling = derived("derived:ceiling", "ceiling at batch 1 (bandwidth / bytes per token)",
                          round(bw["value"] / gb["value"], 2), "tok/s",
                          "copy bandwidth (GB/s, bytes read and written) / weights read per token (GB)", [bw, gb],
                          gpu=bw["gpu"], chip=bw["chip"])
    else:
        ceiling = missing("derived:ceiling", "ceiling at batch 1", "the chip sheet or the byte budget is missing", "R33")
    achieved = [from_row(r, run_label(r)) for r in decode_rows(rows)]
    here = [a for a in achieved if not a["other_chip"]]
    apart = [a for a in achieved if a["other_chip"]]
    fractions = []
    for a in here:
        if ceiling["value"]:
            fractions.append(derived(f"derived:share:{a['id']}", f"share of the ceiling, {a['label']}",
                                     round(100 * a["value"] / ceiling["value"], 1), "%",
                                     "100 x achieved tok/s / ceiling tok/s", [a, ceiling], gpu=a["gpu"], chip=a["chip"]))
    priors, prior_gap = prior_numbers(root, gb["value"])
    if here and ceiling["value"]:
        best = max(here, key=lambda a: a["value"])
        reading = (f"The memory on this chip could feed the 27B teacher about {ceiling['value']:.0f} tokens a second; "
                   f"our best build so far reaches {best['value']:.2f}, about {best['value'] / ceiling['value']:.0%} "
                   "of what the memory allows, so the gap is in the software, not the silicon.")
    elif here:
        reading = "Our builds are measured, but the chip's ceiling is not: the chip sheet has not been committed."
    else:
        reading = "Not measured yet: no batch-1 decode row from the Strix Halo is on main."
    return {"id": "ceiling", "title": "The ceiling the chip allows, against what each build achieves",
            "reading": reading, "state": "measured" if here and ceiling["value"] else ("partial" if here else "empty"),
            "inputs": [bw, gb], "ceiling": ceiling, "achieved": here, "shares": fractions, "priors": priors,
            "prior_gap": prior_gap, "other_chip": apart,
            "awaiting": [a["task"] for a in AWAITED if "ceiling" in a["sections"]]}


# --- section 2: the batching frontier -----------------------------------------------------------------------------

def sweeps(rows: list[dict[str, Any]], metric_suffix: str) -> list[dict[str, Any]]:
    """Rows of one metric grouped by (metric, node, build, session) where the group spans two or more batch sizes."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        if not r["metric"].endswith(metric_suffix) or batch_of(r) is None or r.get("node") == STUDENT_NODE:
            continue
        c = r["conditions"]
        key = (r["metric"], r.get("node"), build_of(c), c.get("session") or c.get("run_id") or "", other_chip(r))
        groups.setdefault(key, []).append(r)
    out = []
    for (metric, node, build, session, old), members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        levels = sorted({batch_of(m) for m in members})
        if len(levels) < 2:
            continue
        points = []
        for b in levels:  # the last line written at a batch size wins
            r = max((m for m in members if batch_of(m) == b), key=lambda m: m["_line"])
            points.append({"batch": b, "number": from_row(r, f"{metric} at {b:g} streams")})
        out.append({"metric": metric, "node": node, "build": build, "session": session, "other_chip": old,
                    "points": points})
    return out


def section_frontier(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    energy = sweeps(rows, ".joules_per_token")
    total = sweeps(rows, ".decode_tok_s_total")
    here = [s for s in energy if not s["other_chip"]]
    ratios = []
    for s in here:
        first, last = s["points"][0], s["points"][-1]
        if first["number"]["value"] and last["number"]["value"]:
            ratios.append(derived(f"derived:ratio:{s['metric']}:{s['session']}",
                                  f"joules per token, {first['batch']:g} stream over {last['batch']:g} streams",
                                  round(first["number"]["value"] / last["number"]["value"], 1), "x",
                                  "J/token at the fewest streams / J/token at the most streams",
                                  [first["number"], last["number"]], gpu=first["number"]["gpu"],
                                  chip=first["number"]["chip"]))
    pending = sorted(p.name for p in (root / DATA_DIR).glob("sweep-request-gfx1151-*.json"))
    if ratios:
        r, s = ratios[0], here[0]
        reading = (f"Serving {s['points'][-1]['batch']:g} conversations at once instead of one cuts the energy for each "
                   f"word by {r['value']:g} times: the chip reads its weights once and shares them across every stream.")
    else:
        reading = "Not measured yet: no joules-per-token sweep across stream counts from the Strix Halo is on main."
    return {"id": "frontier", "title": "The batching frontier: joules per token against concurrent streams",
            "reading": reading, "state": "measured" if here else "empty", "energy": here,
            "throughput": [s for s in total if not s["other_chip"]], "ratios": ratios,
            "other_chip": [s for s in energy + total if s["other_chip"]], "pending_files": pending,
            "awaiting": [a["task"] for a in AWAITED if "frontier" in a["sections"]]}


# --- section 3: where a decode step's time goes -------------------------------------------------------------------

def section_step(root: Path) -> dict[str, Any]:
    path = newest(root / DATA_DIR, f"decode-profile-{CHIP}-*.json")
    base = {"id": "step", "title": "Where a decode step's time goes",
            "awaiting": [a["task"] for a in AWAITED if "step" in a["sections"]], "other_chip": []}
    if path is None:
        return {**base, "state": "empty", "batches": [],
                "reading": "Not measured yet: the decode profile (python -m lab profile-decode) has not committed a "
                           f"result under {DATA_DIR}.",
                "why": f"no {DATA_DIR}/decode-profile-{CHIP}-<date>.json on main"}
    data = json.loads(path.read_text(encoding="utf-8"))
    rel = _rel(root, path)
    gpu, chip = data.get("gpu"), data.get("chip")
    batches, cursor = [], 1
    for key, entry in sorted((data.get("batches") or {}).items(), key=lambda kv: float(kv[0])):
        ms = entry.get("ms_per_step") or {}
        line = line_of(path, '"ms_per_step"', cursor)
        cursor = (line or cursor) + 1
        parts = []
        for bucket, label in STEP_BUCKETS:
            value = _num(ms.get(bucket))
            parts.append({"id": f"profile:{key}:{bucket}", "label": f"{label}, batch {key}", "bucket": bucket,
                          "value": value, "unit": "ms per step", "kind": "measured" if value is not None else NOT_MEASURED,
                          "source": {"file": rel, "line": line, "ref": f"batches.{key}.ms_per_step.{bucket}"},
                          "gpu": gpu, "chip": chip, "conditions": {"batch": key, "breakdown_source":
                                                                    entry.get("breakdown_source")},
                          "ts": data.get("ended"), "low_confidence": False, "reason": None,
                          "other_chip": chip not in (None, CHIP)})
        batches.append({"batch": key, "step_ms": _num(ms.get("step")), "parts": parts})
    measured = [p for b in batches for p in b["parts"] if p["value"] is not None]
    if measured and batches and batches[0]["step_ms"]:
        first = batches[0]
        top = max((p for p in first["parts"] if p["value"] is not None), key=lambda p: p["value"])
        reading = (f"At batch {first['batch']}, one decode step takes {first['step_ms']:.1f} ms and the biggest slice, "
                   f"{top['value'] / first['step_ms']:.0%}, is {top['label'].split(',')[0]}: that is where a faster "
                   "kernel pays off first.")
    else:
        reading = f"{rel} is committed but gives no step breakdown."
    return {**base, "state": "measured" if measured else "empty", "batches": batches, "reading": reading,
            "file": rel, "other_chip": batches if chip not in (None, CHIP) else []}


# --- section 4: engines side by side ------------------------------------------------------------------------------

def section_engines(root: Path, ceiling: dict[str, Any]) -> dict[str, Any]:
    here = ceiling["achieved"]
    families: dict[str, list[dict[str, Any]]] = {}
    for a in here:
        families.setdefault(engine_family(a["conditions"]), []).append(a)
    best = {name: max(items, key=lambda a: a["value"]) for name, items in families.items()}
    names = sorted(best)
    if len(names) >= 2:
        order = sorted(best.values(), key=lambda a: -a["value"])
        reading = (f"On the same chip and model, {engine_family(order[0]['conditions'])} decodes "
                   f"{order[0]['value']:.2f} tokens a second against {order[-1]['value']:.2f} for "
                   f"{engine_family(order[-1]['conditions'])}.")
        state = "measured"
    else:
        only = names[0] if names else None
        reading = (f"Only {only} has been measured on this chip so far; the other engines are not measured, and the "
                   "public prior beside it is other people's numbers, not ours." if only else
                   "Not measured yet: no engine has a batch-1 decode row from the Strix Halo on main.")
        state = "partial" if only else "empty"
    return {"id": "engines", "title": "Engines side by side", "reading": reading, "state": state,
            "engines": [{"engine": n, "best": best[n], "rows": families[n]} for n in names],
            "not_measured": [e for e in ("llama.cpp", "vllm") if e not in best],
            "priors": ceiling["priors"], "other_chip": ceiling["other_chip"],
            "awaiting": [a["task"] for a in AWAITED if "engines" in a["sections"]]}


# --- section 5: the fine-tune run ---------------------------------------------------------------------------------

def section_finetune(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trainer = [r for r in rows if r["metric"].startswith(("trainer.", "telemetry.train."))]
    def curve(word: str) -> list[dict[str, Any]]:
        found = [r for r in trainer if word in r["metric"] and _num(r["conditions"].get("step")) is not None
                 and not other_chip(r)]
        return [from_row(r, f"{r['metric']} at step {r['conditions']['step']}")
                for r in sorted(found, key=lambda r: (_num(r["conditions"]["step"]), r["_line"]))]
    loss, step_time = curve("loss"), curve("step_s") or curve("step_time")
    singles = [from_row(r, r["metric"]) for r in trainer if _num(r["conditions"].get("step")) is None]
    here = [s for s in singles if not s["other_chip"]]
    apart = [s for s in singles if s["other_chip"]]
    if loss:
        reading = (f"The fine-tune's loss went from {loss[0]['value']:g} to {loss[-1]['value']:g} over "
                   f"{len(loss)} logged steps on the Strix Halo.")
        state = "measured"
    else:
        reading = ("Not measured yet: the first real fine-tune on the Strix Halo (L16) has not committed per-step loss "
                   "or step time" + ("; the trainer rows on main are from the old chip and are shown apart."
                                     if apart and not here else "."))
        state = "partial" if here else "empty"
    return {"id": "finetune", "title": "The fine-tune run", "reading": reading, "state": state, "loss": loss,
            "step_time": step_time, "numbers": here, "other_chip": apart,
            "awaiting": [a["task"] for a in AWAITED if "finetune" in a["sections"]]}


# --- section 6: what was tried and dropped ------------------------------------------------------------------------

def section_dropped(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    items = []
    for path in sorted((root / DATA_DIR).glob("ledger*.jsonl")):
        rel = _rel(root, path)
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict) or rec.get("verb") != "experiment result":
                continue
            exp = rec.get("experiment") if isinstance(rec.get("experiment"), dict) else {}
            verdict = exp.get("verdict") or rec.get("outcome")
            items.append({"what": exp.get("id") or exp.get("name") or rec.get("task") or f"ledger seq {rec.get('seq')}",
                          "verdict": verdict, "note": rec.get("note"), "kind": "measured",
                          "source": {"file": rel, "line": n, "ref": f"ledger seq {rec.get('seq')}"},
                          "gpu": (rec.get("conditions") or {}).get("gpu"), "task": rec.get("task")})
    slide = root / "lab" / "slide_numbers.py"
    moved_line = line_of(slide, "LAB_MOVED_AT =")
    old = [r for r in rows if other_chip(r)]
    if moved_line and old:
        items.append({"what": "the R9700 cards (gfx1201, DUB hub) as the lab",
                      "verdict": "left for the Strix Halo pods (gfx1151, SJC) on 2026-10-10",
                      "note": f"{len(old)} rows on main describe it; every section shows them apart",
                      "kind": "record", "source": {"file": "lab/slide_numbers.py", "line": moved_line,
                                                   "ref": "LAB_MOVED_AT, the rule every reader uses"},
                      "gpu": "R9700 (gfx1201)", "task": None})
    ledger_items = [i for i in items if i["kind"] == "measured"]
    reading = (f"{len(ledger_items)} experiments have a recorded verdict." if ledger_items else
               "The experiments' verdicts are not on main yet (the lab's ledger lives on the lab machine); what is on "
               "record is the move off the old chip.")
    return {"id": "dropped", "title": "What was tried and dropped", "reading": reading,
            "state": "measured" if ledger_items else ("partial" if items else "empty"), "items": items,
            "other_chip": [], "awaiting": []}


# --- the pack -----------------------------------------------------------------------------------------------------

def _sources(root: Path) -> list[dict[str, Any]]:
    paths = [root / METRICS, newest(root / DATA_DIR, f"chipsheet-{CHIP}-*.json"),
             newest(root / DATA_DIR, f"decode-profile-{CHIP}-*.json"), root / DATA_DIR / "public-strix-halo-benchmarks.csv",
             root / "lab" / "public_prior.py", root / "lab" / "slide_numbers.py",
             *sorted((root / DATA_DIR).glob("ledger*.jsonl")), *sorted((root / DATA_DIR).glob("sweep-request-gfx1151-*.json"))]
    out = []
    for p in paths:
        if p is not None and p.is_file():
            data = p.read_bytes()
            out.append({"file": _rel(root, p), "sha256": hashlib.sha256(data).hexdigest(),
                        "lines": data.count(b"\n")})
    return out


def _walk(value: Any) -> list[dict[str, Any]]:
    """Every number dict inside a section, in order."""
    found = []
    if isinstance(value, dict):
        if "kind" in value and "source" in value and "value" in value:
            found.append(value)
        for v in value.values():
            found.extend(_walk(v))
    elif isinstance(value, list):
        for v in value:
            found.extend(_walk(v))
    return found


def build_pack(root: Path = REPO) -> dict[str, Any]:
    """The data pack: every section with its numbers, each number with its source file and line."""
    root = Path(root)
    rows, skipped = read_metrics(root / METRICS)
    ceiling = section_ceiling(root, rows)
    sections = [ceiling, section_frontier(root, rows), section_step(root), section_engines(root, ceiling),
                section_finetune(rows), section_dropped(root, rows)]
    numbers, seen = [], set()
    for s in sections:
        for n in _walk(s):
            if n["id"] not in seen:
                seen.add(n["id"])
                numbers.append(n)
    present = {a["task"]: bool(a["glob"] and list((root / DATA_DIR).glob(a["glob"]))) for a in AWAITED}
    times = sorted(t for t in (_ts(r.get("ts")) for r in rows) if t)
    return {
        "format": FORMAT,
        "title": "Under the hood: the Atlas lab on the Strix Halo",
        "chip": {"current": CHIP, "name": "Radeon 8060S (Strix Halo)", "other": "R9700 (gfx1201), the lab until "
                 "2026-10-10, shown apart", "rule": "lab/slide_numbers.py _old_chip"},
        "data_as_of": times[-1] if times else None,
        "sources": _sources(root),
        "skipped": skipped,
        "sections": sections,
        "numbers": numbers,
        "awaited": [{k: v for k, v in a.items() if k != "glob"} | {"file_present": present[a["task"]]}
                    for a in AWAITED],
    }


# --- the page -----------------------------------------------------------------------------------------------------

def esc(text: Any) -> str:
    return html.escape("" if text is None else str(text), quote=True)


def fmt(value: Any, unit: str | None = None) -> str:
    if value is None:
        return NOT_MEASURED
    v = float(value)
    text = f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}" if abs(v) < 10 else f"{v:.1f}"
    return f"{text} {unit}" if unit else text


def gpu_text(n: dict[str, Any]) -> str:
    if n.get("gpu"):
        return f"gpu: {n['gpu']}"
    if n.get("chip"):
        return f"gpu: not recorded (chip: {n['chip']})"
    return "gpu: not recorded"


def href(page_dir: Path, root: Path, rel: str) -> str:
    """A relative link from the page's folder to a committed file, so it opens from disk."""
    import os
    return Path(os.path.relpath(root / rel, page_dir)).as_posix()


class Page:
    def __init__(self, root: Path, page_dir: Path):
        self.root, self.page_dir = root, page_dir

    def cite(self, n: dict[str, Any]) -> str:
        """The source line under a number: file and line as a link, the row's gpu, and its flags."""
        src = n.get("source") or {}
        if not src.get("file"):
            return f'<span class="cite">{esc(n.get("why") or NOT_MEASURED)}</span>'
        where = f"{src['file']}:{src['line']}" if src.get("line") else src["file"]
        bits = [f'<a href="{esc(href(self.page_dir, self.root, src["file"]))}" data-line="{esc(src.get("line") or "")}">'
                f'{esc(where)}</a>', esc(gpu_text(n))]
        if n.get("kind") in ("derived", "prior", "stated"):
            bits.append(esc(n["kind"]))
        if n.get("low_confidence"):
            bits.append(f'<span class="flag">low confidence{": " + esc(n["reason"]) if n.get("reason") else ""}</span>')
        if n.get("other_chip"):
            bits.append('<span class="flag">other chip</span>')
        return f'<span class="cite">{" · ".join(bits)}</span>'

    def stat(self, n: dict[str, Any]) -> str:
        return (f'<div class="stat"><div class="k">{esc(n["label"])}</div>'
                f'<div class="v">{esc(fmt(n["value"], n.get("unit")))}</div>{self.cite(n)}</div>')


def svg_bars(items: list[tuple[str, float | None, str]], line: tuple[str, float] | None, unit: str,
             title: str) -> str:
    """Horizontal bars (label, value, class) with an optional vertical reference line; a None value says so."""
    width, left, right, bar_h, gap = 520, 210, 56, 24, 10
    height = 30 + len(items) * (bar_h + gap) + 24
    peak = max([v for _, v, _ in items if v] + ([line[1]] if line else []) + [1e-9])
    scale = (width - left - right) / (peak * 1.05)
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}" class="chart">',
           f'<title>{esc(title)}</title>']
    y = 20
    for label, value, cls in items:
        short = label if len(label) <= 24 else label[:23] + "\u2026"
        out.append(f'<text x="{left - 8}" y="{y + bar_h * 0.7:.1f}" text-anchor="end" class="lbl">{esc(short)}'
                   f'<title>{esc(label)}</title></text>')
        if value is None:
            out.append(f'<text x="{left + 4}" y="{y + bar_h * 0.7:.1f}" class="nm">{NOT_MEASURED}</text>')
        else:
            w = max(1.0, value * scale)
            out.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="3" class="{cls}"/>')
            out.append(f'<text x="{left + w + 6:.1f}" y="{y + bar_h * 0.7:.1f}" class="val">{esc(fmt(value))}</text>')
        y += bar_h + gap
    if line:
        x = left + line[1] * scale
        out.append(f'<line x1="{x:.1f}" y1="12" x2="{x:.1f}" y2="{y}" class="ref"/>')
        out.append(f'<text x="{x:.1f}" y="{y + 16}" text-anchor="middle" class="reft">{esc(line[0])} '
                   f'{esc(fmt(line[1], unit))}</text>')
    out.append("</svg>")
    return "".join(out)


def svg_frontier(points: list[tuple[float, float]], second: list[tuple[float, float]], ratio: float | None,
                 title: str) -> str:
    """Joules per token (left axis, line) against streams on a log2 axis, aggregate tok/s as bars behind it."""
    width, height, left, right, top, bottom = 520, 300, 48, 44, 24, 44
    xs = [x for x, _ in points]
    lo, hi = math.log2(min(xs)), math.log2(max(xs))
    span = (hi - lo) or 1.0
    def px(x: float) -> float:
        return left + (math.log2(x) - lo) / span * (width - left - right)
    ymax = max(y for _, y in points) * 1.1
    def py(y: float) -> float:
        return top + (1 - y / ymax) * (height - top - bottom)
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}" class="chart">',
           f'<title>{esc(title)}</title>',
           f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" class="axis"/>',
           f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" class="axis"/>']
    if second:
        smax = max(y for _, y in second) * 1.1
        for x, y in second:
            h = y / smax * (height - top - bottom)
            out.append(f'<rect x="{px(x) - 9:.1f}" y="{height - bottom - h:.1f}" width="18" height="{h:.1f}" class="b2"/>')
            out.append(f'<text x="{px(x):.1f}" y="{height - bottom - h - 4:.1f}" text-anchor="middle" class="val2">'
                       f'{esc(fmt(y))}</text>')
        out.append(f'<text x="{width - right + 6}" y="{top + 10}" class="lbl2">tok/s total (bars)</text>')
    path = " ".join(f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(y):.1f}" for i, (x, y) in enumerate(points))
    out.append(f'<path d="{path}" class="ln"/>')
    for x, y in points:
        out.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4.5" class="dot"/>')
        out.append(f'<text x="{px(x) + 7:.1f}" y="{py(y) - 7:.1f}" class="val">{esc(fmt(y))} J</text>')
        out.append(f'<text x="{px(x):.1f}" y="{height - bottom + 16}" text-anchor="middle" class="lbl">{x:g}</text>')
    if ratio:
        y_last = py(points[-1][1])
        out.append(f'<line x1="{left}" y1="{y_last:.1f}" x2="{width - right}" y2="{y_last:.1f}" class="ref"/>')
        out.append(f'<text x="{width - right:.1f}" y="{top + 14}" text-anchor="end" class="reft">'
                   f'{ratio:g} times less energy</text>')
        out.append(f'<text x="{width - right:.1f}" y="{top + 32}" text-anchor="end" class="reft">'
                   f'per token than 1 stream (dashed)</text>')
    out.append(f'<text x="{(left + width - right) / 2:.1f}" y="{height - 8}" text-anchor="middle" class="lbl">'
               'concurrent streams</text>')
    out.append(f'<text x="14" y="{top + (height - top - bottom) / 2:.1f}" class="lbl" transform="rotate(-90 14 '
               f'{top + (height - top - bottom) / 2:.1f})" text-anchor="middle">joules per token</text>')
    out.append("</svg>")
    return "".join(out)


def svg_stack(batches: list[dict[str, Any]], title: str) -> str:
    """One stacked bar per batch size: ms per decode step by bucket."""
    width, left, right, bar_h, gap = 520, 70, 16, 28, 34
    height = 20 + len(batches) * (bar_h + gap) + 40
    peak = max([sum(p["value"] or 0 for p in b["parts"]) for b in batches] + [1e-9])
    scale = (width - left - right) / peak
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}" class="chart">',
           f'<title>{esc(title)}</title>']
    y = 20
    for b in batches:
        out.append(f'<text x="{left - 8}" y="{y + bar_h * 0.7:.1f}" text-anchor="end" class="lbl">batch {esc(b["batch"])}</text>')
        x = left
        for i, p in enumerate(b["parts"]):
            if p["value"]:
                w = p["value"] * scale
                out.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{bar_h}" class="s{i}">'
                           f'<title>{esc(p["label"])}: {esc(fmt(p["value"], "ms"))}</title></rect>')
                x += w
        out.append(f'<text x="{left}" y="{y + bar_h + 16}" class="val">{esc(fmt(b["step_ms"], "ms per step"))}</text>')
        y += bar_h + gap
    lx = left
    for i, (_, label) in enumerate(STEP_BUCKETS):
        out.append(f'<rect x="{lx}" y="{y}" width="10" height="10" class="s{i}"/><text x="{lx + 14}" y="{y + 9}" '
                   f'class="lbl">{esc(label)}</text>')
        lx += 14 + len(label) * 6.2 + 12
    out.append("</svg>")
    return "".join(out)


CSS = """
:root{--bg:#f7f6f2;--card:#fff;--ink:#1d1d1b;--mute:#5d5d58;--line:#d9d6cc;--a:#2f6fb2;--b:#c2620a;--c:#2c8a5a;
--old:#8a8a84;--prior:#9a7bb8;--flag:#a03a1c;--s0:#2f6fb2;--s1:#c2620a;--s2:#2c8a5a;--s3:#9a7bb8;--s4:#b0912c;--s5:#8a8a84}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121416;--card:#1b1e21;--ink:#ecebe6;
--mute:#a9a8a1;--line:#33373b;--a:#6aa6e6;--b:#f0954a;--c:#5cc58f;--old:#7d7d78;--prior:#b99ad8;--flag:#f08a6a;
--s0:#6aa6e6;--s1:#f0954a;--s2:#5cc58f;--s3:#b99ad8;--s4:#d8b84e;--s5:#7d7d78}}
:root[data-theme="dark"]{--bg:#121416;--card:#1b1e21;--ink:#ecebe6;--mute:#a9a8a1;--line:#33373b;--a:#6aa6e6;
--b:#f0954a;--c:#5cc58f;--old:#7d7d78;--prior:#b99ad8;--flag:#f08a6a;--s0:#6aa6e6;--s1:#f0954a;--s2:#5cc58f;
--s3:#b99ad8;--s4:#d8b84e;--s5:#7d7d78}
body:has(#day:checked){--bg:#f7f6f2;--card:#fff;--ink:#1d1d1b;--mute:#5d5d58;--line:#d9d6cc;--a:#2f6fb2;--b:#c2620a;
--c:#2c8a5a;--old:#8a8a84;--prior:#9a7bb8;--flag:#a03a1c;--s0:#2f6fb2;--s1:#c2620a;--s2:#2c8a5a;--s3:#9a7bb8;
--s4:#b0912c;--s5:#8a8a84}
body:has(#night:checked){--bg:#121416;--card:#1b1e21;--ink:#ecebe6;--mute:#a9a8a1;--line:#33373b;--a:#6aa6e6;
--b:#f0954a;--c:#5cc58f;--old:#7d7d78;--prior:#b99ad8;--flag:#f08a6a;--s0:#6aa6e6;--s1:#f0954a;--s2:#5cc58f;
--s3:#b99ad8;--s4:#d8b84e;--s5:#7d7d78}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1040px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:clamp(1.6rem,4vw,2.6rem);line-height:1.15;margin:.2em 0}h2{font-size:clamp(1.2rem,3vw,1.7rem);margin:0 0 .3em}
.lede{color:var(--mute);max-width:70ch}.theme{float:right;font-size:.9rem;color:var(--mute)}
.theme input{accent-color:var(--a)}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px 18px;margin:20px 0}
.n{color:var(--mute);font-weight:600;margin-right:.4em}
.say{font-size:clamp(1.05rem,2.4vw,1.35rem);font-weight:500;margin:.2em 0 1em;max-width:62ch}
.state{display:inline-block;font-size:.75rem;text-transform:uppercase;letter-spacing:.06em;padding:2px 8px;
border-radius:99px;border:1px solid var(--line);color:var(--mute);vertical-align:middle;margin-left:.5em}
.chart{width:100%;height:auto;display:block;margin:8px 0 12px}
.chart text{font-size:16px;fill:var(--ink)}.chart .lbl{fill:var(--mute)}.chart .val{font-weight:600}
.chart .nm{fill:var(--flag);font-style:italic}.chart .reft{fill:var(--b);font-weight:600}
.chart .lbl2,.chart .val2{fill:var(--mute);font-size:13px}
.chart .cur{fill:var(--a)}.chart .old{fill:var(--old)}.chart .prior{fill:var(--prior);opacity:.75}
.chart .ceil{fill:var(--c)}.chart .ref{stroke:var(--b);stroke-width:2;stroke-dasharray:6 4}
.chart .axis{stroke:var(--line)}.chart .ln{fill:none;stroke:var(--a);stroke-width:3}.chart .dot{fill:var(--a)}
.chart .b2{fill:var(--c);opacity:.35}
.chart .s0{fill:var(--s0)}.chart .s1{fill:var(--s1)}.chart .s2{fill:var(--s2)}.chart .s3{fill:var(--s3)}
.chart .s4{fill:var(--s4)}.chart .s5{fill:var(--s5)}
.stats{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
.stat{border-top:2px solid var(--line);padding-top:6px}.stat .k{font-size:.85rem;color:var(--mute)}
.stat .v{font-size:1.4rem;font-weight:650}
a{color:var(--a)}
.cite{display:block;font-size:.75rem;color:var(--mute);overflow-wrap:anywhere}.cite a{color:var(--a)}
.flag{color:var(--flag)}
.apart{border:1px dashed var(--old);border-radius:10px;padding:12px;margin-top:14px}
.apart h3{font-size:.95rem;margin:0 0 6px;color:var(--old)}
.empty{border-left:4px solid var(--flag);padding:8px 12px;background:transparent;color:var(--ink)}
table{border-collapse:collapse;width:100%;font-size:.85rem}td,th{border-bottom:1px solid var(--line);padding:6px 4px;
text-align:left;vertical-align:top}.wrap{overflow-x:auto}
footer{color:var(--mute);font-size:.8rem}
@media (min-width:1600px){body{font-size:19px}main{max-width:1400px}.chart text{font-size:14px}}
"""


def render_html(pack: dict[str, Any], root: Path = REPO, page_dir: Path | None = None) -> str:
    """The self-contained page: inline CSS and SVG, no script, no font, no network reference."""
    root = Path(root)
    page = Page(root, page_dir or root / OUT_DIR)
    secs = {s["id"]: s for s in pack["sections"]}
    body: list[str] = []

    def head(i: int, s: dict[str, Any]) -> str:
        return (f'<section id="{s["id"]}"><h2><span class="n">{i}</span>{esc(s["title"])}'
                f'<span class="state">{esc(s["state"])}</span></h2><p class="say">{esc(s["reading"])}</p>')

    def awaiting(s: dict[str, Any]) -> str:
        names = [a for a in pack["awaited"] if a["task"] in s.get("awaiting", [])]
        if not names:
            return ""
        lis = "".join(f"<li><b>{esc(a['task'])}</b>: {esc(a['what'])} ({esc(a['where'])}): "
                      f"{'file present' if a['file_present'] else NOT_MEASURED}</li>" for a in names)
        return f'<details><summary>Runs that fill this section when their rows land</summary><ul>{lis}</ul></details>'

    def apart(numbers: list[dict[str, Any]], what: str) -> str:
        if not numbers:
            return ""
        return (f'<div class="apart"><h3>Other chip, shown apart: {esc(what)}</h3><div class="stats">'
                + "".join(page.stat(n) for n in numbers) + "</div></div>")

    # 1
    s = secs["ceiling"]
    body.append(head(1, s))
    bars = [(a["label"], a["value"], "cur") for a in s["achieved"]]
    bars += [(p["label"], p["value"], "prior") for p in s["priors"]]
    if s["achieved"] or s["ceiling"]["value"]:
        line = ("ceiling", s["ceiling"]["value"]) if s["ceiling"]["value"] else None
        body.append(svg_bars(bars, line, "tok/s", "Decode tokens per second at batch 1 against the bandwidth ceiling"))
    else:
        body.append(f'<p class="empty">{NOT_MEASURED}: no batch-1 decode row and no chip sheet on main.</p>')
    body.append('<div class="stats">' + "".join(page.stat(n) for n in [*s["inputs"], s["ceiling"], *s["shares"],
                                                                       *s["achieved"], *s["priors"]]) + "</div>")
    body.append(apart(s["other_chip"], "teacher decode on the R9700 (gfx1201), a different card and memory"))
    body.append(awaiting(s) + "</section>")

    # 2
    s = secs["frontier"]
    body.append(head(2, s))
    if s["energy"]:
        sweep = s["energy"][0]
        pts = [(p["batch"], p["number"]["value"]) for p in sweep["points"]]
        thr = next((t for t in s["throughput"] if t["node"] == sweep["node"] and t["session"] == sweep["session"]), None)
        second = [(p["batch"], p["number"]["value"]) for p in thr["points"]] if thr else []
        ratio = s["ratios"][0]["value"] if s["ratios"] else None
        body.append(svg_frontier(pts, [], ratio, "Joules per token against concurrent streams"))
        if second:
            body.append(svg_bars([(f"{x:g} streams", y, "ceil") for x, y in second], None, "tok/s",
                                 "Tokens per second across all streams together"))
        nums = [p["number"] for p in sweep["points"]] + ([p["number"] for p in thr["points"]] if thr else [])
        body.append('<div class="stats">' + "".join(page.stat(n) for n in [*s["ratios"], *nums]) + "</div>")
        for extra in s["energy"][1:]:
            body.append('<div class="stats">' + "".join(page.stat(p["number"]) for p in extra["points"]) + "</div>")
    else:
        body.append(f'<p class="empty">{NOT_MEASURED}: no stream-count sweep with joules per token on main.</p>')
    if s["pending_files"]:
        body.append(f'<p class="cite">Committed but not charted yet: {esc(", ".join(s["pending_files"]))}</p>')
    body.append(apart([p["number"] for o in s["other_chip"] for p in o["points"]], "batching rows from the R9700"))
    body.append(awaiting(s) + "</section>")

    # 3
    s = secs["step"]
    body.append(head(3, s))
    if s["state"] == "measured":
        body.append(svg_stack(s["batches"], "Milliseconds per decode step by where the time goes"))
        body.append('<div class="stats">' + "".join(page.stat(p) for b in s["batches"] for p in b["parts"]) + "</div>")
    else:
        body.append(f'<p class="empty">{NOT_MEASURED}: {esc(s.get("why") or s["reading"])}.</p>')
    body.append(awaiting(s) + "</section>")

    # 4
    s = secs["engines"]
    body.append(head(4, s))
    bars = [(e["engine"], e["best"]["value"], "cur") for e in s["engines"]]
    bars += [(name, None, "cur") for name in s["not_measured"]]
    bars += [(p["label"], p["value"], "prior") for p in s["priors"]]
    if bars:
        body.append(svg_bars(bars, None, "tok/s", "Best batch-1 decode per engine on the Strix Halo, ours and the prior"))
    body.append('<div class="stats">' + "".join(page.stat(e["best"]) for e in s["engines"])
                + "".join(page.stat(missing(f"engine:{n}", f"{n} on the Strix Halo, batch 1",
                                            "no row on main for this engine")) for n in s["not_measured"])
                + "</div>")
    body.append(awaiting(s) + "</section>")

    # 5
    s = secs["finetune"]
    body.append(head(5, s))
    if s["loss"]:
        pts = [(float(n["conditions"]["step"]), n["value"]) for n in s["loss"]]
        body.append(svg_bars([(f"step {x:g}", y, "cur") for x, y in pts], None, "", "Training loss per logged step"))
        body.append('<div class="stats">' + "".join(page.stat(n) for n in s["loss"] + s["step_time"]) + "</div>")
    else:
        body.append(f'<p class="empty">{NOT_MEASURED}: no per-step loss or step time from the Strix Halo on main.</p>')
    if s["numbers"]:
        body.append('<div class="stats">' + "".join(page.stat(n) for n in s["numbers"]) + "</div>")
    body.append(apart(s["other_chip"], "the 20-step LoRA smoke run on one R9700"))
    body.append(awaiting(s) + "</section>")

    # 6
    s = secs["dropped"]
    body.append(head(6, s))
    if s["items"]:
        rows = "".join(f"<tr><td>{esc(i['what'])}</td><td>{esc(i['verdict'])}</td><td>{esc(i['note'] or '')}</td>"
                       f"<td>{page.cite(i)}</td></tr>" for i in s["items"])
        body.append(f'<div class="wrap"><table><tr><th>Tried</th><th>Verdict</th><th>Note</th><th>Source</th></tr>'
                    f"{rows}</table></div>")
    if s["state"] != "measured":
        body.append(f'<p class="empty">{NOT_MEASURED}: no experiment verdict is committed; the lab ledger '
                    "(lab/state/ledger.jsonl) stays on the lab machine.</p>")
    body.append("</section>")

    srcs = "".join(f"<tr><td><a href=\"{esc(href(page.page_dir, root, x['file']))}\">{esc(x['file'])}</a></td>"
                   f"<td>{x['lines']}</td><td><code>{esc(x['sha256'][:12])}</code></td></tr>" for x in pack["sources"])
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>Under the hood</title><style>{CSS}</style></head><body><main>"
        '<div class="theme" role="radiogroup" aria-label="theme"><label><input type="radio" name="t" id="auto" checked> '
        'Auto</label> <label><input type="radio" name="t" id="night"> Night</label> '
        '<label><input type="radio" name="t" id="day"> Day</label></div>'
        f"<h1>{esc(pack['title'])}</h1>"
        f'<p class="lede">Every number below is read from a file committed in this repository and links to its line. '
        f"Rows from the old chip (R9700, gfx1201) are kept apart; a measurement that does not exist yet says "
        f"&ldquo;{NOT_MEASURED}&rdquo;. Data as of {esc(pack['data_as_of'] or 'unknown')}.</p>"
        + "".join(body)
        + f'<footer><h2>Sources</h2><div class="wrap"><table><tr><th>File</th><th>Lines</th><th>sha256</th></tr>{srcs}'
          "</table></div><p>Built by <code>python -m lab report</code>; the data pack beside this page is "
          f"<code>{PACK_NAME}</code> ({FORMAT}).</p></footer></main></body></html>\n")


# --- the verb -----------------------------------------------------------------------------------------------------

def write(root: Path = REPO, out: Path | None = None) -> dict[str, Any]:
    root = Path(root)
    out_dir = Path(out) if out else root / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pack = build_pack(root)
    page = render_html(pack, root, out_dir)
    (out_dir / PACK_NAME).write_text(json.dumps(pack, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    (out_dir / HTML_NAME).write_text(page, encoding="utf-8", newline="\n")
    return {"ok": True, "html": str(out_dir / HTML_NAME), "pack": str(out_dir / PACK_NAME),
            "sections": {s["id"]: s["state"] for s in pack["sections"]}, "numbers": len(pack["numbers"])}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Write the under-the-hood report (one HTML page and its JSON data pack) from committed files only."""
    p = argparse.ArgumentParser(prog="python -m lab report", description=__doc__.splitlines()[0])
    p.add_argument("--root", default=str(REPO), help="the checkout to read (default: this one)")
    p.add_argument("--out", help=f"where to write (default: <root>/{OUT_DIR})")
    args = p.parse_args(argv or [])
    return write(Path(args.root), Path(args.out) if args.out else None)
