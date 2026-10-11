"""The under-the-hood report: one HTML page and a JSON data pack that show the lab's optimization work.

    python -m lab report                      # writes lab/refine/report/under-the-hood.html and .json
    python -m lab report --out DIR            # somewhere else
    python -m lab report --root DIR           # read the committed files of another checkout (tests use a fixture)

Built only from committed files, never from a pod, the hub or the network:

- lab/refine/metrics.jsonl, the lab's measured rows (each number keeps its line);
- the chip sheet lab/refine/research/kernels/data/chipsheet-gfx1151-<date>.json (the newest measured one);
- the decode profile decode-profile-gfx1151-<date>.json in the same folder, when one is committed;
- the public prior: lab/public_prior.py fitted on public-strix-halo-benchmarks.csv, always labelled a prior;
- a fine-tune run's own files, fixtures/demo/<run>-telemetry.jsonl (one telemetry.train.step event per line) and
  <run>-run.json beside it, when a run is committed (L16's are named l16-*);
- any lab ledger copy (ledger*.jsonl) in the data folder, and the `excluded` list of a searchspace-*.json there,
  for what was tried and dropped or ruled out;
- the demo numbers sheet lab/refine/research/telemetry/demo_numbers_<date>.md, named as the page's companion.

The page has six sections, each a chart drawn here as inline SVG, one sentence a non-expert can say aloud and one
key figure: (1) the ceiling the chip allows against what each build achieves at batch 1, (2) the batching frontier,
joules per token against concurrent streams, (3) where a decode step's time goes, (4) engines side by side,
(5) the fine-tune run, (6) what was tried and dropped. A section with no rows says "not measured" and names the
runs still coming (AWAITED, with what each was doing on STILL_COMING_AS_OF); nothing is invented or interpolated.

Every number in the pack is a dict: {id, label, value, unit, kind, source: {file, line, ref}, gpu, chip,
conditions, ts, low_confidence, reason, other_chip}. `kind` is "measured" (a row's own value), "derived"
(worked out here from other numbers, with `formula` and `inputs`), "stated" (an estimate a committed file states,
such as the teacher's weight bytes), "prior" (the public prior's prediction, never a measurement) or
"not measured" (value None, with `why`). `gpu` is the row's conditions.gpu (for a fine-tune step, the device its
run file names), `chip` its conditions.chip; either may be None.
Rows from another chip (the R9700s, by lab/slide_numbers.py's own rule, which the Learning lab's bench reader
uses too, the A100 reference, any GPU a row names) are never drawn with Strix Halo rows: they sit in each section's
`other_chip` list and in a dashed panel of their own on the page.

The data pack's shape is documented in lab/README.md, section "Under the hood report".
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Any

FORMAT = "atlas.lab_report/1"
REPO = Path(__file__).resolve().parents[1]
METRICS = "lab/refine/metrics.jsonl"
DATA_DIR = "lab/refine/research/kernels/data"
SHEET_DIR = "lab/refine/research/telemetry"
RUN_DIR = "fixtures/demo"
OUT_DIR = "lab/refine/report"
HTML_NAME = "under-the-hood.html"
PACK_NAME = "under-the-hood.json"
CHIP = "gfx1151"
CHIP_NAME = "Radeon 8060S (Strix Halo, gfx1151)"
STUDENT_NODE = "jetson"
LAB_MOVED_AT = "2026-10-10T20:25:00Z"
NOT_MEASURED = "not measured"
TEACHER_TARGET = "teacher-27b-int4"
PRIOR_BACKENDS = ("vllm", "llama.cpp-rocm", "llama.cpp-vulkan")
ENGINE_NAMES = {"vllm": "vLLM", "llama.cpp": "llama.cpp", "sglang": "SGLang"}
EXPECTED_ENGINES = ("vllm", "llama.cpp")
STEP_BUCKETS = (("w4a16_matmul", "quantized matmul"), ("attention", "attention"), ("other_gpu", "other GPU work"),
                ("sampling", "sampling"), ("copies", "copies"), ("host_side", "host side (GPU idle)"))
RUN_FACTS = (("trainer.samples_per_s", "samples/s"), ("trainer.peak_mem_gb", "GiB"), ("trainer.train_s", "s"),
             ("trainer.steps", "steps"), ("trainer.examples", "examples"))


# --- reading the committed files -----------------------------------------------------------------------------------

def _ts(text: Any) -> str | None:
    return text if isinstance(text, str) and re.match(r"\d{4}-\d\d-\d\dT", text) else None


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


PODS = ("pod-a", "pod-b", "pod-c", "pod-d")  # the lab's Strix Halo pods, by role and by account


def chip_of(row: dict[str, Any]) -> str:
    """Which chip a row describes: "gfx1151" (the lab today), "R9700 (gfx1201)", the GPU the row names, or "Jetson".

    A row naming gfx1151, Strix Halo or the Radeon 8060S is the lab's. A row naming gfx1201 or the R9700, or an
    untagged pod row measured before the move, is the old chip (lab/slide_numbers.py's `_old_chip`, the rule the
    Learning lab's bench reader imports). Any other GPU the row names (the A100 reference, say) is its own chip, and
    so is a row whose conditions.other_chip is true. An untagged row is the lab's only when it comes from a pod."""
    if row.get("node") == STUDENT_NODE:
        return "Jetson"
    c = row.get("conditions") or {}
    named = " ".join(str(c.get(k) or "") for k in ("gpu", "chip"))
    if any(word in named for word in ("gfx1151", "Strix", "8060S")) and c.get("other_chip") is not True:
        return CHIP
    try:
        from lab.slide_numbers import _old_chip
    except ImportError:  # httpx missing: the same rule, restated
        def _old_chip(rec: dict) -> bool:
            gpu = str((rec.get("conditions") or {}).get("gpu", ""))
            if "gfx1201" in gpu or "R9700" in gpu:
                return True
            return bool(_ts(rec.get("ts"))) and str(rec["ts"]) < LAB_MOVED_AT
    if "gfx1201" in named or "R9700" in named:
        return "R9700 (gfx1201)"
    if named.strip():
        return named.strip().split(",")[0]
    if _old_chip(row):
        return "another chip (the row says so)" if c.get("other_chip") is True else "R9700 (gfx1201)"
    return CHIP if row.get("node") in PODS else f"not recorded (node {row.get('node')})"


def other_chip(row: dict[str, Any]) -> bool:
    """True for a row from any chip but the lab's Strix Halo (gfx1151); the Jetson is not one of the lab's cards."""
    return chip_of(row) not in (CHIP, "Jetson")


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
    """vllm, llama.cpp, sglang or the engine's first word; "not recorded" when the row names none.

    lab/bench.py writes llama.cpp's engine as "llamacpp" (its --engine choice) and its server is llama-server; both
    are llama.cpp here, so its rows fill the engine a reader was told is "not measured"."""
    e = str(c.get("engine") or "").lower()
    if any(word in e for word in ("llama.cpp", "llamacpp", "llama-server", "llama_cpp")):
        return "llama.cpp"
    for name in ("vllm", "sglang"):
        if name in e:
            return name
    return e.split("@")[0].split()[0] if e.strip() else "not recorded"


def build_parts(c: dict[str, Any]) -> dict[str, Any]:
    """What tells one build from another, as far as the row records it: the engine family, its commit (seven hex
    digits, from "engine@commit", conditions.engine_commit or llama-server's "build: N (hex)" line), the patch files
    named in the engine string or in conditions.patch_hash (without the folder and ".diff"), conditions.backend
    (vulkan or rocm for llama.cpp) and a released version (conditions.vllm on the stock-wheel rows)."""
    engine = str(c.get("engine") or "")
    commit = None
    if m := re.search(r"@([0-9a-f]{7,40})", engine):
        commit = m.group(1)[:7]
    stated = c.get("engine_commit")
    if commit is None and isinstance(stated, str):
        if re.fullmatch(r"[0-9a-f]{7,40}", stated):
            commit = stated[:7]
        elif m := re.search(r"\(([0-9a-f]{7,40})\)", stated):
            commit = m.group(1)[:7]
    patches = set(re.findall(r"([\w.-]+)\.diff", engine))
    if isinstance(c.get("patch_hash"), dict):
        patches |= {re.sub(r"\.diff$", "", str(k).replace("\\", "/").split("/")[-1]) for k in c["patch_hash"]}
    backend = c.get("backend") if isinstance(c.get("backend"), str) and c["backend"].strip() else None
    version = c.get("vllm") if isinstance(c.get("vllm"), str) and c["vllm"].strip() else None
    return {"family": engine_family(c), "commit": commit, "patches": sorted(patches), "backend": backend,
            "version": version}


def build_label(c: dict[str, Any]) -> str:
    """A build's name for a person: "vLLM b1388b1 + gfx1201 patch", "llama.cpp vulkan 1a2b3c4", "vLLM 0.31.0";
    a row that names an engine but no commit or version says "build not recorded"."""
    p = build_parts(c)
    if p["family"] == "not recorded":
        return "engine not recorded"
    bits = [ENGINE_NAMES.get(p["family"], p["family"])]
    if p["backend"]:
        bits.append({"vulkan": "Vulkan", "rocm": "ROCm"}.get(p["backend"].lower(), p["backend"]))
    if p["commit"] or p["version"]:
        bits.append(p["commit"] or p["version"])
    label = " ".join(bits)
    if p["patches"]:
        label += " + " + ", ".join(re.sub(r"^vllm-", "", name) for name in p["patches"]) + " patch"
    if not (p["commit"] or p["version"]):
        label += ", build not recorded"
    return label


def asked_of(c: dict[str, Any]) -> str | None:
    """What a row was asked: its prompt ("text prompt"), or its prompt set ("bench-v1 prompt set"), or None."""
    if isinstance(c.get("prompt"), str) and c["prompt"].strip():
        return f"{c['prompt'].strip()} prompt"
    ps = c.get("prompt_set")
    name = ps.get("id") if isinstance(ps, dict) else ps
    if isinstance(name, str) and name.strip():
        return f"{name.strip()} prompt set"
    if isinstance(c.get("prompt_class"), str) and c["prompt_class"].strip():
        return f"{c['prompt_class'].strip()} prompts"
    return None


def setup_of(c: dict[str, Any]) -> str:
    """The few conditions a reader needs beside a number, when the row records them, in one short phrase."""
    bits = [asked_of(c)]
    if _num(c.get("max_tokens")) is not None:
        bits.append(f"{c['max_tokens']:g} tokens out")
    if c.get("thinking") is not None:
        bits.append("thinking " + ("on" if c["thinking"] in (True, "on") else "off"))
    if isinstance(c.get("statistic"), str):
        bits.append(c["statistic"])
    return ", ".join(b for b in bits if b)


def run_label(row: dict[str, Any]) -> str:
    """A short name for one batch-1 decode row: its build, what was asked, and the node."""
    c = row["conditions"]
    return f"{build_label(c)}, {asked_of(c) or 'prompt not recorded'}, {row.get('node') or 'node not recorded'}"


def from_row(row: dict[str, Any], label: str) -> dict[str, Any]:
    c = row["conditions"]
    low = bool(row.get("low_confidence"))
    return {
        "id": f"metrics:{row['_line']}", "label": label, "value": row["value"], "unit": row.get("unit"),
        "kind": "measured", "source": {"file": METRICS, "line": row["_line"], "ref": row.get("source")},
        "gpu": c.get("gpu"), "chip": c.get("chip"), "node": row.get("node"), "metric": row["metric"],
        "conditions": c, "ts": row.get("ts"), "low_confidence": low,
        "reason": c.get("low_confidence_reason") if low else None, "other_chip": other_chip(row),
        "chip_label": chip_of(row),
    }


def derived(nid: str, label: str, value: float | None, unit: str, formula: str, inputs: list[dict[str, Any]],
            gpu: str | None = None, chip: str | None = None) -> dict[str, Any]:
    first = inputs[0]["source"] if inputs else {"file": None, "line": None, "ref": None}
    labels = {i.get("chip_label") for i in inputs if i.get("chip_label")}
    return {"id": nid, "label": label, "value": value, "unit": unit, "kind": "derived", "formula": formula,
            "inputs": [i["id"] for i in inputs], "source": dict(first), "gpu": gpu, "chip": chip,
            "conditions": {}, "ts": None, "low_confidence": any(i.get("low_confidence") for i in inputs),
            "reason": None, "other_chip": any(i.get("other_chip") for i in inputs),
            "chip_label": labels.pop() if len(labels) == 1 else None}


def missing(nid: str, label: str, why: str, task: str | None = None) -> dict[str, Any]:
    return {"id": nid, "label": label, "value": None, "unit": None, "kind": NOT_MEASURED, "why": why, "task": task,
            "source": {"file": None, "line": None, "ref": None}, "gpu": None, "chip": None, "conditions": {},
            "ts": None, "low_confidence": False, "reason": None, "other_chip": False}


# --- the runs still coming -----------------------------------------------------------------------------------------

# What each awaited run was doing on this date, in words; a run that has landed says so from the files instead.
STILL_COMING_AS_OF = "2026-10-11"


def _is_new_vllm(c: dict[str, Any]) -> bool:
    p = build_parts(c)
    return p["family"] == "vllm" and bool(p["commit"]) and not p["commit"].startswith("b1388b1")


def _is_backport(c: dict[str, Any]) -> bool:
    p = build_parts(c)
    return p["family"] == "vllm" and "vllm-gfx1151" in p["patches"]


def _is_llama(c: dict[str, Any]) -> bool:
    return engine_family(c) == "llama.cpp"


def _is_sweep_cell(c: dict[str, Any]) -> bool:
    return isinstance(c.get("slice"), str) and isinstance(c.get("cell"), str)


# The runs whose rows a later build picks up, with what each adds. `glob` (in the data folder), `path` (a glob from
# the root) and `rows` (a test on a Strix Halo row's conditions) say how a build knows the run has landed.
AWAITED = (
    {"task": "R60", "what": "the teacher on the new vLLM build with the fast decode kernel",
     "status": "being measured on pod C", "sections": ["ceiling", "engines"],
     "where": f"rows in {METRICS} whose engine commit is not b1388b1", "rows": _is_new_vllm},
    {"task": "R58", "what": "llama.cpp (Vulkan and ROCm) on the same chip and prompts as vLLM",
     "status": "measured on both pods; its rows are not committed yet", "sections": ["ceiling", "engines"],
     "where": f"rows in {METRICS} whose conditions.engine names llama.cpp", "rows": _is_llama},
    {"task": "R53", "what": "the first vLLM commit with the fast decode kernel backported (lab/vllm-gfx1151.diff)",
     "status": "the patch is on main; its benchmark rows are not committed yet", "sections": ["ceiling", "engines"],
     "where": f"rows in {METRICS} whose patch is vllm-gfx1151.diff", "rows": _is_backport},
    {"task": "R52", "what": "the request-side settings sweep (batch size, thinking, prompt length)",
     "status": "the sweep tool is on main; the pod run is not committed yet", "sections": ["frontier"],
     "where": f"{DATA_DIR}/sweep-request-gfx1151-<date>.json, rows in {METRICS}",
     "glob": "sweep-request-gfx1151-*.json", "rows": _is_sweep_cell},
    {"task": "R47", "what": "the decode profile at batch 1 and batch 4",
     "status": "the profiler is on main; the pod run is not committed yet", "sections": ["step"],
     "where": f"{DATA_DIR}/decode-profile-gfx1151-<date>.json", "glob": "decode-profile-gfx1151-*.json"},
    {"task": "L16", "what": "the first real fine-tune's loss and step time, step by step",
     "status": "its run files are not on main yet", "sections": ["finetune"],
     "where": f"{RUN_DIR}/l16-telemetry.jsonl and {RUN_DIR}/l16-run.json", "path": f"{RUN_DIR}/*-telemetry.jsonl"},
)


def awaited(root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """AWAITED with, per run, whether it has landed and the evidence: its file, or how many Strix Halo rows match."""
    here = [r for r in rows if chip_of(r) == CHIP]
    out = []
    for a in AWAITED:
        files = sorted(_rel(root, p) for p in (root / DATA_DIR).glob(a["glob"])) if a.get("glob") else []
        files += sorted(_rel(root, p) for p in root.glob(a["path"])) if a.get("path") else []
        matched = [r["_line"] for r in here if a.get("rows") and a["rows"](r["conditions"])]
        evidence = ", ".join(files) if files else (
            f"{len(matched)} {'row' if len(matched) == 1 else 'rows'} in {METRICS} (from line {matched[0]})"
            if matched else None)
        out.append({"task": a["task"], "what": a["what"], "where": a["where"], "sections": a["sections"],
                    "status": a["status"], "as_of": STILL_COMING_AS_OF, "file_present": bool(files),
                    "landed": bool(files or matched), "evidence": evidence})
    return out


def _awaiting(section: str) -> list[str]:
    return [a["task"] for a in AWAITED if section in a["sections"]]


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
            "ts": sheet.get("ended"), "low_confidence": False, "reason": None, "other_chip": sheet.get("gfx") != CHIP,
            "chip_label": sheet.get("gfx")}


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
                    "gpu": "Radeon 8060S (other people's machines, published)", "chip": CHIP,
                    "conditions": {"gb_per_token": gb, "fit_rows": fit.rows, "fit_sources": fit.sources,
                                   "fit_r2": round(fit.r2, 3), "fit_error_factor": round(math.exp(fit.rms_log), 2)},
                    "ts": None, "low_confidence": True, "reason": "a prior fitted to published benchmarks, not ours",
                    "other_chip": False})
    return out, None


def decode_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Teacher batch-1 decode rows (per stream), oldest first."""
    found = [r for r in rows if r["metric"].endswith(".decode_tok_s") and r.get("node") != STUDENT_NODE
             and batch_of(r) == 1]
    return sorted(found, key=lambda r: (str(r.get("ts")), r["_line"]))


def section_ceiling(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    bw, gb = chip_bandwidth(root), bytes_per_token(root)
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
    share_of = {f["inputs"][0]: f for f in fractions}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for a in here:  # oldest first, so a newer build is a later bar
        grouped.setdefault(build_label(a["conditions"]), []).append(a)
    builds = []
    for label, members in grouped.items():
        best = max(members, key=lambda a: a["value"])
        builds.append({"build": label, "engine": engine_family(best["conditions"]), "best": best, "rows": members,
                       "share": share_of.get(best["id"])})
    priors, prior_gap = prior_numbers(root, gb["value"])
    notes = []
    if ceiling["value"]:
        notes.append(f"The ceiling is {bw['value']:g} GB/s (the chip sheet's copy over its largest buffer, bytes read "
                     f"and written) divided by {gb['value']:g} GB of weights read for each token (an estimate): "
                     f"{ceiling['value']:.1f} tokens a second at batch 1.")
        for p in priors:
            if p["value"] > ceiling["value"]:
                factor = p["conditions"]["fit_error_factor"]
                inside = p["value"] / ceiling["value"] <= factor
                notes.append(f"The {PRIOR_NAMES.get(p['backend'], p['backend'])} prior ({p['value']:.1f}) sits above "
                             f"the ceiling. The prior is a fit to {p['conditions']['fit_rows']} published rows "
                             f"(typical error a factor of {factor:g}) and the ceiling rests on an estimate, so "
                             + ("the two agree within the fit's own error." if inside else
                                "one of the two is off by more than the fit's own error."))
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
            "inputs": [bw, gb], "ceiling": ceiling, "achieved": here, "shares": fractions, "builds": builds,
            "priors": priors, "prior_gap": prior_gap, "notes": notes, "other_chip": apart,
            "awaiting": _awaiting("ceiling")}


# --- section 2: the batching frontier -----------------------------------------------------------------------------

def sweeps(rows: list[dict[str, Any]], metric_suffix: str) -> list[dict[str, Any]]:
    """Rows of one metric grouped by (metric, node, build, session, slice) where the group spans two or more batch
    sizes. The lab's chip first; within a chip, the sweep with the most stream counts, then the widest span, then
    the one measured first, so the headline sweep does not change when a second pod repeats it."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        if not r["metric"].endswith(metric_suffix) or batch_of(r) is None or r.get("node") == STUDENT_NODE:
            continue
        c = r["conditions"]
        key = (r["metric"], r.get("node"), build_of(c), c.get("session") or c.get("run_id") or "",
               c.get("slice") if isinstance(c.get("slice"), str) else "", chip_of(r))
        groups.setdefault(key, []).append(r)
    out = []
    for (metric, node, build, session, _, chip), members in groups.items():
        levels = sorted({batch_of(m) for m in members})
        if len(levels) < 2:
            continue
        points = []
        for b in levels:  # the last line written at a batch size wins
            r = max((m for m in members if batch_of(m) == b), key=lambda m: m["_line"])
            points.append({"batch": b, "number": from_row(r, f"{metric} at {b:g} {'stream' if b == 1 else 'streams'}")})
        first = min(m["_line"] for m in members)
        out.append({"metric": metric, "prefix": metric[: -len(metric_suffix)], "node": node, "build": build,
                    "session": session, "other_chip": chip != CHIP, "chip_label": chip,
                    "label": f"{node}, {build_label(members[0]['conditions'])}", "first_line": first,
                    "points": points})
    return sorted(out, key=lambda s: (s["other_chip"], -len(s["points"]),
                                      -(s["points"][-1]["batch"] / s["points"][0]["batch"]), s["first_line"]))


def mate(sweep: dict[str, Any], others: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The sweep of another metric from the same run: same metric prefix, node and session."""
    key = (sweep["prefix"], sweep["node"], sweep["session"])
    return next((o for o in others if (o["prefix"], o["node"], o["session"]) == key), None)


def _ratio(nid: str, label: str, sweep: dict[str, Any], formula: str, flip: bool = False) -> dict[str, Any] | None:
    first, last = sweep["points"][0], sweep["points"][-1]
    a, b = first["number"]["value"], last["number"]["value"]
    if not a or not b:
        return None
    return derived(nid, label.format(lo=f"{first['batch']:g}", hi=f"{last['batch']:g}"),
                   round(b / a if flip else a / b, 1), "x", formula, [first["number"], last["number"]],
                   gpu=first["number"]["gpu"], chip=first["number"]["chip"])


def section_frontier(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    energy = sweeps(rows, ".joules_per_token")
    total = sweeps(rows, ".decode_tok_s_total")
    power = sweeps(rows, ".watts_avg")
    here = [s for s in energy if not s["other_chip"]]
    ratios, more_work, other_ratios = [], [], []
    for s in energy:
        r = _ratio(f"derived:ratio:{s['metric']}:{s['node']}:{s['session']}",
                   "joules per token, {lo} stream over {hi} streams", s,
                   "J/token at the fewest streams / J/token at the most streams")
        if r:
            (other_ratios if s["other_chip"] else ratios).append(r)
    for s in here:
        t = mate(s, total)
        r = t and _ratio(f"derived:words:{t['metric']}:{t['node']}:{t['session']}",
                         "tokens per second in all, {hi} streams over {lo} stream", t,
                         "tok/s over all streams at the most streams / at the fewest", flip=True)
        if r:
            more_work.append(r)
    pending = sorted(p.name for p in (root / DATA_DIR).glob("sweep-request-gfx1151-*.json"))
    changes = [from_row(r, f"{r['metric']}: {r['conditions'].get('from')} against {r['conditions'].get('against')}")
               for r in rows if r["metric"].endswith("_change_pct") and chip_of(r) == CHIP]
    if ratios:
        r, s = ratios[0], here[0]
        reading = (f"Serving {s['points'][-1]['batch']:g} conversations at once instead of one cuts the energy for each "
                   f"word by {r['value']:g} times")
        watts, words = mate(s, power), mate(s, total)
        lo = watts and watts["points"][0]["number"]["value"]
        hi = watts and watts["points"][-1]["number"]["value"]
        if lo and hi and words and more_work and max(lo, hi) / min(lo, hi) <= 1.1:
            reading += (f": the chip draws about the same {(lo + hi) / 2:.0f} watts either way and turns out "
                        f"{more_work[0]['value']:.0f} times as many words")
        reading += "."
        if len(here) > 1 and len(ratios) > 1:
            reading += (f" A second sweep, on {here[1]['node']}, gives {ratios[1]['value']:g} times at "
                        f"{here[1]['points'][-1]['batch']:g} streams.")
    else:
        reading = "Not measured yet: no joules-per-token sweep across stream counts from the Strix Halo is on main."
    return {"id": "frontier", "title": "The batching frontier: joules per token against concurrent streams",
            "reading": reading, "state": "measured" if here else "empty", "energy": here,
            "throughput": [s for s in total if not s["other_chip"]],
            "power": [s for s in power if not s["other_chip"]], "ratios": ratios, "more_work": more_work,
            "changes": changes, "other_chip": [s for s in energy + total if s["other_chip"]],
            "other_ratios": other_ratios, "pending_files": pending, "awaiting": _awaiting("frontier")}


# --- section 3: where a decode step's time goes -------------------------------------------------------------------

def section_step(root: Path) -> dict[str, Any]:
    path = newest(root / DATA_DIR, f"decode-profile-{CHIP}-*.json")
    base = {"id": "step", "title": "Where a decode step's time goes", "awaiting": _awaiting("step"), "other_chip": []}
    if path is None:
        return {**base, "state": "empty", "batches": [],
                "reading": "Not measured yet: the decode profile (python -m lab profile-decode) has not committed a "
                           f"result under {DATA_DIR}.",
                "why": f"no {DATA_DIR}/decode-profile-{CHIP}-<date>.json on main"}
    data = json.loads(path.read_text(encoding="utf-8"))
    rel = _rel(root, path)
    gpu, chip = data.get("gpu"), data.get("chip")
    apart = chip not in (None, CHIP)
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
                          "gpu": gpu, "chip": chip, "node": data.get("node"),
                          "conditions": {"batch": key, "breakdown_source": entry.get("breakdown_source")},
                          "ts": data.get("ended"), "low_confidence": False, "reason": None,
                          "other_chip": apart, "chip_label": chip or CHIP})
        batches.append({"batch": key, "step_ms": _num(ms.get("step")), "parts": parts})
    measured = [p for b in batches for p in b["parts"] if p["value"] is not None]
    top = None
    if measured and batches and batches[0]["step_ms"]:
        first = batches[0]
        biggest = max((p for p in first["parts"] if p["value"] is not None), key=lambda p: p["value"])
        name = biggest["label"].split(",")[0]
        top = derived(f"derived:step_share:{first['batch']}:{biggest['bucket']}",
                      f"share of a decode step in {name}, batch {first['batch']}",
                      round(100 * biggest["value"] / first["step_ms"], 1), "%",
                      "100 x the bucket's ms per step / the step's ms", [biggest], gpu=gpu, chip=chip)
        reading = (f"At batch {first['batch']}, one decode step takes {first['step_ms']:,.0f} ms and the biggest slice, "
                   f"{top['value']:.0f}%, is {name}: that is where a faster kernel pays off first.")
    else:
        reading = f"{rel} is committed but gives no step breakdown."
    return {**base, "state": "measured" if measured else "empty", "batches": batches, "reading": reading,
            "top": top, "file": rel, "other_chip": batches if apart else []}


# --- section 4: engines side by side ------------------------------------------------------------------------------

def section_engines(root: Path, ceiling: dict[str, Any]) -> dict[str, Any]:
    here = ceiling["achieved"]
    families: dict[str, list[dict[str, Any]]] = {}
    for a in here:
        families.setdefault(engine_family(a["conditions"]), []).append(a)
    unnamed = families.pop("not recorded", [])  # a row that names no engine cannot stand for one
    # Like with like: when two or more engines have a row for the same prompt, each engine's bar is its best row
    # under that prompt; an engine with no such row keeps its best row and the difference is said aloud.
    asked: dict[str, set[str]] = {}
    for name, items in families.items():
        for a in items:
            asked.setdefault(asked_of(a["conditions"]) or "", set()).add(name)
    shared = max(asked, key=lambda k: (len(asked[k]), k != "", k)) if asked else None
    if shared is None or len(asked[shared]) < 2:
        shared = None
    best = {}
    for name, items in families.items():
        same = [a for a in items if shared is not None and (asked_of(a["conditions"]) or "") == shared]
        best[name] = max(same or items, key=lambda a: a["value"])
    names = sorted(best, key=lambda n: (-best[n]["value"], n))
    differences = []
    if len(names) >= 2:  # were the engines asked the same thing? say so when the rows differ
        for what, read in (("prompts", asked_of), ("tokens out", lambda c: c.get("max_tokens")),
                           ("thinking", lambda c: c.get("thinking")),
                           ("weights", lambda c: c.get("quantization") or c.get("checkpoint") or c.get("model"))):
            seen = {name: read(best[name]["conditions"]) for name in names}
            if len({json.dumps(v, sort_keys=True, default=str) for v in seen.values()}) > 1:
                differences.append({"what": what, "by_engine": {n: (v if v is not None else "not recorded")
                                                                 for n, v in seen.items()}})
        top, low = names[0], names[-1]
        reading = (f"On the same chip, {ENGINE_NAMES.get(top, top)} decodes {best[top]['value']:.2f} tokens a second "
                   f"against {best[low]['value']:.2f} for {ENGINE_NAMES.get(low, low)}"
                   + ("; their rows differ in " + " and ".join(d["what"] for d in differences)
                      + ", so read the gap with that in mind." if differences else "."))
        state = "measured"
    elif names:
        only = names[0]
        absent = [ENGINE_NAMES.get(e, e) for e in EXPECTED_ENGINES if e != only]
        reading = (f"Only {ENGINE_NAMES.get(only, only)} has a measured row on this chip so far, at "
                   f"{best[only]['value']:.2f} tokens a second; {' and '.join(absent)} "
                   f"{'is' if len(absent) == 1 else 'are'} not measured here yet, and the public prior beside it is "
                   "other people's numbers, not ours.")
        state = "partial"
    else:
        reading = "Not measured yet: no engine has a batch-1 decode row from the Strix Halo on main."
        state = "empty"
    return {"id": "engines", "title": "Engines side by side", "reading": reading, "state": state,
            "engines": [{"engine": n, "name": ENGINE_NAMES.get(n, n), "best": best[n], "rows": families[n],
                         "builds": sorted({build_label(a["conditions"]) for a in families[n]}),
                         "setup": setup_of(best[n]["conditions"])} for n in names],
            "compared_on": shared or None, "differences": differences, "unnamed": unnamed,
            "not_measured": [e for e in EXPECTED_ENGINES if e not in best],
            "priors": ceiling["priors"], "other_chip": ceiling["other_chip"], "awaiting": _awaiting("engines")}


# --- section 5: the fine-tune run ---------------------------------------------------------------------------------

def train_runs(root: Path) -> list[dict[str, Any]]:
    """Every committed fine-tune run: fixtures/demo/<name>-telemetry.jsonl, one telemetry.train.step event per line
    (step, loss, step_ms), with <name>-run.json beside it for the device and the run's totals. Oldest first."""
    folder = root / RUN_DIR
    runs = []
    for tel in sorted(folder.glob("*-telemetry.jsonl")) if folder.is_dir() else []:
        name = tel.name[: -len("-telemetry.jsonl")]
        run_path = folder / f"{name}-run.json"
        try:
            run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else {}
        except ValueError:
            run = {}
        refs = run.get("refs") if isinstance(run, dict) and isinstance(run.get("refs"), dict) else {}
        device = refs.get("device") if isinstance(refs.get("device"), str) else None
        rel = _rel(root, tel)
        loss, step_time, run_id, node, last_ts = [], [], None, None, None
        for n, line in enumerate(tel.read_text(encoding="utf-8").splitlines(), 1):
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if not isinstance(ev, dict) or ev.get("type") != "telemetry.train.step" or _num(ev.get("step")) is None:
                continue
            run_id, node, last_ts = run_id or ev.get("run_id"), node or ev.get("node"), _ts(ev.get("ts")) or last_ts
            as_row = {"node": ev.get("node"), "ts": ev.get("ts"), "conditions": {"gpu": device} if device else {}}
            base = {"kind": "measured", "gpu": device, "chip": CHIP if device and CHIP in device else None,
                    "node": ev.get("node"), "ts": ev.get("ts"), "low_confidence": False, "reason": None,
                    "source": {"file": rel, "line": n, "ref": f"telemetry.train.step, run {ev.get('run_id')}"},
                    "conditions": {"step": ev["step"], "run_id": ev.get("run_id"), "tokens": ev.get("tokens")},
                    "other_chip": other_chip(as_row), "chip_label": chip_of(as_row)}
            if _num(ev.get("loss")) is not None:
                loss.append({**base, "id": f"train:{name}:{n}:loss", "label": f"loss at step {ev['step']:g}",
                             "value": ev["loss"], "unit": "loss", "metric": "telemetry.train.step.loss"})
            if _num(ev.get("step_ms")) is not None:
                step_time.append({**base, "id": f"train:{name}:{n}:step_ms", "label": f"step time at step {ev['step']:g}",
                                  "value": ev["step_ms"], "unit": "ms", "metric": "telemetry.train.step.step_ms"})
        if not loss and not step_time:
            continue
        sample = (loss or step_time)[0]
        facts = []
        metrics = run.get("metrics") if isinstance(run, dict) and isinstance(run.get("metrics"), dict) else {}
        for key, unit in RUN_FACTS:
            if _num(metrics.get(key)) is not None:
                facts.append({"id": f"train:{name}:{key}", "label": key, "value": metrics[key], "unit": unit,
                              "kind": "measured", "metric": key, "gpu": device, "chip": sample["chip"], "node": node,
                              "source": {"file": _rel(root, run_path), "line": line_of(run_path, f'"{key}"'),
                                         "ref": f"metrics.{key}, run {run_id}"},
                              "conditions": {"run_id": run_id, "base_model": refs.get("base_model"),
                                             "task": refs.get("task")},
                              "ts": last_ts, "low_confidence": False, "reason": None,
                              "other_chip": sample["other_chip"], "chip_label": sample["chip_label"]})
        runs.append({"name": name, "run_id": run_id, "node": node, "gpu": device, "file": rel,
                     "run_file": _rel(root, run_path) if run_path.is_file() else None, "ts": last_ts,
                     "other_chip": sample["other_chip"], "chip_label": sample["chip_label"],
                     "loss": sorted(loss, key=lambda x: x["conditions"]["step"]),
                     "step_time": sorted(step_time, key=lambda x: x["conditions"]["step"]), "facts": facts})
    return sorted(runs, key=lambda r: (str(r["ts"]), r["name"]))


def section_finetune(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    trainer = [r for r in rows if r["metric"].startswith(("trainer.", "telemetry.train."))]

    def curve(word: str) -> list[dict[str, Any]]:
        found = [r for r in trainer if word in r["metric"] and _num(r["conditions"].get("step")) is not None
                 and not other_chip(r)]
        return [from_row(r, f"{r['metric']} at step {r['conditions']['step']}")
                for r in sorted(found, key=lambda r: (_num(r["conditions"]["step"]), r["_line"]))]

    runs = train_runs(root)
    ours = [r for r in runs if not r["other_chip"]]
    run = ours[-1] if ours else None  # the newest run on the lab's chip is the one charted
    loss = run["loss"] if run else curve("loss")
    step_time = run["step_time"] if run else (curve("step_s") or curve("step_time"))
    singles = [from_row(r, r["metric"]) for r in trainer if _num(r["conditions"].get("step")) is None]
    here = [s for s in singles if not s["other_chip"]] + (run["facts"] if run else [])
    apart = [s for s in singles if s["other_chip"]]
    apart += [n for r in runs if r["other_chip"] for n in r["facts"] + r["loss"] + r["step_time"]]
    median = None
    if step_time and all(n["unit"] == "ms" for n in step_time):
        median = derived(f"derived:train:{run['name'] if run else 'rows'}:median_step", "median step time",
                         round(statistics.median(n["value"] for n in step_time) / 1000, 1), "s",
                         "median of the logged step times, in seconds", step_time,
                         gpu=step_time[0]["gpu"], chip=step_time[0]["chip"])
    if loss:
        reading = (f"The fine-tune's loss went from {loss[0]['value']:.3g} to {loss[-1]['value']:.3g} over "
                   f"{len(loss)} logged steps on the Strix Halo"
                   + (f", about {median['value']:.0f} seconds a step." if median else "."))
        state = "measured"
    else:
        reading = ("Not measured yet: the first real fine-tune on the Strix Halo (L16) has not committed per-step loss "
                   "or step time" + ("; the trainer rows on main are from the old chip and are shown apart."
                                     if apart and not here else "."))
        state = "partial" if here else "empty"
    return {"id": "finetune", "title": "The fine-tune run", "reading": reading, "state": state, "loss": loss,
            "step_time": step_time, "median_step": median, "numbers": here,
            "run": {k: run[k] for k in ("name", "run_id", "node", "gpu", "file", "run_file")} if run else None,
            "other_chip": apart, "awaiting": _awaiting("finetune")}


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
    ruled_out = []
    for path in sorted((root / DATA_DIR).glob("searchspace-*.json")):
        try:
            space = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        excluded = space.get("excluded") if isinstance(space, dict) else None
        for i, entry in enumerate(excluded if isinstance(excluded, list) else []):
            if not isinstance(entry, dict) or not isinstance(entry.get("setting"), str):
                continue
            line = (line_of(path, f'"setting": {json.dumps(entry["setting"])}')
                    or line_of(path, json.dumps(entry["setting"])))
            ruled_out.append({"what": entry["setting"], "verdict": "left out of the search before any run",
                              "note": entry.get("why"), "kind": "ruled out",
                              "source": {"file": _rel(root, path), "line": line,
                                         "ref": f"excluded[{i}] of {space.get('name') or path.name}"},
                              "gpu": None, "task": None})
    items += ruled_out
    slide = root / "lab" / "slide_numbers.py"
    moved_line = line_of(slide, "LAB_MOVED_AT =")
    old = [r for r in rows if chip_of(r) == "R9700 (gfx1201)"]
    moved = bool(moved_line and old)
    if moved:
        items.append({"what": "the R9700 cards (gfx1201, DUB hub) as the lab",
                      "verdict": "left for the Strix Halo pods (gfx1151, SJC) on 2026-10-10",
                      "note": f"{len(old)} rows on main describe it; every section shows them apart",
                      "kind": "record", "source": {"file": "lab/slide_numbers.py", "line": moved_line,
                                                   "ref": "LAB_MOVED_AT, the rule every reader uses"},
                      "gpu": "R9700 (gfx1201)", "task": None})
    ledger_items = [i for i in items if i["kind"] == "measured"]
    if ledger_items:
        reading = f"{len(ledger_items)} experiments have a recorded verdict."
    else:
        record = ([f"{len(ruled_out)} settings ruled out before any pod time was spent on them"] if ruled_out else [])
        record += ["the move off the old chip"] if moved else []
        reading = ("The experiments' verdicts are not on main yet (the lab's ledger lives on the lab machine)"
                   + ("; what is on record is " + " and ".join(record) + "." if record else "."))
    return {"id": "dropped", "title": "What was tried and dropped", "reading": reading,
            "state": "measured" if ledger_items else ("partial" if items else "empty"), "items": items,
            "other_chip": [], "awaiting": []}


# --- the pack -----------------------------------------------------------------------------------------------------

def _sources(root: Path) -> list[dict[str, Any]]:
    data, runs = root / DATA_DIR, root / RUN_DIR
    spaces = [p for p in sorted(data.glob("searchspace-*.json")) if '"excluded"' in p.read_text(encoding="utf-8")]
    paths = [("rows", root / METRICS), ("chip sheet", newest(data, f"chipsheet-{CHIP}-*.json")),
             ("decode profile", newest(data, f"decode-profile-{CHIP}-*.json")),
             ("public prior data", data / "public-strix-halo-benchmarks.csv"),
             ("public prior model", root / "lab" / "public_prior.py"), ("chip rule", root / "lab" / "slide_numbers.py"),
             *(("ledger", p) for p in sorted(data.glob("ledger*.jsonl"))),
             *(("settings sweep", p) for p in sorted(data.glob("sweep-request-gfx1151-*.json"))),
             *(("search space", p) for p in spaces),
             *(("fine-tune run", p) for p in sorted([*runs.glob("*-telemetry.jsonl"), *runs.glob("*-run.json")])),
             ("demo numbers sheet", newest(root / SHEET_DIR, "demo_numbers_*.md"))]
    out = []
    for role, p in paths:
        if p is not None and p.is_file():
            data_bytes = p.read_bytes()
            out.append({"file": _rel(root, p), "role": role, "sha256": hashlib.sha256(data_bytes).hexdigest(),
                        "lines": data_bytes.count(b"\n")})
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
                section_finetune(root, rows), section_dropped(root, rows)]
    numbers, seen = [], set()
    for s in sections:
        for n in _walk(s):
            if n["id"] not in seen:
                seen.add(n["id"])
                numbers.append(n)
    times = sorted(t for t in (_ts(r.get("ts")) for r in rows) if t)
    return {
        "format": FORMAT,
        "title": "Under the hood: the Atlas lab on the Strix Halo",
        "chip": {"current": CHIP, "name": "Radeon 8060S (Strix Halo)",
                 "other": "every other chip, shown apart by name: the R9700 (gfx1201, the lab until 2026-10-10), the "
                          "A100 reference and any GPU a row names",
                 "rule": "lab/report.py chip_of, on top of lab/slide_numbers.py _old_chip"},
        "data_as_of": times[-1] if times else None,
        "sources": _sources(root),
        "skipped": skipped,
        "sections": sections,
        "numbers": numbers,
        "awaited": awaited(root, rows),
    }


# --- the charts ---------------------------------------------------------------------------------------------------

W = 340  # every chart's viewBox width: close to 1:1 on a phone, scaled up by the page's CSS on a wide screen
STACK_CLS = ("c1", "c2", "c3", "c4", "c5", "cg")  # one fixed colour per decode-step bucket; host side is the gray
STATE_WORDS = {"measured": "measured", "partial": "partly measured", "empty": NOT_MEASURED}
PRIOR_NAMES = {"vllm": "vLLM", "llama.cpp-rocm": "llama.cpp ROCm", "llama.cpp-vulkan": "llama.cpp Vulkan"}


def esc(text: Any) -> str:
    return html.escape("" if text is None else str(text), quote=True)


def fmt(value: Any, unit: str | None = None) -> str:
    if value is None:
        return NOT_MEASURED
    v = float(value)
    text = f"{v:,.0f}" if abs(v) >= 100 else f"{v:.1f}" if abs(v) >= 10 else f"{v:.3g}"
    if unit == "x":
        return f"{text} times"
    return f"{text} {unit}" if unit else text


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _nice(top: float, ticks: int = 4) -> tuple[float, float]:
    """(step, axis top) of a linear axis from zero that covers `top` in about `ticks` round steps."""
    if top <= 0:
        return 1.0, 1.0
    raw = top / ticks
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag * (1 + 1e-9))
    return step, step * math.ceil(top / step - 1e-9)


def _open(height: float, title: str) -> str:
    return (f'<svg viewBox="0 0 {W} {height:.0f}" role="img" aria-label="{esc(title)}" class="chart">'
            f"<title>{esc(title)}</title>")


def _bar(x: float, y: float, w: float, h: float, cls: str, tip: str) -> str:
    """A horizontal bar from its baseline at x: square at the baseline, rounded at the data end."""
    r = min(4.0, w / 2, h / 2)
    return (f'<path d="M{x:.1f},{y:.1f}h{w - r:.1f}a{r:.1f},{r:.1f} 0 0 1 {r:.1f},{r:.1f}v{h - 2 * r:.1f}'
            f'a{r:.1f},{r:.1f} 0 0 1 -{r:.1f},{r:.1f}h-{w - r:.1f}z" class="{cls}"><title>{esc(tip)}</title></path>')


def _column(x: float, base: float, w: float, h: float, cls: str, tip: str) -> str:
    """A column standing on `base`: square at the baseline, rounded at the top."""
    r = min(4.0, w / 2, h / 2)
    return (f'<path d="M{x:.1f},{base:.1f}v-{h - r:.1f}a{r:.1f},{r:.1f} 0 0 1 {r:.1f},-{r:.1f}h{w - 2 * r:.1f}'
            f'a{r:.1f},{r:.1f} 0 0 1 {r:.1f},{r:.1f}v{h - r:.1f}z" class="{cls}"><title>{esc(tip)}</title></path>')


def svg_bars(items: list[dict[str, Any]], ref: tuple[str, float] | None, unit: str, title: str) -> str:
    """Horizontal bars, each under its own full label. An item is {"label", "sub", "value", "cls"} (a None value
    says "not measured") or {"header": text} for a group title; `ref` is a (name, value) line across all of them."""
    left, right, bar_h = 1, 58, 14
    top = 22 if ref else 2
    peak = max([i["value"] for i in items if i.get("value")] + ([ref[1]] if ref else []) + [1e-9])
    scale = (W - left - right) / peak
    body, y = [], top
    for it in items:
        if "header" in it:
            body.append(f'<text x="{left}" y="{y + 12}" class="hd">{esc(it["header"])}</text>')
            y += 22
            continue
        body.append(f'<text x="{left}" y="{y + 11}" class="lbl">{esc(_clip(it["label"], 50))}'
                    f'<title>{esc(it["label"])}</title></text>')
        y += 16
        if it.get("sub"):
            body.append(f'<text x="{left}" y="{y + 9}" class="sub">{esc(_clip(it["sub"], 58))}</text>')
            y += 13
        y += 2
        if it.get("value") is None:
            body.append(f'<text x="{left}" y="{y + 11}" class="nm">{NOT_MEASURED}</text>')
        else:
            w = max(2.0, it["value"] * scale)
            body.append(_bar(left, y, w, bar_h, it["cls"], f'{it["label"]}: {fmt(it["value"], unit)}'))
            body.append(f'<text x="{left + w + 6:.1f}" y="{y + 11.5}" class="val">{esc(fmt(it["value"]))}</text>')
        y += bar_h + 12
    out = [_open(y, title)]
    if ref:  # drawn first, so every bar and value sits over it
        x = left + ref[1] * scale
        out.append(f'<line x1="{x:.1f}" y1="18" x2="{x:.1f}" y2="{y - 8}" class="ref"/>')
        anchor = "end" if x > W * 0.55 else "start"
        out.append(f'<text x="{x:.1f}" y="12" text-anchor="{anchor}" class="val">{esc(ref[0])} '
                   f'{esc(fmt(ref[1], unit))}</text>')
    return "".join(out + body) + "</svg>"


def svg_xy(series: list[dict[str, Any]], *, xlog: bool, xname: str, yname: str, yfmt: Any, title: str,
           ref: float | None = None, note: tuple[str, str] | None = None, height: int = 232) -> str:
    """Lines over one y axis from zero. A series is {"points": [(x, y)], "line": class, "fill": class}; the first
    one's first and last points carry their values. `ref` is a dashed level, `note` two lines at the top right."""
    left, right, top, bottom = 36, 14, 34, 38
    xs = sorted({x for s in series for x, _ in s["points"]})
    fx = math.log2 if xlog else float
    lo, span = fx(xs[0]), (fx(xs[-1]) - fx(xs[0])) or 1.0
    step, ymax = _nice(max(y for s in series for _, y in s["points"]))

    def px(x: float) -> float:
        return left + (fx(x) - lo) / span * (W - left - right)

    def py(y: float) -> float:
        return top + (1 - y / ymax) * (height - top - bottom)

    out = [_open(height, title), f'<text x="0" y="12" class="ax">{esc(yname)}</text>']
    for i in range(int(round(ymax / step)) + 1):
        t = i * step
        out.append(f'<line x1="{left}" y1="{py(t):.1f}" x2="{W - right}" y2="{py(t):.1f}" '
                   f'class="{"base" if i == 0 else "grid"}"/>')
        out.append(f'<text x="{left - 5}" y="{py(t) + 4:.1f}" text-anchor="end" class="ax">{t:g}</text>')
    every = math.ceil(len(xs) / 7)
    for i, x in enumerate(xs):
        if i % every == 0 or x == xs[-1]:
            out.append(f'<text x="{px(x):.1f}" y="{height - bottom + 15}" text-anchor="middle" class="ax">{x:g}</text>')
    out.append(f'<text x="{(left + W - right) / 2:.1f}" y="{height - 4}" text-anchor="middle" class="ax">'
               f"{esc(xname)}</text>")
    if ref is not None:
        out.append(f'<line x1="{left}" y1="{py(ref):.1f}" x2="{W - right}" y2="{py(ref):.1f}" class="ref"/>')
    for s in series:
        path = " ".join(f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(y):.1f}" for i, (x, y) in enumerate(s["points"]))
        out.append(f'<path d="{path}" class="ln {s["line"]}"/>')
        for x, y in s["points"]:
            out.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4" class="dot {s["fill"]}">'
                       f"<title>{x:g} {esc(xname)}: {esc(yfmt(y))}</title></circle>")
    (x0, y0), (x1, y1) = series[0]["points"][0], series[0]["points"][-1]
    out.append(f'<text x="{px(x0) + 3:.1f}" y="{py(y0) - 9:.1f}" class="val">{esc(yfmt(y0))}</text>')
    out.append(f'<text x="{px(x1):.1f}" y="{py(y1) - 10:.1f}" text-anchor="end" class="val">{esc(yfmt(y1))}</text>')
    if note:
        out.append(f'<text x="{W - right}" y="{top + 12}" text-anchor="end" class="val">{esc(note[0])}</text>')
        out.append(f'<text x="{W - right}" y="{top + 27}" text-anchor="end" class="ax">{esc(note[1])}</text>')
    return "".join(out) + "</svg>"


def svg_columns(points: list[tuple[float, float]], *, xname: str, yname: str, title: str, cls: str = "c1",
                height: int = 176) -> str:
    """One column per x, its value on the cap."""
    left, right, top, bottom = 4, 4, 34, 36
    slot = (W - left - right) / len(points)
    width = min(24.0, slot * 0.6)
    peak = max(v for _, v in points) or 1e-9
    base = height - bottom
    out = [_open(height, title), f'<text x="0" y="12" class="ax">{esc(yname)}</text>',
           f'<line x1="{left}" y1="{base}" x2="{W - right}" y2="{base}" class="base"/>']
    for i, (x, v) in enumerate(points):
        cx = left + slot * (i + 0.5)
        h = max(2.0, v / peak * (base - top))
        out.append(_column(cx - width / 2, base, width, h, cls, f"{x:g} {xname}: {fmt(v)}"))
        out.append(f'<text x="{cx:.1f}" y="{base - h - 5:.1f}" text-anchor="middle" class="val">{esc(fmt(v))}</text>')
        out.append(f'<text x="{cx:.1f}" y="{base + 15}" text-anchor="middle" class="ax">{x:g}</text>')
    out.append(f'<text x="{W / 2:.0f}" y="{height - 4}" text-anchor="middle" class="ax">{esc(xname)}</text>')
    return "".join(out) + "</svg>"


def svg_stack(batches: list[dict[str, Any]], title: str) -> str:
    """One stacked bar per batch size: milliseconds of a decode step by bucket, a gap between the slices."""
    left, bar_h = 1, 22
    peak = max([max(b["step_ms"] or 0, sum(p["value"] or 0 for p in b["parts"])) for b in batches] + [1e-9])
    scale = (W - 2 * left) / peak
    body, y = [], 2
    for b in batches:
        body.append(f'<text x="{left}" y="{y + 12}" class="lbl">batch {esc(b["batch"])}: '
                    f'{esc(fmt(b["step_ms"], "ms"))} a step</text>')
        y += 20
        x = float(left)
        for i, p in enumerate(b["parts"]):
            if p["value"]:
                w = p["value"] * scale
                body.append(f'<rect x="{x:.1f}" y="{y}" width="{max(w - 2, 0.6):.1f}" height="{bar_h}" rx="2" '
                            f'class="{STACK_CLS[i]}"><title>{esc(p["label"])}: {esc(fmt(p["value"], "ms"))}</title></rect>')
                x += w
        y += bar_h + 14
    return _open(y, title) + "".join(body) + "</svg>"


# --- the page -----------------------------------------------------------------------------------------------------

LIGHT = {"page": "#f9f9f7", "card": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "mute": "#6f6d68",
         "line": "#e1e0d9", "axis": "#c3c2b7", "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#eda100",
         "s5": "#e87ba4", "gray": "#898781", "link": "#1c5cab", "flag": "#a3301f"}
DARK = {"page": "#0d0d0d", "card": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "mute": "#a3a199",
        "line": "#2c2c2a", "axis": "#4a4a46", "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70", "s4": "#c98500",
        "s5": "#d55181", "gray": "#898781", "link": "#86b6ef", "flag": "#f0906f"}


def _vars(theme: dict[str, str]) -> str:
    return ";".join(f"--{k}:{v}" for k, v in theme.items())


CSS = (
    f":root{{{_vars(LIGHT)}}}"
    f"@media (prefers-color-scheme:dark){{:root:not([data-theme=\"light\"]){{{_vars(DARK)}}}}}"
    f":root[data-theme=\"dark\"]{{{_vars(DARK)}}}"
    f"body:has(#day:checked){{{_vars(LIGHT)}}}body:has(#night:checked){{{_vars(DARK)}}}"
    """
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%;font-size:16px}
body{margin:0;min-height:100vh;background:var(--page);color:var(--ink);
font:1rem/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:18px 16px 64px}
h1{font-size:clamp(1.6rem,4.2vw,2.5rem);line-height:1.15;margin:.3em 0}
h2{font-size:clamp(1.2rem,2.6vw,1.6rem);line-height:1.25;margin:0 0 .35em}
h3{font-size:.95rem;margin:0 0 .4em}a{color:var(--link)}
.lede{color:var(--ink2);max-width:74ch;margin:.2em 0 1em}
.theme{float:right;font-size:.85rem;color:var(--ink2);margin:.5em 0 .4em 1em}.theme input{accent-color:var(--s1)}
.toc{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin:0;padding:0;list-style:none;clear:both}
.toc a{display:block;padding:6px 10px;border:1px solid var(--line);border-radius:10px;background:var(--card);
color:var(--ink);text-decoration:none;font-size:.9rem}.toc b{color:var(--ink2);margin-right:.35em}
.toc .st{display:table;margin:2px 0 0}
.st{display:inline-block;font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.05em;
padding:1px 7px;border-radius:99px;border:1px solid var(--axis);color:var(--ink2);vertical-align:middle;
margin-left:.5em;white-space:nowrap}.st-measured{border-color:var(--s1);color:var(--ink)}
.st-empty{border-color:var(--flag);color:var(--flag)}
section{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 16px;margin:18px 0}
.n{color:var(--ink2);font-weight:600;margin-right:.4em}
.say{font-size:clamp(1.1rem,2.3vw,1.4rem);line-height:1.35;font-weight:500;margin:.2em 0 .9em;max-width:62ch}
.body{display:grid;gap:4px 18px;align-items:start}figure{margin:0 0 14px}
figcaption{font-size:.8rem;color:var(--ink2);margin-top:6px;overflow-wrap:anywhere}
.chart{width:100%;max-width:640px;height:auto;display:block}
.chart text{font-size:13px;fill:var(--ink)}.chart .sub,.chart .ax{font-size:11.5px;fill:var(--mute)}
.chart .hd{font-size:11px;font-weight:700;fill:var(--ink2);letter-spacing:.04em;text-transform:uppercase}
.chart .val{font-weight:650;paint-order:stroke;stroke:var(--card);stroke-width:4px;stroke-linejoin:round}
.chart .nm{fill:var(--flag);font-style:italic}.chart .grid{stroke:var(--line)}.chart .base{stroke:var(--axis)}
.chart .ref{stroke:var(--ink2);stroke-width:1.5;stroke-dasharray:5 4;fill:none}
.chart .c1{fill:var(--s1)}.chart .c2{fill:var(--s2)}.chart .c3{fill:var(--s3)}.chart .c4{fill:var(--s4)}
.chart .c5{fill:var(--s5)}.chart .cg{fill:var(--gray)}
.chart .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.chart .l1{stroke:var(--s1)}.chart .l2{stroke:var(--s2)}.chart .lg{stroke:var(--gray)}
.chart .dot{stroke:var(--card);stroke-width:2}
.key{border-left:4px solid var(--s1);padding:2px 0 2px 14px;margin:0 0 14px}
.key .big{font-size:clamp(2.2rem,6vw,3.4rem);font-weight:700;line-height:1.05}
.key .kl{max-width:38ch;margin:.2em 0 .3em}
.note{font-size:.88rem;color:var(--ink2);margin:.5em 0;max-width:72ch}
.cite{display:block;font-size:.78rem;color:var(--ink2);overflow-wrap:anywhere}
.tag{display:inline-block;font-size:.7rem;border:1px solid var(--axis);border-radius:4px;padding:0 5px;
white-space:nowrap}.tag.oc{border-style:dashed}
.empty{border-left:4px solid var(--flag);padding:6px 12px;margin:0 0 14px;font-weight:600}
.coming{border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin:0 0 12px;font-size:.9rem}
.coming h3 span{font-weight:400;color:var(--ink2)}
.coming ul{margin:0;padding-left:1.1em}.coming li{margin:.3em 0}.coming .when{color:var(--ink2)}
.items{list-style:none;margin:0 0 16px;padding:0}
.items li{padding:9px 0;border-top:1px solid var(--line)}.items li:first-child{border-top:0}
.items .k{font-weight:600}.items .verdict{color:var(--ink2);font-size:.9rem}.items .why{font-size:.92rem}
details{margin-top:14px;border-top:1px solid var(--line);padding-top:10px}
summary{cursor:pointer;font-size:.9rem;color:var(--ink2)}
.nums{list-style:none;margin:8px 0 0;padding:0}
.nums li{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:2px 12px;padding:8px 0;
border-top:1px solid var(--line);align-items:baseline}.nums li:first-child{border-top:0}
.nums .k{font-size:.9rem}.nums .v{font-weight:650;white-space:nowrap;font-variant-numeric:tabular-nums}
.nums .cite{grid-column:1/-1}
.legend{list-style:none;margin:8px 0 0;padding:0;font-size:.85rem;display:flex;flex-wrap:wrap;gap:4px 16px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
.sw.c1{background:var(--s1)}.sw.c2{background:var(--s2)}.sw.c3{background:var(--s3)}.sw.c4{background:var(--s4)}
.sw.c5{background:var(--s5)}.sw.cg{background:var(--gray)}
.apart{border:1px dashed var(--gray);border-radius:10px;padding:12px;margin-top:14px}
.apart h3{color:var(--ink2)}.apart details{border-top:0;margin-top:6px;padding-top:0}.apart .chart{max-width:440px}
table{border-collapse:collapse;width:100%;font-size:.85rem}
td,th{border-bottom:1px solid var(--line);padding:6px 10px 6px 0;text-align:left;vertical-align:top}
td.num{white-space:nowrap;font-variant-numeric:tabular-nums}.wrap{overflow-x:auto}
footer{color:var(--ink2);font-size:.85rem;margin-top:28px}
@media (min-width:900px){
.toc{grid-template-columns:repeat(6,minmax(0,1fr))}
.nums li{grid-template-columns:minmax(0,1.1fr) 9em minmax(0,1.9fr)}.nums .cite{grid-column:auto}
.body{grid-template-columns:minmax(0,1.2fr) minmax(0,.8fr);grid-template-rows:auto 1fr;gap:4px 30px;
grid-template-areas:"viz keys" "viz more"}.viz{grid-area:viz}.keys{grid-area:keys}.more{grid-area:more}
.body.one{grid-template-columns:minmax(0,1fr);grid-template-rows:auto;grid-template-areas:"keys" "more"}
.items li{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.5fr);gap:2px 28px}
section{padding:24px 26px}}
@media (min-width:1600px){html{font-size:20px}main{max-width:1560px}.chart{max-width:680px}}
""")


def href(page_dir: Path, root: Path, rel: str) -> str:
    """A relative link from the page's folder to a committed file, so it opens from disk."""
    return Path(os.path.relpath(root / rel, page_dir)).as_posix()


class Page:
    def __init__(self, root: Path, page_dir: Path, pack: dict[str, Any]):
        self.root, self.page_dir, self.pack = root, page_dir, pack

    def link(self, rel: str, line: Any = None, text: str | None = None) -> str:
        where = text or (f"{rel}:{line}" if line else rel)
        return (f'<a href="{esc(href(self.page_dir, self.root, rel))}" data-line="{esc(line or "")}">'
                f"{esc(where)}</a>")

    @staticmethod
    def chip_text(n: dict[str, Any]) -> str:
        if n.get("kind") == "stated":
            return "not a measurement, so no gpu"
        if n.get("gpu"):
            return f"gpu: {n['gpu']}"
        if n.get("chip"):
            return f"chip: {n['chip']} (the row records no gpu)"
        if n.get("other_chip") and n.get("chip_label"):
            return f"gpu: not recorded in the row; {n['chip_label']} by the date it was measured"
        return "gpu: not recorded"

    def cite(self, n: dict[str, Any], reason: bool = True) -> str:
        """The source line under a number: file and line as a link, the row's gpu, its conditions and its flags."""
        src = n.get("source") or {}
        if not src.get("file"):
            return f'<span class="cite">{esc(n.get("why") or NOT_MEASURED)}</span>'
        if n.get("kind") in ("ruled out", "record"):  # a decision on file, not a number: the line is the citation
            return f'<span class="cite">{self.link(src["file"], src.get("line"))}</span>'
        bits = [self.link(src["file"], src.get("line")), esc(self.chip_text(n))]
        setup = setup_of(n.get("conditions") or {})
        if setup:
            bits.append(esc(setup))
        if n.get("kind") in ("derived", "prior", "stated"):
            bits.append(esc(n["kind"]) + (": " + esc(n["formula"]) if n.get("formula") else ""))
        if n.get("low_confidence"):
            why = ": " + esc(n["reason"]) if reason and n.get("reason") else ""
            bits.append(f'<span class="tag">low confidence</span>{why}')
        if n.get("other_chip"):
            bits.append('<span class="tag oc">other chip</span>')
        return f'<span class="cite">{" · ".join(bits)}</span>'

    def nums(self, numbers: list[dict[str, Any]], summary: str | None = None) -> str:
        """Every number with its source row, as a list; under a summary line it folds away until asked for.
        A low-confidence reason every row shares is said once above the list instead of on each row."""
        if not numbers:
            return ""
        reasons = {n.get("reason") for n in numbers if n.get("low_confidence")}
        shared = reasons.pop() if len(reasons) == 1 and sum(bool(n.get("low_confidence")) for n in numbers) > 2 else None
        rows = "".join(f'<li><span class="k">{esc(n["label"])}</span>'
                       f'<span class="v">{esc(fmt(n["value"], n.get("unit")))}</span>{self.cite(n, not shared)}</li>'
                       for n in numbers)
        note = f'<p class="note">The rows marked low confidence share one reason: {esc(shared)}.</p>' if shared else ""
        inner = f'{note}<ul class="nums">{rows}</ul>'
        if summary is None:
            return inner
        return (f"<details><summary>{esc(summary)} ({len(numbers)} numbers, each with its file, line and chip)"
                f"</summary>{inner}</details>")

    def sources(self, numbers: list[dict[str, Any]]) -> str:
        """The files a set of numbers came from, each with its line or line range, as links."""
        files: dict[str, list[int]] = {}
        for n in numbers:
            src = n.get("source") or {}
            if src.get("file"):
                files.setdefault(src["file"], []).extend([src["line"]] if src.get("line") else [])
        parts = []
        for rel, lines in files.items():  # the file's name is enough under a chart; the list below has the path
            lines, name = sorted(set(lines)), rel.rsplit("/", 1)[-1]
            text = name if not lines else (f"{name}:{lines[0]}" + (f" to :{lines[-1]}" if len(lines) > 1 else ""))
            parts.append(self.link(rel, lines[0] if lines else None, text))
        return ", ".join(parts)

    @staticmethod
    def chips(numbers: list[dict[str, Any]]) -> str:
        names: list[str] = []
        for n in numbers:
            label = n.get("chip_label")
            name = CHIP_NAME if label == CHIP else (label or n.get("gpu") or n.get("chip") or "a chip the row does not name")
            if name not in names:
                names.append(name)
        return " and ".join(names)

    def measured_on(self, numbers: list[dict[str, Any]], lead: str = "Measured on") -> str:
        return f"{esc(lead)} {esc(self.chips(numbers))}: {self.sources(numbers)}." if numbers else ""

    def key(self, big: str, label: str, n: dict[str, Any] | None = None) -> str:
        return (f'<div class="key"><div class="big">{esc(big)}</div><div class="kl">{esc(label)}</div>'
                f'{self.cite(n) if n else ""}</div>')

    def coming(self, s: dict[str, Any]) -> str:
        """The runs that will fill this section and have not landed, with what each was doing on the stated date."""
        names = [a for a in self.pack["awaited"] if a["task"] in s.get("awaiting", []) and not a["landed"]]
        if not names:
            return ""
        lis = "".join(f'<li><b>{esc(a["task"])}</b>: {esc(a["what"])}. <span class="when">{esc(a["status"])}.</span>'
                      "</li>" for a in names)
        return (f'<div class="coming"><h3>Still coming <span>(as of {esc(names[0]["as_of"])}; {NOT_MEASURED} here '
                f"until the rows are committed)</span></h3><ul>{lis}</ul></div>")

    def apart(self, numbers: list[dict[str, Any]], what: str, extra: dict[str, str] | None = None) -> str:
        """One dashed panel per other chip, named, so its rows never sit on a Strix Halo chart."""
        chips: dict[str, list[dict[str, Any]]] = {}
        for n in numbers:
            chips.setdefault(n.get("chip_label") or "other chip", []).append(n)
        out = []
        for chip, items in chips.items():
            inner = self.nums(items) if len(items) <= 4 else self.nums(items, "The rows from this chip")
            out.append(f'<div class="apart"><h3>Other chip, shown apart: {esc(chip)}</h3>'
                       f'<p class="note">{esc(what)}</p>{(extra or {}).get(chip, "")}{inner}</div>')
        return "".join(out)


def _figure(svg: str, caption: str) -> str:
    return f"<figure>{svg}<figcaption>{caption}</figcaption></figure>"


def _body(viz: str, key: str, more: str) -> str:
    """A section's body: the key figure, the chart and the rest. On a phone they stack in that order; on a wide
    screen the chart takes the left column and the key figure sits beside it with the rest under it."""
    parts = "".join(f'<div class="{cls}">{inner}</div>' for cls, inner in (("keys", key), ("viz", viz), ("more", more))
                    if inner)
    return f'<div class="body{"" if viz else " one"}">{parts}</div>'


def render_html(pack: dict[str, Any], root: Path = REPO, page_dir: Path | None = None) -> str:
    """The self-contained page: inline CSS and SVG, no script, no font, no network reference."""
    root = Path(root)
    page = Page(root, page_dir or root / OUT_DIR, pack)
    secs = {s["id"]: s for s in pack["sections"]}
    body: list[str] = []

    def head(i: int, s: dict[str, Any]) -> str:
        return (f'<section id="{s["id"]}"><h2><span class="n">{i}</span>{esc(s["title"])}'
                f'<span class="st st-{esc(s["state"])}">{esc(STATE_WORDS[s["state"]])}</span></h2>'
                f'<p class="say">{esc(s["reading"])}</p>')

    def empty(text: str) -> str:
        return f'<p class="empty">{NOT_MEASURED}: {esc(text)}</p>'

    def notes(lines: list[str]) -> str:
        return "".join(f'<p class="note">{esc(line)}</p>' for line in lines)

    # 1: the ceiling against each build
    s = secs["ceiling"]
    body.append(head(1, s))
    bars: list[dict[str, Any]] = []
    if s["builds"]:
        bars.append({"header": "Our builds, measured on the Strix Halo"})
        for b in s["builds"]:
            n = b["best"]
            sub = ", ".join(x for x in (asked_of(n["conditions"]), n.get("node")) if x)
            bars.append({"label": b["build"], "value": n["value"], "cls": "c1",
                         "sub": f"best of {len(b['rows'])} rows: {sub}" if len(b["rows"]) > 1 else sub})
    if s["priors"]:
        bars.append({"header": "Public prior: other people's runs, not ours"})
        bars += [{"label": PRIOR_NAMES.get(p["backend"], p["backend"]), "value": p["value"], "cls": "cg"}
                 for p in s["priors"]]
    viz = key = more = ""
    if s["achieved"] or s["ceiling"]["value"]:
        ref = ("ceiling", s["ceiling"]["value"]) if s["ceiling"]["value"] else None
        caption = page.measured_on([b["best"] for b in s["builds"]], "Blue bars: measured on")
        if ref:
            caption += f" Dashed line: the ceiling, worked out from {page.sources(s['inputs'])}."
        if s["priors"]:
            caption += (" Gray bars: a fit to published runs on other people's machines with this chip, never a "
                        f"measurement of ours ({page.sources(s['priors'])}).")
        viz = _figure(svg_bars(bars, ref, "tok/s", "Decode tokens per second at batch 1, each build against the "
                                                    "ceiling the chip's memory allows"), caption)
    if s["state"] == "measured":
        best = max(s["achieved"], key=lambda a: a["value"])
        share = next(f for f in s["shares"] if f["inputs"][0] == best["id"])
        key = page.key(f"{best['value'] / s['ceiling']['value']:.0%}", f"of the ceiling: {best['value']:.2f} of about "
                       f"{s['ceiling']['value']:.1f} tokens a second at batch 1", share)
    else:
        key = empty("no batch-1 decode row from the Strix Halo, or no chip sheet, is on main.")
    more = notes(s["notes"]) + page.coming(s)
    body.append(_body(viz, key, more))
    body.append(page.nums([*s["inputs"], s["ceiling"], *s["shares"], *s["achieved"], *s["priors"]],
                          "The numbers behind this chart"))
    body.append(page.apart(s["other_chip"], "The teacher's batch-1 decode on a different card and memory; not "
                                            "comparable bar for bar with the Strix Halo."))
    body.append("</section>")

    # 2: the batching frontier
    s = secs["frontier"]
    body.append(head(2, s))
    viz = key = more = ""
    shown: list[dict[str, Any]] = []
    if s["energy"]:
        first = s["energy"][0]
        series = [{"points": [(p["batch"], p["number"]["value"]) for p in sw["points"]], "line": f"l{i + 1}",
                   "fill": f"c{i + 1}", "label": sw["label"]} for i, sw in enumerate(s["energy"][:2])]
        ratio = s["ratios"][0] if s["ratios"] else None
        last = first["points"][-1]
        note = (f"{ratio['value']:g} times less", f"energy per token at {last['batch']:g} streams (dashed)") if ratio else None
        chart = svg_xy(series, xlog=True, xname="concurrent streams", yname="joules per token",
                       yfmt=lambda v: f"{fmt(v)} J", title="Joules per token against concurrent streams",
                       ref=last["number"]["value"], note=note)
        legend = ""
        if len(series) > 1:
            legend = '<ul class="legend">' + "".join(f'<li><span class="sw c{i + 1}"></span>{esc(x["label"])}</li>'
                                                     for i, x in enumerate(series)) + "</ul>"
        energy_numbers = [p["number"] for sw in s["energy"][:2] for p in sw["points"]]
        viz = _figure(chart + legend, page.measured_on(energy_numbers))
        shown += [*s["ratios"], *s["more_work"], *energy_numbers]
        if ratio:
            key = page.key(f"{ratio['value']:g}×", f"less energy for each token at {last['batch']:g} streams than "
                           f"at {first['points'][0]['batch']:g}: {fmt(first['points'][0]['number']['value'])} J down "
                           f"to {fmt(last['number']['value'])} J", ratio)
        words = mate(first, s["throughput"])
        if words:
            numbers = [p["number"] for p in words["points"]]
            more += _figure(svg_columns([(p["batch"], p["number"]["value"]) for p in words["points"]],
                                        xname="concurrent streams", yname="tokens per second, all streams together",
                                        title="Tokens per second across all streams together"),
                            page.measured_on(numbers))
            shown += numbers
        watts = mate(first, s["power"])
        shown += [p["number"] for p in watts["points"]] if watts else []
        for extra in s["throughput"] + s["power"]:
            shown += [p["number"] for p in extra["points"] if p["number"] not in shown]
        for extra in s["energy"][2:]:
            shown += [p["number"] for p in extra["points"]]
        shown += s["changes"]
    else:
        key = empty("no stream-count sweep with joules per token from the Strix Halo is on main.")
    if s["pending_files"]:
        more += (f'<p class="note">Committed and not charted here: {esc(", ".join(s["pending_files"]))}; its headline '
                 "rows are in the list below when the metrics file carries them.</p>")
    more += page.coming(s)
    body.append(_body(viz, key, more))
    body.append(page.nums(shown, "The numbers behind these charts"))
    charts: dict[str, str] = {}
    for sw in s["other_chip"]:
        if sw["metric"].endswith(".joules_per_token") and sw["chip_label"] not in charts:
            pts = [(p["batch"], p["number"]["value"]) for p in sw["points"]]
            r = next((x for x in s["other_ratios"] if x["inputs"][0] == sw["points"][0]["number"]["id"]), None)
            note = (f"{r['value']:g} times less", f"at {pts[-1][0]:g} streams, on this other chip") if r else None
            charts[sw["chip_label"]] = _figure(
                svg_xy([{"points": pts, "line": "lg", "fill": "cg"}], xlog=True, xname="concurrent streams",
                       yname="joules per token", yfmt=lambda v: f"{fmt(v)} J", height=190, note=note,
                       title=f"Joules per token against concurrent streams on {sw['chip_label']}"),
                page.measured_on([p["number"] for p in sw["points"]]))
    body.append(page.apart([*s["other_ratios"], *(p["number"] for o in s["other_chip"] for p in o["points"])],
                           "The same sweep on another chip, drawn on its own axes.", charts))
    body.append("</section>")

    # 3: where a decode step goes
    s = secs["step"]
    body.append(head(3, s))
    if s["state"] == "measured" and not s["other_chip"]:
        parts = [p for b in s["batches"] for p in b["parts"]]
        cells = "".join(
            f'<tr><td><span class="sw {STACK_CLS[i]}"></span>{esc(label)}</td>'
            + "".join(f'<td class="num">{esc(fmt(b["parts"][i]["value"], "ms"))}'
                      + (f' ({b["parts"][i]["value"] / b["step_ms"]:.0%})' if b["parts"][i]["value"] is not None
                         and b["step_ms"] else "") + "</td>" for b in s["batches"]) + "</tr>"
            for i, (_, label) in enumerate(STEP_BUCKETS))
        heads = "".join(f"<th>batch {esc(b['batch'])}</th>" for b in s["batches"])
        table = f'<div class="wrap"><table><tr><th>Where the step goes</th>{heads}</tr>{cells}</table></div>'
        viz = _figure(svg_stack(s["batches"], "Milliseconds of one decode step by where the time goes") + table,
                      page.measured_on(parts))
        key = page.key(f"{s['top']['value']:.0f}%", s["top"]["label"], s["top"]) if s.get("top") else ""
        body.append(_body(viz, key, page.coming(s)))
        body.append(page.nums(parts, "The numbers behind this chart"))
    else:
        body.append(_body("", empty((s.get("why") or s["reading"]) + "."), page.coming(s)))
        if s["other_chip"]:
            body.append(page.apart([p for b in s["other_chip"] for p in b["parts"]],
                                   "A decode profile from another chip."))
    body.append("</section>")

    # 4: engines side by side
    s = secs["engines"]
    body.append(head(4, s))
    bars = []
    if s["engines"] or s["not_measured"]:
        bars.append({"header": "Measured on the Strix Halo, best batch-1 row"})
        bars += [{"label": e["name"], "sub": build_label(e["best"]["conditions"]), "value": e["best"]["value"],
                  "cls": "c1"} for e in s["engines"]]
        bars += [{"label": ENGINE_NAMES.get(n, n), "value": None, "cls": "c1"} for n in s["not_measured"]]
    if s["priors"]:
        bars.append({"header": "Public prior: other people's runs, not ours"})
        bars += [{"label": PRIOR_NAMES.get(p["backend"], p["backend"]), "value": p["value"], "cls": "cg"}
                 for p in s["priors"]]
    caption = page.measured_on([e["best"] for e in s["engines"]], "Blue bars: measured on")
    if s["priors"]:
        caption += f" Gray bars: the public prior, never a measurement of ours ({page.sources(s['priors'])})."
    viz = _figure(svg_bars(bars, None, "tok/s", "Best batch-1 decode per engine on the Strix Halo, with the public "
                                                 "prior beside it"), caption)
    key = more = ""
    if len(s["engines"]) >= 2:
        order = sorted(s["engines"], key=lambda e: -e["best"]["value"])
        key = page.key(f"{fmt(order[0]['best']['value'])} tok/s", f"{order[0]['name']}, the fastest engine measured "
                       "on this chip at batch 1" + (f", on the {s['compared_on']}" if s["compared_on"] else ""),
                       order[0]["best"])
        more += notes([f"The rows differ in {d['what']}: "
                       + "; ".join(f"{ENGINE_NAMES.get(k, k)} {v}" for k, v in d["by_engine"].items()) + "."
                       for d in s["differences"]])
    elif s["engines"]:
        only = s["engines"][0]
        key = page.key(f"{fmt(only['best']['value'])} tok/s", f"{only['name']}, the one engine measured on this "
                       "chip so far", only["best"])
    if s["not_measured"]:
        more += empty(", ".join(ENGINE_NAMES.get(n, n) for n in s["not_measured"])
                      + ": no committed row from the Strix Halo for this engine.")
    more += page.coming(s)
    body.append(_body(viz if bars else "", key, more))
    body.append(page.nums([*(n for e in s["engines"] for n in e["rows"]), *s["unnamed"]],
                          "Every batch-1 decode row on this chip, by engine"))
    body.append("</section>")

    # 5: the fine-tune run
    s = secs["finetune"]
    body.append(head(5, s))
    viz = key = more = ""
    if s["loss"]:
        pts = [(float(n["conditions"]["step"]), n["value"]) for n in s["loss"]]
        viz = _figure(svg_xy([{"points": pts, "line": "l1", "fill": "c1"}], xlog=False, xname="training step",
                             yname="training loss", yfmt=fmt, title="Training loss at each logged step"),
                      page.measured_on(s["loss"]))
        key = page.key(f"{fmt(pts[0][1])} to {fmt(pts[-1][1])}", f"training loss, first to last of {len(pts)} "
                       "logged steps", s["loss"][-1])
        if len(s["step_time"]) > 1 and all(n["unit"] == "ms" for n in s["step_time"]):
            times = [(float(n["conditions"]["step"]), n["value"] / 1000) for n in s["step_time"]]
            more += _figure(svg_xy([{"points": times, "line": "l1", "fill": "c1"}], xlog=False, xname="training step",
                                   yname="seconds per step", yfmt=lambda v: f"{fmt(v)} s", height=190,
                                   title="Seconds per training step"), page.measured_on(s["step_time"]))
    else:
        key = empty("no per-step loss or step time from the Strix Halo is on main.")
    more += page.coming(s)
    body.append(_body(viz, key, more))
    body.append(page.nums([*([s["median_step"]] if s["median_step"] else []), *s["numbers"], *s["loss"],
                           *s["step_time"]], "The numbers behind this run"))
    body.append(page.apart(s["other_chip"], "Trainer rows from another chip; a different card, so not a baseline "
                                            "for the Strix Halo run."))
    body.append("</section>")

    # 6: tried and dropped
    s = secs["dropped"]
    body.append(head(6, s))
    groups = (("measured", "Measured, with a recorded verdict"), ("ruled out", "Ruled out before any run"),
              ("record", "On record"))
    for kind, title in groups:
        items = [i for i in s["items"] if i["kind"] == kind]
        if kind == "measured" and not items:
            body.append(f"<h3>{esc(title)}</h3>" + empty("no experiment verdict is committed; the lab ledger "
                                                         "(lab/state/ledger.jsonl) stays on the lab machine."))
        if items:
            lis = "".join(f'<li><div><div class="k">{esc(i["what"])}</div>'
                          + (f'<div class="verdict">{esc(i["verdict"])}</div>' if kind != "ruled out" else "")
                          + f'</div><div><div class="why">{esc(i["note"] or "")}</div>{page.cite(i)}</div></li>'
                          for i in items)
            body.append(f'<h3>{esc(title)} ({len(items)})</h3><ul class="items">{lis}</ul>')
    body.append("</section>")

    toc = "".join(f'<li><a href="#{s["id"]}"><b>{i}</b>{esc(short)}<span class="st st-{esc(s["state"])}">'
                  f'{esc(STATE_WORDS[s["state"]])}</span></a></li>'
                  for i, (s, short) in enumerate(zip(pack["sections"], ("Ceiling", "Batching", "Decode step", "Engines",
                                                                        "Fine-tune", "Dropped")), 1))
    as_of = (pack["data_as_of"] or "unknown").replace("T", " ").replace("Z", " UTC")
    sheet = next((x["file"] for x in pack["sources"] if x.get("role") == "demo numbers sheet"), None)
    srcs = "".join(f"<tr><td>{page.link(x['file'])}</td><td>{esc(x['role'])}</td><td class=\"num\">{x['lines']}</td>"
                   f"<td><code>{esc(x['sha256'][:12])}</code></td></tr>" for x in pack["sources"])
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>Under the hood</title><style>{CSS}</style></head><body><main>"
        '<div class="theme" role="radiogroup" aria-label="theme"><label><input type="radio" name="t" id="auto" checked> '
        'Auto</label> <label><input type="radio" name="t" id="night"> Night</label> '
        '<label><input type="radio" name="t" id="day"> Day</label></div>'
        f"<h1>{esc(pack['title'])}</h1>"
        f'<p class="lede">Every number below is read from a file committed in this repository and names its file and '
        f"line. The lab's chip is the {esc(CHIP_NAME)}; rows from any other chip (the R9700 the lab left, the A100 "
        f"reference) sit in dashed panels of their own. A measurement that does not exist yet says "
        f"&ldquo;{NOT_MEASURED}&rdquo;. Newest row: {esc(as_of)}.</p>"
        f'<ul class="toc">{toc}</ul>'
        + "".join(body)
        + f'<footer><h2>Sources</h2><div class="wrap"><table><tr><th>File</th><th>Used for</th><th>Lines</th>'
          f"<th>sha256</th></tr>{srcs}</table></div>"
        + (f"<p>The slide sentences that use these numbers are in {page.link(sheet)}.</p>" if sheet else "")
        + "<p>Built by <code>python -m lab report</code>; the data pack beside this page is "
          f"<code>{PACK_NAME}</code> ({FORMAT}).</p></footer></main></body></html>\n")


# --- the verb -----------------------------------------------------------------------------------------------------

def write(root: Path = REPO, out: Path | None = None, check: bool = False) -> dict[str, Any]:
    """Build both files and write them; with `check`, write nothing and say which of the files on disk a build
    would change (the committed page is stale once new rows land, until the verb is run again)."""
    root = Path(root)
    out_dir = Path(out) if out else root / OUT_DIR
    pack = build_pack(root)
    built = {PACK_NAME: json.dumps(pack, indent=1, ensure_ascii=False) + "\n",
             HTML_NAME: render_html(pack, root, out_dir)}
    answer = {"html": str(out_dir / HTML_NAME), "pack": str(out_dir / PACK_NAME),
              "sections": {s["id"]: s["state"] for s in pack["sections"]}, "numbers": len(pack["numbers"]),
              "still_coming": [a["task"] for a in pack["awaited"] if not a["landed"]]}
    if check:
        stale = [name for name, text in built.items() if not (out_dir / name).is_file()
                 or (out_dir / name).read_bytes().replace(b"\r\n", b"\n") != text.encode("utf-8")]
        return {"ok": not stale, "outcome": "stale" if stale else "current", "stale": stale, **answer}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in built.items():
        (out_dir / name).write_text(text, encoding="utf-8", newline="\n")
    return {"ok": True, **answer}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Write the under-the-hood report (one HTML page and its JSON data pack) from committed files only."""
    p = argparse.ArgumentParser(prog="python -m lab report", description=__doc__.splitlines()[0])
    p.add_argument("--root", default=str(REPO), help="the checkout to read (default: this one)")
    p.add_argument("--out", help=f"where to write (default: <root>/{OUT_DIR})")
    p.add_argument("--check", action="store_true",
                   help="write nothing: exit 1 when the files there differ from what a build would write now")
    args = p.parse_args(argv or [])
    return write(Path(args.root), Path(args.out) if args.out else None, check=args.check)
