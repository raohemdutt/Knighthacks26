"""The wavefront sheet's file format, the gfx1151 occupancy arithmetic, and the parsers for what the compiler and the
profiler report. Standard library only (plus chipsheet_format for the spread statistics).

This file is uploaded to the pod next to wavefront.py and the kernel-class files (lab.controller run-script --with),
so it imports nothing from the repo except chipsheet_format, which is uploaded beside it too.

A sheet is one JSON object:

    format        FORMAT ("atlas.wavefront/1")
    gfx           the chip rocminfo named (gfx1151 on the SJC pods)
    node          the ledger name of the machine it ran on (pod-a, pod-b)
    mode          full, quick or self-check (a self-check sheet holds parsed fixtures and invented timings)
    date          YYYY-MM-DD (UTC) of the run, the date in the file name
    started, ended  ISO 8601 UTC times of the run
    gpu_seconds   wall seconds the run held the card
    budget_s      the time budget; a case that would start past it is written as skipped, never cut short
    command       the command line that ran it
    host          the host name
    versions      {torch, torch_hip, triton, transformers, fla, python}: text or None
    classes       the kernel classes the run covered, in run order
    rows          a list of rows (below)
    notes         free text lines

A row is one case of one kernel class at one value of its lever:

    id          dotted lowercase words, unique in the sheet (gdn.impl_torch.b4_t256); the ledger metric is wave.<id>
    class       one of CLASSES
    lever       the one parameter the class varies (impl, split_k, num_stages, mode)
    setting     the lever's value in this row (torch, fla, 4, 2, graph)
    what        one line saying what was timed
    unit        "s" (seconds per call; for the launch class, seconds per op)
    value       the median of samples; None when the row failed or was skipped
    status      ok, failed or skipped
    error       why it failed or was skipped, else None
    params      the shapes and fixed settings
    samples     every repeat's value
    repeats     how many samples
    spread      chipsheet_format.stats(samples)
    first_call  wall seconds of the first call (compilation, library find steps, lazy loading), else None
    compile     {n_regs, n_spills, shared_bytes, num_warps, num_stages} of a Triton kernel, else None
    occupancy   occupancy(...) of that compile record, else None
    derived     other numbers from the same samples (per-layer milliseconds, ratio to the byte floor)
    command     the exact command that produced it
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

try:  # in the repo
    from lab.kernels.chipsheet import chipsheet_format as cf
except ImportError:  # on the pod: uploaded side by side
    import chipsheet_format as cf  # type: ignore[no-redef]

FORMAT = "atlas.wavefront/1"
MODES = ("full", "quick", "self-check")
CLASSES = ("gdn", "conv", "launch", "tile", "lora")
STATUSES = ("ok", "failed", "skipped")
NODES = ("pod-a", "pod-b")
SHEET_KEYS = ("format", "gfx", "node", "mode", "date", "started", "ended", "gpu_seconds", "budget_s", "command",
              "host", "versions", "classes", "rows", "notes")
ROW_KEYS = ("id", "class", "lever", "setting", "what", "unit", "value", "status", "error", "params", "samples",
            "repeats", "spread", "first_call", "compile", "occupancy", "derived", "command")
VERSION_KEYS = ("torch", "torch_hip", "triton", "transformers", "fla", "python")
COMPILE_KEYS = ("n_regs", "n_spills", "shared_bytes", "num_warps", "num_stages")
ID_RE = cf.ID_RE
DATE_RE = cf.DATE_RE

# gfx1151 in wave32, from LLVM's AMDGPU target files as lab/refine/research/kernels/strix_halo_hardware.md section 2
# and 3 record them: 1,536 VGPRs per SIMD lane slot, allocated in granules of 24, at most 256 per wave and 16 waves
# per SIMD; 128 KiB of LDS per WGP allocated in 1,024-byte granules, at most 64 KiB per workgroup.
WAVE_SIZE = 32
TOTAL_VGPRS = 1536
VGPR_GRANULE = 24
MAX_VGPRS = 256
MAX_WAVES_PER_SIMD = 16
LDS_PER_WGP = 131072
LDS_PER_WORKGROUP = 65536
LDS_GRANULE = 1024
# Two CUs of two SIMDs each per WGP: inferred from AMD's 768 KiB register-file figure, not stated in any opened
# source (strix_halo_hardware.md section 2), so every LDS-derived wave count says it rests on this.
SIMDS_PER_WGP = 4
WGPS = 20  # the HIP runtime's multiProcessorCount on gfx1151 (chip sheet cache.hip.compute_units)
# Device copy bandwidth from 256 MiB up on pod D (chip sheet copy.d2d.size_1gib), for byte floors
COPY_GBS = 209.5


class Skipped(Exception):
    """A case that cannot run here (a package not installed, a shape too large): the row is written as skipped."""


# --- occupancy --------------------------------------------------------------------------------------------------

def waves_by_vgprs(vgprs: int | None) -> int | None:
    """Waves per SIMD that a kernel using `vgprs` VGPRs per lane can keep resident in wave32 on gfx1151 (LLVM's
    min(max(total / alignTo(vgprs, granule), 1), max waves)); None for an unknown count."""
    if vgprs is None or vgprs <= 0:
        return None if vgprs is None else MAX_WAVES_PER_SIMD
    rounded = -(-min(int(vgprs), MAX_VGPRS) // VGPR_GRANULE) * VGPR_GRANULE
    return min(max(TOTAL_VGPRS // rounded, 1), MAX_WAVES_PER_SIMD)


def workgroups_by_lds(lds_bytes: int | None) -> int | None:
    """Workgroups a WGP can hold by LDS alone; None when the kernel uses none (LDS does not limit it) or the
    count is unknown, 0 when one workgroup asks for more than a workgroup may address."""
    if not lds_bytes:
        return None
    if lds_bytes > LDS_PER_WORKGROUP:
        return 0
    return LDS_PER_WGP // (-(-int(lds_bytes) // LDS_GRANULE) * LDS_GRANULE)


def occupancy(vgprs: int | None, lds_bytes: int | None, threads: int | None,
              workgroups: int | None = None) -> dict[str, Any]:
    """What limits residency of one kernel: waves per SIMD by VGPRs, workgroups per WGP by LDS, and the waves per
    SIMD that the LDS limit allows for workgroups of `threads` work-items (on SIMDS_PER_WGP, an inference); the
    smaller of the two, which one binds, and whether `workgroups` (the grid) covers the chip's 20 WGPs."""
    by_vgpr = waves_by_vgprs(vgprs)
    by_lds = workgroups_by_lds(lds_bytes)
    waves_per_group = max(1, -(-int(threads) // WAVE_SIZE)) if threads else None
    lds_waves = None
    if by_lds is not None and waves_per_group is not None:
        lds_waves = min(MAX_WAVES_PER_SIMD, (by_lds * waves_per_group) // SIMDS_PER_WGP)
    limits = {name: v for name, v in (("vgpr", by_vgpr), ("lds", lds_waves)) if v is not None}
    waves = min(limits.values()) if limits else None
    bound = min(limits, key=limits.get) if limits else None
    return {"vgprs": vgprs, "lds_bytes": lds_bytes, "threads": threads, "waves_per_group": waves_per_group,
            "waves_per_simd_by_vgpr": by_vgpr, "workgroups_per_wgp_by_lds": by_lds,
            "waves_per_simd_by_lds": lds_waves, "waves_per_simd": waves, "bound_by": bound,
            "grid_workgroups": workgroups,
            "fills_wgps": None if workgroups is None else workgroups >= WGPS,
            "assumes_simds_per_wgp": SIMDS_PER_WGP if lds_waves is not None else None}


# --- compiler and profiler output -------------------------------------------------------------------------------

def compile_record(meta: Mapping[str, Any]) -> dict[str, Any]:
    """A Triton kernel's compile facts from what the compiled handle exposes (n_regs, n_spills) and its metadata
    (shared, num_warps, num_stages); accepts either Triton's own names or ours."""
    def pick(*names: str) -> int | None:
        for name in names:
            value = meta.get(name)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
        return None

    return {"n_regs": pick("n_regs"), "n_spills": pick("n_spills"), "shared_bytes": pick("shared_bytes", "shared"),
            "num_warps": pick("num_warps"), "num_stages": pick("num_stages")}


def compile_occupancy(record: Mapping[str, Any], workgroups: int | None = None) -> dict[str, Any]:
    warps = record.get("num_warps")
    return occupancy(record.get("n_regs"), record.get("shared_bytes"),
                     None if warps is None else int(warps) * WAVE_SIZE, workgroups)


def remark_occupancy(remarks: Mapping[str, Mapping[str, Any]], threads: int = 256) -> dict[str, dict[str, Any]]:
    """Occupancy per kernel from hipcc's -Rpass-analysis=kernel-resource-usage remarks as
    chipsheet_format.parse_resource_remarks returns them (VGPRs and LDS size in bytes per kernel)."""
    out = {}
    for name, fields in remarks.items():
        vgprs = _first_int(fields, "VGPRs", "VGPRs: ")
        lds = _first_int(fields, "LDS Size [bytes/block]", "LDS Size")
        found = occupancy(vgprs, lds, threads)
        found["compiler_occupancy"] = _first_int(fields, "Occupancy [waves/SIMD]", "Occupancy")
        out[name] = found
    return out


def _first_int(fields: Mapping[str, Any], *names: str) -> int | None:
    for name in names:
        value = fields.get(name)
        if value is None:
            continue
        match = re.match(r"\s*(\d+)", str(value))
        if match:
            return int(match.group(1))
    return None


TRACE_COLUMNS = ("Kernel_Name", "Start_Timestamp", "End_Timestamp", "Workgroup_Size_X", "Workgroup_Size_Y",
                 "Workgroup_Size_Z", "Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z", "LDS_Block_Size", "VGPR_Count")


def _cell_int(row: Mapping[str, str], *names: str) -> int | None:
    for name in names:
        text = (row.get(name) or "").strip()
        if text:
            try:
                return int(float(text))
            except ValueError:
                continue
    return None


def trace_kernels(text: str) -> list[dict[str, Any]]:
    """The rows of a rocprofv3 kernel trace CSV (Kernel_Name, Start_Timestamp and End_Timestamp in nanoseconds,
    workgroup and grid sizes in work-items, LDS_Block_Size or Group_Segment_Size, VGPR_Count or Arch_VGPR_Count;
    the column names lab/trace_profile.py reads), one dict per launch, in start order."""
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        start, end = _cell_int(row, "Start_Timestamp"), _cell_int(row, "End_Timestamp")
        if start is None or end is None or end < start:
            continue
        block = [_cell_int(row, f"Workgroup_Size_{axis}") or 1 for axis in "XYZ"]
        items = [_cell_int(row, f"Grid_Size_{axis}") or (0 if axis == "X" else 1) for axis in "XYZ"]
        groups = 1
        for size, width in zip(items, block):
            groups *= -(-size // width) if width else size
        name = (row.get("Kernel_Name") or "").strip()
        out.append({"name": name[:-3] if name.endswith(".kd") else name, "start_ns": start, "dur_us": (end - start) / 1e3,
                    "threads": block[0] * block[1] * block[2], "workgroups": groups,
                    "lds_bytes": _cell_int(row, "LDS_Block_Size", "Group_Segment_Size", "LDS_Size"),
                    "vgprs": _cell_int(row, "VGPR_Count", "Arch_VGPR_Count", "arch_vgpr_count")})
    out.sort(key=lambda k: k["start_ns"])
    return out


def trace_occupancy(kernels: Sequence[Mapping[str, Any]], top: int = 25) -> dict[str, Any]:
    """Per kernel name: launches, total and median microseconds, share of kernel time, and the occupancy its most
    common VGPR, LDS and workgroup shape imply; the top `top` names by total time, plus the launch count overall
    (the number a launch-count lever reduces) and the median kernel duration."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for kernel in kernels:
        groups.setdefault(kernel["name"], []).append(kernel)
    total = sum(k["dur_us"] for k in kernels)
    rows = []
    for name, items in groups.items():
        durations = sorted(k["dur_us"] for k in items)

        def common(key: str) -> Any:
            values = [k.get(key) for k in items if k.get(key) is not None]
            return max(set(values), key=values.count) if values else None

        found = occupancy(common("vgprs"), common("lds_bytes"), common("threads"), common("workgroups"))
        rows.append({"name": name, "launches": len(items), "total_us": round(sum(durations), 3),
                     "median_us": round(statistics.median(durations), 3),
                     "share": round(sum(durations) / total, 4) if total else None, "occupancy": found})
    rows.sort(key=lambda r: -r["total_us"])
    every = sorted(k["dur_us"] for k in kernels)
    return {"launches": len(kernels), "kernel_us": round(total, 3),
            "median_us": round(statistics.median(every), 3) if every else None,
            "names": len(rows), "top": rows[:top]}


# --- rows and sheets --------------------------------------------------------------------------------------------

def row(id: str, klass: str, lever: str, setting: Any, what: str, *, command: str, params: Mapping[str, Any],
        samples: Sequence[float] | None = None, status: str = "ok", error: str | None = None,
        first_call: float | None = None, compile: Mapping[str, Any] | None = None,
        occupancy: Mapping[str, Any] | None = None, derived: Mapping[str, Any] | None = None,
        unit: str = "s") -> dict[str, Any]:
    values = [float(s) for s in (samples or [])]
    spread = cf.stats(values)
    return {"id": id, "class": klass, "lever": lever, "setting": setting, "what": what, "unit": unit,
            "value": spread["median"] if status == "ok" else None, "status": status, "error": error,
            "params": dict(params), "samples": values, "repeats": len(values), "spread": spread,
            "first_call": first_call, "compile": dict(compile) if compile else None,
            "occupancy": dict(occupancy) if occupancy else None, "derived": dict(derived or {}), "command": command}


def row_id(klass: str, lever: str, setting: Any, shape: str) -> str:
    """gdn + impl + torch + b4_t256 -> gdn.impl_torch.b4_t256 (lowercase, dots between words)."""
    word = re.sub(r"[^a-z0-9_]+", "_", f"{lever}_{setting}".lower()).strip("_")
    return f"{klass}.{word}.{shape.lower()}"


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate(sheet: Any) -> list[str]:
    """Every problem with a sheet, as strings; an empty list means it conforms to this file's format."""
    if not isinstance(sheet, Mapping):
        return ["the sheet is not a JSON object"]
    problems = [f"missing key {key}" for key in SHEET_KEYS if key not in sheet]
    if sheet.get("format") != FORMAT:
        problems.append(f"format is {sheet.get('format')!r}, expected {FORMAT}")
    if sheet.get("mode") not in MODES:
        problems.append(f"mode {sheet.get('mode')!r} is not one of {', '.join(MODES)}")
    if sheet.get("node") not in NODES and sheet.get("mode") != "self-check":
        problems.append(f"node {sheet.get('node')!r} is not one of {', '.join(NODES)}")
    if not isinstance(sheet.get("date"), str) or not DATE_RE.fullmatch(sheet.get("date") or ""):
        problems.append("date is not YYYY-MM-DD")
    for name in sheet.get("classes") or []:
        if name not in CLASSES:
            problems.append(f"class {name!r} is not one of {', '.join(CLASSES)}")
    seen: set[str] = set()
    rows = sheet.get("rows")
    if not isinstance(rows, list):
        return problems + ["rows is not a list"]
    for index, item in enumerate(rows):
        where = f"row {index}"
        if not isinstance(item, Mapping):
            problems.append(f"{where} is not an object")
            continue
        problems += [f"{where}: missing key {key}" for key in ROW_KEYS if key not in item]
        rid = item.get("id")
        if not isinstance(rid, str) or not ID_RE.fullmatch(rid):
            problems.append(f"{where}: id {rid!r} is not dotted lowercase words")
        elif rid in seen:
            problems.append(f"{where}: id {rid} repeats")
        else:
            seen.add(rid)
            where = rid
        if item.get("class") not in CLASSES:
            problems.append(f"{where}: class {item.get('class')!r} is not one of {', '.join(CLASSES)}")
        elif isinstance(rid, str) and not rid.startswith(item["class"] + "."):
            problems.append(f"{where}: id does not start with its class")
        status = item.get("status")
        if status not in STATUSES:
            problems.append(f"{where}: status {status!r} is not one of {', '.join(STATUSES)}")
        samples = item.get("samples") or []
        if not all(_finite(s) and s >= 0 for s in samples):
            problems.append(f"{where}: a sample is not a finite non-negative number")
        if status == "ok":
            if not samples:
                problems.append(f"{where}: ok with no samples")
            elif not _finite(item.get("value")):
                problems.append(f"{where}: ok with no value")
            elif abs(item["value"] - statistics.median(samples)) > 1e-12 * max(1.0, abs(item["value"])):
                problems.append(f"{where}: value is not the median of its samples")
            if item.get("repeats") != len(samples):
                problems.append(f"{where}: repeats does not count its samples")
        elif not item.get("error"):
            problems.append(f"{where}: {status} with no error saying why")
        record = item.get("compile")
        if record is not None and (not isinstance(record, Mapping) or set(record) != set(COMPILE_KEYS)):
            problems.append(f"{where}: compile does not hold exactly {', '.join(COMPILE_KEYS)}")
    return problems


def file_stem(sheet: Mapping[str, Any]) -> str:
    suffix = "-selfcheck" if sheet.get("mode") == "self-check" else ""
    return f"wavefront-{sheet.get('gfx') or 'nogfx'}-{sheet.get('date')}{suffix}"


def write(sheet: Mapping[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    """Write <stem>.json and <stem>.md into out_dir after validating; a sheet that does not conform raises."""
    problems = validate(sheet)
    if problems:
        raise ValueError("the sheet does not conform: " + "; ".join(problems[:5]))
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    stem = file_stem(sheet)
    json_path, md_path = out / f"{stem}.json", out / f"{stem}.md"
    json_path.write_text(json.dumps(sheet, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(sheet), encoding="utf-8")
    return json_path, md_path


def read(path: str | Path) -> dict[str, Any]:
    """Read a sheet back and validate it; a sheet that does not conform raises."""
    sheet = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = validate(sheet)
    if problems:
        raise ValueError(f"{path} does not conform: " + "; ".join(problems[:5]))
    return sheet


TITLES = {"gdn": "Gated DeltaNet chunk rule, one layer, forward plus backward",
          "conv": "Depthwise causal conv, one layer, forward plus backward",
          "launch": "Per-op launch cost, eager against one HIP graph replay",
          "tile": "fla-like chunk scan tile (Triton), num_stages",
          "lora": "LoRA A projection (Triton), split-K"}


def _num(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 1e-3:
            return f"{value * 1e6:.3g} us"
        if abs(value) < 1:
            return f"{value * 1e3:.3g} ms"
        return f"{value:.3g} s"
    return str(value)


def _plain(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_markdown(sheet: Mapping[str, Any]) -> str:
    """The sheet as a Markdown page: one table per kernel class, each row with its lever setting, median and spread,
    first call, compile facts and the occupancy they imply."""
    lines = [f"# Wavefront sheet: {sheet.get('gfx')} on {sheet.get('node')}, {sheet.get('date')}", "",
             f"- Mode: {sheet.get('mode')}; GPU time {_plain(sheet.get('gpu_seconds'))} s of a "
             f"{_plain(sheet.get('budget_s'))} s budget",
             f"- Command: `{sheet.get('command')}`",
             f"- Versions: " + ", ".join(f"{k} {v}" for k, v in (sheet.get("versions") or {}).items() if v), ""]
    rows = sheet.get("rows") or []
    for klass in sheet.get("classes") or []:
        mine = [r for r in rows if r.get("class") == klass]
        if not mine:
            continue
        lines += [f"## {TITLES.get(klass, klass)}", "",
                  "| id | lever = setting | median | spread (min to max, rel) | n | first call | VGPRs / spills | "
                  "LDS bytes | waves per SIMD (bound) | status |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for item in mine:
            spread = item.get("spread") or {}
            record = item.get("compile") or {}
            occ = item.get("occupancy") or {}
            spread_text = "-" if not spread.get("n") else \
                f"{_num(spread.get('min'))} to {_num(spread.get('max'))}, {_plain(spread.get('rel_spread'))}"
            regs = "-" if not record else f"{_plain(record.get('n_regs'))} / {_plain(record.get('n_spills'))}"
            waves = "-" if not occ else f"{_plain(occ.get('waves_per_simd'))} ({occ.get('bound_by') or '-'})"
            status = item.get("status") if item.get("status") == "ok" else f"{item.get('status')}: {item.get('error')}"
            lines.append(f"| {item.get('id')} | {item.get('lever')} = {item.get('setting')} | {_num(item.get('value'))} "
                         f"| {spread_text} | {spread.get('n', 0)} | {_num(item.get('first_call'))} | {regs} | "
                         f"{_plain(record.get('shared_bytes')) if record else '-'} | {waves} | "
                         f"{str(status).replace('|', '/')} |")
        lines.append("")
    if sheet.get("notes"):
        lines += ["## Notes", ""] + [f"- {note}" for note in sheet["notes"]] + [""]
    return "\n".join(lines)


def by_id(sheet: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {item["id"]: item for item in sheet.get("rows") or []}


def ratio(rows: Mapping[str, Mapping[str, Any]], slow: str, fast: str) -> float | None:
    """value(slow) / value(fast) for two ok rows, the speedup a lever setting buys; None when either is missing."""
    a, b = rows.get(slow), rows.get(fast)
    if not a or not b or a.get("status") != "ok" or b.get("status") != "ok" or not b.get("value"):
        return None
    return a["value"] / b["value"]


def classes_of(items: Iterable[Mapping[str, Any]]) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item.get("class") not in seen:
            seen.append(item.get("class"))
    return seen
