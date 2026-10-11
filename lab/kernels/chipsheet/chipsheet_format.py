"""The chip sheet's file format and the parsers for what the pod's tools print. Standard library only.

This file is uploaded to the pod next to chipsheet.py (lab.controller run-script --with), so it imports nothing from
the repo; the dispatcher side (lab/kernels/chipsheet/ingest.py) imports it as lab.kernels.chipsheet.chipsheet_format.

A sheet is one JSON object:

    format        FORMAT ("atlas.chipsheet/1")
    gfx           the chip rocminfo named (gfx1151 on the SJC pods)
    node          the ledger name of the machine it ran on (pod-a, pod-b)
    mode          full, quick or self-check (a self-check sheet holds parsed fixtures and invented timings)
    date          YYYY-MM-DD (UTC) of the run, the date in the file name
    started, ended  ISO 8601 UTC times of the run
    gpu_seconds   wall seconds the run held the card
    budget_s      the time budget the run was given; measurements past it are skipped, not cut short
    command       the command line that ran it
    host          the host name
    versions      {rocm, hip, hipcc, torch, torch_hip, triton, python, amd_smi}: text or None
    device        {rocminfo: parsed GPU agent (parse_rocminfo), hip: hipDeviceProp fields the probe printed}
    power         {reader: what read the watts, idle: {section: idle watt summary}}
    measurements  a list of entries (below)
    notes         free text lines

A measurement entry:

    id        dotted lowercase words, unique in the sheet (copy.d2d.size_64mib); the ledger metric is chip.<id>
    section   one of SECTIONS
    what      one line saying what was measured
    unit      the unit of value and of every sample
    value     the median of samples, or a single reading for a fact; None when the entry failed or was skipped
    status    ok, failed or skipped
    error     why it failed or was skipped, else None
    source    timing (timed repeats), probe (the HIP probe program), tool (rocminfo, amd-smi, hipcc), torch (a
              torch or HIP runtime property)
    command   the exact command that produced it (the run command with --only for timings, the tool's argv for facts)
    op        the exact operation timed or read (code, kernel and launch shape)
    params    the inputs (shapes, sizes, dtypes, configs)
    samples   every repeat's value
    repeats   how many samples
    spread    stats(samples): min, max, median, mean, stdev, p10, p90, rel_spread ((max - min) / median)
    derived   other numbers computed from the same samples (weight GB/s of a decode GEMM, relative throughput)
    watts     the power window: idle_w (the section's idle mean), loaded_mean_w, loaded_max_w, samples, or error
"""
from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

FORMAT = "atlas.chipsheet/1"
MODES = ("full", "quick", "self-check")
SECTIONS = ("copy", "matmul", "lora", "dequant", "lds", "wave", "occupancy", "cache", "versions")
STATUSES = ("ok", "failed", "skipped")
SOURCES = ("timing", "probe", "tool", "torch")
SHEET_KEYS = ("format", "gfx", "node", "mode", "date", "started", "ended", "gpu_seconds", "budget_s", "command",
              "host", "versions", "device", "power", "measurements", "notes")
ENTRY_KEYS = ("id", "section", "what", "unit", "value", "status", "error", "source", "command", "op", "params",
              "samples", "repeats", "spread", "derived", "watts")
VERSION_KEYS = ("rocm", "hip", "hipcc", "torch", "torch_hip", "triton", "python", "amd_smi")
NODES = ("pod-a", "pod-b")
ID_RE = re.compile(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
GFX_RE = re.compile(r"\bgfx[0-9a-f]{3,4}\b")
MIB = 1024 * 1024


# --- numbers ----------------------------------------------------------------------------------------------------

def stats(samples: Sequence[float]) -> dict[str, float | int | None]:
    """The spread of a list of repeats: min, max, median, mean, stdev (0 for one sample), p10, p90 (nearest rank)
    and rel_spread, (max - min) / median (None when the median is 0). Every key is None for no samples."""
    values = sorted(float(v) for v in samples if v is not None and math.isfinite(float(v)))
    keys = ("n", "min", "max", "median", "mean", "stdev", "p10", "p90", "rel_spread")
    if not values:
        return {key: (0 if key == "n" else None) for key in keys}
    median = statistics.median(values)

    def rank(q: float) -> float:
        return values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))]

    return {"n": len(values), "min": values[0], "max": values[-1], "median": median,
            "mean": statistics.fmean(values), "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p10": rank(0.10), "p90": rank(0.90),
            "rel_spread": (values[-1] - values[0]) / median if median else None}


def size_label(nbytes: int) -> str:
    """A byte count as the word an id uses: 1048576 -> 1mib, 4294967296 -> 4gib, 1536 -> 1536b."""
    if nbytes % (1024 * MIB) == 0:
        return f"{nbytes // (1024 * MIB)}gib"
    if nbytes % MIB == 0:
        return f"{nbytes // MIB}mib"
    if nbytes % 1024 == 0:
        return f"{nbytes // 1024}kib"
    return f"{nbytes}b"


def _int_of(text: str | None) -> int | None:
    """The leading whole number of a rocminfo value such as '64(0x40) KB' or '2900'."""
    if text is None:
        return None
    match = re.match(r"\s*(\d+)", text)
    return int(match.group(1)) if match else None


def _bytes_of(text: str | None) -> int | None:
    """A rocminfo size such as '32(0x20) KB' in bytes; a bare number is taken as bytes."""
    number = _int_of(text)
    if number is None or text is None:
        return None
    unit = text.upper()
    if "GB" in unit:
        return number * 1024 * MIB
    if "MB" in unit:
        return number * MIB
    if "KB" in unit:
        return number * 1024
    return number


# --- rocminfo ---------------------------------------------------------------------------------------------------

def rocminfo_agents(text: str) -> list[dict[str, Any]]:
    """Every agent block of rocminfo's output: its top-level fields as written, plus 'cache' (the Cache Info lines)
    and 'pools' (one dict per Pool, its Segment, Size and the rest)."""
    agents: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    section: str | None = None
    for raw in text.splitlines():
        line = raw.replace("\t", "    ").rstrip()
        stripped = line.strip()
        if re.fullmatch(r"Agent \d+", stripped):
            current = {"cache": {}, "pools": []}
            agents.append(current)
            section = None
            continue
        if current is None or not stripped:
            continue
        indent = len(line) - len(line.lstrip(" "))
        if re.fullmatch(r"Pool \d+", stripped):
            current["pools"].append({})
            continue
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key, value = key.strip(), value.strip()
        if indent <= 2:
            section = key if key in ("Cache Info", "Pool Info", "ISA Info") else None
            if section is None:
                current[key] = value
            continue
        if section == "Cache Info" and value:
            current["cache"][key] = value
        elif section == "Pool Info" and value:
            if not current["pools"]:
                current["pools"].append({})
            current["pools"][-1][key] = value
    return agents


def parse_rocminfo(text: str) -> dict[str, Any] | None:
    """The first GPU agent of rocminfo's output in the sheet's terms, or None when there is none: gfx, marketing
    name, compute units, SIMDs per CU, wavefront size, workgroup max size, max waves per CU, cacheline bytes, max
    clock, cache sizes in bytes by level and the LDS (the GROUP segment pool) in bytes."""
    for agent in rocminfo_agents(text):
        if agent.get("Device Type", "").upper() != "GPU":
            continue
        name = agent.get("Name", "")
        gfx = GFX_RE.search(name)
        lds = next((_bytes_of(pool.get("Size")) for pool in agent["pools"]
                    if pool.get("Segment", "").upper().startswith("GROUP")), None)
        return {
            "gfx": gfx.group(0) if gfx else name or None,
            "marketing_name": agent.get("Marketing Name") or None,
            "compute_units": _int_of(agent.get("Compute Unit")),
            "simds_per_cu": _int_of(agent.get("SIMDs per CU")),
            "wavefront_size": _int_of(agent.get("Wavefront Size")),
            "workgroup_max_size": _int_of(agent.get("Workgroup Max Size")),
            "max_waves_per_cu": _int_of(agent.get("Max Waves Per CU")),
            "cacheline_bytes": _int_of(agent.get("Cacheline Size")),
            "max_clock_mhz": _int_of(agent.get("Max Clock Freq. (MHz)")),
            "cache_bytes": {level.lower(): _bytes_of(size) for level, size in agent["cache"].items()},
            "lds_bytes": lds,
        }
    return None


# --- versions ---------------------------------------------------------------------------------------------------

def parse_hipcc_version(text: str) -> dict[str, str | None]:
    """`hipcc --version`: the HIP version line and the clang (AMD clang) version line."""
    hip = re.search(r"HIP version:\s*(\S+)", text)
    clang = re.search(r"clang version\s+(\S+)", text)
    return {"hip": hip.group(1) if hip else None, "clang": clang.group(1) if clang else None}


def parse_rocm_version(text: str) -> str | None:
    """/opt/rocm/.info/version (one line such as 7.13.0-71) or the ROCm field of `amd-smi version`."""
    field = re.search(r"ROCm version:\s*([0-9][^\s|]*)", text)
    if field:
        return field.group(1)
    first = text.strip().splitlines()[0].strip() if text.strip() else ""
    return first if re.match(r"\d+\.\d+", first) else None


# --- the HIP probe ----------------------------------------------------------------------------------------------

def parse_probe_lines(text: str) -> list[dict[str, Any]]:
    """The JSON objects the probe program prints, one per line, in order; every other line is left out."""
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get("probe"), str):
            out.append(value)
    return out


_REMARK_NAME = re.compile(r"remark:.*?:\s*Function Name:\s*(\S+)")
_REMARK_FIELD = re.compile(r"remark:.*?:\d+:\d+:\s+(.+?):\s*(\S+)(?:\s+\[-Rpass[^\]]*\])?\s*$")


def parse_resource_remarks(text: str) -> dict[str, dict[str, Any]]:
    """What `-Rpass-analysis=kernel-resource-usage` says per kernel: {kernel: {VGPRs, SGPRs, AGPRs,
    ScratchSize [bytes/lane], Occupancy [waves/SIMD], LDS Size [bytes/block], ...}}, whole numbers as ints."""
    kernels: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        name = _REMARK_NAME.search(line)
        if name:
            current = kernels.setdefault(name.group(1), {})
            continue
        field = _REMARK_FIELD.search(line)
        if field and current is not None:
            key, value = field.group(1).strip(), field.group(2)
            current[key] = int(value) if re.fullmatch(r"-?\d+", value) else value
    return kernels


# --- the sheet --------------------------------------------------------------------------------------------------

def entry(id: str, section: str, what: str, unit: str, *, source: str, command: str, op: str,
          params: Mapping[str, Any] | None = None, samples: Sequence[float] = (), value: float | None = None,
          status: str = "ok", error: str | None = None, derived: Mapping[str, Any] | None = None,
          watts: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One measurement entry. value defaults to the median of samples; a fact passes value and one sample."""
    clean = [float(s) for s in samples]
    spread = stats(clean)
    if value is None and status == "ok":
        value = spread["median"]
    return {"id": id, "section": section, "what": what, "unit": unit, "value": value, "status": status,
            "error": error, "source": source, "command": command, "op": op, "params": dict(params or {}),
            "samples": clean, "repeats": len(clean), "spread": spread, "derived": dict(derived or {}),
            "watts": dict(watts) if watts is not None else None}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate(sheet: Any) -> list[str]:
    """Every way the sheet breaks the format, as one line each; an empty list means it is a valid sheet."""
    if not isinstance(sheet, Mapping):
        return [f"a sheet is a JSON object, not {type(sheet).__name__}"]
    problems: list[str] = []
    for key in SHEET_KEYS:
        if key not in sheet:
            problems.append(f"{key} is missing")
    if sheet.get("format") != FORMAT:
        problems.append(f"format must be {FORMAT!r}, not {sheet.get('format')!r}")
    if sheet.get("mode") not in MODES:
        problems.append(f"mode must be one of {', '.join(MODES)}, not {sheet.get('mode')!r}")
    if sheet.get("node") not in NODES:
        problems.append(f"node must be one of {', '.join(NODES)}, not {sheet.get('node')!r}")
    if not isinstance(sheet.get("date"), str) or not DATE_RE.fullmatch(sheet["date"]):
        problems.append(f"date must be YYYY-MM-DD, not {sheet.get('date')!r}")
    if not isinstance(sheet.get("gfx"), str) or not GFX_RE.fullmatch(sheet["gfx"]):
        problems.append(f"gfx must name a chip such as gfx1151, not {sheet.get('gfx')!r}")
    for key in ("gpu_seconds", "budget_s"):
        if key in sheet and not _finite(sheet[key]):
            problems.append(f"{key} must be a finite number, not {sheet[key]!r}")
    versions = sheet.get("versions")
    if not isinstance(versions, Mapping):
        problems.append("versions must be an object")
    else:
        for key in VERSION_KEYS:
            if key not in versions:
                problems.append(f"versions.{key} is missing")
            elif versions[key] is not None and not isinstance(versions[key], str):
                problems.append(f"versions.{key} must be text or null")
    measurements = sheet.get("measurements")
    if not isinstance(measurements, list):
        problems.append("measurements must be a list")
        measurements = []
    seen: set[str] = set()
    for n, item in enumerate(measurements):
        where = f"measurements[{n}]"
        if not isinstance(item, Mapping):
            problems.append(f"{where} must be an object")
            continue
        where = f"{where} ({item.get('id')})"
        for key in ENTRY_KEYS:
            if key not in item:
                problems.append(f"{where}: {key} is missing")
        mid = item.get("id")
        if not isinstance(mid, str) or not ID_RE.fullmatch(mid):
            problems.append(f"{where}: id must be dotted lowercase words starting with a letter")
        elif mid in seen:
            problems.append(f"{where}: id is used twice")
        else:
            seen.add(mid)
        if item.get("section") not in SECTIONS:
            problems.append(f"{where}: section must be one of {', '.join(SECTIONS)}")
        if item.get("status") not in STATUSES:
            problems.append(f"{where}: status must be one of {', '.join(STATUSES)}")
        if item.get("source") not in SOURCES:
            problems.append(f"{where}: source must be one of {', '.join(SOURCES)}")
        for key in ("what", "unit", "command", "op"):
            if not isinstance(item.get(key), str) or not item.get(key, "").strip():
                problems.append(f"{where}: {key} must be non-empty text")
        samples = item.get("samples")
        if not isinstance(samples, list) or not all(_finite(s) for s in samples):
            problems.append(f"{where}: samples must be a list of finite numbers")
        elif item.get("repeats") != len(samples):
            problems.append(f"{where}: repeats must equal the number of samples")
        if item.get("status") == "ok":
            if not _finite(item.get("value")):
                problems.append(f"{where}: an ok entry needs a finite value")
            if isinstance(samples, list) and not samples:
                problems.append(f"{where}: an ok entry needs at least one sample")
        else:
            if item.get("value") is not None:
                problems.append(f"{where}: a {item.get('status')} entry has no value")
            if not isinstance(item.get("error"), str) or not item["error"].strip():
                problems.append(f"{where}: a {item.get('status')} entry says why in error")
        for key in ("params", "spread", "derived"):
            if not isinstance(item.get(key), Mapping):
                problems.append(f"{where}: {key} must be an object")
        if item.get("watts") is not None and not isinstance(item.get("watts"), Mapping):
            problems.append(f"{where}: watts must be an object or null")
    try:
        json.dumps(sheet, allow_nan=False)
    except (TypeError, ValueError) as exc:
        problems.append(f"the sheet is not plain JSON: {exc}")
    return problems


def file_stem(sheet: Mapping[str, Any]) -> str:
    """chipsheet-<gfx>-<date>, the name both files share; a self-check sheet says so in its name."""
    stem = f"chipsheet-{sheet['gfx']}-{sheet['date']}"
    return f"{stem}-self-check" if sheet.get("mode") == "self-check" else stem


def write(sheet: Mapping[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    """Write <stem>.json and <stem>.md into out_dir after validating the sheet; ValueError lists the problems and
    nothing is written. Returns both paths."""
    problems = validate(sheet)
    if problems:
        raise ValueError("not a valid chip sheet: " + "; ".join(problems))
    folder = Path(out_dir).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    stem = file_stem(sheet)
    json_path, md_path = folder / f"{stem}.json", folder / f"{stem}.md"
    json_path.write_text(json.dumps(sheet, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(sheet), encoding="utf-8")
    return json_path, md_path


def read(path: str | Path) -> dict[str, Any]:
    """A sheet read back from its JSON file; ValueError when it is not a valid sheet."""
    sheet = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = validate(sheet)
    if problems:
        raise ValueError(f"{path} is not a valid chip sheet: " + "; ".join(problems))
    return sheet


# --- the Markdown summary ---------------------------------------------------------------------------------------

SECTION_TITLES = {
    "versions": "Versions",
    "cache": "Caches and device limits",
    "copy": "Device-to-device copy bandwidth",
    "matmul": "fp16 and bf16 matmul, 27B decode shapes",
    "lora": "fp16 and bf16 matmul, 4B LoRA step shapes",
    "dequant": "int4-to-fp16 dequantize plus matmul (plain Triton kernel)",
    "lds": "LDS",
    "wave": "Wavefront size the compiler chose",
    "occupancy": "Occupancy curve",
}


def _num(value: Any) -> str:
    if not _finite(value):
        return "-"
    value = float(value)
    if value.is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3g}"


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _watts(item: Mapping[str, Any]) -> str:
    watts = item.get("watts") or {}
    if watts.get("error"):
        return "n/a"
    return f"{_num(watts.get('loaded_mean_w'))} / {_num(watts.get('loaded_max_w'))} ({_num(watts.get('idle_w'))})"


def render_markdown(sheet: Mapping[str, Any]) -> str:
    """The sheet as a Markdown summary: the run, the versions and device, then one table per section with every
    number, its spread, its watts and the exact command and operation that produced it."""
    lines = [f"# Chip sheet: {sheet['gfx']} on {sheet['node']}, {sheet['date']}", ""]
    if sheet.get("mode") == "self-check":
        lines += ["**Self-check sheet: parsed fixtures and invented timings, not a measurement.**", ""]
    lines += [
        f"- Mode: {sheet['mode']}; GPU time {_num(sheet.get('gpu_seconds'))} s of a {_num(sheet.get('budget_s'))} s budget",
        f"- Run: {sheet.get('started')} to {sheet.get('ended')} on host {sheet.get('host')}",
        f"- Command: `{sheet.get('command')}`",
        f"- Watts reader: {_cell(json.dumps((sheet.get('power') or {}).get('reader')))}",
        "",
        "## Versions",
        "",
        "| What | Version |",
        "|---|---|",
    ]
    for key in VERSION_KEYS:
        lines.append(f"| {key} | {_cell((sheet.get('versions') or {}).get(key) or '-')} |")
    rocm = ((sheet.get("device") or {}).get("rocminfo")) or {}
    hip = ((sheet.get("device") or {}).get("hip")) or {}
    lines += ["", "## Device", "", "| Field | rocminfo | HIP runtime |", "|---|---|---|"]
    pairs = [("marketing name", rocm.get("marketing_name"), hip.get("name")),
             ("compute units", rocm.get("compute_units"), hip.get("multiProcessorCount")),
             ("wavefront size", rocm.get("wavefront_size"), hip.get("warpSize")),
             ("workgroup max size", rocm.get("workgroup_max_size"), hip.get("maxThreadsPerBlock")),
             ("LDS bytes", rocm.get("lds_bytes"), hip.get("sharedMemPerBlock")),
             ("L2 bytes", (rocm.get("cache_bytes") or {}).get("l2"), hip.get("l2CacheSize")),
             ("max clock MHz", rocm.get("max_clock_mhz"),
              hip["clockRate_khz"] / 1000 if _finite(hip.get("clockRate_khz")) else None)]
    def shown(value: Any) -> str:
        return "-" if value is None else _num(value) if _finite(value) else _cell(value)

    for name, left, right in pairs:
        lines.append(f"| {name} | {shown(left)} | {shown(right)} |")
    items = list(sheet.get("measurements") or [])
    for section in SECTIONS:
        rows = [m for m in items if m.get("section") == section]
        if not rows:
            continue
        lines += ["", f"## {SECTION_TITLES[section]}", "",
                  "| id | value | unit | spread (min to max, rel) | n | W loaded mean / max (idle) | command | op |",
                  "|---|---|---|---|---|---|---|---|"]
        for m in rows:
            spread = m.get("spread") or {}
            if m.get("status") != "ok":
                value = f"{m.get('status')}: {_cell(m.get('error'))}"
                span = "-"
            else:
                value = _num(m.get("value"))
                span = f"{_num(spread.get('min'))} to {_num(spread.get('max'))}, {_num(spread.get('rel_spread'))}"
            lines.append(f"| {m['id']} | {value} | {_cell(m.get('unit'))} | {span} | {m.get('repeats')} | "
                         f"{_watts(m)} | `{_cell(m.get('command'))}` | {_cell(m.get('op'))} |")
    failed = [m for m in items if m.get("status") != "ok"]
    lines += ["", "## Not measured", ""]
    lines += [f"- {m['id']}: {m.get('status')}, {_cell(m.get('error'))}" for m in failed] or ["- nothing"]
    notes = list(sheet.get("notes") or [])
    if notes:
        lines += ["", "## Notes", ""] + [f"- {_cell(note)}" for note in notes]
    return "\n".join(lines) + "\n"


def by_id(sheet: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """The sheet's entries by id."""
    return {m["id"]: m for m in sheet.get("measurements", []) if isinstance(m, Mapping) and "id" in m}


def sections_of(items: Iterable[Mapping[str, Any]]) -> list[str]:
    """The sections that hold at least one entry, in SECTIONS order."""
    present = {m.get("section") for m in items}
    return [s for s in SECTIONS if s in present]
