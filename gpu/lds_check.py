"""R12: the LDS bytes of every compiled kernel a change touches, against the 65,536-byte workgroup limit.

    python gpu/lds_check.py --arm stages|block|stock --triton-cache DIR [--serve-log FILE]
        [--code-object FILE ...] [--code-object-notes FILE ...] [--readelf PATH] [--out FILE]

Two sources, whichever the build has:
- Triton kernels (the unified attention kernels R12 changes): each compiled kernel leaves
  <TRITON_CACHE_DIR>/<hash>/<kernel name>.json, whose `shared` field is the LDS bytes the kernel allocates; `num_stages`,
  `num_warps` and `arch` come from the same file. `__grp__*.json` files are the cache's group index and are skipped.
- HIP code objects (.hsaco, or one extracted from a fat binary with roc-obj): `llvm-readelf --notes` prints the AMDGPU
  metadata, where each kernel's `.group_segment_fixed_size` is its static LDS bytes. --code-object runs readelf
  (--readelf, else /opt/rocm/llvm/bin/llvm-readelf, else llvm-readelf on PATH); --code-object-notes reads text it
  printed earlier.

Failures, each counted once: a kernel over 65,536 bytes; a kernel the arm touches that no source shows (it was never
compiled, so it cannot be shown to fit); and every resource failure in the serve log (Triton's OutOfResources for
shared memory, LLVM's "local memory (N) exceeds limit"). The record goes to --out as JSON and its summary is the last
line printed. Exit 0 when there is no failure, 1 when there is one, 2 when an input cannot be read.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

LDS_LIMIT_BYTES = 65536   # per workgroup on gfx11 and gfx12 (lab/refine/research/kernels/strix_halo_hardware.md s.3)
SCHEMA = "gpu.r12.lds/1"
# The compiled kernels each arm changes against the stock build (see gpu/patches/).
TOUCHED = {
    "stages": ["kernel_unified_attention_3d"],
    "block": ["kernel_unified_attention_3d", "reduce_segments"],
    "stock": ["kernel_unified_attention_2d", "kernel_unified_attention_3d", "reduce_segments"],
}
READELF_CANDIDATES = ("/opt/rocm/llvm/bin/llvm-readelf", "llvm-readelf")

_TRITON_OOR = re.compile(r"OutOfResources: out of resource: shared memory, Required: (\d+), Hardware limit: (\d+)")
_LLVM_OOR = re.compile(r"local memory \((\d+)\) exceeds limit \((\d+)\)(?: in function '([^']+)')?")


class InputError(Exception):
    pass


def read_triton_cache(root: Path) -> list[dict[str, Any]]:
    """One row per compiled kernel metadata file under a Triton cache directory."""
    root = Path(root)
    if not root.is_dir():
        raise InputError(f"no Triton cache at {root}")
    rows = []
    for path in sorted(root.glob("*/*.json")):
        if path.name.startswith("__grp__"):
            continue
        try:
            meta = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise InputError(f"cannot read {path}: {exc}") from exc
        if not isinstance(meta, dict) or "shared" not in meta or "name" not in meta:
            continue
        target = meta.get("target") if isinstance(meta.get("target"), dict) else {}
        rows.append({"kernel": meta["name"], "lds_bytes": int(meta["shared"]), "source": "triton.metadata.shared",
                     "num_stages": meta.get("num_stages"), "num_warps": meta.get("num_warps"),
                     "arch": target.get("arch") or meta.get("arch"), "path": str(path.relative_to(root))})
    return rows


def parse_readelf_notes(text: str, *, path: str) -> list[dict[str, Any]]:
    """Kernels and their .group_segment_fixed_size from `llvm-readelf --notes` output."""
    arch = None
    m = re.search(r"amdhsa\.target:\s+\S*--(gfx[0-9a-f]+)", text)
    if m:
        arch = m.group(1)
    rows: list[dict[str, Any]] = []
    kernel_indent = None
    current: dict[str, Any] | None = None
    in_kernels = False
    for line in text.splitlines():
        if line.startswith("amdhsa.kernels:"):
            in_kernels = True
            continue
        if in_kernels and line and not line.startswith(" "):
            in_kernels = False
        if not in_kernels:
            continue
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if stripped.startswith("- ") and (kernel_indent is None or indent == kernel_indent):
            kernel_indent = indent
            current = {"kernel": None, "lds_bytes": None}
            rows.append(current)
            stripped = stripped[2:].lstrip()
            indent += 2
        if current is None or indent != kernel_indent + 2:
            continue
        key, _, value = stripped.partition(":")
        if key == ".name":
            current["kernel"] = value.strip()
        elif key == ".group_segment_fixed_size":
            current["lds_bytes"] = int(value.strip())
    return [{"kernel": r["kernel"], "lds_bytes": r["lds_bytes"], "source": ".group_segment_fixed_size",
             "num_stages": None, "num_warps": None, "arch": arch, "path": path}
            for r in rows if r["kernel"] and r["lds_bytes"] is not None]


def readelf_notes(code_object: Path, readelf: str | None) -> str:
    tool = readelf or next((t for t in READELF_CANDIDATES if shutil.which(t) or Path(t).is_file()), None)
    if tool is None:
        raise InputError("no llvm-readelf: pass --readelf or use --code-object-notes")
    run = subprocess.run([tool, "--notes", str(code_object)], capture_output=True, text=True)
    if run.returncode != 0:
        raise InputError(f"{tool} --notes {code_object} failed: {run.stderr.strip()[:300]}")
    return run.stdout


def resource_failures(log_text: str) -> list[dict[str, Any]]:
    out = []
    for n, line in enumerate(log_text.splitlines(), 1):
        m = _TRITON_OOR.search(line)
        if m:
            out.append({"line": n, "required_bytes": int(m.group(1)), "limit_bytes": int(m.group(2)), "kernel": None,
                        "kind": "triton OutOfResources"})
            continue
        m = _LLVM_OOR.search(line)
        if m:
            out.append({"line": n, "required_bytes": int(m.group(1)), "limit_bytes": int(m.group(2)),
                        "kernel": m.group(3), "kind": "llvm local memory"})
    return out


def check(arm: str, kernels: list[dict[str, Any]], failures_in_log: list[dict[str, Any]],
          limit: int = LDS_LIMIT_BYTES) -> dict[str, Any]:
    for k in kernels:
        k["fits"] = k["lds_bytes"] <= limit
        k["headroom_bytes"] = limit - k["lds_bytes"]
    touched = TOUCHED.get(arm, [])
    seen = {k["kernel"] for k in kernels}
    missing = [name for name in touched if name not in seen]
    over = [k for k in kernels if not k["fits"]]
    failures = len(over) + len(missing) + len(failures_in_log)
    return {"schema": SCHEMA, "arm": arm, "limit_bytes": limit, "touched": touched, "kernels": kernels,
            "touched_max_lds_bytes": max((k["lds_bytes"] for k in kernels if k["kernel"] in touched), default=None),
            "missing": missing, "over_limit": [k["kernel"] for k in over], "resource_failures": failures_in_log,
            "failures": failures, "pass": failures == 0}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", required=True, help="stages, block or stock (picks the kernels that must be present)")
    ap.add_argument("--triton-cache", action="append", default=[], help="a TRITON_CACHE_DIR (repeatable)")
    ap.add_argument("--code-object", action="append", default=[], help="an .hsaco to read with llvm-readelf")
    ap.add_argument("--code-object-notes", action="append", default=[], help="saved `llvm-readelf --notes` output")
    ap.add_argument("--readelf", default=None)
    ap.add_argument("--serve-log", action="append", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    try:
        kernels: list[dict[str, Any]] = []
        for root in args.triton_cache:
            kernels += read_triton_cache(Path(root))
        for co in args.code_object:
            kernels += parse_readelf_notes(readelf_notes(Path(co), args.readelf), path=str(co))
        for notes in args.code_object_notes:
            kernels += parse_readelf_notes(Path(notes).read_text(), path=str(notes))
        log_fails = []
        for log in args.serve_log:
            log_fails += [{**f, "log": str(log)} for f in resource_failures(Path(log).read_text(errors="replace"))]
    except (InputError, OSError) as exc:
        print(json.dumps({"schema": SCHEMA, "error": str(exc)}))
        return 2
    if not kernels and not log_fails:
        print(json.dumps({"schema": SCHEMA, "error": "no kernel metadata found in any input"}))
        return 2
    record = check(args.arm, kernels, log_fails)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
    for k in kernels:
        print(f"{'ok  ' if k['fits'] else 'OVER'} {k['kernel']}: {k['lds_bytes']} B of {LDS_LIMIT_BYTES} "
              f"({k['source']}, stages={k['num_stages']})")
    print(json.dumps({key: record[key] for key in ("arm", "failures", "missing", "over_limit", "pass")}))
    return 0 if record["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
