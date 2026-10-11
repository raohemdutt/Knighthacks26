"""capacity: what one chip offers a fine-tune, probed in under two minutes and written as a capacity record.

    python -m lab capacity --fake --node pod-b                         # the R33 sample sheet, no chip (tests, rehearsal)
    python -m lab capacity --sheet SHEET.json --facts NODE.facts.txt --node pod-b   # files pulled from a pod
    python -m lab capacity ... --out FILE                              # default <state dir>/capacity/capacity-<node>.json

A probe reads the chip through a chip client: the chip sheet in its short mode (R33's `chipsheet.py run --quick`,
inside 110 s of GPU time) and the memory a new job can take. From those it takes the five things the model ladder
(lab/ladder.py) needs: free memory, the best achieved device-to-device copy bandwidth, the best achieved bf16 matmul
rate, the chip's name and the ROCm version. The record also carries the node, its role (pod-a the teacher, pod-b the
trainer, anything else a friend's chip in the Circle), the probe's own wall-clock and the sheet's GPU seconds, and
whether the two together stay inside two minutes.

Every number is a placement.Num view: value, unit, the file and key it came from, and its kind (measured from an R33
sheet, sample from the sample sheet, recorded from a facts file, unknown with the reason). Nothing is invented: the
sample sheet has measured neither bandwidth nor matmul rate, so both stay unknown until R33 commits a sheet.

Two clients exist. capacity_fake.FakeChipClient answers with a sheet on disk (the sample by default). FilesClient
reads a quick sheet and the output of `bash lab/pod_a_teacher.sh facts` that lab/R44_REAL_RUN.md pulls from a pod.
This module calls no pod, no hub and no network address; the pod steps are the REAL_RUN file's commands.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from lab import placement
from lab.placement import Num, relative, unknown

FORMAT = "capacity/1"
BUDGET_S = 120.0
ROLES = {"pod-a": "teacher", "pod-b": "trainer"}
FRIEND = "friend"

_ROCM_IN_TORCH = re.compile(r"\+rocm([0-9][0-9.]*)")


class ChipClient(Protocol):
    """How a probe reaches a chip. sheet() returns the path of a chip sheet (quick=True: R33's short mode);
    memory() returns free_gib and visible_gib (numbers or {"value": ...}), their source and kind."""

    name: str

    def sheet(self, quick: bool = True) -> Path: ...

    def memory(self) -> Mapping[str, Any]: ...


class FilesClient:
    """A chip client over files already pulled from a pod: a quick chip sheet, and optionally the facts output."""

    def __init__(self, sheet: Path | str, facts: Path | str | None = None) -> None:
        self.path = Path(sheet)
        self.facts = Path(facts) if facts is not None else None

    @property
    def name(self) -> str:
        return f"files: {relative(self.path)}" + (f", {relative(self.facts)}" if self.facts else "")

    def sheet(self, quick: bool = True) -> Path:
        return self.path

    def memory(self) -> dict[str, Any]:
        if self.facts is None:
            return {"source": "no facts file given"}
        facts = placement.read_facts_text(self.facts.read_text(encoding="utf-8"))
        mem = placement._MEM.search(facts.get("mem", ""))
        torch_mem = placement._TORCH_MEM.search(facts.get("torch_gpu", ""))
        return {"free_gib": float(mem.group(2)) if mem else None,
                "visible_gib": float(torch_mem.group(1)) if torch_mem else None,
                "source": f"{relative(self.facts)}: mem (MemAvailable) and torch_gpu mem_gib", "kind": "recorded"}


# --- reading a sheet --------------------------------------------------------------------------------------------

def _value(value: Any) -> float | None:
    return placement._number(value)


def _best_measurement(sheet: Mapping[str, Any], *, sections: tuple[str, ...], unit: str,
                      where: Callable[[str], bool] = lambda mid: True) -> tuple[float, str] | None:
    """The largest ok value among the measurements of an R33 sheet in those sections, and its entry id."""
    best: tuple[float, str] | None = None
    for item in sheet.get("measurements") or []:
        if not isinstance(item, Mapping) or item.get("status") != "ok" or item.get("section") not in sections:
            continue
        mid = str(item.get("id") or "")
        value = _value(item.get("value"))
        if value is None or str(item.get("unit")) != unit or not where(mid):
            continue
        if best is None or value > best[0]:
            best = (value, mid)
    return best


def chip_name(sheet: Mapping[str, Any]) -> str | None:
    chip = sheet.get("chip") if isinstance(sheet.get("chip"), Mapping) else {}
    device = sheet.get("device") if isinstance(sheet.get("device"), Mapping) else {}
    rocminfo = device.get("rocminfo") if isinstance(device.get("rocminfo"), Mapping) else {}
    hip = device.get("hip") if isinstance(device.get("hip"), Mapping) else {}
    for name in (chip.get("name"), rocminfo.get("marketing_name"), hip.get("name")):
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None


def rocm_version(sheet: Mapping[str, Any], name: str) -> dict[str, Any]:
    """versions.rocm, else the ROCm a torch build string names (2.9.1+rocm7.13.0), with where it came from."""
    versions = sheet.get("versions") if isinstance(sheet.get("versions"), Mapping) else {}
    if isinstance(versions.get("rocm"), str) and versions["rocm"]:
        return {"value": versions["rocm"], "source": f"{name}: versions.rocm"}
    match = _ROCM_IN_TORCH.search(str(versions.get("torch") or ""))
    if match:
        return {"value": match.group(1), "source": f"{name}: versions.torch {versions['torch']} (the ROCm torch was "
                                                   "built for; the sheet has no versions.rocm)"}
    return {"value": None, "source": f"{name}: no versions.rocm and no ROCm torch build"}


def _memory_num(memory: Mapping[str, Any], key: str, fallback: Num) -> Num:
    value = _value(memory.get(key))
    if value is None:
        return fallback
    return Num(value, "GiB", f"{memory.get('source')}: {key}", str(memory.get("kind") or "recorded"))


# --- the probe --------------------------------------------------------------------------------------------------

def probe(client: ChipClient, *, node: str, role: str | None = None,
          clock: Callable[[], float] = time.monotonic, now: dt.datetime | None = None) -> dict[str, Any]:
    """Probe one chip through the client and return its capacity record (see the module docstring)."""
    start = clock()
    path = client.sheet(quick=True)
    memory = client.memory()
    wall = clock() - start
    name = relative(path)
    sheet = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(sheet, Mapping):
        raise ValueError(f"{name} is not a JSON object")
    fields = placement.read_chipsheet(path)
    kind = "sample" if fields.sample else "measured"

    bandwidth = fields.get("bandwidth_gb_s")
    if not bandwidth.known:
        best = _best_measurement(sheet, sections=("copy",), unit="GB/s")
        bandwidth = (Num(best[0], "GB/s", f"{name}: measurements {best[1]} (the best ok copy)", kind) if best
                     else unknown("GB/s", f"{name}: no achieved copy bandwidth (R33 measures it)"))
    matmul = fields.get("matmul_tflops")
    if not matmul.known:
        best = _best_measurement(sheet, sections=("matmul", "lora"), unit="TFLOP/s", where=lambda mid: ".bf16." in mid)
        matmul = (Num(best[0], "TFLOPS", f"{name}: measurements {best[1]} (the best ok bf16 matmul)", kind) if best
                  else unknown("TFLOPS", f"{name}: no achieved bf16 matmul rate (R33 measures it)"))
    free = _memory_num(memory, "free_gib", fields.get("free_gib"))
    visible = _memory_num(memory, "visible_gib", fields.get("visible_gib"))

    chip = sheet.get("chip") if isinstance(sheet.get("chip"), Mapping) else {}
    gpu_seconds = _value(sheet.get("gpu_seconds"))
    total = wall + (gpu_seconds or 0.0)
    moment = (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc).replace(microsecond=0)
    return {
        "format": FORMAT,
        "node": node,
        "role": role or ROLES.get(node, FRIEND),
        "probed_at": moment.isoformat().replace("+00:00", "Z"),
        "sample": fields.sample,
        "chip": {"name": chip_name(sheet), "gfx": chip.get("gfx") or sheet.get("gfx") or fields.gfx},
        "rocm": rocm_version(sheet, name),
        "free_gib": free.view(),
        "visible_gib": visible.view(),
        "bandwidth_gb_s": bandwidth.view(),
        "bf16_tflops": matmul.view(),
        "probe": {"mode": "quick", "client": client.name, "wall_s": round(wall, 3), "sheet_gpu_s": gpu_seconds,
                  "budget_s": BUDGET_S, "within_budget": total <= BUDGET_S},
    }


def write_record(record: Mapping[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def read_record(path: Path | str) -> dict[str, Any]:
    """A capacity record from its file; ValueError when the file is not one."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} record")
    return record


def default_out(node: str) -> Path:
    from lab.machine import ledger   # the state dir as every lab verb resolves it
    return ledger.resolve_state_dir() / "capacity" / f"capacity-{node}.json"


# --- the command line -------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lab capacity", description=main.__doc__, allow_abbrev=False)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fake", action="store_true", help="the fake client: the R33 sample sheet, no chip")
    source.add_argument("--sheet", metavar="FILE", help="a quick chip sheet pulled from the pod (R33 --quick)")
    parser.add_argument("--facts", metavar="FILE", help="with --sheet: `bash lab/pod_a_teacher.sh facts` output")
    parser.add_argument("--node", required=True, help="pod-a, pod-b, or a friend's chip's name")
    parser.add_argument("--role", help="the node's role (default: pod-a teacher, pod-b trainer, else friend)")
    parser.add_argument("--out", metavar="FILE", help="the record's file (default <state dir>/capacity/capacity-<node>.json)")
    return parser


def main(argv: list[str]) -> dict[str, Any]:
    """Probe one chip (free memory, bandwidth, bf16 matmul, name, ROCm) in under two minutes; write a capacity record."""
    args = build_parser().parse_args(argv)
    if args.facts and not args.sheet:
        raise SystemExit("--facts goes with --sheet")
    if args.fake:
        from lab.capacity_fake import FakeChipClient
        client: ChipClient = FakeChipClient()
    else:
        client = FilesClient(args.sheet, args.facts)
    record = probe(client, node=args.node, role=args.role)
    path = write_record(record, args.out or default_out(args.node))
    return {"ok": True, "outcome": "probed", "record": str(path), "within_budget": record["probe"]["within_budget"],
            "free_gib": record["free_gib"]["value"], "bf16_tflops": record["bf16_tflops"]["value"],
            "sample": record["sample"]}
