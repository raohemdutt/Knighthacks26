"""Wavefront microbenchmarks for the student's training step on gfx1151: one kernel class per file, one lever per
class, results as rows with their spread, the compile facts of every Triton kernel and the occupancy they imply.

    python3 wavefront.py run --node pod-b --only gdn [--quick] [--out ~/wavefront] [--budget-s 900]
    python3 wavefront.py self-check [--out DIR] [--fixtures DIR]
    python3 wavefront.py pod-script --only gdn [--account D] [--node pod-b] [--quick]
    python3 wavefront.py occupancy <rocprofv3 kernel_trace.csv> [--top 25]

On the pod it runs from the Jupyter root next to wavefront_format.py, chipsheet_format.py and the class files
(gdn.py, conv.py, launch.py, tile.py, lora.py), which lab.controller run-script --with uploads; pod_wavefront.sh is
the script it runs, and `pod-script` prints the one-line controller command for a class. In the repo it also runs as
`python -m lab.kernels.wavefront.wavefront`.

The classes, each in its own file with its lever:

  gdn     Gated DeltaNet chunk rule, one layer, forward plus backward; impl torch (transformers' reference) or fla
  conv    depthwise causal conv, one layer, forward plus backward; impl torch (F.conv1d, MIOpen) or fla (Triton)
  launch  per-op cost of a chain of tiny kernels; mode eager or graph (one HIP graph replay)
  tile    a Triton tile shaped like fla's state scan; num_stages 1 or 2 at num_warps 4
  lora    the LoRA A projection x @ A (rank 16); split_k torch (reference), 1 or 4 (Triton)

Every case times its first call apart (compilation, library find steps, lazy loading), then repeats: 15 samples
(--quick 5) after 3 warm-up calls (--quick 2), each sample enough back-to-back calls between two events to last about
20 ms. A case that would start past the budget (900 s, --quick 240 s) is written as skipped, never cut short; a case
whose package is missing is skipped with the reason; a failure is a failed row and the run goes on. It writes
<out>/wavefront-<gfx>-<date>.json and .md and prints one JSON line naming them.

Node gate: run refuses (exit 2) unless --node is pod-a or pod-b, the machine is x86_64 and rocminfo names gfx1151;
nothing imports torch before that. self-check never touches a GPU: it parses the fixtures in fixtures/lab/wavefront/
(a Triton compile record list, a rocprofv3 kernel trace, the chip sheet's hipcc remarks), builds a sheet with invented
timings for every class's quick cases, writes it, reads it back and exits 0. occupancy reads a kernel trace R34's
profile leaves and prints the top kernels with their launches, time and the occupancy their VGPRs and LDS imply.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

try:  # in the repo
    from lab.kernels.chipsheet import chipsheet_format as cf
    from lab.kernels.wavefront import conv, gdn, launch, lora, tile
    from lab.kernels.wavefront import wavefront_format as wf
except ImportError:  # on the pod: every file uploaded side by side
    import chipsheet_format as cf  # type: ignore[no-redef]
    import conv  # type: ignore[no-redef]
    import gdn  # type: ignore[no-redef]
    import launch  # type: ignore[no-redef]
    import lora  # type: ignore[no-redef]
    import tile  # type: ignore[no-redef]
    import wavefront_format as wf  # type: ignore[no-redef]

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2] if len(HERE.parents) > 2 else HERE
FIXTURES = REPO / "fixtures" / "lab" / "wavefront"
REMARKS_FIXTURE = REPO / "fixtures" / "lab" / "chipsheet" / "remarks_w32_synthetic.txt"
MODULES = {"gdn": gdn, "conv": conv, "launch": launch, "tile": tile, "lora": lora}
POD_GFX = "gfx1151"
DEFAULT_OUT = "~/wavefront"
FULL_BUDGET_S = 900.0
QUICK_BUDGET_S = 240.0
EXIT_GATE = 2
# The files a pod run needs next to the script, repo-relative; pod-script names each with --with.
POD_FILES = ("lab/kernels/wavefront/wavefront.py", "lab/kernels/wavefront/wavefront_format.py",
             "lab/kernels/wavefront/gdn.py", "lab/kernels/wavefront/conv.py", "lab/kernels/wavefront/launch.py",
             "lab/kernels/wavefront/tile.py", "lab/kernels/wavefront/lora.py",
             "lab/kernels/chipsheet/chipsheet_format.py")
POD_SCRIPT = "lab/kernels/wavefront/pod_wavefront.sh"
# GPU minutes each class takes at full size, and the slot minutes asked for it (upload, imports, first-call
# compilation and the controller's own start included).
SLOT_MINUTES = {"gdn": 20, "conv": 15, "launch": 10, "tile": 15, "lora": 15}


def log(message: str) -> None:
    print(f"[wavefront] {message}", file=sys.stderr, flush=True)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_tool(argv: Sequence[str], timeout_s: float = 60.0) -> tuple[int, str]:
    try:
        done = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout_s)
        return done.returncode, done.stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, f"{type(exc).__name__}: {exc}"


def rocminfo_gfx() -> str | None:
    path = shutil.which("rocminfo") or ("/opt/rocm/bin/rocminfo" if Path("/opt/rocm/bin/rocminfo").exists() else None)
    if path is None:
        return None
    code, out = run_tool([path])
    found = cf.GFX_RE.findall(out) if code == 0 else []
    return found[0] if found else None


def gate(node: str, gfx: str | None, machine: str | None = None) -> str | None:
    """Why this machine must not run the measurements, or None when it is a gfx1151 pod under a known name.
    x86_64 is AMD64 where the system spells it so (Windows); such a machine is then refused by its chip."""
    machine = machine or platform.machine()
    if node not in wf.NODES:
        return f"--node must be one of {', '.join(wf.NODES)}, not {node!r}"
    if machine.lower() not in ("x86_64", "amd64"):
        return f"this machine is {machine}, not a pod (x86_64); the Jetson is aarch64"
    if gfx != POD_GFX:
        return f"rocminfo names {gfx or 'no gfx'}, not {POD_GFX}: not a Strix Halo pod"
    return None


def versions() -> dict[str, str | None]:
    found: dict[str, str | None] = {key: None for key in wf.VERSION_KEYS}
    found["python"] = platform.python_version()
    for key, module in (("torch", "torch"), ("triton", "triton"), ("transformers", "transformers"), ("fla", "fla")):
        try:
            found[key] = str(getattr(__import__(module), "__version__", "unknown"))
        except Exception:
            found[key] = None
    try:
        import torch

        found["torch_hip"] = getattr(torch.version, "hip", None)
    except Exception:
        pass
    return found


class Timer:
    """Times a callable on the GPU: the first call by wall clock around a synchronize, then `repeats` samples of
    enough back-to-back calls between two events to last about target_s each."""

    def __init__(self, repeats: int, warmup: int, target_s: float = 0.02) -> None:
        self.repeats, self.warmup, self.target_s = repeats, warmup, target_s

    def first(self, fn: Callable[[], Any]) -> float:
        import torch

        torch.cuda.synchronize()
        began = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return time.perf_counter() - began

    def repeat(self, fn: Callable[[], Any], inner: int | None = None) -> list[float]:
        import torch

        for _ in range(self.warmup):
            fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        if inner is None:
            start.record()
            fn()
            end.record()
            end.synchronize()
            one = max(start.elapsed_time(end) / 1e3, 1e-7)
            inner = max(1, min(1000, int(self.target_s / one)))
        samples = []
        for _ in range(self.repeats):
            start.record()
            for _ in range(inner):
                fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / 1e3 / inner)
        return samples


def run_case(module: Any, case: dict[str, Any], timer: Timer, command: str, left_s: float) -> dict[str, Any]:
    """One case as a row: ok with samples, skipped (budget or a missing package) or failed with the error."""
    common = dict(command=command, params=case["params"])
    if left_s < 5:
        return wf.row(case["id"], module.CLASS, module.LEVER, case["setting"], module.WHAT, status="skipped",
                      error=f"budget: {left_s:.0f} s left", **common)
    try:
        found = module.measure(case, timer)
    except wf.Skipped as exc:
        return wf.row(case["id"], module.CLASS, module.LEVER, case["setting"], module.WHAT, status="skipped",
                      error=str(exc)[:400], **common)
    except Exception as exc:
        try:
            import torch

            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass
        return wf.row(case["id"], module.CLASS, module.LEVER, case["setting"], module.WHAT, status="failed",
                      error=f"{type(exc).__name__}: {exc}"[:400], **common)
    record = found.get("compile")
    occ = wf.compile_occupancy(record, found.get("workgroups")) if record else None
    return wf.row(case["id"], module.CLASS, module.LEVER, case["setting"], module.WHAT, samples=found["samples"],
                  first_call=found.get("first_call"), compile=record, occupancy=occ, derived=found.get("derived"),
                  **common)


def selected(only: str | None) -> list[str]:
    if not only:
        return list(wf.CLASSES)
    names = [name.strip() for name in only.split(",") if name.strip()]
    unknown = [name for name in names if name not in wf.CLASSES]
    if unknown:
        raise SystemExit(f"unknown class {', '.join(unknown)}; the classes are {', '.join(wf.CLASSES)}")
    return names


def run_main(args: argparse.Namespace, command: str) -> int:
    gfx = rocminfo_gfx()
    why = gate(args.node, gfx)
    if why:
        print(json.dumps({"ok": False, "outcome": "refused", "error": why}))
        return EXIT_GATE
    classes = selected(args.only)
    budget = args.budget_s or (QUICK_BUDGET_S if args.quick else FULL_BUDGET_S)
    timer = Timer(5 if args.quick else 15, 2 if args.quick else 3)
    started, t0 = utc_now(), time.monotonic()
    rows = []
    for name in classes:
        module = MODULES[name]
        for case in module.cases(args.quick):
            log(f"{case['id']} ({time.monotonic() - t0:.0f} s of {budget:.0f})")
            row_command = command if args.only else f"{command} --only {name}"
            rows.append(run_case(module, case, timer, row_command, budget - (time.monotonic() - t0)))
    sheet = {"format": wf.FORMAT, "gfx": gfx, "node": args.node, "mode": "quick" if args.quick else "full",
             "date": args.date or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"), "started": started,
             "ended": utc_now(), "gpu_seconds": round(time.monotonic() - t0, 1), "budget_s": budget,
             "command": command, "host": socket.gethostname(), "versions": versions(), "classes": classes,
             "rows": rows, "notes": [f"occupancy from LDS assumes {wf.SIMDS_PER_WGP} SIMDs per WGP (inferred)"]}
    json_path, md_path = wf.write(sheet, args.out)
    counts = {status: sum(r["status"] == status for r in rows) for status in wf.STATUSES}
    print(json.dumps({"ok": True, "outcome": "measured", "json": str(json_path), "md": str(md_path), **counts}))
    return 0


def invented_samples(index: int, n: int = 5) -> list[float]:
    """Deterministic made-up timings for the self-check sheet (never a measurement)."""
    base = 1e-4 * (index + 1)
    return [base * (1 + 0.01 * ((i * 7) % 5)) for i in range(n)]


def self_check_sheet(fixtures: Path = FIXTURES, remarks: Path = REMARKS_FIXTURE) -> dict[str, Any]:
    """A sheet built from the fixtures and invented timings, plus what the parsers made of the fixtures."""
    compiled = json.loads((fixtures / "triton_compile_synthetic.json").read_text(encoding="utf-8"))
    trace = wf.trace_occupancy(wf.trace_kernels((fixtures / "rocprof_kernel_trace_synthetic.csv")
                                                .read_text(encoding="utf-8")))
    remark = wf.remark_occupancy(cf.parse_resource_remarks(remarks.read_text(encoding="utf-8")))
    rows, index = [], 0
    for name in wf.CLASSES:
        module = MODULES[name]
        for case in module.cases(True):
            record = compiled.get(case["id"])
            record = wf.compile_record(record) if record else None
            rows.append(wf.row(case["id"], module.CLASS, module.LEVER, case["setting"], module.WHAT,
                               command="self-check", params=case["params"], samples=invented_samples(index),
                               first_call=0.5, compile=record,
                               occupancy=wf.compile_occupancy(record, 512) if record else None))
            index += 1
    sheet = {"format": wf.FORMAT, "gfx": POD_GFX, "node": "pod-b", "mode": "self-check",
             "date": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"), "started": utc_now(), "ended": utc_now(),
             "gpu_seconds": 0.0, "budget_s": QUICK_BUDGET_S, "command": "wavefront.py self-check", "host": "self-check",
             "versions": {key: None for key in wf.VERSION_KEYS}, "classes": list(wf.CLASSES), "rows": rows,
             "notes": ["self-check: parsed fixtures and invented timings, not a measurement"]}
    return {"sheet": sheet, "trace": trace, "remarks": remark}


def self_check_main(args: argparse.Namespace) -> int:
    found = self_check_sheet(Path(args.fixtures))
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="wavefront-selfcheck-"))
    json_path, md_path = wf.write(found["sheet"], out)
    back = wf.read(json_path)
    same = back == json.loads(json.dumps(found["sheet"]))
    top = found["trace"]["top"][0] if found["trace"]["top"] else {}
    print(json.dumps({"ok": same, "outcome": "self-check" if same else "mismatch", "gpu": rocminfo_gfx(),
                      "json": str(json_path), "md": str(md_path), "rows": len(back["rows"]),
                      "trace_launches": found["trace"]["launches"], "trace_top": top.get("name"),
                      "remark_kernels": sorted(found["remarks"])}))
    return 0 if same else 1


def pod_command(classes: Sequence[str], account: str = "D", node: str = "pod-b", quick: bool = False) -> dict[str, Any]:
    """The one-line controller command that runs these classes on a pod, and the slot minutes to ask for."""
    minutes = sum(SLOT_MINUTES[name] for name in classes)
    timeout = max(600, minutes * 60)
    parts = ["python", "-m", "lab.controller", "run-script", "--account", account, "--timeout", str(timeout)]
    for path in POD_FILES:
        parts += ["--with", path]
    parts += [POD_SCRIPT, "run", "--node", node, "--only", ",".join(classes)]
    if quick:
        parts.append("--quick")
    return {"command": shlex.join(parts), "slot_minutes": minutes, "classes": list(classes)}


def pod_script_main(args: argparse.Namespace) -> int:
    print(json.dumps({"ok": True, **pod_command(selected(args.only), args.account, args.node, args.quick)}))
    return 0


def occupancy_main(args: argparse.Namespace) -> int:
    text = Path(args.trace).read_text(encoding="utf-8", errors="replace")
    found = wf.trace_occupancy(wf.trace_kernels(text), top=args.top)
    print(json.dumps({"ok": True, "outcome": "occupancy", "trace": args.trace, **found}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wavefront.py", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="verb", required=True)
    run = sub.add_parser("run", help="measure on this pod (node gate first)")
    run.add_argument("--node", required=True, help="the ledger name of this pod: pod-a or pod-b")
    run.add_argument("--only", help=f"comma list of classes: {', '.join(wf.CLASSES)} (default all)")
    run.add_argument("--quick", action="store_true", help=f"fewer cases and repeats, budget {QUICK_BUDGET_S:g} s")
    run.add_argument("--out", default=DEFAULT_OUT, help=f"where the sheet goes (default {DEFAULT_OUT})")
    run.add_argument("--budget-s", type=float, help="time budget in seconds (default 900, --quick 240)")
    run.add_argument("--date", help="the date in the file name (default today, UTC)")
    check = sub.add_parser("self-check", help="no GPU: parse the fixtures, write and re-read a sheet")
    check.add_argument("--out", help="where to write it (default a new temporary folder)")
    check.add_argument("--fixtures", default=str(FIXTURES), help="the fixture folder")
    pod = sub.add_parser("pod-script", help="print the one-line controller command for these classes")
    pod.add_argument("--only", required=True, help="comma list of classes")
    pod.add_argument("--account", default="D", help="hub account letter (default D, the trainer pod)")
    pod.add_argument("--node", default="pod-b", help="ledger name of that pod (default pod-b)")
    pod.add_argument("--quick", action="store_true")
    occ = sub.add_parser("occupancy", help="top kernels of a rocprofv3 kernel trace with their occupancy")
    occ.add_argument("trace", help="a rocprofv3 *_kernel_trace.csv")
    occ.add_argument("--top", type=int, default=25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verb == "run":
        return run_main(args, "python3 wavefront.py " + shlex.join(list(argv if argv is not None else sys.argv[1:])))
    if args.verb == "self-check":
        return self_check_main(args)
    if args.verb == "pod-script":
        return pod_script_main(args)
    return occupancy_main(args)


if __name__ == "__main__":
    sys.exit(main())
