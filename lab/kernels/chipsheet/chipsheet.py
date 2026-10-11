"""Measure the chip sheet of one AMD GPU pod, or self-check the parsing and file writing on a machine with no GPU.

    python3 chipsheet.py run --node pod-b [--quick] [--out ~/chipsheet] [--only copy,matmul] [--budget-s 840]
    python3 chipsheet.py self-check [--out DIR]

On the pod it runs from the Jupyter root next to chipsheet_format.py, probe.hip and pod_power.py (lab.controller
run-script --with uploads them; lab/kernels/chipsheet/pod_chipsheet.sh is the script it runs). In the repo it also
runs as `python -m lab.kernels.chipsheet.chipsheet`. It needs the pod image's torch, triton, hipcc, rocminfo and
amd-smi, and nothing else.

What `run` measures, each number with its exact command, its repeats and their spread, and the amd-smi watts of the
window it ran in (idle watts are read before each section):

  copy       device-to-device copy bandwidth, torch dst.copy_(src) on uint8 buffers from 1 MiB to 4 GiB; the value is
             DRAM traffic (each byte read once and written once, 2 x size / time), derived.copy_gbs is size / time
  matmul     fp16 and bf16 torch.nn.functional.linear at the 27B decode shapes (hidden 5120, attention 6144, MLP
             17408, lm_head 248320; lab/refine/research/kernels/w4a16_decode_gfx1201.md section 1.2) at batch 1, 4,
             8 and 16; TFLOP/s, derived.weight_gbs is the weight bytes read per second
  lora       fp16 and bf16 GEMMs of one 4B LoRA step (hidden 2560, rank 16, 4 x 256 tokens as lab/smoke_lora.py
             trains; the MLP width is read from the checkpoint's config.json when the pod has it)
  dequant    a plain Triton W4A16 kernel (int4 symmetric, group 128, unpacked to fp16 in registers, tl.dot) at the
             27B decode shapes, checked against torch on a dequantized copy first
  lds        the LDS size rocminfo and HIP report, and the largest dynamic LDS a workgroup launches with (probe lds)
  wave       the wavefront size the compiler chose for the probe kernel built for wave32 and for wave64
  occupancy  a latency-bound pointer chase at growing LDS per workgroup and growing live VGPRs: achieved loads per
             second (relative to the best point, the timing proxy for achieved occupancy) beside the compiler's
             VGPR count and occupancy remark and hipOccupancyMaxActiveBlocksPerMultiprocessor
  cache      cache sizes as rocminfo and the HIP runtime report them
  versions   ROCm, HIP, hipcc, torch, triton, python and amd-smi

The run keeps inside its budget (840 s, --quick 110 s): a measurement that would start past it is written as
skipped, never cut short. It writes <out>/chipsheet-<gfx>-<date>.json and .md and prints one JSON line naming them.
The dispatcher pulls the JSON back and runs `python -m lab chipsheet ingest`, which commits the files under
lab/refine/research/kernels/data/ and writes the ledger rows on the sheet's node.

Node gate: run refuses (exit 2) unless --node is pod-a or pod-b and rocminfo names gfx1151; nothing touches the GPU
before that. self-check never touches a GPU: it says whether this machine has one, parses the recorded tool output in
fixtures/lab/chipsheet/, builds a sheet with invented timings, writes and re-reads it, and exits 0.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

if (Path(__file__).resolve().parent / "pod_power.py").exists():  # on the pod: uploaded side by side
    import chipsheet_format as fmt  # type: ignore[no-redef]
    import pod_power  # type: ignore[no-redef]
else:  # in the repo, run as a module or as a file
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from lab import pod_power
    from lab.kernels.chipsheet import chipsheet_format as fmt

HERE = Path(__file__).resolve().parent
PROBE_SOURCE = HERE / "probe.hip"
REPO = HERE.parents[2] if len(HERE.parents) > 2 else HERE
FIXTURES = REPO / "fixtures" / "lab" / "chipsheet"
# the amd-smi power reading recorded on pod C (Strix Halo), which the self-check parses
POWER_FIXTURE = REPO / "fixtures" / "telemetry" / "pod_power" / "strixhalo_amdsmi_metric_power.txt"
POD_GFX = "gfx1151"
NODES = fmt.NODES
DEFAULT_OUT = "~/chipsheet"
FULL_BUDGET_S = 840.0
QUICK_BUDGET_S = 110.0
EXIT_GATE = 2

# 27B decode GEMMs as (name, K, N): lab/refine/research/kernels/w4a16_decode_gfx1201.md section 1.2 (hidden 5120,
# 24 query heads of 256 = 6144, MLP width 17408, vocabulary 248320).
DECODE_SHAPES = (("k5120_n6144", 5120, 6144), ("k6144_n5120", 6144, 5120),
                 ("k5120_n17408", 5120, 17408), ("k17408_n5120", 17408, 5120))
LM_HEAD = ("k5120_n248320", 5120, 248320)
DECODE_BATCHES = (1, 4, 8, 16)
QUICK_DECODE = (DECODE_SHAPES[0], DECODE_SHAPES[3])
QUICK_BATCHES = (1, 16)
# 4B LoRA step: hidden 2560 (lab/refine/research/learning/lora_recipe.md), rank 16 and 4 x 256 tokens (smoke_lora.py)
LORA_HIDDEN = 2560
LORA_RANK = 16
LORA_TOKENS = 4 * 256
LORA_CONFIG = "~/models/Qwen3.5-4B/config.json"
COPY_SIZES_MIB = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
QUICK_COPY_MIB = (1, 64, 1024)
DEQUANT_GROUP = 128
DEQUANT_CONFIGS = ({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64}, {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 128})
OCC_LDS_BYTES = (0, 4096, 8192, 16384, 24576, 32768, 49152, 65536)
QUICK_OCC_LDS = (0, 16384, 65536)
OCC_VGPRS = (4, 16, 32, 64, 96, 128, 192, 240)
QUICK_OCC_VGPRS = (4, 64, 240)
PROBE_TIMEOUT_S = 120.0
COMPILE_TIMEOUT_S = 300.0


def log(message: str) -> None:
    print(f"[chipsheet {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_tool(argv: Sequence[str], timeout_s: float = 60.0) -> tuple[int, str, str]:
    """(exit code, stdout, stderr) of one command; 127 and the reason when it cannot be started or times out."""
    try:
        done = subprocess.run(list(argv), capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout_s, check=False)
    except FileNotFoundError as exc:
        return 127, "", str(exc)
    except subprocess.TimeoutExpired:
        return 124, "", f"{argv[0]} gave no answer within {timeout_s:g} s"
    return done.returncode, done.stdout, done.stderr


def tool(name: str) -> str | None:
    """The program for a ROCm tool: on PATH, else under /opt/rocm/bin (amd-smi is off PATH on the SJC image)."""
    return pod_power.find_tool(name)


# --- what this machine is ---------------------------------------------------------------------------------------

def read_rocminfo() -> tuple[str | None, str | None]:
    """(rocminfo's output, None) or (None, why it could not be read)."""
    program = tool("rocminfo")
    if program is None:
        return None, "rocminfo is not installed (not on PATH or under /opt/rocm/bin)"
    code, out, err = run_tool([program], timeout_s=30.0)
    if code != 0:
        return None, f"rocminfo exited {code}: {(err or out).strip()[-200:]}"
    return out, None


def gate(node: str, rocminfo_text: str | None, why: str | None, machine: str = platform.machine()) -> str | None:
    """Why `run` may not measure here, or None when it may: a known pod node, not the Jetson (aarch64), and a
    rocminfo that names gfx1151, the SJC pods' Strix Halo. Any doubt refuses."""
    if node not in NODES:
        return f"--node must be one of {', '.join(NODES)}, got {node!r}"
    if machine.lower() in ("aarch64", "arm64"):
        return f"this is an {machine} machine (the Jetson is aarch64), not a pod"
    if rocminfo_text is None:
        return f"cannot tell what GPU this is: {why}"
    agent = fmt.parse_rocminfo(rocminfo_text)
    if agent is None:
        return "rocminfo lists no GPU agent"
    if agent.get("gfx") != POD_GFX:
        return f"rocminfo names {agent.get('gfx')!r}, not {POD_GFX}: this is not a Strix Halo pod"
    return None


def gpu_presence() -> dict[str, Any]:
    """What this machine says about a GPU, without touching one: rocminfo found or not, torch importable or not."""
    have: dict[str, Any] = {"rocminfo": tool("rocminfo"), "hipcc": shutil.which("hipcc") or tool("hipcc")}
    try:
        import torch  # noqa: F401 - only whether it imports

        have["torch"] = True
    except ImportError:
        have["torch"] = False
    have["gpu"] = bool(have["rocminfo"]) and bool(have["torch"])
    return have


# --- versions ---------------------------------------------------------------------------------------------------

def collect_versions(hipcc_text: str | None) -> dict[str, str | None]:
    versions: dict[str, str | None] = dict.fromkeys(fmt.VERSION_KEYS)
    versions["python"] = platform.python_version()
    info = Path("/opt/rocm/.info/version")
    if info.exists():
        versions["rocm"] = fmt.parse_rocm_version(info.read_text(encoding="utf-8", errors="replace"))
    smi = tool("amd-smi")
    if smi:
        code, out, _ = run_tool([smi, "version"], timeout_s=30.0)
        if code == 0 and out.strip():
            versions["amd_smi"] = " ".join(out.split())[:300]
            versions["rocm"] = versions["rocm"] or fmt.parse_rocm_version(out)
    if hipcc_text:
        parsed = fmt.parse_hipcc_version(hipcc_text)
        versions["hip"] = parsed["hip"]
        versions["hipcc"] = parsed["clang"]
    try:
        import torch

        versions["torch"] = torch.__version__
        versions["torch_hip"] = getattr(torch.version, "hip", None)
    except ImportError:
        pass
    try:
        import triton

        versions["triton"] = triton.__version__
    except ImportError:
        pass
    return versions


# --- watts ------------------------------------------------------------------------------------------------------

class Power:
    """amd-smi (or rocm-smi) read in a thread for the whole run, through lab/pod_power.py's reader choice. A window
    is the readings between two monotonic times; a reading is stamped at the middle of its tool call."""

    def __init__(self, period_s: float = 0.25) -> None:
        self.period_s = period_s
        self.samples: list[tuple[float, float]] = []
        self.errors = 0
        self.reader: Any = None
        self.attempts: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            self.reader, self.attempts = pod_power.choose_reader(0)
        except Exception as exc:  # a reader that cannot even be chosen costs the watts, never the run
            self.attempts = [{"error": f"{type(exc).__name__}: {exc}"}]
        if self.reader is None:
            return
        self._thread = threading.Thread(target=self._loop, name="chipsheet-power", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            before = time.monotonic()
            try:
                reading = self.reader.read()
                if reading.socket_w is not None:
                    self.samples.append(((before + time.monotonic()) / 2, float(reading.socket_w)))
            except Exception:
                self.errors += 1
            self._stop.wait(max(0.0, self.period_s - (time.monotonic() - before)))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def describe(self) -> dict[str, Any]:
        if self.reader is None:
            return {"tool": None, "error": "no reader read a power number", "attempts": self.attempts}
        return {**self.reader.describe(), "period_s": self.period_s, "read_errors": self.errors}

    def window(self, start: float, end: float) -> list[float]:
        return [w for t, w in list(self.samples) if start <= t <= end]

    def summary(self, start: float, end: float, idle_w: float | None) -> dict[str, Any]:
        if self.reader is None:
            return {"error": "no power reader", "idle_w": None}
        watts = self.window(start, end)
        if not watts:
            return {"error": "no reading landed inside the window", "idle_w": idle_w, "samples": 0}
        return {"idle_w": idle_w, "loaded_mean_w": sum(watts) / len(watts), "loaded_max_w": max(watts),
                "samples": len(watts)}


# --- the run ----------------------------------------------------------------------------------------------------

class Run:
    """One measuring run: the budget, the power windows and the entries, section by section."""

    def __init__(self, args: argparse.Namespace, command: str) -> None:
        self.args = args
        self.quick = bool(args.quick)
        self.command = command
        self.budget_s = args.budget_s if args.budget_s else (QUICK_BUDGET_S if self.quick else FULL_BUDGET_S)
        self.repeats = 5 if self.quick else 15
        self.warmup = 2 if self.quick else 3
        self.min_load_s = 0.6 if self.quick else 1.5
        self.idle_s = 1.0 if self.quick else 3.0
        self.t0 = time.monotonic()
        self.power = Power()
        self.idle: dict[str, Any] = {}
        self.entries: list[dict[str, Any]] = []
        self.notes: list[str] = []
        self.work = Path(args.out).expanduser() / "build"
        self.section = ""
        self.idle_w: float | None = None

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def left(self) -> float:
        return self.budget_s - self.elapsed()

    def cmd(self, section: str) -> str:
        return f"{self.command} --only {section}"

    def begin(self, section: str) -> None:
        """Read the idle watts before a section, with nothing running on the card."""
        self.section = section
        log(f"section {section} ({self.elapsed():.0f} s used of {self.budget_s:.0f})")
        if self.power.reader is None or self.left() < self.idle_s + 5:
            self.idle[section] = {"error": "no power reader" if self.power.reader is None else "skipped: budget"}
            self.idle_w = None
            return
        start = time.monotonic()
        time.sleep(self.idle_s)
        watts = self.power.window(start, time.monotonic())
        self.idle_w = sum(watts) / len(watts) if watts else None
        self.idle[section] = {"idle_w": self.idle_w, "samples": len(watts), "seconds": self.idle_s}

    def add(self, item: dict[str, Any]) -> None:
        self.entries.append(item)

    def measure(self, mid: str, what: str, unit: str, op: str, params: Mapping[str, Any],
                body: Callable[[], tuple[list[float], dict[str, Any]]], *, need_s: float = 3.0) -> None:
        """Time one thing: body returns (samples in unit, derived numbers). A failure becomes a failed entry and the
        run goes on; a measurement that would start past the budget becomes a skipped one."""
        section = self.section
        if self.left() < need_s:
            self.add(fmt.entry(mid, section, what, unit, source="timing", command=self.cmd(section), op=op,
                               params=params, status="skipped",
                               error=f"budget: {self.elapsed():.0f} s of {self.budget_s:.0f} s used"))
            return
        start = time.monotonic()
        try:
            samples, derived = body()
        except Exception as exc:
            self.add(fmt.entry(mid, section, what, unit, source="timing", command=self.cmd(section), op=op,
                               params=params, status="failed", error=f"{type(exc).__name__}: {exc}"[:400],
                               watts=self.power.summary(start, time.monotonic(), self.idle_w)))
            _free_gpu()
            return
        self.add(fmt.entry(mid, section, what, unit, source="timing", command=self.cmd(section), op=op,
                           params=params, samples=samples, derived=derived,
                           watts=self.power.summary(start, time.monotonic(), self.idle_w)))

    def fact(self, mid: str, what: str, unit: str, value: float | int | None, *, source: str, command: str, op: str,
             params: Mapping[str, Any] | None = None, error: str | None = None, section: str | None = None) -> None:
        """A number a tool or the runtime reports, read once; None becomes a failed entry with the reason."""
        where = section or self.section
        if value is None:
            self.add(fmt.entry(mid, where, what, unit, source=source, command=command, op=op, params=params,
                               status="failed", error=error or "the tool printed no value"))
        else:
            self.add(fmt.entry(mid, where, what, unit, source=source, command=command, op=op, params=params,
                               samples=[float(value)], value=float(value)))


def _free_gpu() -> None:
    try:
        import torch

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:
        pass


def time_torch(fn: Callable[[], Any], *, repeats: int, warmup: int, min_load_s: float,
               target_s: float = 0.002) -> list[float]:
    """Seconds per call of fn, one sample per repeat; each repeat times enough back-to-back calls between two
    events to last about target_s, then fn keeps running until min_load_s has passed so the watts are read."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    one = max(start.elapsed_time(end) / 1e3, 1e-7)
    inner = max(1, min(2000, int(target_s / one)))
    began = time.monotonic()
    samples = []
    for _ in range(repeats):
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 1e3 / inner)
    while time.monotonic() - began < min_load_s:
        for _ in range(inner):
            fn()
        torch.cuda.synchronize()
    return samples


# --- sections ---------------------------------------------------------------------------------------------------

def section_copy(run: Run) -> None:
    import torch

    run.begin("copy")
    for mib in (QUICK_COPY_MIB if run.quick else COPY_SIZES_MIB):
        nbytes = mib * fmt.MIB

        def body(nbytes: int = nbytes) -> tuple[list[float], dict[str, Any]]:
            src = torch.empty(nbytes, dtype=torch.uint8, device="cuda").random_(0, 255)
            dst = torch.empty_like(src)
            try:
                secs = time_torch(lambda: dst.copy_(src), repeats=run.repeats, warmup=run.warmup,
                                  min_load_s=run.min_load_s)
            finally:
                del src, dst
                _free_gpu()
            copy = fmt.stats([nbytes / s / 1e9 for s in secs])
            return [2 * nbytes / s / 1e9 for s in secs], {"copy_gbs": copy["median"]}

        run.measure(f"copy.d2d.size_{fmt.size_label(nbytes)}", f"device-to-device copy of {fmt.size_label(nbytes)}",
                    "GB/s", f"torch: dst.copy_(src), uint8[{nbytes}] on cuda:0, value = 2 x bytes / time",
                    {"bytes": nbytes}, body, need_s=run.min_load_s + 2)


def _linear_body(run: Run, m: int, k: int, n: int, dtype_name: str) -> Callable[[], tuple[list[float], dict[str, Any]]]:
    def body() -> tuple[list[float], dict[str, Any]]:
        import torch
        import torch.nn.functional as F

        dtype = getattr(torch, dtype_name)
        x = torch.randn(m, k, dtype=dtype, device="cuda")
        w = torch.randn(n, k, dtype=dtype, device="cuda") * 0.02
        try:
            secs = time_torch(lambda: F.linear(x, w), repeats=run.repeats, warmup=run.warmup,
                              min_load_s=run.min_load_s)
        finally:
            del x, w
            _free_gpu()
        median = fmt.stats(secs)["median"]
        return [2 * m * n * k / s / 1e12 for s in secs], {"weight_gbs": n * k * 2 / median / 1e9,
                                                          "seconds_median": median}
    return body


def section_matmul(run: Run) -> None:
    run.begin("matmul")
    shapes = QUICK_DECODE if run.quick else DECODE_SHAPES + (LM_HEAD,)
    batches = QUICK_BATCHES if run.quick else DECODE_BATCHES
    for dtype_name in ("float16", "bfloat16"):
        short = "fp16" if dtype_name == "float16" else "bf16"
        for name, k, n in shapes:
            for m in batches:
                run.measure(f"matmul.{short}.decode27b.{name}.m{m}",
                            f"{short} decode GEMM, batch {m}, K {k} to N {n}", "TFLOP/s",
                            f"torch.nn.functional.linear(x[{m},{k}], w[{n},{k}]) {dtype_name}, value = 2MNK / time",
                            {"m": m, "k": k, "n": n, "dtype": dtype_name}, _linear_body(run, m, k, n, dtype_name),
                            need_s=run.min_load_s + 2)


def lora_widths(config_path: str) -> tuple[list[tuple[str, int, int]], str]:
    """The (name, K, N) projections of a 4B LoRA step and where the MLP width came from: the square 2560 one always,
    the MLP pair when the checkpoint's config.json names intermediate_size."""
    shapes = [("k2560_n2560", LORA_HIDDEN, LORA_HIDDEN)]
    path = Path(config_path).expanduser()
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return shapes, f"MLP width unknown: {path} could not be read ({type(exc).__name__})"
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    width = text.get("intermediate_size")
    hidden = text.get("hidden_size")
    if not isinstance(width, int) or (isinstance(hidden, int) and hidden != LORA_HIDDEN):
        return shapes, f"MLP width unknown: {path} names intermediate_size {width!r}, hidden_size {hidden!r}"
    shapes += [(f"k{LORA_HIDDEN}_n{width}", LORA_HIDDEN, width), (f"k{width}_n{LORA_HIDDEN}", width, LORA_HIDDEN)]
    return shapes, f"MLP width {width} from {path}"


def section_lora(run: Run) -> None:
    run.begin("lora")
    shapes, where = lora_widths(run.args.lora_config)
    run.notes.append(f"lora: {where}")
    if run.quick:
        shapes = shapes[:1]
    t, r = LORA_TOKENS, LORA_RANK
    for dtype_name in ("float16", "bfloat16"):
        short = "fp16" if dtype_name == "float16" else "bf16"
        for name, k, n in shapes:
            # (step, M, K, N): base forward and input grad, then the low-rank A and B products and their weight grads
            steps = (("fwd", t, k, n), ("dgrad", t, n, k), ("a_fwd", t, k, r), ("b_fwd", t, r, n),
                     ("a_wgrad", k, t, r), ("b_wgrad", r, t, n))
            if run.quick:
                steps = steps[:1] + steps[2:4]
            for step, m, kk, nn in steps:
                run.measure(f"lora.{short}.lora4b.{name}.{step}",
                            f"{short} 4B LoRA {step} GEMM of the {name} projection: [{m},{kk}] x [{kk},{nn}]",
                            "TFLOP/s",
                            f"torch.nn.functional.linear(x[{m},{kk}], w[{nn},{kk}]) {dtype_name}, value = 2MNK / time",
                            {"m": m, "k": kk, "n": nn, "dtype": dtype_name, "rank": r, "tokens": t, "step": step},
                            _linear_body(run, m, kk, nn, dtype_name), need_s=run.min_load_s + 2)


def _w4a16_kernel() -> Any:
    """The plain Triton W4A16 kernel: A fp16 [M, K]; QW uint8 [K/2, N], row k/2 holding k (low nibble) and k + 1
    (high nibble), each an int4 code q with weight (q - 8) * scale; S fp16 [K/GROUP, N]; C fp16 [M, N]. Even and
    odd K columns of A meet the low and high nibbles in two tl.dot calls, so no interleave is needed."""
    import triton
    import triton.language as tl

    @triton.jit
    def w4a16(a_ptr, qw_ptr, s_ptr, c_ptr, M, N, K,
              stride_am, stride_ak, stride_qk, stride_qn, stride_sg, stride_sn, stride_cm, stride_cn,
              GROUP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K // 2)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            q = tl.load(qw_ptr + (k0 // 2 + rk)[:, None] * stride_qk + rn[None, :] * stride_qn,
                        mask=rn[None, :] < N, other=0)
            s = tl.load(s_ptr + (k0 // GROUP) * stride_sg + rn * stride_sn, mask=rn < N, other=0.0).to(tl.float32)
            lo = (((q & 0xF).to(tl.float32) - 8.0) * s[None, :]).to(tl.float16)
            hi = ((((q >> 4) & 0xF).to(tl.float32) - 8.0) * s[None, :]).to(tl.float16)
            ke = k0 + 2 * rk
            a_even = tl.load(a_ptr + rm[:, None] * stride_am + ke[None, :] * stride_ak, mask=rm[:, None] < M,
                             other=0.0)
            a_odd = tl.load(a_ptr + rm[:, None] * stride_am + (ke + 1)[None, :] * stride_ak, mask=rm[:, None] < M,
                            other=0.0)
            acc += tl.dot(a_even, lo)
            acc += tl.dot(a_odd, hi)
        tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc.to(tl.float16),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))

    return w4a16


def _dequant_body(run: Run, kernel: Any, m: int, k: int, n: int, cfg: Mapping[str, int]
                  ) -> Callable[[], tuple[list[float], dict[str, Any]]]:
    def body() -> tuple[list[float], dict[str, Any]]:
        import torch
        import triton

        gen = torch.Generator(device="cuda").manual_seed(m * 1000003 + k * 31 + n)
        codes = torch.randint(0, 16, (k, n), dtype=torch.uint8, device="cuda", generator=gen)
        qw = (codes[0::2] | (codes[1::2] << 4)).contiguous()
        scales = (torch.rand(k // DEQUANT_GROUP, n, device="cuda", generator=gen) * 0.02 + 0.001).half()
        a = torch.randn(m, k, device="cuda", generator=gen).half()
        c = torch.empty(m, n, dtype=torch.float16, device="cuda")
        grid = (triton.cdiv(m, cfg["BLOCK_M"]), triton.cdiv(n, cfg["BLOCK_N"]))

        def call() -> None:
            kernel[grid](a, qw, scales, c, m, n, k, a.stride(0), a.stride(1), qw.stride(0), qw.stride(1),
                         scales.stride(0), scales.stride(1), c.stride(0), c.stride(1), GROUP=DEQUANT_GROUP, **cfg)

        try:
            call()
            torch.cuda.synchronize()
            weights = (codes.float() - 8.0) * scales.float().repeat_interleave(DEQUANT_GROUP, 0)
            ref = a.float() @ weights
            rel = float((c.float() - ref).norm() / ref.norm().clamp_min(1e-12))
            del weights, ref
            if rel > 1e-2:
                raise ValueError(f"wrong answer: relative L2 error {rel:.3g} against torch on the dequantized weights")
            secs = time_torch(call, repeats=run.repeats, warmup=run.warmup, min_load_s=run.min_load_s)
        finally:
            del codes, qw, scales, a, c
            _free_gpu()
        median = fmt.stats(secs)["median"]
        weight_bytes = k * n // 2 + (k // DEQUANT_GROUP) * n * 2
        return [2 * m * n * k / s / 1e12 for s in secs], {"weight_gbs": weight_bytes / median / 1e9,
                                                          "rel_l2_error": rel, "seconds_median": median}
    return body


def section_dequant(run: Run) -> None:
    run.begin("dequant")
    try:
        kernel = _w4a16_kernel()
    except Exception as exc:
        run.fact("dequant.w4a16.compile", "the Triton W4A16 kernel compiles", "ok", None, source="timing",
                 command=run.cmd("dequant"), op="triton.jit(w4a16)", error=f"{type(exc).__name__}: {exc}"[:400])
        return
    shapes = QUICK_DECODE if run.quick else DECODE_SHAPES
    batches = QUICK_BATCHES if run.quick else DECODE_BATCHES
    configs = DEQUANT_CONFIGS[:1] if run.quick else DEQUANT_CONFIGS
    for cfg in configs:
        tag = f"bn{cfg['BLOCK_N']}_bk{cfg['BLOCK_K']}"
        for name, k, n in shapes:
            for m in batches:
                run.measure(f"dequant.w4a16.decode27b.{name}.m{m}.{tag}",
                            f"int4-to-fp16 dequantize plus matmul (Triton), batch {m}, K {k} to N {n}, {tag}",
                            "TFLOP/s",
                            f"triton w4a16[{m},{k}]x[{k}/2 packed,{n}] group {DEQUANT_GROUP}, "
                            f"BLOCK_M {cfg['BLOCK_M']} BLOCK_N {cfg['BLOCK_N']} BLOCK_K {cfg['BLOCK_K']}, "
                            "value = 2MNK / time",
                            {"m": m, "k": k, "n": n, "group": DEQUANT_GROUP, **cfg},
                            _dequant_body(run, kernel, m, k, n, cfg), need_s=run.min_load_s + 8)


# --- the HIP probe ----------------------------------------------------------------------------------------------

def compile_probe(run: Run, wave: int) -> tuple[Path | None, str, str]:
    """(the binary or None, the exact compile command, its stderr): the resource-usage remarks are on stderr."""
    hipcc = shutil.which("hipcc") or tool("hipcc")
    run.work.mkdir(parents=True, exist_ok=True)
    binary = run.work / f"probe_w{wave}"
    flags = ["-mwavefrontsize64"] if wave == 64 else ["-mno-wavefrontsize64"]
    argv = [hipcc or "hipcc", "-O3", f"--offload-arch={POD_GFX}", *flags, "-Rpass-analysis=kernel-resource-usage",
            "-o", str(binary), str(PROBE_SOURCE)]
    command = shlex.join(argv)
    if hipcc is None:
        return None, command, "hipcc is not installed"
    code, out, err = run_tool(argv, timeout_s=COMPILE_TIMEOUT_S)
    if (code != 0 or not binary.exists()) and wave == 32:
        # wave32 is the gfx11 default: a compiler that refuses the explicit flag builds it without one
        first = f"{command} failed (exit {code}); "
        argv = [a for a in argv if a != "-mno-wavefrontsize64"]
        command = shlex.join(argv)
        code, out, err = run_tool(argv, timeout_s=COMPILE_TIMEOUT_S)
        if code == 0 and binary.exists():
            run.notes.append(f"wave: {first}built with the compiler default instead: {command}")
    if code != 0 or not binary.exists():
        return None, command, f"exit {code}: {(err or out).strip()[-600:]}"
    return binary, command, err


def probe(binary: Path, *args: Any, timeout_s: float = PROBE_TIMEOUT_S) -> tuple[list[dict[str, Any]], str, str]:
    """(the probe's JSON lines, the exact command, the error text when it failed)."""
    argv = [str(binary), *map(str, args)]
    code, out, err = run_tool(argv, timeout_s=timeout_s)
    lines = fmt.parse_probe_lines(out)
    errors = [line.get("error") for line in lines if line.get("probe") == "error"]
    problem = "" if code == 0 and not errors else f"exit {code}: {'; '.join(map(str, errors)) or err.strip()[-300:]}"
    return lines, shlex.join(argv), problem


def section_probes(run: Run, agent: Mapping[str, Any] | None, rocminfo_cmd: str) -> dict[str, Any]:
    """lds, wave, occupancy and cache: build the probe for wave32 and wave64, then ask it. Returns the HIP props."""
    run.begin("wave")
    builds: dict[int, tuple[Path | None, str, str]] = {}
    if run.left() >= 40:  # both builds at once: hipcc is most of this section's time
        threads = [threading.Thread(target=lambda w=wave: builds.__setitem__(w, compile_probe(run, w)))
                   for wave in (32, 64)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    for wave in (32, 64):
        if wave not in builds:
            run.fact(f"wave.w{wave}.compiled", f"wavefront size compiled for a wave{wave} build", "lanes", None,
                     source="probe", command="hipcc", op="probe wave", error="skipped: budget")
            continue
        binary, command, err = builds[wave]
        if binary is None:
            run.fact(f"wave.w{wave}.compiled", f"wavefront size compiled for a wave{wave} build", "lanes", None,
                     source="probe", command=command, op="probe wave", error=f"compile failed: {err}")
            continue
        lines, pcmd, problem = probe(binary, "wave")
        found = next((line for line in lines if line.get("probe") == "wave"), None)
        run.fact(f"wave.w{wave}.compiled", f"__builtin_amdgcn_wavefrontsize() of a kernel built for wave{wave}",
                 "lanes", found.get("compiled_wave") if found else None, source="probe",
                 command=f"{command} && {pcmd}", op="wave_probe<<<1,256>>>: __builtin_amdgcn_wavefrontsize()",
                 error=problem or None)
        run.fact(f"wave.w{wave}.lanes", f"lanes counted in one wave of a wave{wave} build (max __lane_id() + 1)",
                 "lanes", found.get("lanes") if found else None, source="probe", command=f"{command} && {pcmd}",
                 op="wave_probe<<<1,256>>>: atomicMax(lanes, __lane_id() + 1)", error=problem or None)
    remarks = fmt.parse_resource_remarks(builds[32][2]) if 32 in builds and builds[32][0] else {}
    w32 = builds.get(32, (None, "", ""))[0]
    hip: dict[str, Any] = {}

    run.begin("cache")
    if w32 is not None:
        lines, pcmd, problem = probe(w32, "props")
        hip = next((line for line in lines if line.get("probe") == "props"), {})
        hip_cmd = pcmd
    else:
        problem, hip_cmd = "the probe did not build", "probe props"
    cache = (agent or {}).get("cache_bytes") or {}
    for level in ("l1", "l2", "l3"):
        run.fact(f"cache.rocminfo.{level}_bytes", f"{level.upper()} cache size rocminfo reports", "bytes",
                 cache.get(level), source="tool", command=rocminfo_cmd, op=f"rocminfo GPU agent, Cache Info {level.upper()}",
                 error=f"rocminfo printed no {level.upper()}")
    run.fact("cache.rocminfo.cacheline_bytes", "cacheline size rocminfo reports", "bytes",
             (agent or {}).get("cacheline_bytes"), source="tool", command=rocminfo_cmd, op="rocminfo Cacheline Size")
    run.fact("cache.hip.l2_bytes", "L2 size the HIP runtime reports", "bytes", hip.get("l2CacheSize"), source="probe",
             command=hip_cmd, op="hipGetDeviceProperties: l2CacheSize", error=problem or None)
    run.fact("cache.hip.compute_units", "compute units the HIP runtime reports", "CUs",
             hip.get("multiProcessorCount"), source="probe", command=hip_cmd,
             op="hipGetDeviceProperties: multiProcessorCount", error=problem or None)
    run.fact("cache.rocminfo.compute_units", "compute units rocminfo reports", "CUs",
             (agent or {}).get("compute_units"), source="tool", command=rocminfo_cmd, op="rocminfo Compute Unit")

    run.begin("lds")
    run.fact("lds.rocminfo.group_segment_bytes", "LDS size rocminfo reports (GROUP segment pool)", "bytes",
             (agent or {}).get("lds_bytes"), source="tool", command=rocminfo_cmd, op="rocminfo Pool Info: Segment GROUP, Size")
    run.fact("lds.hip.shared_per_block_bytes", "LDS per workgroup the HIP runtime reports", "bytes",
             hip.get("sharedMemPerBlock"), source="probe", command=hip_cmd, op="hipGetDeviceProperties: sharedMemPerBlock",
             error=problem or None)
    run.fact("lds.hip.shared_per_cu_bytes", "LDS per CU the HIP runtime reports", "bytes",
             hip.get("maxSharedMemoryPerMultiProcessor"), source="probe", command=hip_cmd,
             op="hipGetDeviceProperties: maxSharedMemoryPerMultiProcessor", error=problem or None)
    if w32 is not None and run.left() > 10:
        lines, pcmd, problem = probe(w32, "lds")
        found = next((line for line in lines if line.get("probe") == "lds"), None)
        run.fact("lds.max_launch_bytes", "largest dynamic LDS a 256-thread workgroup launched and ran with", "bytes",
                 found.get("max_ok_bytes") if found else None, source="probe", command=pcmd,
                 op="lds_probe<<<1,256,bytes>>> doubling from 1 KiB then bisecting to 128 B; launch error, sync and "
                    "read-back checked",
                 params={k: v for k, v in (found or {}).items() if k != "probe"}, error=problem or None)
    else:
        run.fact("lds.max_launch_bytes", "largest dynamic LDS a 256-thread workgroup launched and ran with", "bytes",
                 None, source="probe", command="probe lds", op="lds_probe", error="the probe did not build or budget")

    run.begin("occupancy")
    lds_points = QUICK_OCC_LDS if run.quick else OCC_LDS_BYTES
    vgpr_points = QUICK_OCC_VGPRS if run.quick else OCC_VGPRS
    occ: list[tuple[str, dict[str, Any] | None, str, str, dict[str, Any]]] = []
    for kind, points in (("occ_lds", lds_points), ("occ_vgpr", vgpr_points)):
        for point in points:
            mid = f"occupancy.{'lds' if kind == 'occ_lds' else 'vgpr'}.p_{point}"
            if w32 is None or run.left() < 6:
                occ.append((mid, None, f"probe {kind} {point}", "the probe did not build" if w32 is None
                            else f"budget: {run.elapsed():.0f} s of {run.budget_s:.0f} s used", {}))
                continue
            start = time.monotonic()
            lines, pcmd, problem = probe(w32, kind, point, run.repeats, int(run.min_load_s * 1000))
            found = next((line for line in lines if line.get("probe") == kind), None)
            watts = run.power.summary(start, time.monotonic(), run.idle_w)
            occ.append((mid, found, pcmd, problem, watts))
    rates = [_loads_per_s(found) for _, found, _, problem, _ in occ if found and not problem]
    best = max((fmt.stats(r)["median"] or 0.0 for r in rates), default=0.0)
    for mid, found, pcmd, problem, watts in occ:
        is_lds = ".lds." in mid
        point = int(mid.rsplit("_", 1)[1])
        op = (f"occ_lds<<<CUs x 32, 256, {point} B LDS>>>: 512-step pointer chase over 256 MiB" if is_lds else
              f"occ_vgpr_{point}<<<CUs x 32, 256>>>: 512-step pointer chase with {point} floats kept live")
        what = (f"achieved chase loads per second with {point} B of LDS per workgroup" if is_lds else
                f"achieved chase loads per second with {point} floats kept live per thread")
        if not found or problem:
            run.add(fmt.entry(mid, "occupancy", what, "Gload/s", source="probe", command=pcmd, op=op,
                              status="skipped" if "budget" in (problem or "") else "failed",
                              error=problem or "the probe printed nothing", watts=watts or None))
            continue
        samples = _loads_per_s(found)
        median = fmt.stats(samples)["median"] or 0.0
        derived: dict[str, Any] = {"relative_throughput": median / best if best else None,
                                   "hip_blocks_per_cu": found.get("blocks_per_cu"),
                                   "hip_waves_per_cu": (found.get("blocks_per_cu") or 0) * 256 // 32}
        usage = remarks.get(found.get("kernel", ""), {})
        for key, name in (("VGPRs", "vgprs"), ("SGPRs", "sgprs"), ("Occupancy [waves/SIMD]", "remark_waves_per_simd"),
                          ("ScratchSize [bytes/lane]", "scratch_bytes_per_lane"), ("VGPRs Spill", "vgpr_spill"),
                          ("LDS Size [bytes/block]", "static_lds_bytes")):
            if key in usage:
                derived[name] = usage[key]
        params = {k: v for k, v in found.items() if k not in ("probe", "ms")}
        run.add(fmt.entry(mid, "occupancy", what, "Gload/s", source="probe", command=pcmd, op=op, params=params,
                          samples=samples, derived=derived, watts=watts))
    return hip


def _loads_per_s(found: Mapping[str, Any]) -> list[float]:
    """Chase loads per second (in billions) of each timed launch the occupancy probe printed."""
    loads = found["blocks"] * found["threads"] * found["iters"]
    return [loads / (ms / 1e3) / 1e9 for ms in found.get("ms", []) if ms and ms > 0]


# --- the commands -----------------------------------------------------------------------------------------------

ALL_SECTIONS = ("copy", "matmul", "lora", "dequant", "probes")


def run_main(args: argparse.Namespace, command: str) -> int:
    rocminfo_text, why = read_rocminfo()
    refused = gate(args.node, rocminfo_text, why)
    if refused:
        print(json.dumps({"ok": False, "outcome": "refused", "error": refused}))
        return EXIT_GATE
    agent = fmt.parse_rocminfo(rocminfo_text or "")
    only = [s.strip() for s in (args.only or ",".join(ALL_SECTIONS)).split(",") if s.strip()]
    unknown = [s for s in only if s not in ALL_SECTIONS + ("lds", "wave", "occupancy", "cache")]
    if unknown:
        print(json.dumps({"ok": False, "outcome": "usage", "error": f"unknown sections {unknown}"}))
        return 2
    run = Run(args, command)
    started = utc_now()
    hipcc = shutil.which("hipcc") or tool("hipcc")
    hipcc_text = run_tool([hipcc, "--version"])[1] if hipcc else None
    versions = collect_versions(hipcc_text)
    run.power.start()
    hip: dict[str, Any] = {}
    rocminfo_cmd = tool("rocminfo") or "rocminfo"
    try:
        import torch
    except ImportError as exc:
        run.power.stop()
        print(json.dumps({"ok": False, "outcome": "no_torch", "error": str(exc)}))
        return 1
    if not torch.cuda.is_available():
        run.power.stop()
        print(json.dumps({"ok": False, "outcome": "no_gpu", "error": "rocminfo names gfx1151 but torch sees no GPU"}))
        return 1
    try:
        for name, fn in (("copy", section_copy), ("matmul", section_matmul), ("lora", section_lora),
                         ("dequant", section_dequant)):
            if name in only:
                fn(run)
        if any(s in only for s in ("probes", "lds", "wave", "occupancy", "cache")):
            hip = section_probes(run, agent, rocminfo_cmd)
    finally:
        run.power.stop()
    sheet = {
        "format": fmt.FORMAT, "gfx": (agent or {}).get("gfx") or POD_GFX, "node": args.node,
        "mode": "quick" if run.quick else "full", "date": args.date or started[:10], "started": started,
        "ended": utc_now(), "gpu_seconds": round(run.elapsed(), 1), "budget_s": run.budget_s, "command": command,
        "host": socket.gethostname(), "versions": versions, "device": {"rocminfo": agent, "hip": hip},
        "power": {"reader": run.power.describe(), "idle": run.idle}, "measurements": run.entries,
        "notes": run.notes,
    }
    json_path, md_path = fmt.write(sheet, args.out)
    counts = {status: sum(1 for m in run.entries if m["status"] == status) for status in fmt.STATUSES}
    print(json.dumps({"ok": True, "outcome": "measured", "json": str(json_path), "markdown": str(md_path),
                      "gpu_seconds": sheet["gpu_seconds"], "counts": counts}))
    return 0


def self_check_sheet(fixtures: Path = FIXTURES, power_fixture: Path = POWER_FIXTURE) -> dict[str, Any]:
    """A sheet built from the recorded tool output in fixtures/lab/chipsheet/ and invented timings: every parser and
    the writer run, nothing is measured."""
    rocminfo_text = (fixtures / "rocminfo_gfx1151_synthetic.txt").read_text(encoding="utf-8")
    agent = fmt.parse_rocminfo(rocminfo_text)
    if gate("pod-b", rocminfo_text, None, machine="x86_64") is not None:
        raise ValueError("the recorded rocminfo does not pass the node gate")
    hipcc = fmt.parse_hipcc_version((fixtures / "hipcc_version_synthetic.txt").read_text(encoding="utf-8"))
    rocm = fmt.parse_rocm_version((fixtures / "amdsmi_version_synthetic.txt").read_text(encoding="utf-8"))
    probes = fmt.parse_probe_lines((fixtures / "probe_w32_synthetic.txt").read_text(encoding="utf-8"))
    remarks = fmt.parse_resource_remarks((fixtures / "remarks_w32_synthetic.txt").read_text(encoding="utf-8"))
    power = pod_power.parse_amdsmi(power_fixture.read_text(encoding="utf-8"))
    hip = next(line for line in probes if line["probe"] == "props")
    lds = next(line for line in probes if line["probe"] == "lds")
    wave = next(line for line in probes if line["probe"] == "wave")
    occ = [line for line in probes if line["probe"] in ("occ_lds", "occ_vgpr")]
    idle = power[0].socket_w
    watts = {"idle_w": idle, "loaded_mean_w": idle, "loaded_max_w": idle, "samples": 1}
    cmd = "python3 chipsheet.py self-check"
    items = [
        fmt.entry("copy.d2d.size_64mib", "copy", "device-to-device copy of 64mib (invented timing)", "GB/s",
                  source="timing", command=cmd, op="self-check: no GPU", params={"bytes": 64 * fmt.MIB},
                  samples=[100.0, 101.0, 99.0], derived={"copy_gbs": 50.0}, watts=watts),
        fmt.entry("matmul.fp16.decode27b.k5120_n6144.m1", "matmul", "fp16 decode GEMM (invented timing)", "TFLOP/s",
                  source="timing", command=cmd, op="self-check: no GPU", params={"m": 1, "k": 5120, "n": 6144},
                  samples=[0.2, 0.21], watts=watts),
        fmt.entry("dequant.w4a16.decode27b.k5120_n6144.m1.bn64_bk64", "dequant", "W4A16 (not run)", "TFLOP/s",
                  source="timing", command=cmd, op="self-check: no GPU", status="skipped", error="self-check: no GPU"),
        fmt.entry("lds.rocminfo.group_segment_bytes", "lds", "LDS rocminfo reports", "bytes", source="tool",
                  command="rocminfo", op="Pool Info GROUP", samples=[agent["lds_bytes"]], value=agent["lds_bytes"]),
        fmt.entry("lds.max_launch_bytes", "lds", "largest dynamic LDS launched", "bytes", source="probe",
                  command="probe lds", op="lds_probe", samples=[lds["max_ok_bytes"]], value=lds["max_ok_bytes"],
                  params={k: v for k, v in lds.items() if k != "probe"}),
        fmt.entry("wave.w32.compiled", "wave", "wavefront size compiled", "lanes", source="probe",
                  command="probe wave", op="wave_probe", samples=[wave["compiled_wave"]], value=wave["compiled_wave"]),
        fmt.entry("cache.rocminfo.l2_bytes", "cache", "L2 rocminfo reports", "bytes", source="tool",
                  command="rocminfo", op="Cache Info L2", samples=[agent["cache_bytes"]["l2"]],
                  value=agent["cache_bytes"]["l2"]),
        fmt.entry("cache.hip.l2_bytes", "cache", "L2 HIP reports", "bytes", source="probe", command="probe props",
                  op="l2CacheSize", samples=[hip["l2CacheSize"]], value=hip["l2CacheSize"]),
    ]
    for found in occ:
        samples = _loads_per_s(found)
        usage = remarks.get(found["kernel"], {})
        items.append(fmt.entry(f"occupancy.{'lds' if found['probe'] == 'occ_lds' else 'vgpr'}.p_{found['param']}",
                               "occupancy", "chase loads per second (recorded probe output)", "Gload/s",
                               source="probe", command="probe", op=found["kernel"], samples=samples,
                               derived={"vgprs": usage.get("VGPRs"),
                                        "remark_waves_per_simd": usage.get("Occupancy [waves/SIMD]")}))
    return {
        "format": fmt.FORMAT, "gfx": agent["gfx"], "node": "pod-b", "mode": "self-check",
        "date": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"), "started": utc_now(), "ended": utc_now(),
        "gpu_seconds": 0.0, "budget_s": QUICK_BUDGET_S, "command": cmd, "host": socket.gethostname(),
        "versions": {**dict.fromkeys(fmt.VERSION_KEYS), "rocm": rocm, "hip": hipcc["hip"], "hipcc": hipcc["clang"],
                     "python": platform.python_version()},
        "device": {"rocminfo": agent, "hip": hip}, "power": {"reader": {"tool": "fixture"}, "idle": {}},
        "measurements": items, "notes": ["self-check: parsed fixtures, invented timings"],
    }


def self_check_main(args: argparse.Namespace) -> int:
    presence = gpu_presence()
    out = Path(args.out).expanduser() if args.out else Path(tempfile.mkdtemp(prefix="chipsheet-self-check-"))
    try:
        sheet = self_check_sheet(Path(args.fixtures))
        json_path, md_path = fmt.write(sheet, out)
        back = fmt.read(json_path)
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        print(json.dumps({"ok": False, "outcome": "self_check_failed", "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps({"ok": True, "outcome": "self_checked", "gpu": presence["gpu"],
                      "note": "no GPU here: parsed the recorded output and wrote the files, measured nothing"
                      if not presence["gpu"] else "a GPU is present; self-check still measured nothing",
                      "json": str(json_path), "markdown": str(md_path), "entries": len(back["measurements"])}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chipsheet.py", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="measure this pod's chip (node gate first)")
    run.add_argument("--node", required=True, help="the ledger name of this pod: pod-a or pod-b")
    run.add_argument("--quick", action="store_true", help=f"fewer points, budget {QUICK_BUDGET_S:g} s")
    run.add_argument("--out", default=DEFAULT_OUT, help=f"where the sheet goes (default {DEFAULT_OUT})")
    run.add_argument("--only", help=f"comma list of sections: {', '.join(ALL_SECTIONS)} (probes = lds, wave, "
                                    "occupancy and cache)")
    run.add_argument("--budget-s", type=float, help="time budget in seconds (default 840, --quick 110)")
    run.add_argument("--date", help="the date in the file name (default today, UTC)")
    run.add_argument("--lora-config", default=LORA_CONFIG, help=f"the 4B checkpoint's config.json ({LORA_CONFIG})")
    check = sub.add_parser("self-check", help="no GPU: parse recorded output, write and re-read a sheet")
    check.add_argument("--out", help="where to write it (default a new temporary folder)")
    check.add_argument("--fixtures", default=str(FIXTURES), help="the recorded tool output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.command == "self-check":
        return self_check_main(args)
    command = shlex.join(["python3", "chipsheet.py", *argv])
    return run_main(args, command)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
