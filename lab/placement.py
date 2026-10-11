"""placement: what runs where on the two pods, from the pods' facts, a chip sheet per pod and the GPU Queue.

    python -m lab placement                                  # the plan, as text
    python -m lab placement --json                           # the same plan as one JSON object
    python -m lab placement --check "L3: smoke LoRA" --gpu A  # does a slot request contradict the plan? (exit 1: yes)
    python -m lab placement --check "R20: baseline matrix" --gpu B --kind bench --json

The plan answers five questions with a reason per line: which pod serves the teacher, and where the power sampler,
the benchmark runs, the Ax trials and the LoRA jobs go; then which jobs may share a chip and under what memory
margin. Every number in it carries the file it came from (in the text, a [n] that the Numbers list at the end
resolves; in the JSON, the `numbers` list). The rules and why they hold are the research note
lab/refine/research/plumbing/placement.md.

Inputs, all read, none written:

- The pods' facts: a folder (--facts, else $LAB_PLACEMENT_FACTS, else <state dir>/placement when it holds a
  pods.json, else the recorded fixtures/lab/placement) with a pods.json that names, per pod role (pod-a, pod-b),
  the files the existing helpers produce: the answer of `python -m lab.controller status --probe-port 8000`
  (status.json: is the pod up, which accelerator, which model answers through the proxy), the output of
  `bash lab/pod_a_teacher.sh facts` on the pod (<node>.facts.txt: GPU, memory visible to torch, memory available,
  listeners, idle watts) and, where the pod's sampler ran, its telemetry.gpu.power sample file (lab/pod_power.py:
  the watts now are the mean of its last minute). A live snapshot is the same files written today.
- One chip sheet per pod: the JSON lab/chipsheet writes (R33) under lab/refine/research/kernels/data/, the newest
  chipsheet-<gfx>-<date>.json for the pod's chip, else chipsheet-sample.json, which documents the fields read here
  (memory.free_gib, memory.visible_gib, bandwidth.copy_d2d_peak_gb_s, matmul.bf16_peak_tflops, power.idle_w,
  power.loaded_w). A value is a number or {"value": number, ...}; every other field is ignored. --chipsheet
  pod-b=FILE names one.
- The GPU Queue rows, through lab/machine/slots.py and its Notion client (NOTION_TOKEN), or --queue-file with the
  pages as the Notion API returns them, or none with --no-queue. Running rows say what holds a card now; Queued rows
  are checked against the plan.
- The measured numbers the rules use, from lab/refine/metrics.jsonl (by metric and node, with the line), and two
  settings of lab/pod_a_teacher.sh (the teacher's --gpu-memory-utilization default, the checkpoint size).

--check takes a slot request's text (its Request "<task>: <what>", or the --what alone) and the card it asks for
(--gpu A or B), works out the kind of job from its words (or --kind) and says whether the plan puts that kind on
that card. `python -m lab slot request` asks the same question before it adds a row and refuses a contradicting
request unless --override says why.

Every call answers on stdout (text, or one JSON object with --json) and exits 0, or 1 when --check finds a
contradiction or an input cannot be read, or 2 for bad arguments. Nothing here starts, stops or calls a pod or the
hub; the queue is only read.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import os
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from lab import pod_power
from lab.controller import parse_time

__all__ = [
    "KINDS", "MEASUREMENTS", "NODES", "GPU_OF", "Num", "ChipSheet", "Pod", "Job", "Plan", "Rules",
    "classify", "read_chipsheet", "read_facts_text", "load_pods", "build_plan", "check", "default_plan",
    "check_request", "render", "main", "run",
]

CLI_VERB = "placement"   # registers python -m lab placement
REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_FACTS = REPO_ROOT / "fixtures" / "lab" / "placement"
CHIPSHEET_DIR = REPO_ROOT / "lab" / "refine" / "research" / "kernels" / "data"
SAMPLE_CHIPSHEET = CHIPSHEET_DIR / "chipsheet-sample.json"
METRICS_FILE = REPO_ROOT / "lab" / "refine" / "metrics.jsonl"
INTERFERENCE_PREFIX = "interference-gfx1151"
TEACHER_SCRIPT = REPO_ROOT / "lab" / "pod_a_teacher.sh"
README_FILE = REPO_ROOT / "lab" / "README.md"
NOTE = "lab/refine/research/plumbing/placement.md"
ENV_FACTS = "LAB_PLACEMENT_FACTS"
MANIFEST = "pods.json"

#: The kinds of job the plan places. A sampler is lab/pod_power.py; bench is a benchmark run (lab/bench.py, a
#: baseline matrix, a profile, the chip sheet); ax is one or more Ax trials (lab/optimize_settings.py); lora is a
#: LoRA training or adapter job (the L3 trainer, lab/smoke_lora.py).
KINDS = ("teacher", "sampler", "bench", "ax", "lora")
MEASUREMENTS = ("bench", "ax")
NODES = ("pod-a", "pod-b")
#: The queue's card letter of each pod role (lab/machine/slots.py NODES, the other way round).
GPU_OF = {"pod-a": "A", "pod-b": "B"}
NODE_OF = {letter: node for node, letter in GPU_OF.items()}
#: The sampler's own names for the pods (lab/pod_power.py LEDGER_NODE, the other way round).
SAMPLER_NODE = {ledger: sampler for sampler, ledger in pod_power.LEDGER_NODE.items()}
TEACHER_MODEL = "teacher"
TEACHER_PORT = 8000
SAMPLER_PORT = pod_power.DEFAULT_PORT
POWER_WINDOW_S = 60.0

#: How the words of a request name its kind, tried in this order: an Ax trial is also a benchmark, a benchmark of
#: the teacher still a benchmark, a sampler run under the teacher's load still the sampler.
KIND_WORDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ax", re.compile(r"\bax\b|\bbotorch\b|\boptimi[sz]\w*|\btrials?\b", re.IGNORECASE)),
    ("bench", re.compile(r"\bbench\w*|\bbaselines?\b|\bsweeps?\b|\bprofil\w*|\bchip[ -]?sheet\b|\broofline\b|"
                         r"\bmatrix\b|\bmeasur\w*", re.IGNORECASE)),
    ("lora", re.compile(r"\blora\b|\btrain\w*|\bfine-?tun\w*|\bfinetun\w*|\bsft\b|\badapters?\b|\bdistill\w*",
                        re.IGNORECASE)),
    ("sampler", re.compile(r"\bsampler\b|\bpod_power\b", re.IGNORECASE)),
    ("teacher", re.compile(r"\bteacher\b|\bvllm\b|\bserv(?:e|er|es|ing)\b", re.IGNORECASE)),
)


def classify(text: str) -> str | None:
    """The kind of job a request's words name (see KIND_WORDS), or None when they name none."""
    for kind, pattern in KIND_WORDS:
        if pattern.search(text or ""):
            return kind
    return None


# --- numbers and where they came from ----------------------------------------------------------------------------

def relative(path: Path | str) -> str:
    """A path as the plan names it: repo-relative with forward slashes when it is inside the repo."""
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


@dataclasses.dataclass(frozen=True)
class Num:
    """A number the plan uses and its trace. kind is recorded (read from a file of facts), measured (a measurement
    row), sample (from the sample chip sheet, not a measurement), derived (worked out from other numbers, the
    formula in source), rule (a setting of this module, its reason in the research note) or unknown."""

    value: float | None
    unit: str
    source: str
    kind: str = "recorded"

    @property
    def known(self) -> bool:
        return self.value is not None and math.isfinite(self.value)

    def text(self) -> str:
        if not self.known:
            return "unknown"
        value = round(float(self.value), 2)
        shown = f"{int(value)}" if value == int(value) else f"{value:g}"
        return f"{shown} {self.unit}".strip()

    def view(self) -> dict[str, Any]:
        return {"value": self.value, "unit": self.unit, "source": self.source, "kind": self.kind}


def unknown(unit: str, why: str) -> Num:
    return Num(None, unit, why, "unknown")


def derived(value: float | None, unit: str, formula: str) -> Num:
    return Num(None if value is None else round(value, 3), unit, formula, "derived")


class Trace:
    """Numbers as a plan's text cites them: each distinct number gets an id the Numbers list resolves."""

    def __init__(self) -> None:
        self.numbers: list[Num] = []

    def say(self, num: Num) -> str:
        """'58 GiB [3]': the number with the id of its trace."""
        if num not in self.numbers:
            self.numbers.append(num)
        return f"{num.text()} [{self.numbers.index(num) + 1}]"

    def views(self) -> list[dict[str, Any]]:
        return [{"id": n, **num.view()} for n, num in enumerate(self.numbers, 1)]


# --- measured numbers from the metrics file and the teacher's script --------------------------------------------

def metric(name: str, *, node: str | None = None, where: Mapping[str, Any] | None = None,
           path: Path = METRICS_FILE) -> Num:
    """The first row of lab/refine/metrics.jsonl with that metric, node and conditions (where), as a Num whose
    source is the file and line; unknown when no row matches or the file cannot be read."""
    conditions = dict(where or {})
    unit = ""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return unknown(unit, f"{relative(path)} cannot be read: {exc}")
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("metric") != name or (node is not None and row.get("node") != node):
            continue
        held = row.get("conditions") if isinstance(row.get("conditions"), Mapping) else {}
        if any(held.get(key) != value for key, value in conditions.items()):
            continue
        value = row.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return Num(float(value), str(row.get("unit") or ""), f"{relative(path)}:{number} ({name}, {row.get('node')})",
                       "measured")
    return unknown(unit, f"no {name} row{' on ' + node if node else ''} in {relative(path)}")


def script_setting(name: str, path: Path = TEACHER_SCRIPT) -> Num:
    """A NAME="${NAME:-value}" default of lab/pod_a_teacher.sh as a Num with the file and line."""
    pattern = re.compile(rf'^{re.escape(name)}="\$\{{{re.escape(name)}:-([0-9.]+)\}}"')
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return unknown("", f"{relative(path)} cannot be read: {exc}")
    for number, line in enumerate(lines, 1):
        match = pattern.match(line.strip())
        if match:
            return Num(float(match.group(1)), "", f"{relative(path)}:{number} ({name} default, the teacher's "
                       "--gpu-memory-utilization)", "recorded")
    return unknown("", f"no {name} default in {relative(path)}")


def text_number(pattern: str, unit: str, what: str, path: Path, kind: str = "recorded") -> Num:
    """The number the first line of a text file matching pattern holds in its first group, traced to that line;
    unknown when no line matches. Finding the line, rather than naming it, keeps the trace right when the file
    grows above it."""
    regex = re.compile(pattern)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return unknown(unit, f"{relative(path)} cannot be read: {exc}")
    for number, line in enumerate(lines, 1):
        match = regex.search(line)
        if match:
            return Num(float(match.group(1)), unit, f"{relative(path)}:{number} ({what})", kind)
    return unknown(unit, f"no line in {relative(path)} gives {what}")


# --- the measured interference -----------------------------------------------------------------------------------

def newest_interference(folder: Path = CHIPSHEET_DIR) -> Path | None:
    """The newest interference-gfx1151-<date>.json that lab/interference.py collect wrote, or None."""
    found = sorted(p for p in Path(folder).glob(f"{INTERFERENCE_PREFIX}-*.json")
                   if re.fullmatch(rf"{INTERFERENCE_PREFIX}-\d{{4}}-\d{{2}}-\d{{2}}\.json", p.name))
    return found[-1] if found else None


def read_interference(path: Path | str) -> dict[str, Num] | None:
    """The numbers of an interference result the sharing line uses, each traced to its field in the file; None when
    the file is not an interference result or its comparison is missing."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, Mapping) or data.get("schema") != "lab.interference/1":
        return None
    arms, compare = data.get("arms") or {}, data.get("compare") or {}
    both = arms.get("both") or {}
    fields = {
        "share": (((data.get("serve") or {}).get("share")), "", "serve.share"),
        "decode_alone": ((arms.get("decode") or {}).get("decode_tok_s"), "tok/s", "arms.decode.decode_tok_s"),
        "decode_shared": ((both.get("decode") or {}).get("decode_tok_s"), "tok/s", "arms.both.decode.decode_tok_s"),
        "decode_change": (compare.get("decode_tok_s_change_pct"), "%", "compare.decode_tok_s_change_pct"),
        "train_alone": ((arms.get("train") or {}).get("samples_per_s"), "samples/s", "arms.train.samples_per_s"),
        "train_shared": ((both.get("train") or {}).get("samples_per_s"), "samples/s",
                         "arms.both.train.samples_per_s"),
        "train_change": (compare.get("samples_per_s_change_pct"), "%", "compare.samples_per_s_change_pct"),
        "time_change": (compare.get("time_change_pct"), "%", "compare.time_change_pct"),
        "energy_change": (compare.get("energy_change_pct"), "%", "compare.energy_change_pct"),
    }
    out: dict[str, Num] = {}
    for key, (value, unit, field) in fields.items():
        number = _number(value)
        out[key] = (Num(number, unit, f"{relative(path)}: {field}", "measured") if number is not None
                    else unknown(unit, f"{relative(path)} has no {field}"))
    if not (out["time_change"].known and out["energy_change"].known):
        return None
    return out


def interference_sentence(m: Mapping[str, Num], say: Callable[[Num], str], path: Path | str) -> str:
    """What the measured run says about a teacher and a LoRA job on one chip, in one sentence."""
    shares = m["time_change"].value < 0 and m["energy_change"].value <= 0
    return (f"Measured on pod D with the teacher at a share of {say(m['share'])} ({relative(path)}): beside the LoRA "
            f"the teacher's decode went from {say(m['decode_alone'])} to {say(m['decode_shared'])} "
            f"({say(m['decode_change'])}) and the LoRA's samples per second from {say(m['train_alone'])} to "
            f"{say(m['train_shared'])} ({say(m['train_change'])}); doing the work together changed its time by "
            f"{say(m['time_change'])} and its energy by {say(m['energy_change'])} against one after the other, so "
            "sharing "
            + ("pays." if shares else "does not pay: run them one after the other."))


# --- the rules ---------------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Rules:
    """The settings the plan applies. margin and host_reserve_gib are judgement, not measurement: the research note
    says why these values, and R43's interference run replaces them with measured ones."""

    margin: float = 0.5               # on a measured peak: max_memory_allocated leaves out the allocator's cache
    host_reserve_gib: float = 4.0     # the jobs' own host memory, which the unified pool also serves
    teacher_share: float | None = None    # None: the default of lab/pod_a_teacher.sh

    def margin_num(self) -> Num:
        return Num(self.margin * 100, "%", f"rule margin ({NOTE}#memory-margin)", "rule")

    def reserve_num(self) -> Num:
        return Num(self.host_reserve_gib, "GiB", f"rule host_reserve_gib ({NOTE}#memory-margin)", "rule")


# --- chip sheets -------------------------------------------------------------------------------------------------

#: Where each field the plan reads may sit in a chip sheet, tried in order.
CHIP_FIELDS: dict[str, tuple[tuple[str, str], ...]] = {
    "visible_gib": (("memory", "visible_gib"), ("memory", "total_gib")),
    "free_gib": (("memory", "free_gib"), ("memory", "available_gib")),
    "bandwidth_gb_s": (("bandwidth", "copy_d2d_peak_gb_s"), ("bandwidth", "peak_gb_s")),
    "matmul_tflops": (("matmul", "bf16_peak_tflops"), ("matmul", "peak_tflops")),
    "idle_w": (("power", "idle_w"),),
    "loaded_w": (("power", "loaded_w"),),
}
CHIP_UNITS = {"visible_gib": "GiB", "free_gib": "GiB", "bandwidth_gb_s": "GB/s", "matmul_tflops": "TFLOPS",
              "idle_w": "W", "loaded_w": "W"}


@dataclasses.dataclass(frozen=True)
class ChipSheet:
    """The fields of one chip sheet the plan reads, each a Num traced to the file and key."""

    path: str
    sample: bool
    gfx: str | None
    fields: Mapping[str, Num]
    problem: str | None = None

    def get(self, name: str) -> Num:
        return self.fields.get(name) or unknown(CHIP_UNITS.get(name, ""), f"{self.path}: no {name}")


def _number(value: Any) -> float | None:
    if isinstance(value, Mapping):
        value = value.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _best_of(items: Any, key: str, *, where: Callable[[Mapping[str, Any]], bool] = lambda item: True) -> float | None:
    """The largest number under key among the mappings of a list (a sheet's per-size or per-shape rows)."""
    if not isinstance(items, list):
        return None
    found = [_number(item.get(key)) for item in items if isinstance(item, Mapping) and where(item)]
    found = [value for value in found if value is not None]
    return max(found) if found else None


def read_chipsheet(path: Path | str) -> ChipSheet:
    """The fields the plan needs from a chip sheet. Tolerant: extra fields are ignored, a missing or null field is
    unknown, and a file that cannot be read is a sheet with every field unknown and the problem named."""
    path = Path(path)
    name = relative(path)
    try:
        sheet = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return ChipSheet(name, False, None, {}, f"cannot read {name}: {type(exc).__name__}: {exc}")
    if not isinstance(sheet, Mapping):
        return ChipSheet(name, False, None, {}, f"{name} is not a JSON object")
    sample = sheet.get("sample") is True
    kind = "sample" if sample else "measured"
    chip = sheet.get("chip") if isinstance(sheet.get("chip"), Mapping) else {}
    fields: dict[str, Num] = {}
    for field, paths in CHIP_FIELDS.items():
        for section, key in paths:
            part = sheet.get(section)
            value = _number(part.get(key)) if isinstance(part, Mapping) else None
            if value is not None:
                fields[field] = Num(value, CHIP_UNITS[field], f"{name}: {section}.{key}", kind)
                break
    if "bandwidth_gb_s" not in fields:
        bandwidth = sheet.get("bandwidth") if isinstance(sheet.get("bandwidth"), Mapping) else {}
        best = _best_of(bandwidth.get("copy_d2d"), "gb_s")
        if best is not None:
            fields["bandwidth_gb_s"] = Num(best, "GB/s", f"{name}: the largest bandwidth.copy_d2d[].gb_s", kind)
    if "matmul_tflops" not in fields:
        matmul = sheet.get("matmul") if isinstance(sheet.get("matmul"), Mapping) else {}
        best = _best_of(matmul.get("shapes"), "tflops", where=lambda item: str(item.get("dtype", "")).lower() == "bf16")
        if best is not None:
            fields["matmul_tflops"] = Num(best, "TFLOPS", f"{name}: the largest bf16 matmul.shapes[].tflops", kind)
    gfx = chip.get("gfx") if isinstance(chip.get("gfx"), str) else None
    return ChipSheet(name, sample, gfx, fields)


def default_chipsheet(gfx: str | None, folder: Path = CHIPSHEET_DIR) -> Path:
    """The newest chipsheet-<gfx>-<date>.json in the folder for that chip, else the sample."""
    if gfx:
        found = sorted(folder.glob(f"chipsheet-{gfx}-*.json"))
        if found:
            return found[-1]
    return folder / SAMPLE_CHIPSHEET.name


# --- the pods' facts ---------------------------------------------------------------------------------------------

def read_facts_text(text: str) -> dict[str, str]:
    """The `key: value` lines of `bash lab/pod_a_teacher.sh facts` output; lines starting with # are notes."""
    facts: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if sep and key.strip() and key.strip() not in facts:
            facts[key.strip()] = value.strip()
    return facts


_MEM = re.compile(r"([0-9.]+)\s*GB total,\s*([0-9.]+)\s*GB available")
_TORCH_MEM = re.compile(r"mem_gib=([0-9.]+)")


@dataclasses.dataclass
class Pod:
    """What the plan knows about one pod role. Every number is a Num; up is None when no status says."""

    node: str
    account: str | None = None
    up: bool | None = None
    up_source: str = ""
    started: str | None = None
    accelerator: str | None = None
    gfx: str | None = None
    gpu: str | None = None
    serving: tuple[str, ...] = ()
    serving_source: str = ""
    visible: Num = dataclasses.field(default_factory=lambda: unknown("GiB", "no facts"))
    available: Num = dataclasses.field(default_factory=lambda: unknown("GiB", "no facts"))
    facts_time: dt.datetime | None = None
    facts_source: str = ""
    facts_server_running: bool | None = None
    listening: frozenset[int] = frozenset()
    free_ports: frozenset[int] = frozenset()
    home: tuple[str, ...] = ()
    watts: Num = dataclasses.field(default_factory=lambda: unknown("W", "no power reading"))
    sampler_seen: str | None = None
    chip: ChipSheet | None = None

    @property
    def teacher_serving(self) -> bool:
        return TEACHER_MODEL in self.serving

    @property
    def holds_teacher_build(self) -> bool:
        return {"vllm", "models"} <= set(self.home)


def _facts_into(pod: Pod, path: Path) -> None:
    name = relative(path)
    facts = read_facts_text(path.read_text(encoding="utf-8"))
    pod.facts_source = name
    pod.facts_time = parse_time(facts.get("date_utc"))
    pod.gfx = facts.get("gfx") or pod.gfx
    pod.gpu = facts.get("gpu") or pod.gpu
    mem = _MEM.search(facts.get("mem", ""))
    if mem:
        pod.available = Num(float(mem.group(2)), "GiB", f"{name}: mem (MemAvailable when the facts were read)")
    torch_mem = _TORCH_MEM.search(facts.get("torch_gpu", ""))
    if torch_mem:
        pod.visible = Num(float(torch_mem.group(1)), "GiB", f"{name}: torch_gpu mem_gib (visible to torch)")
    words = facts.get("listeners", "").split()
    pod.listening = frozenset(int(word) for word in words if word.isdigit())
    pod.free_ports = frozenset(int(word[3:]) for word in words if word.startswith("no-") and word[3:].isdigit())
    pod.home = tuple(facts.get("home_holds", "").split())
    running = facts.get("server_running", "").lower()
    pod.facts_server_running = {"yes": True, "no": False}.get(running)
    reads = [value for value in (pod_power.number(word) for word in facts.get("idle_socket_power_w", "").split())
             if value is not None]
    if reads:
        pod.watts = Num(reads[-1], "W", f"{name}: idle_socket_power_w (the last of {len(reads)} amd-smi reads)")


def _status_into(pods: Mapping[str, Pod], path: Path) -> None:
    name = relative(path)
    answer = json.loads(path.read_text(encoding="utf-8"))
    accounts = answer.get("accounts") if isinstance(answer, Mapping) else None
    if not isinstance(accounts, Mapping):
        raise ValueError(f"{name} is not the answer of python -m lab.controller status (no accounts)")
    for letter, entry in accounts.items():
        if not isinstance(entry, Mapping) or entry.get("node") not in pods:
            continue
        pod = pods[entry["node"]]
        pod.account = pod.account or letter
        if entry.get("reachable") is False:
            pod.up, pod.up_source = None, f"{name}: account {letter} unreachable ({entry.get('error')})"
            continue
        pod.up = bool(entry.get("running")) and bool(entry.get("ready")) and not entry.get("pending")
        pod.up_source = f"{name}: account {letter}"
        pod.started = entry.get("started")
        pod.accelerator = entry.get("accelerator")
        serving = entry.get("serving") if isinstance(entry.get("serving"), Mapping) else None
        if serving is not None:
            healthy = serving.get("health") == 200
            models = [m for m in serving.get("models") or [] if isinstance(m, str)]
            pod.serving = tuple(models) if healthy else ()
            pod.serving_source = (f"{name}: account {letter} serving on port {serving.get('port')}, "
                                  f"health {serving.get('health')}")


def _power_into(pod: Pod, path: Path) -> None:
    name = relative(path)
    events = pod_power.read_samples([path])
    node = SAMPLER_NODE.get(pod.node, pod.node)
    mine = [e for e in events if e.get("type") == pod_power.GPU_POWER and e.get("node") == node]
    if not mine:
        return
    end = max(pod_power.parse_ts(str(e["ts"])) for e in mine)
    start = end - dt.timedelta(seconds=POWER_WINDOW_S)
    window = pod_power.summarize(mine, node=node, start=start, end=end)
    if window.get("mean_w") is not None:
        pod.watts = Num(float(window["mean_w"]), "W", f"{name}: mean socket_w of the last {POWER_WINDOW_S:g} s of "
                        f"samples ({window['start']} to {window['end']}, {window['samples']} samples)")
    pod.sampler_seen = f"{name}: samples from {node} up to {window['end']}"


def facts_folder(given: str | os.PathLike[str] | None = None, environ: Mapping[str, str] | None = None) -> Path:
    """Where the pods' facts are: given, else $LAB_PLACEMENT_FACTS, else <state dir>/placement when it holds a
    pods.json, else the recorded fixtures."""
    environ = os.environ if environ is None else environ
    if given:
        return Path(given)
    if (environ.get(ENV_FACTS) or "").strip():
        return Path(environ[ENV_FACTS].strip())
    from lab.machine import ledger   # the state dir as every lab verb resolves it
    live = ledger.resolve_state_dir() / "placement"
    return live if (live / MANIFEST).is_file() else FIXTURE_FACTS


def load_pods(folder: Path, chipsheets: Mapping[str, str | os.PathLike[str]] | None = None) -> tuple[
        dict[str, Pod], dict[str, Any]]:
    """The pods described by a facts folder's pods.json, each with its chip sheet; and the manifest itself."""
    manifest_path = folder / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("pods") if isinstance(manifest, Mapping) else None
    if not isinstance(entries, Mapping):
        raise ValueError(f"{relative(manifest_path)} names no pods")
    pods = {node: Pod(node) for node in NODES}
    for node, entry in entries.items():
        if node not in pods or not isinstance(entry, Mapping):
            continue
        pods[node].account = entry.get("account")
        if entry.get("facts"):
            _facts_into(pods[node], folder / entry["facts"])
    if manifest.get("status"):
        _status_into(pods, folder / manifest["status"])
    for node, entry in entries.items():
        if node in pods and isinstance(entry, Mapping) and entry.get("power"):
            _power_into(pods[node], folder / entry["power"])
    chipsheets = dict(chipsheets or {})
    for node, pod in pods.items():
        pod.chip = read_chipsheet(chipsheets.get(node) or default_chipsheet(pod.gfx))
    return pods, manifest


# --- jobs --------------------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Job:
    """A job that holds a pod now: what it is, what memory it holds and how the plan knows."""

    kind: str
    node: str
    label: str
    memory: Num
    since_facts: bool     # started after the pod's facts were read, so the facts' free memory still counts it free


def queue_rows(slots_list: Iterable[Any]) -> list[Any]:
    return [row for row in slots_list if getattr(row, "status", None) in ("Running", "Queued")]


# --- the plan ----------------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Line:
    job: str
    pods: list[str]
    decision: str
    reason: str

    def view(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Share:
    jobs: tuple[str, str]
    allowed: bool
    on_paper: bool
    reason: str
    margin: Num | None = None
    measured: dict[str, Any] | None = None

    def view(self) -> dict[str, Any]:
        out = {"jobs": list(self.jobs), "allowed": self.allowed, "on_paper": self.on_paper, "reason": self.reason}
        if self.margin is not None:
            out["margin"] = self.margin.view()
        if self.measured is not None:
            out["measured"] = self.measured
        return out


@dataclasses.dataclass
class Plan:
    """A placement plan: the pods as read, one line per kind of job, the sharing rules, the queue rows checked
    against it, the levers that would free capacity, and the trace of every number."""

    as_of: str
    facts: dict[str, Any]
    pods: dict[str, Pod]
    lines: list[Line]
    sharing: list[Share]
    where: dict[str, list[str]]
    queue: dict[str, Any]
    levers: list[str]
    notes: list[str]
    trace: Trace
    rules: Rules

    def line(self, kind: str) -> Line | None:
        return next((line for line in self.lines if line.job == kind), None)

    def view(self) -> dict[str, Any]:
        pods = {}
        for node, pod in self.pods.items():
            pods[node] = {
                "account": pod.account, "up": pod.up, "up_source": pod.up_source, "started": pod.started,
                "accelerator": pod.accelerator, "gfx": pod.gfx, "serving": list(pod.serving),
                "visible": pod.visible.view(), "available": pod.available.view(), "watts": pod.watts.view(),
                "facts": pod.facts_source, "facts_time": None if pod.facts_time is None else _iso(pod.facts_time),
                "sampler": pod.sampler_seen, "home_holds_teacher_build": pod.holds_teacher_build,
                "chipsheet": None if pod.chip is None else {"path": pod.chip.path, "sample": pod.chip.sample,
                                                            "problem": pod.chip.problem},
            }
        return {"ok": True, "outcome": "plan", "as_of": self.as_of, "facts": self.facts, "pods": pods,
                "lines": [line.view() for line in self.lines], "sharing": [share.view() for share in self.sharing],
                "where": self.where, "queue": self.queue, "levers": self.levers, "notes": self.notes,
                "rules": {"margin": self.rules.margin, "host_reserve_gib": self.rules.host_reserve_gib,
                          "note": NOTE},
                "numbers": self.trace.views()}


def _iso(moment: dt.datetime) -> str:
    return moment.astimezone(dt.timezone.utc).replace(microsecond=0, tzinfo=None).isoformat() + "Z"


def _min_num(a: Num, b: Num) -> Num:
    """The smaller of two known numbers, traced to both; the known one when only one is."""
    if a.known and b.known:
        low = a if a.value <= b.value else b
        return Num(low.value, low.unit, f"the smaller of ({a.source}) and ({b.source})", "derived")
    return a if a.known else b


def build_plan(pods: Mapping[str, Pod], *, rows: Sequence[Any] | None = None, rules: Rules | None = None,
               manifest: Mapping[str, Any] | None = None, folder: Path | None = None,
               now: dt.datetime | None = None, metrics_path: Path = METRICS_FILE,
               script_path: Path = TEACHER_SCRIPT, readme_path: Path = README_FILE,
               interference: Path | str | None = None, interference_dir: Path = CHIPSHEET_DIR) -> Plan:
    """The plan for these pods. rows are the GPU Queue's rows (lab/machine/slots.Slot); None when the queue was
    not read. interference is the measured interference file (lab/interference.py), else the newest in
    interference_dir, else none. Pure apart from reading the metrics file, the teacher's script and that file."""
    rules = rules or Rules()
    now = now or dt.datetime.now(dt.timezone.utc)
    t = Trace()
    notes: list[str] = []
    levers: list[str] = []
    up = [node for node in NODES if pods[node].up]

    # what the rules need, each traced
    share = (Num(rules.teacher_share, "", "rule --teacher-share (given)", "rule") if rules.teacher_share is not None
             else script_setting("GPU_UTIL", script_path))
    lora_peak = metric("trainer.peak_mem_gb", path=metrics_path)
    margin = rules.margin_num()
    reserve = rules.reserve_num()
    lora_budget = (derived(lora_peak.value * (1 + rules.margin), "GiB",
                           f"LoRA peak x (1 + margin): {lora_peak.text()} x {1 + rules.margin:g}")
                   if lora_peak.known else unknown("GiB", "no LoRA peak memory measured"))
    # the checkpoint's size as the serve script states it (R29 loaded 17.98 GiB of weights), read as GiB to err high
    weights = text_number(r"W4A16, ([0-9.]+) GB\)", "GiB", "the teacher checkpoint's size", script_path)
    warm_start = metric("teacher.engine_init_s", node="pod-c", path=metrics_path)
    cold_start = text_number(r"engine ready ([0-9.]+) s after a cold start", "s",
                             "the teacher's engine start on pod D with no compile cache", readme_path, "measured")
    decode = metric("teacher.decode_tok_s", node="pod-c", where={"prompt": "text"}, path=metrics_path)
    decode_watts = metric("teacher.watts_avg", node="pod-c", where={"prompt": "text"}, path=metrics_path)
    # the T6 row: pod C (the teacher's role) idle with the teacher loaded, read by lab/pod_power.py; the other
    # pod-a row of that metric is the R9700's
    idle_loaded = metric("pod.idle_watts", node="pod-a", where={"sampler": "lab/pod_power.py 1"},
                         path=metrics_path)
    interference_path = Path(interference) if interference is not None else newest_interference(interference_dir)
    measured = read_interference(interference_path) if interference_path is not None else None

    def free_memory(pod: Pod) -> Num:
        base = pod.available
        if pod.chip is not None:
            base = _min_num(base, pod.chip.get("free_gib"))
        return base

    def visible(pod: Pod) -> Num:
        if pod.visible.known:
            return pod.visible
        return pod.chip.get("visible_gib") if pod.chip is not None else pod.visible

    def teacher_hold(pod: Pod) -> Num:
        vis = visible(pod)
        if not (share.known and vis.known):
            return unknown("GiB", "the teacher's share or the memory visible to torch is unknown")
        return derived(share.value * vis.value, "GiB", f"the teacher's share x memory visible to torch: "
                       f"{share.text()} x {vis.text()} (vLLM reserves this at start: vllm/v1/worker/utils.py "
                       "request_memory)")

    # jobs that hold each pod now
    by_card: dict[str, list[Any]] = {node: [] for node in NODES}
    for row in rows or ():
        node = next((n for n, letter in GPU_OF.items() if str(getattr(row, "gpu", "")).startswith(letter + " ")),
                    None)
        if node is not None:
            by_card[node].append(row)
    jobs: dict[str, list[Job]] = {node: [] for node in NODES}
    for node in NODES:
        pod = pods[node]
        if pod.teacher_serving:
            jobs[node].append(Job("teacher", node, f"teacher (vLLM on port {TEACHER_PORT})", teacher_hold(pod),
                                  pod.facts_server_running is not True))
        if pod.sampler_seen:
            jobs[node].append(Job("sampler", node, "power sampler (lab/pod_power.py)", Num(0, "GiB", "lab/pod_power.py "
                                  "(stdlib, CPU only: no GPU memory)", "recorded"), False))
        for row in by_card[node]:
            if row.status != "Running":
                continue
            kind = classify(row.request) or "other"
            started = getattr(row, "started", None)
            later = pod.facts_time is None or started is None or started > pod.facts_time
            jobs[node].append(Job(kind, node, f"queue row {row.request!r} (Running)",
                                  unknown("GiB", "a queue slot: it holds the card until the slot ends"), later))

    def free_now(pod: Pod) -> Num:
        """Free memory now: what the facts found free, less what the pod's services (the teacher, the sampler)
        started since they were read. A queue slot holds its card only until the slot ends, and the queue runs one
        at a time, so it makes the pod busy (see busy()) rather than smaller."""
        base = free_memory(pod)
        taken = [job for job in jobs[pod.node] if job.kind in ("teacher", "sampler") and job.since_facts
                 and job.memory.known and job.memory.value]
        if not base.known:
            return base
        if not taken:
            return base
        left = base.value - sum(job.memory.value for job in taken)
        parts = " - ".join([base.text()] + [f"{job.memory.text()} ({job.kind})" for job in taken])
        return derived(max(0.0, left), "GiB", f"free when the facts were read, less what started since: {parts}")

    def busy(node: str) -> str:
        """' It is busy now: <the Running queue rows>; the queue keeps the order.' or nothing."""
        held = [job.label for job in jobs[node] if job.kind not in ("teacher", "sampler")]
        return f" It is busy now: {', '.join(held)}; the queue keeps the order." if held else ""

    # 1. the teacher
    serving = [node for node in up if pods[node].teacher_serving]
    teacher_node: str | None
    if serving:
        teacher_node = "pod-a" if "pod-a" in serving else serving[0]
        pod = pods[teacher_node]
        hold = teacher_hold(pod)
        role = " and is the teacher's role" if teacher_node == "pod-a" else ""
        reason = (f"it serves the teacher now ({pod.serving_source}){role}; moving it costs an "
                  f"engine start of {t.say(warm_start)} warm or {t.say(cold_start)} cold, with no teacher for the box "
                  f"meanwhile. It holds {t.say(hold)} of the {t.say(free_memory(pod))} the pod had free.")
        if decode.known:
            reason += f" It decodes {t.say(decode)} at {t.say(decode_watts)}"
            bandwidth = pod.chip.get("bandwidth_gb_s") if pod.chip else unknown("GB/s", "no chip sheet")
            if bandwidth.known and weights.known:
                ceiling = derived(bandwidth.value / weights.value, "tok/s",
                                  f"bandwidth / weights read per token: {bandwidth.text()} / {weights.text()}")
                reason += f", against a bandwidth ceiling of {t.say(ceiling)}"
            reason += "."
        if len(serving) > 1:
            other = [node for node in serving if node != teacher_node][0]
            notes.append(f"{other} also answers as teacher ({pods[other].serving_source}): a benchmark copy or a "
                         "second server; the box's service is " + teacher_node + "'s.")
        decision = f"{teacher_node} (stays)"
    else:
        ready = [node for node in up if pods[node].holds_teacher_build] or up
        teacher_node = "pod-a" if "pod-a" in ready else (ready[0] if ready else None)
        if teacher_node is None:
            decision, reason = "nowhere", "no pod is up (status), so nothing can serve it"
        else:
            pod = pods[teacher_node]
            decision = f"{teacher_node} (start it)"
            reason = (f"no pod serves the teacher; {teacher_node} is up"
                      + (" and holds the build and the checkpoint (home_holds vllm, models)" if pod.holds_teacher_build
                         else "") + f"; an engine start takes {t.say(warm_start)} warm, {t.say(cold_start)} cold, "
                      f"and needs {t.say(teacher_hold(pod))} free at start.")
    teacher_line = Line("teacher", [teacher_node] if teacher_node else [], decision, reason)

    # the exclusive card: the up pod that does not serve the box's teacher
    exclusive = next((node for node in ("pod-b", "pod-a") if node in up and node != teacher_node), None)
    if teacher_node and teacher_node != "pod-a" and "pod-a" not in up:
        notes.append("pod-a is not up (status), so the teacher's role falls to " + teacher_node + ".")
    fallback = next((node for node in NODES if node != teacher_node and node in up and pods[node].holds_teacher_build),
                    None)
    if teacher_node and fallback:
        teacher_line.reason += (f" If {teacher_node} goes down, {fallback} holds the same build and checkpoint "
                                f"({pods[fallback].facts_source}: home_holds vllm, models) and starts it in about "
                                f"{t.say(cold_start)} cold.")

    # 2. the sampler
    sampler_pods = list(up)
    parts = []
    for node in up:
        pod = pods[node]
        if pod.sampler_seen:
            parts.append(f"{node} has one ({pod.sampler_seen})")
        elif SAMPLER_PORT in pod.listening:
            parts.append(f"{node} has one (port {SAMPLER_PORT} listens, {pod.facts_source})")
        else:
            parts.append(f"{node} has none recorded: start it (bash lab/pod_power.sh) before the first measured slot")
    sampler_line = Line("sampler", sampler_pods, ", ".join(sampler_pods) if sampler_pods else "nowhere",
                        "it holds no GPU memory (lab/pod_power.py: stdlib, CPU only, amd-smi once a second), so it "
                        "runs beside anything, and every measured job needs one on its pod: " + "; ".join(parts) + ".")

    # 3. and 4. benchmarks and Ax trials
    if exclusive:
        pod = pods[exclusive]
        busy_text = busy(exclusive)
        bench_reason = (f"a benchmark measures the chip's tok/s and watts, so it gets the chip alone with only the "
                        f"sampler beside it: an exclusive slot on GPU {GPU_OF[exclusive]}, the card the teacher does "
                        f"not hold ({t.say(free_now(pod))} free, {t.say(pod.watts)} at the last reading)."
                        f"{busy_text}")
        if teacher_node:
            bench_reason += (f" {teacher_node} takes only read-only requests to the live teacher, without a slot; "
                             "those measure the service as it runs.")
        ax_reason = (f"each Ax trial is a benchmark or training run (lab/optimize_settings.py drives lab/bench.py or "
                     f"the trainer), so trials take exclusive slots on GPU {GPU_OF[exclusive]} like benchmarks: one "
                     "trial per 15-minute slot, or a declared batch in a 60-minute slot.")
        bench_line = Line("bench", [exclusive], f"{exclusive} (alone)", bench_reason)
        ax_line = Line("ax", [exclusive], f"{exclusive} (alone)", ax_reason)
    else:
        why = "no pod is up" if not up else "the only pod up serves the teacher"
        bench_line = Line("bench", [], "wait", f"{why}, and a benchmark needs a chip to itself.")
        ax_line = Line("ax", [], "wait", f"{why}, and an Ax trial is a benchmark or training run.")

    # 5. LoRA jobs, and whether the teacher's pod could take one
    s_max: Num | None = None
    teacher_lora: Share | None = None
    if teacher_node:
        tpod = pods[teacher_node]
        free = free_memory(tpod)
        hold = teacher_hold(tpod)
        if tpod.teacher_serving and tpod.facts_server_running is True and free.known and hold.known:
            # the facts were read with the teacher up, so what they found free is already without its share
            free = derived(free.value + hold.value, "GiB", f"free when the facts were read with the teacher up, plus "
                           f"its share: {free.text()} + {hold.text()}")
        vis = visible(tpod)
        if free.known and vis.known and lora_budget.known:
            s_max = derived(math.floor((free.value - lora_budget.value - reserve.value) / vis.value * 100) / 100, "",
                            f"(free - LoRA budget - host reserve) / visible: ({free.text()} - {lora_budget.text()} - "
                            f"{reserve.text()}) / {vis.text()}, rounded down")
        need = (derived(hold.value + lora_budget.value + reserve.value, "GiB",
                        f"teacher + LoRA budget + host reserve: {hold.text()} + {lora_budget.text()} + {reserve.text()}")
                if hold.known and lora_budget.known else unknown("GiB", "a part is unknown"))
        paper = (derived(weights.value + lora_budget.value + reserve.value, "GiB",
                         f"teacher weights + LoRA budget + host reserve: {weights.text()} + {lora_budget.text()} + "
                         f"{reserve.text()}") if lora_budget.known and weights.known
                 else unknown("GiB", "no LoRA peak or no checkpoint size"))
        allowed = bool(need.known and free.known and need.value <= free.value)
        pays = measured is None or (measured["time_change"].value < 0 and measured["energy_change"].value <= 0)
        on_paper = bool(paper.known and free.known and paper.value <= free.value)
        margin_left = (derived(free.value - need.value, "GiB", f"free - needed: {free.text()} - {need.text()}")
                       if need.known and free.known else None)
        if allowed and not pays:
            allowed = False
            reason = (f"not on {teacher_node}: they fit (the teacher's {t.say(hold)}, a LoRA budget of "
                      f"{t.say(lora_budget)} and a {t.say(reserve)} host reserve come to {t.say(need)} of "
                      f"{t.say(free)} free), but " + interference_sentence(measured, t.say, interference_path))
        elif allowed:
            reason = (f"yes on {teacher_node}: the teacher's {t.say(hold)}, a LoRA budget of {t.say(lora_budget)} "
                      f"({t.say(lora_peak)} peak plus a {t.say(margin)} margin) and a {t.say(reserve)} host reserve "
                      f"come to {t.say(need)} of {t.say(free)} free. Start the teacher first: vLLM checks free memory "
                      "only at start. " + (interference_sentence(measured, t.say, interference_path) if measured
                                           else "What each slows the other is unmeasured (R43)."))
        else:
            reason = (f"not on {teacher_node} today: the teacher's {t.say(hold)} (share {t.say(share)}), a LoRA "
                      f"budget of {t.say(lora_budget)} ({t.say(lora_peak)} peak plus a {t.say(margin)} margin) and a "
                      f"{t.say(reserve)} host reserve come to {t.say(need)}, more than the {t.say(free)} free.")
            if on_paper and s_max is not None and s_max.known:
                reason += (f" On paper they fit: the weights {t.say(weights)} plus the rest are {t.say(paper)}; with "
                           f"the teacher served at a share of at most {t.say(s_max)} they would share. "
                           + (interference_sentence(measured, t.say, interference_path) if measured else
                              "Whether the teacher starts at its 16k context at that share, and what each slows the "
                              "other, is unmeasured (R43)."))
        teacher_lora = Share(("teacher", "lora"), allowed, on_paper, reason, margin_left,
                             {"file": relative(interference_path), **{k: v.view() for k, v in measured.items()}}
                             if measured else None)
    if exclusive:
        epod = pods[exclusive]
        efree = free_now(epod)
        fits = bool(efree.known and lora_budget.known and lora_budget.value + reserve.value <= efree.value)
        lora_margin = (derived(efree.value - lora_budget.value - reserve.value, "GiB",
                               f"free - LoRA budget - host reserve: {efree.text()} - {lora_budget.text()} - "
                               f"{reserve.text()}") if efree.known and lora_budget.known else None)
        lora_reason = (f"GPU {GPU_OF[exclusive]} is the exclusive card: a LoRA job needs {t.say(lora_budget)} "
                       f"({t.say(lora_peak)} peak, measured on the R9700 smoke run, plus a {t.say(margin)} margin) and "
                       f"{exclusive} has {t.say(efree)} free"
                       + (f", leaving {t.say(lora_margin)} after the {t.say(reserve)} host reserve." if fits and
                          lora_margin else "; that is not enough, so the job waits or shrinks.") + busy(exclusive))
        lora_pods = [exclusive]
        if teacher_lora is not None and teacher_lora.allowed and teacher_node:
            lora_pods.append(teacher_node)
            lora_reason += f" {teacher_node} may take one beside the teacher too (see sharing)."
        elif teacher_node and teacher_lora is not None:
            lora_reason += (f" Not {teacher_node}: the teacher holds {t.say(teacher_hold(pods[teacher_node]))} of its "
                            f"{t.say(free_memory(pods[teacher_node]))}, leaving {t.say(free_now(pods[teacher_node]))}.")
        lora_line = Line("lora", lora_pods, " or ".join(lora_pods), lora_reason)
    elif teacher_lora is not None and teacher_lora.allowed and teacher_node:
        lora_line = Line("lora", [teacher_node], f"{teacher_node} (beside the teacher)", teacher_lora.reason)
    else:
        lora_line = Line("lora", [], "wait", "no pod is free of the teacher and the teacher's pod has no room.")

    lines = [teacher_line, sampler_line, bench_line, ax_line, lora_line]
    where = {line.job: list(line.pods) for line in lines}

    # which jobs may share a chip
    sharing: list[Share] = []
    for other in ("teacher", "lora", "bench", "ax"):
        sharing.append(Share(("sampler", other), True, True, "the sampler holds no GPU memory and reads amd-smi once "
                             "a second; it shares any chip."))
    if teacher_lora is not None:
        sharing.append(teacher_lora)
    for a, b, what in (("teacher", "bench", "a benchmark"), ("teacher", "ax", "an Ax trial")):
        sharing.append(Share((a, b), False, False, f"{what} measures the chip; the live teacher's traffic moves "
                             "its tok/s and watts, and a teacher copy for the measurement belongs on the exclusive "
                             "card."))
    for a, b in (("lora", "bench"), ("lora", "ax"), ("bench", "ax"), ("bench", "bench"), ("ax", "ax")):
        sharing.append(Share((a, b), False, False, "a measurement gets the chip alone: a second job moves its tok/s, "
                             "samples per second and watts."))
    two_lora = (derived(2 * lora_budget.value + reserve.value, "GiB", f"2 x {lora_budget.text()} + {reserve.text()}")
                if lora_budget.known else unknown("GiB", "no LoRA peak"))
    sharing.append(Share(("lora", "lora"), False, False, f"two LoRA jobs fit in memory on paper ({t.say(two_lora)}) but "
                         "both are compute-bound, so together they finish no sooner than one after the other, and "
                         "each run's samples per second stops meaning anything."))
    sharing.append(Share(("teacher", "teacher"), False, False, f"a second vLLM at the same share asks for "
                         f"{t.say(teacher_hold(pods[teacher_node])) if teacher_node else 'the same share'} more at "
                         "start and is refused while the first holds it (vllm/v1/worker/utils.py request_memory)."))

    # levers: what would free capacity
    if (teacher_node and exclusive and teacher_lora is not None and not teacher_lora.allowed and teacher_lora.on_paper
            and s_max is not None and s_max.known and share.known and s_max.value < share.value):
        levers.append(
            f"{teacher_node} idles at {t.say(idle_loaded)} with the teacher loaded between requests, while the teacher "
            f"reserves {t.say(teacher_hold(pods[teacher_node]))}. Serving it at a share of at most {t.say(s_max)} "
            f"instead of {t.say(share)} would let LoRA jobs run on {teacher_node} while {exclusive} runs "
            f"benchmarks, so training need not wait behind measurements. It costs a teacher restart "
            f"({t.say(warm_start)} without the teacher) and Ilan's go; "
            + (f"the decode slowdown measured beside a LoRA is {t.say(measured['decode_change'])}." if measured
               else "the decode slowdown is R43's measurement."))
    if exclusive and not pods[exclusive].sampler_seen and SAMPLER_PORT not in pods[exclusive].listening:
        levers.append(f"Start the sampler on {exclusive} once and leave it up, so every slot there is measured "
                      "without a start-up in the slot.")

    # the queue, checked against the plan
    queue: dict[str, Any]
    if rows is None:
        queue = {"read": False, "rows": []}
    else:
        checked = []
        for row in queue_rows(rows):
            node = next((n for n, letter in GPU_OF.items() if str(row.gpu).startswith(letter + " ")), None)
            kind = classify(row.request)
            entry = {"row": row.id, "request": row.request, "status": row.status, "gpu": row.gpu, "node": node,
                     "kind": kind}
            if kind is not None and node is not None:
                entry["consistent"] = node in where.get(kind, [])
            checked.append(entry)
        queue = {"read": True, "rows": checked}

    stamp = None
    if manifest:
        stamp = manifest.get("recorded")
    times = [pod.facts_time for pod in pods.values() if pod.facts_time is not None]
    facts = {"folder": relative(folder) if folder else None, "recorded": stamp,
             "oldest_facts": _iso(min(times)) if times else None,
             "age_hours": round((now - min(times)).total_seconds() / 3600, 1) if times else None}
    if any(pod.chip is not None and pod.chip.sample for pod in pods.values()):
        notes.append("The chip sheet is the sample (chipsheet-sample.json): its numbers are hour-0 facts and earlier "
                     "measurements, not R33's; bandwidth and matmul rate stay unknown until R33 commits a sheet.")
    for pod in pods.values():
        if pod.chip is not None and pod.chip.problem:
            notes.append(f"{pod.node}: {pod.chip.problem}")
    return Plan(_iso(now), facts, dict(pods), lines, sharing, where, queue, levers, notes, t, rules)


# --- checking a request ------------------------------------------------------------------------------------------

_REF = re.compile(r" \[\d+\]")


def plain(text: str) -> str:
    """A plan's sentence without its [n] references, for an answer that does not carry the Numbers list."""
    return _REF.sub("", text or "")


def check(plan: Plan, *, gpu: str, text: str, kind: str | None = None) -> dict[str, Any]:
    """Whether a slot request for that card (A or B) with those words contradicts the plan. The answer carries
    contradicts (bool), the kind and how it was found, the plan's line for it and the reason."""
    letter = gpu.strip().upper()[:1]
    node = NODE_OF.get(letter)
    if node is None:
        raise ValueError(f"{gpu!r} is not a card of the queue; use A or B")
    found = kind if kind else classify(text)
    how = "given" if kind else ("words" if found else None)
    if found is None:
        return {"contradicts": False, "verdict": "no_line", "kind": None, "gpu": letter, "node": node,
                "reason": "the request's words name no kind of job the plan places (teacher, sampler, bench, ax, "
                          "lora); name one with --kind to check it"}
    line = plan.line(found)
    pods = list(line.pods) if line else []
    reason = plain(line.reason) if line else ""
    out: dict[str, Any] = {"kind": found, "kind_from": how, "gpu": letter, "node": node,
                           "plan": {"pods": pods, "decision": line.decision if line else None, "reason": reason}}
    if node in pods:
        return {"contradicts": False, "verdict": "consistent", **out,
                "reason": f"the plan puts {found} on {node} (GPU {letter})"}
    if pods:
        cards = " or ".join(f"GPU {GPU_OF[p]} ({p})" for p in pods)
        why = f"the plan puts {found} on {cards}, not on {node}: {reason}".strip()
        next_step = f"ask for GPU {GPU_OF[pods[0]]} instead, or add --override \"why it must run on {node}\""
    else:
        why = f"the plan has no pod for {found} now: {reason}".strip()
        next_step = f"wait until the plan has a pod for it, or add --override \"why it must run on {node} now\""
    return {"contradicts": True, "verdict": "contradicts", **out, "reason": why, "next": next_step}


# --- building a plan from the defaults ---------------------------------------------------------------------------

def default_plan(*, facts: str | os.PathLike[str] | None = None, chipsheets: Mapping[str, Any] | None = None,
                 rows: Sequence[Any] | None = None, rules: Rules | None = None,
                 environ: Mapping[str, str] | None = None, now: dt.datetime | None = None,
                 interference: str | os.PathLike[str] | None = None) -> Plan:
    """The plan from the facts folder (see facts_folder), the pods' chip sheets and the queue rows given."""
    folder = facts_folder(facts, environ)
    pods, manifest = load_pods(folder, chipsheets)
    return build_plan(pods, rows=rows, rules=rules, manifest=manifest, folder=folder, now=now,
                      interference=interference)


def check_request(*, gpu: str, text: str, kind: str | None = None, environ: Mapping[str, str] | None = None,
                  now: dt.datetime | None = None) -> dict[str, Any]:
    """check() against the default plan, for the slot request verb. A plan that cannot be built never blocks a
    request: the answer is verdict unavailable, with the error."""
    try:
        plan = default_plan(environ=environ, now=now)
    except Exception as exc:  # the queue must keep working when the facts are missing or broken
        return {"contradicts": False, "verdict": "unavailable", "kind": kind or classify(text),
                "reason": f"no placement plan: {type(exc).__name__}: {exc}"}
    answer = check(plan, gpu=gpu, text=text, kind=kind)
    answer["facts"] = plan.facts
    return answer


# --- text --------------------------------------------------------------------------------------------------------

def render(plan: Plan) -> str:
    """The plan as text: pods, one line per job, sharing, the queue, levers, notes and the numbers' sources."""
    t = plan.trace
    out = [f"Placement plan as of {plan.as_of}; facts from {plan.facts.get('folder')} "
           f"(recorded {plan.facts.get('recorded') or 'unknown'}, oldest pod facts {plan.facts.get('oldest_facts')}, "
           f"{plan.facts.get('age_hours')} h old). Rules: {NOTE}.", "", "Pods"]
    for node, pod in plan.pods.items():
        state = {True: "up", False: "down", None: "unknown"}[pod.up]
        chip = pod.chip.path + (" (sample)" if pod.chip and pod.chip.sample else "") if pod.chip else "none"
        serving = ("serves " + ", ".join(pod.serving)) if pod.serving else "no model answering"
        out.append(f"  {node} (account {pod.account}, {pod.gpu or pod.accelerator or 'GPU unknown'}, {pod.gfx}): "
                   f"{state}{' since ' + pod.started if pod.started else ''}; {serving}; "
                   f"{t.say(pod.available)} free when its facts were read, {t.say(pod.visible)} visible; "
                   f"{t.say(pod.watts)} at the last reading; "
                   f"chip sheet {chip}")
    out += ["", "Plan"]
    for line in plan.lines:
        out.append(f"  {line.job:<8} -> {line.decision}")
        out.append(f"             {line.reason}")
    out += ["", "May share a chip"]
    for share in plan.sharing:
        word = "yes" if share.allowed else ("no today, yes on paper" if share.on_paper else "no")
        out.append(f"  {share.jobs[0]} + {share.jobs[1]}: {word}. {share.reason}")
    out += ["", "Queue"]
    if not plan.queue.get("read"):
        out.append("  not read (--no-queue, or no NOTION_TOKEN)")
    elif not plan.queue["rows"]:
        out.append("  no Queued or Running rows")
    for entry in plan.queue.get("rows", []):
        verdict = {True: "fits the plan", False: "CONTRADICTS the plan", None: "no line in the plan"}[
            entry.get("consistent")]
        out.append(f"  {entry['status']} on {entry['gpu']}: {entry['request']} ({entry['kind'] or 'kind unknown'}, "
                   f"{verdict})")
    if plan.levers:
        out += ["", "Levers"] + [f"  - {lever}" for lever in plan.levers]
    if plan.notes:
        out += ["", "Notes"] + [f"  - {note}" for note in plan.notes]
    out += ["", "Numbers"]
    for view in t.views():
        out.append(f"  [{view['id']}] {Num(view['value'], view['unit'], view['source'], view['kind']).text()} "
                   f"({view['kind']}): {view['source']}")
    return "\n".join(out)


def render_check(answer: Mapping[str, Any]) -> str:
    verdict = answer.get("verdict")
    head = {"consistent": "consistent", "contradicts": "CONTRADICTS the plan", "no_line": "no line in the plan",
            "unavailable": "no plan"}.get(verdict, str(verdict))
    lines = [f"{head}: {answer.get('reason')}"]
    if answer.get("next"):
        lines.append(f"next: {answer['next']}")
    return "\n".join(lines)


# --- the command line --------------------------------------------------------------------------------------------

def _chipsheet_arg(value: str) -> tuple[str, str]:
    node, sep, path = value.partition("=")
    if not sep or node.strip() not in NODES or not path.strip():
        raise argparse.ArgumentTypeError(f"{value!r} is not <pod-a|pod-b>=<path to a chip sheet>")
    return node.strip(), path.strip()


def _fraction(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from None
    if not 0 <= number <= 5:
        raise argparse.ArgumentTypeError(f"{value!r} is out of range")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lab placement", description=main.__doc__, allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="answer with one JSON object instead of text")
    parser.add_argument("--check", metavar="REQUEST", help="a slot request's text: does it contradict the plan?")
    parser.add_argument("--gpu", default="B", metavar="{A,B}", help="the card the checked request asks for (default B)")
    parser.add_argument("--kind", choices=KINDS, help="the kind of the checked request, instead of its words")
    parser.add_argument("--facts", metavar="DIR", help="the pods' facts folder (default: $LAB_PLACEMENT_FACTS, "
                        "<state dir>/placement, else fixtures/lab/placement)")
    parser.add_argument("--chipsheet", action="append", type=_chipsheet_arg, default=[], metavar="POD=FILE",
                        help="the chip sheet of a pod (repeatable); default the newest for its chip, else the sample")
    queue = parser.add_mutually_exclusive_group()
    queue.add_argument("--queue-file", metavar="FILE", help="GPU Queue pages as the Notion API returns them (a list)")
    queue.add_argument("--no-queue", action="store_true", help="do not read the GPU Queue")
    parser.add_argument("--margin", type=_fraction, default=Rules.margin,
                        help=f"margin on a measured peak, as a fraction (default {Rules.margin:g})")
    parser.add_argument("--host-reserve", type=_fraction, default=Rules.host_reserve_gib, metavar="GIB",
                        help=f"GiB kept for the jobs' host memory (default {Rules.host_reserve_gib:g})")
    parser.add_argument("--teacher-share", type=_fraction, metavar="FRACTION",
                        help="the teacher's --gpu-memory-utilization (default: lab/pod_a_teacher.sh's)")
    parser.add_argument("--interference", metavar="FILE", help="the measured interference file (default: the newest "
                        f"{INTERFERENCE_PREFIX}-<date>.json beside the chip sheets)")
    parser.add_argument("--out", metavar="FILE", help="also write the plan as JSON to this file")
    parser.add_argument("--now", help="compute as of this time (ISO 8601 with a UTC offset; replays and tests)")
    return parser


def _read_queue(args: argparse.Namespace, client: Any, environ: Mapping[str, str]) -> tuple[list[Any] | None, str | None]:
    """The queue rows the plan reads, and why none were when they were not."""
    from lab.machine import slots   # imported here: slots imports this module to check requests
    if args.no_queue:
        return None, None
    if args.queue_file:
        pages = json.loads(Path(args.queue_file).read_text(encoding="utf-8"))
        pages = pages.get("results", pages) if isinstance(pages, Mapping) else pages
        return [slots.slot_from_page(page) for page in pages], None
    client = client or slots.connect(environ)
    if client is None:
        return None, "the GPU Queue was not read: no NOTION_TOKEN"
    queue = slots.SlotQueue(client, data_source=slots.data_source_of(environ))
    try:
        return queue.board(slots.GPUS["A"]) + queue.board(slots.GPUS["B"]), None
    except slots.notion.NotionError as exc:
        return None, f"the GPU Queue could not be read: {exc.kind}: {exc}"


def run(argv: Sequence[str], *, client: Any = None, environ: Mapping[str, str] | None = None,
        now: dt.datetime | None = None) -> tuple[str, dict[str, Any], int]:
    """One placement call: (the text, the JSON answer, the exit code). client replaces the Notion client the token
    would make."""
    environ = os.environ if environ is None else environ
    args = build_parser().parse_args(list(argv))
    if args.now:
        now = parse_time(args.now)
        if now is None:
            raise SystemExit(f"--now {args.now!r} is not an ISO 8601 time")
    rules = Rules(margin=args.margin, host_reserve_gib=args.host_reserve, teacher_share=args.teacher_share)
    try:
        rows, queue_problem = _read_queue(args, client, environ)
        plan = default_plan(facts=args.facts, chipsheets=dict(args.chipsheet), rows=rows, rules=rules,
                            environ=environ, now=now, interference=args.interference)
    except (OSError, ValueError, KeyError) as exc:
        answer = {"ok": False, "outcome": "error", "error": f"{type(exc).__name__}: {exc}"}
        return f"placement: {answer['error']}", answer, 1
    if queue_problem:
        plan.notes.append(queue_problem)
    text = render(plan)          # first: the text cites the pods' numbers too, so the view lists them
    view = plan.view()
    if args.out:
        Path(args.out).write_text(json.dumps(view, indent=2) + "\n", encoding="utf-8")
    if args.check is not None:
        try:
            answer = check(plan, gpu=args.gpu, text=args.check, kind=args.kind)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        answer = {"ok": not answer["contradicts"], "outcome": answer["verdict"], **answer, "facts": plan.facts}
        return render_check(answer), answer, 0 if answer["ok"] else 1
    return text, view, 0


def main(argv: Sequence[str]) -> int:
    """Which pod serves the teacher and where the sampler, benchmarks, Ax trials and LoRA jobs go (--check a request)."""
    as_json = "--json" in argv
    try:
        text, answer, code = run(argv)
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            return 2
        raise
    print(json.dumps(answer) if as_json else text)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
