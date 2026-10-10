"""The R9700 power sampler that runs on an AUP pod, and the parsers and window summary it is built on.

On the pod (python3, nothing to install):

    python3 pod_power.py check [--save DIR]                 # one read with each tool and option set, as JSON
    python3 pod_power.py serve --node gpu-a [--period-ms 1000] [--port 8790] [--minutes N]
    python3 pod_power.py alive --pidfile ~/atlas-power/sampler.pid   # the pid, exit 0, only if it is this sampler

What it reads. `amd-smi metric --power --clock --temperature --usage --energy`, once per period, in one process, so
power, clocks, temperatures, GFX activity and the energy counter come from one instant. SOCKET_POWER is the AMD SMI
library's socket_power: current_socket_power where the card reports one, else average_socket_power (amd_smi.cc at
therock-7.13); AMD documents only the average for Navi cards, so on the R9700 it is the firmware's averaged power,
in whole watts. `--energy` exists only on bare-metal Linux (amdsmi_parser.py at therock-7.13), so the read goes on
without it where the CLI refuses it, and then without `--usage`, then with `--power` alone. When amd-smi reads no
power at all, `rocm-smi --showpower --showtemp --showclocks --showuse` is the fallback; its "Average Graphics
Package Power" comes from the same firmware average. The tool and options are chosen once at start-up by trying
them in that order, and every sample names them (`source`, `power_field`).

Each read becomes one `telemetry.gpu.power` event (contracts/telemetry.py) as JSON, stamped with the pod's clock at
the middle of the read. A read that fails is still a sample, with every reading null, so a gap shows as missing
readings rather than as missing time.

Distinct updates. Every read is a poll; a poll whose readings differ from the previous successful one is a distinct
update. Repeats are normal (whole watts, a firmware average, a 100 ms metrics cache inside one amd-smi process), so
`polls` and `updates` are both on every sample and a window counts its updates, not its samples.

Stale under load. A sample is under load when GFX activity is at least LOAD_ACTIVITY_PCT (or, where activity is not
readable, when power is at least LOAD_WATTS). Under load the readings move; a sample under load whose readings have
not changed for `stale_after_s` is marked stale, and the run says it saw one. A flag to look at, not a verdict.

The energy counter. Over the first `check_window_s` after start-up, and once over the same length of continuous load
after `check_settle_s` of it, the sampler compares the counter's delta with power integrated over the same samples.
The counter advanced when its delta is positive and it never went backwards; it agrees when it advanced and the
ratio is within `tolerance` (10 percent) of 1. The sample that closes a check carries it (`energy_check`), and every
sample says whether the counter may be used for energy (`counter_ok`): null until the start-up check ends, then true
only while every check so far agreed. summarize() takes energy from the counter only when every sample in the window
says it may, and otherwise integrates power; the row says which.

Where samples go. Appended to <dir>/samples-<UTC day>.jsonl (default ~/atlas-power on the persistent home) and kept
in memory for the HTTP endpoint on 127.0.0.1:<port>, which the hub proxy exposes as
https://dub.aupcloud.io/user/<account>/proxy/<port>/ to anyone holding the account's hub token:

    GET /samples?after=<seq>&limit=<n>   the samples after seq, oldest first: {"seq": n, "event": {...}}
    GET /status                          the counters, the chosen tool, the energy checks so far

Every answer carries `pod_received` and `pod_sent` (the pod's clock) so a reader can measure the clock offset. The
server logs no requests: the proxy passes the hub token through in the headers.

Is it running. The pid file lives on the persistent home and outlives the container, so after a relaunch its
number can belong to any process. `alive` (alive_sampler_pid) answers only from /proc/<pid>/cmdline: the pid counts
only while its command line is pod_power.py serve. It never signals the pid (os.kill with signal 0 is not a probe on
every platform), and lab/pod_power.sh asks it before start, stop and status.

Node gate: `serve` refuses to start unless --node is gpu-a or gpu-b and rocminfo names gfx1201 (an R9700 pod).
Stdlib only and one file, because the team repo cannot be cloned on a pod (lab/README.md); upload this file next
to lab/pod_power.sh, which starts and stops it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qs, urlsplit

SAMPLER = "lab/pod_power.py 1"  # program and version, on every sample
GPU_POWER = "telemetry.gpu.power"
NODES = ("gpu-a", "gpu-b")  # telemetry's names for the two pods; the lab ledger calls them pod-a and pod-b
LEDGER_NODE = {"gpu-a": "pod-a", "gpu-b": "pod-b"}
R9700_GFX = "gfx1201"

DEFAULT_PERIOD_MS = 1000.0
DEFAULT_PORT = 8790
DEFAULT_HOST = "127.0.0.1"
DEFAULT_DIR = "~/atlas-power"
STALE_AFTER_S = 10.0
CHECK_WINDOW_S = 10.0
CHECK_SETTLE_S = 5.0
CHECK_TOLERANCE = 0.10
LOAD_ACTIVITY_PCT = 50.0
LOAD_WATTS = 100.0
READ_TIMEOUT_S = 10.0
KEEP = 20_000  # samples the endpoint can hand out: about 5.5 hours at one a second
MAX_LIMIT = 5_000

AMDSMI = "amd-smi"
ROCMSMI = "rocm-smi"
AMDSMI_OPTIONS: tuple[tuple[str, ...], ...] = (
    ("metric", "--power", "--clock", "--temperature", "--usage", "--energy"),
    ("metric", "--power", "--clock", "--temperature", "--usage"),
    ("metric", "--power", "--clock", "--temperature"),
    ("metric", "--power"),
)
ROCMSMI_OPTIONS: tuple[tuple[str, ...], ...] = (
    ("--showpower", "--showtemp", "--showclocks", "--showuse"),
    ("--showpower",),
)
ROCM_BIN = Path("/opt/rocm/bin")

PROC_ROOT = Path("/proc")
SAMPLER_SCRIPT = "pod_power.py"  # the program name a live sampler's command line carries
SAMPLER_SERVE = "serve"  # and the subcommand right after it
ALIVE_EXIT_YES = 0  # `alive`: the pid file names a running sampler (its pid is printed)
ALIVE_EXIT_NO = 1  # `alive`: no pid file, no such process, or the pid is some other program now
ALIVE_EXIT_UNKNOWN = 2  # `alive`: the pid file or /proc could not be read; the launcher must not guess

_NUMBER = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)")
_SENTINELS = {float(0xFFFF), float(0xFFFFFFFF), float(0xFFFFFFFFFFFFFFFF)}  # unsupported-field markers, never data


class ReadError(Exception):
    """A tool ran but gave no reading for the GPU asked for."""


# --- readings ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Reading:
    """One read of one GPU: None for anything the tool could not read."""

    socket_w: float | None = None
    gfx_mhz: float | None = None
    mem_mhz: float | None = None
    temps_c: Mapping[str, float] = field(default_factory=dict)
    throttled: bool | None = None
    gfx_activity_pct: float | None = None
    energy_j: float | None = None
    power_field: str | None = None

    def key(self) -> tuple:
        """What has to change for a read to count as a distinct update."""
        return (self.socket_w, self.gfx_mhz, self.mem_mhz, tuple(sorted(self.temps_c.items())), self.throttled,
                self.gfx_activity_pct, self.energy_j)

    def empty(self) -> bool:
        return all(v is None for v in (self.socket_w, self.gfx_mhz, self.mem_mhz, self.throttled,
                                       self.gfx_activity_pct, self.energy_j)) and not self.temps_c


def number(value: Any) -> float | None:
    """The leading number of a reading such as "40 W", "35 \N{DEGREE SIGN}C" or "2350"; None for N/A, text or a sentinel."""
    if value is None or isinstance(value, (bool, dict, list, tuple)):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
    else:
        match = _NUMBER.match(str(value))
        if not match:
            return None
        out = float(match.group(1))
    if not math.isfinite(out) or out in _SENTINELS:
        return None
    return out


# --- amd-smi ----------------------------------------------------------------------------------------------------

def amdsmi_tree(text: str) -> dict[int, dict[str, Any]]:
    """amd-smi's human-readable output as nested dicts, one per `GPU: <n>` block, keys upper-cased.

    The CLI prints each device as indented `KEY: value` lines and `KEY:` lines that open a section
    (amdsmi_logger.py custom_dump at therock-7.13). Lines before the first device (warnings) and lines
    without a colon (list items) are left out."""
    devices: dict[int, dict[str, Any]] = {}
    stack: list[tuple[int, dict[str, Any]]] = []
    for raw in text.splitlines():
        line = raw.replace("\t", "    ").rstrip()
        if ":" not in line or not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        key, value = key.strip().upper(), value.strip()
        if key == "GPU" and re.fullmatch(r"\d+", value):
            device: dict[str, Any] = {}
            devices[int(value)] = device
            stack = [(indent, device)]
            continue
        if not stack:
            continue
        if indent <= stack[0][0]:
            stack = []  # back at the device's own level without a new GPU line: not ours
            continue
        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if value:
            parent[key] = value
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
    return devices


def _section(tree: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = tree.get(name)
    return value if isinstance(value, Mapping) else {}


def parse_amdsmi(text: str) -> dict[int, Reading]:
    """Readings per GPU from `amd-smi metric` text with any of --power, --clock, --temperature, --usage, --energy."""
    out: dict[int, Reading] = {}
    for gpu, device in amdsmi_tree(text).items():
        power = _section(device, "POWER")
        clock = _section(device, "CLOCK")
        temps = {name.lower(): value for name, raw in _section(device, "TEMPERATURE").items()
                 if (value := number(raw)) is not None}
        throttle = str(power.get("THROTTLE_STATUS", "")).strip().upper()
        out[gpu] = Reading(
            socket_w=number(power.get("SOCKET_POWER")),
            gfx_mhz=number(_section(clock, "GFX_0").get("CLK")),
            mem_mhz=number(_section(clock, "MEM_0").get("CLK")),
            temps_c=temps,
            throttled={"THROTTLED": True, "UNTHROTTLED": False}.get(throttle),
            gfx_activity_pct=number(_section(device, "USAGE").get("GFX_ACTIVITY")),
            energy_j=number(_section(device, "ENERGY").get("TOTAL_ENERGY_CONSUMPTION")),
            power_field="socket_power" if "SOCKET_POWER" in power else None,
        )
    return out


# --- rocm-smi ---------------------------------------------------------------------------------------------------

_ROCM_LINE = re.compile(r"^GPU\[(\d+)\]\s*:\s*(.*)$")
_ROCM_POWER = re.compile(r"(Average|Current Socket) Graphics Package Power \(W\)")
_ROCM_TEMP = re.compile(r"Temperature \(Sensor (.+)\) \(C\)")
_ROCM_CLOCK = re.compile(r"(sclk|mclk) clock level: \S+")
_ROCM_MHZ = re.compile(r"\(\s*([-+]?\d+(?:\.\d+)?)\s*Mhz\)", re.IGNORECASE)
_ROCM_SENSORS = {"edge": "edge", "junction": "hotspot", "memory": "mem"}  # amd-smi's names for the same sensors


def parse_rocmsmi(text: str) -> dict[int, Reading]:
    """Readings per GPU from `rocm-smi` text with any of --showpower, --showtemp, --showclocks, --showuse.

    Lines look like `GPU[0]\\t\\t: Average Graphics Package Power (W): 40.0` (rocm_smi.py printLog at therock-7.13)."""
    found: dict[int, dict[str, Any]] = {}
    for raw in text.splitlines():
        match = _ROCM_LINE.match(raw.strip())
        if not match:
            continue
        gpu, rest = int(match.group(1)), match.group(2)
        label, sep, value = rest.rpartition(":")
        if not sep:
            continue
        label, value = label.strip(), value.strip()
        got = found.setdefault(gpu, {"temps_c": {}})
        if power := _ROCM_POWER.fullmatch(label):
            got["socket_w"] = number(value)
            got["power_field"] = f"{power.group(1).lower().replace(' ', '_')}_graphics_package_power"
        elif temp := _ROCM_TEMP.fullmatch(label):
            reading = number(value)
            if reading is not None:
                sensor = temp.group(1).strip()
                got["temps_c"][_ROCM_SENSORS.get(sensor.lower(), sensor.lower().replace(" ", "_"))] = reading
        elif clock := _ROCM_CLOCK.fullmatch(label):
            mhz = _ROCM_MHZ.search(value)
            got["gfx_mhz" if clock.group(1) == "sclk" else "mem_mhz"] = float(mhz.group(1)) if mhz else None
        elif label == "GPU use (%)":
            got["gfx_activity_pct"] = number(value)
    return {gpu: Reading(**fields) for gpu, fields in found.items()}


# --- running the tools ------------------------------------------------------------------------------------------

def run_command(argv: Sequence[str], timeout_s: float = READ_TIMEOUT_S) -> str:
    """stdout of one tool call; a non-zero exit, a timeout or a missing program raises."""
    try:
        done = subprocess.run(list(argv), capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ReadError(f"{argv[0]} gave no answer within {timeout_s:g} s") from exc
    if done.returncode != 0:
        last = (done.stderr or done.stdout or "").strip().splitlines()[-1:] or [""]
        raise ReadError(f"{argv[0]} exited {done.returncode}: {last[0][:200]}")
    return done.stdout


def find_tool(name: str, which: Callable[[str], str | None] = shutil.which) -> str | None:
    """The program to run for a tool: on PATH, else under /opt/rocm/bin."""
    found = which(name)
    if found:
        return found
    fallback = ROCM_BIN / name
    return str(fallback) if fallback.exists() else None


class CommandReader:
    """One tool with one option set, read for one GPU."""

    def __init__(self, tool: str, program: str, options: Sequence[str], parse: Callable[[str], dict[int, Reading]],
                 gpu: int = 0, run: Callable[[Sequence[str]], str] = run_command) -> None:
        self.source = tool
        self.program = program
        self.options = tuple(options)
        self.parse = parse
        self.gpu = gpu
        self.run = run

    def read(self) -> Reading:
        text = self.run([self.program, *self.options])
        readings = self.parse(text)
        if self.gpu not in readings:
            raise ReadError(f"{self.source} printed no GPU {self.gpu}")
        return readings[self.gpu]

    def describe(self) -> dict[str, Any]:
        return {"tool": self.source, "options": list(self.options), "gpu": self.gpu}


def candidates(gpu: int = 0, run: Callable[[Sequence[str]], str] = run_command,
               which: Callable[[str], str | None] = shutil.which) -> list[CommandReader | dict[str, Any]]:
    """Every reader worth trying, best first; a tool that is not installed is a dict saying so."""
    out: list[CommandReader | dict[str, Any]] = []
    for tool, option_sets, parse in ((AMDSMI, AMDSMI_OPTIONS, parse_amdsmi), (ROCMSMI, ROCMSMI_OPTIONS, parse_rocmsmi)):
        program = find_tool(tool, which)
        if program is None:
            out.append({"tool": tool, "error": "not installed (not on PATH or under /opt/rocm/bin)"})
            continue
        out.extend(CommandReader(tool, program, options, parse, gpu=gpu, run=run) for options in option_sets)
    return out


def choose_reader(gpu: int = 0, run: Callable[[Sequence[str]], str] = run_command,
                  which: Callable[[str], str | None] = shutil.which) -> tuple[CommandReader | None, list[dict[str, Any]]]:
    """The first reader that reads a power number, and what each attempt gave."""
    attempts: list[dict[str, Any]] = []
    for reader in candidates(gpu, run, which):
        if isinstance(reader, dict):
            attempts.append(reader)
            continue
        try:
            reading = reader.read()
        except (ReadError, OSError) as exc:
            attempts.append({**reader.describe(), "error": str(exc)})
            continue
        attempts.append({**reader.describe(), "socket_w": reading.socket_w})
        if reading.socket_w is not None:
            return reader, attempts
    return None, attempts


# --- time -------------------------------------------------------------------------------------------------------

def iso(t: float) -> str:
    """A Unix time as UTC ISO 8601 with milliseconds and a Z."""
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_ts(text: str) -> datetime:
    """ISO 8601 with a Z or an offset, as an aware datetime."""
    out = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return out if out.tzinfo else out.replace(tzinfo=timezone.utc)


# --- the energy counter check -----------------------------------------------------------------------------------

def trapezoid_j(points: Sequence[tuple[float, float]]) -> float | None:
    """Joules under (seconds, watts) points by the trapezoid rule; None with fewer than two points."""
    if len(points) < 2:
        return None
    return sum((t2 - t1) * (w1 + w2) / 2.0 for (t1, w1), (t2, w2) in zip(points, points[1:]))


def energy_check(window: Sequence[tuple[float, float | None, float | None]], phase: str,
                 tolerance: float = CHECK_TOLERANCE) -> dict[str, Any]:
    """Does the energy counter advance, and does its delta agree with integrated power within `tolerance`?

    window: (seconds, socket_w, counter_j) per sample, oldest first. The counter is compared over the span between
    its first and last readings, and power is integrated over the samples in that span. A counter that went
    backwards between two reads did not advance (a wrapped or broken counter is not a usable one)."""
    counted = [(t, e) for t, _, e in window if e is not None]
    advanced: bool | None = None
    delta: float | None = None
    integrated: float | None = None
    if len(counted) >= 2:
        delta = counted[-1][1] - counted[0][1]
        went_back = any(b < a for (_, a), (_, b) in zip(counted, counted[1:]))
        advanced = delta > 0 and not went_back
        start, end = counted[0][0], counted[-1][0]
        inside = [(t, w) for t, w, _ in window if start <= t <= end]
        if all(w is not None for _, w in inside):
            integrated = trapezoid_j([(t, float(w)) for t, w in inside])
    else:
        inside = [(t, w) for t, w, _ in window]
        if all(w is not None for _, w in inside):
            integrated = trapezoid_j([(t, float(w)) for t, w in inside])
    ratio = delta / integrated if delta is not None and integrated else None
    agrees = bool(advanced) and ratio is not None and abs(ratio - 1.0) <= tolerance
    span = window[-1][0] - window[0][0] if window else 0.0
    return {"phase": phase, "window_s": round(span, 3), "samples": len(window), "advanced": advanced,
            "counter_delta_j": None if delta is None else round(delta, 3),
            "integrated_j": None if integrated is None else round(integrated, 3),
            "ratio": None if ratio is None else round(ratio, 4), "tolerance": tolerance, "agrees": agrees}


# --- the sampler ------------------------------------------------------------------------------------------------

class PodSampler:
    """Turns reads into telemetry.gpu.power samples: counts, load and staleness, the energy counter checks.

    poll() reads once and records the sample; record() takes a reading made elsewhere (tests, a replay).
    Thread-safe for one polling thread and any number of readers of samples_after() and status()."""

    def __init__(self, reader: Any, node: str, *, period_ms: float = DEFAULT_PERIOD_MS,
                 stale_after_s: float = STALE_AFTER_S, check_window_s: float = CHECK_WINDOW_S,
                 check_settle_s: float = CHECK_SETTLE_S, tolerance: float = CHECK_TOLERANCE,
                 load_activity_pct: float = LOAD_ACTIVITY_PCT, load_watts: float = LOAD_WATTS,
                 clock: Callable[[], float] = time.time, sink: Callable[[dict[str, Any]], None] | None = None,
                 keep: int = KEEP, new_id: Callable[[], str] = lambda: uuid.uuid4().hex) -> None:
        if node not in NODES:
            raise ValueError(f"unknown node {node!r}; expected one of {', '.join(NODES)}")
        if period_ms <= 0:
            raise ValueError("period_ms must be more than 0")
        self.reader = reader
        self.node = node
        self.period_ms = float(period_ms)
        self.stale_after_s = stale_after_s
        self.check_window_s = check_window_s
        self.check_settle_s = check_settle_s
        self.tolerance = tolerance
        self.load_activity_pct = load_activity_pct
        self.load_watts = load_watts
        self.clock = clock
        self.sink = sink
        self.new_id = new_id
        self.started = clock()
        self.polls = 0
        self.updates = 0
        self.errors = 0
        self.last_error: str | None = None
        self.stale_seen = False
        self.counter_ok: bool | None = None
        self.checks: list[dict[str, Any]] = []
        self.sink_errors = 0
        self._last_key: tuple | None = None
        self._last_change: float | None = None
        self._phase: str | None = "start-up"
        self._window: list[tuple[float, float | None, float | None]] = []
        self._load_since: float | None = None
        self._first_t: float | None = None
        self._last_t: float | None = None
        self._seq = 0
        self._buffer: deque[tuple[int, dict[str, Any]]] = deque(maxlen=keep)
        self._lock = threading.Lock()

    @property
    def source(self) -> str:
        return getattr(self.reader, "source", "unknown")

    def poll(self) -> dict[str, Any]:
        """Read once and record the sample, stamped at the middle of the read."""
        before = self.clock()
        error = None
        try:
            reading = self.reader.read()
        except Exception as exc:  # a failed read is a sample with no readings, never a crash
            reading, error = Reading(), f"{type(exc).__name__}: {exc}"
        after = self.clock()
        return self.record(reading, (before + after) / 2.0, error=error)

    def record(self, reading: Reading, t: float, error: str | None = None) -> dict[str, Any]:
        self.polls += 1
        if error is not None:
            self.errors += 1
            self.last_error = error[:300]
        if not reading.empty():
            key = reading.key()
            if key != self._last_key:
                self.updates += 1
                self._last_change = t
            self._last_key = key
        under_load = self._under_load(reading)
        stale = None
        if under_load is not None:
            stale = bool(under_load and self._last_change is not None and t - self._last_change >= self.stale_after_s)
            self.stale_seen = self.stale_seen or stale
        check = self._energy_check(t, reading, under_load)
        self._first_t = t if self._first_t is None else self._first_t
        self._last_t = t
        event = {
            "event_id": self.new_id(), "type": GPU_POWER, "version": 1, "ts": iso(t), "node": self.node,
            "source": self.source, "period_ms": self.period_ms, "socket_w": reading.socket_w,
            "gfx_mhz": reading.gfx_mhz, "mem_mhz": reading.mem_mhz, "throttled": reading.throttled,
            "power_field": reading.power_field, "temps_c": dict(reading.temps_c), "energy_counter_j": reading.energy_j,
            "energy_counter_ns": None, "received_ts": None, "sampler": SAMPLER, "sampler_started": iso(self.started),
            "polls": self.polls, "updates": self.updates, "gfx_activity_pct": reading.gfx_activity_pct,
            "under_load": under_load, "stale": stale, "counter_ok": self.counter_ok, "energy_check": check,
        }
        with self._lock:
            self._seq += 1
            self._buffer.append((self._seq, event))
        if self.sink is not None:
            try:
                self.sink(event)
            except OSError as exc:  # a full disk must not stop the samples the endpoint serves
                self.sink_errors += 1
                self.last_error = f"sink: {type(exc).__name__}: {exc}"[:300]
        return event

    def _under_load(self, reading: Reading) -> bool | None:
        if reading.gfx_activity_pct is not None:
            return reading.gfx_activity_pct >= self.load_activity_pct
        if reading.socket_w is not None:
            return reading.socket_w >= self.load_watts
        return None

    def _energy_check(self, t: float, reading: Reading, under_load: bool | None) -> dict[str, Any] | None:
        if self._phase is None:
            return None
        if self._phase == "load":
            if under_load is not True:
                self._load_since, self._window = None, []
                return None
            if self._load_since is None:
                self._load_since = t
            if t - self._load_since < self.check_settle_s:
                return None
        self._window.append((t, reading.socket_w, reading.energy_j))
        if t - self._window[0][0] < self.check_window_s:
            return None
        phase, window = self._phase, self._window
        result = energy_check(window, phase, self.tolerance)
        self.checks.append({**result, "ended": iso(t)})
        self.counter_ok = result["agrees"] if phase == "start-up" else bool(self.counter_ok) and result["agrees"]
        self._phase, self._window = ("load" if phase == "start-up" else None), []
        return result

    # --- what the endpoint serves ---

    def samples_after(self, after: int, limit: int = 1000) -> dict[str, Any]:
        limit = max(1, min(int(limit), MAX_LIMIT))
        with self._lock:
            first = self._buffer[0][0] if self._buffer else None
            start = 0 if first is None else max(0, after - first + 1)
            items = [self._buffer[i] for i in range(start, min(len(self._buffer), start + limit))]
            last = self._seq
        return {"ok": True, "node": self.node, "sampler": SAMPLER, "sampler_started": iso(self.started),
                "first_seq": first, "last_seq": last, "samples": [{"seq": s, "event": e} for s, e in items]}

    def measured_period_ms(self) -> float | None:
        if self._first_t is None or self._last_t is None or self.polls < 2:
            return None
        return round((self._last_t - self._first_t) * 1000.0 / (self.polls - 1), 3)

    def status(self) -> dict[str, Any]:
        describe = getattr(self.reader, "describe", None)
        return {"ok": True, "node": self.node, "sampler": SAMPLER, "sampler_started": iso(self.started),
                "reader": describe() if callable(describe) else {"tool": self.source},
                "period_ms": self.period_ms, "measured_period_ms": self.measured_period_ms(), "polls": self.polls,
                "updates": self.updates, "errors": self.errors, "last_error": self.last_error,
                "sink_errors": self.sink_errors, "stale_seen": self.stale_seen, "counter_ok": self.counter_ok,
                "energy_checks": list(self.checks), "last_seq": self._seq}


def run_loop(sampler: PodSampler, stop: threading.Event, *, monotonic: Callable[[], float] = time.monotonic,
             max_polls: int | None = None) -> None:
    """Poll at the sampler's period until `stop` is set. A read that overruns its slot skips the missed slots
    instead of catching up in a burst, so the measured period shows the overrun."""
    period = sampler.period_ms / 1000.0
    next_t = monotonic()
    done = 0
    while not stop.is_set():
        sampler.poll()
        done += 1
        if max_polls is not None and done >= max_polls:
            return
        next_t += period
        now = monotonic()
        if next_t < now:
            next_t += math.ceil((now - next_t) / period) * period
        stop.wait(next_t - now)


class DailyJsonl:
    """Appends each sample as one line to <dir>/samples-<UTC day of the sample>.jsonl."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: Mapping[str, Any]) -> None:
        path = self.directory / f"samples-{str(event['ts'])[:10]}.jsonl"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, separators=(",", ":")) + "\n")


# --- the HTTP endpoint ------------------------------------------------------------------------------------------

def make_handler(sampler: PodSampler, clock: Callable[[], float] = time.time) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "atlas-pod-power/1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - the base class's name
            return  # nothing about requests is logged: the hub proxy forwards the account token in the headers

        def do_GET(self) -> None:  # noqa: N802 - the base class's name
            received = iso(clock())
            parts = urlsplit(self.path)
            path = parts.path.rstrip("/") or "/"
            query = parse_qs(parts.query)
            try:
                if path == "/samples":
                    body = sampler.samples_after(int(query.get("after", ["0"])[0]),
                                                 int(query.get("limit", ["1000"])[0]))
                elif path in ("/", "/status"):
                    body = sampler.status()
                else:
                    return self._send(404, {"ok": False, "error": f"no such path {path}; try /samples or /status"})
            except ValueError:
                return self._send(400, {"ok": False, "error": "after and limit must be whole numbers"})
            body["pod_received"] = received
            body["pod_sent"] = iso(clock())
            self._send(200, body)

        def _send(self, status: int, body: Mapping[str, Any]) -> None:
            data = json.dumps(body, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    return Handler


# --- window summary (what a stored row records) -----------------------------------------------------------------

def _event_key(e: Mapping[str, Any]) -> tuple:
    temps = e.get("temps_c") or {}
    return (e.get("socket_w"), e.get("gfx_mhz"), e.get("mem_mhz"), tuple(sorted(temps.items())), e.get("throttled"),
            e.get("gfx_activity_pct"), e.get("energy_counter_j"))


def _empty(e: Mapping[str, Any]) -> bool:
    return all(e.get(k) is None for k in ("socket_w", "gfx_mhz", "mem_mhz", "throttled", "gfx_activity_pct",
                                          "energy_counter_j")) and not e.get("temps_c")


def _one_or_all(values: Iterable[Any]) -> Any:
    """The one value the samples share, or all the different ones; a failed read's null says nothing here."""
    seen = sorted({json.dumps(v, sort_keys=True) for v in values if v is not None})
    out = [json.loads(v) for v in seen]
    return None if not out else out[0] if len(out) == 1 else out


def summarize(events: Iterable[Mapping[str, Any]], *, node: str | None = None, start: datetime | None = None,
              end: datetime | None = None) -> dict[str, Any]:
    """What a stored row needs from the telemetry.gpu.power samples of one node in [start, end].

    Watts are the samples' socket_w; distinct updates are counted the way the sampler counts them; the measured
    period and the longest gap come from the samples' own times, so a sampler restart or a proxy gap inside the
    window shows; energy comes from the counter only when every sample says the counter may be used, else from
    integrating power. `error` stays "unknown" until a reference run measures the sampler's error."""
    rows = []
    for e in events:
        if e.get("type") != GPU_POWER or (node is not None and e.get("node") != node):
            continue
        ts = parse_ts(str(e["ts"]))
        if (start is None or ts >= start) and (end is None or ts <= end):
            rows.append((ts, e))
    rows.sort(key=lambda item: item[0])
    if not rows:
        raise ValueError("no telemetry.gpu.power samples" + (f" for {node}" if node else "") + " in the window")
    nodes = {e.get("node") for _, e in rows}
    if len(nodes) > 1:
        raise ValueError(f"samples from {len(nodes)} nodes ({', '.join(sorted(map(str, nodes)))}); name one with node=")
    times = [ts.timestamp() for ts, _ in rows]
    gaps_ms = [(b - a) * 1000.0 for a, b in zip(times, times[1:])]
    watts = [(t, float(e["socket_w"])) for t, (_, e) in zip(times, rows) if e.get("socket_w") is not None]
    values = [w for _, w in watts]
    updates, last = 0, None
    for _, e in rows:
        if _empty(e):
            continue
        key = _event_key(e)
        if key != last:
            updates += 1
        last = key
    starts = [e.get("sampler_started") for _, e in rows]
    restarts = sum(1 for a, b in zip(starts, starts[1:]) if a != b)
    counters = [(t, e.get("energy_counter_j")) for t, (_, e) in zip(times, rows)]
    use_counter = (all(e.get("counter_ok") is True for _, e in rows) and len(rows) >= 2
                   and all(c is not None for _, c in counters)
                   and all(b >= a for (_, a), (_, b) in zip(counters, counters[1:])))
    if use_counter:
        energy_j, energy_source = counters[-1][1] - counters[0][1], "energy_counter"
    else:
        energy_j, energy_source = trapezoid_j(watts), "avg_power_integral"
    first, final = rows[0][1], rows[-1][1]
    return {
        "node": rows[0][1].get("node"),
        "start": iso(times[0]), "end": iso(times[-1]), "duration_s": round(times[-1] - times[0], 3),
        "samples": len(rows), "readings": len(values), "missing": len(rows) - len(values), "distinct_updates": updates,
        "mean_w": None if not values else round(sum(values) / len(values), 3),
        "min_w": None if not values else min(values), "max_w": None if not values else max(values),
        "sampler": _one_or_all(e.get("sampler") for _, e in rows),
        "source": _one_or_all(e.get("source") for _, e in rows),
        "power_field": _one_or_all(e.get("power_field") for _, e in rows),
        "period_ms": _one_or_all(e.get("period_ms") for _, e in rows),
        "measured_period_ms": None if not gaps_ms else round(sum(gaps_ms) / len(gaps_ms), 3),
        "max_gap_ms": None if not gaps_ms else round(max(gaps_ms), 3),
        "restarts": restarts,
        "start_temps_c": dict(first.get("temps_c") or {}), "end_temps_c": dict(final.get("temps_c") or {}),
        "under_load_samples": sum(1 for _, e in rows if e.get("under_load") is True),
        "stale_samples": sum(1 for _, e in rows if e.get("stale") is True),
        "counter_ok": final.get("counter_ok"),
        "energy_j": None if energy_j is None else round(energy_j, 3), "energy_source": energy_source,
        "error": "unknown",
    }


def read_samples(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Every JSON line of the files (pod sample files or the Jetson's event log); other event types pass through."""
    out: list[dict[str, Any]] = []
    for path in paths:
        for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{Path(path).name}:{n}: not JSON: {exc.msg}") from exc
    return out


# --- node gate --------------------------------------------------------------------------------------------------

def gate(node: str, rocminfo: Callable[[], str]) -> str | None:
    """Why this process may not sample here, or None when it may: a known node, on an R9700."""
    if node not in NODES:
        return f"--node must be one of {', '.join(NODES)}, got {node!r}"
    try:
        text = rocminfo()
    except (ReadError, OSError) as exc:
        return f"cannot tell what GPU this is: rocminfo failed ({exc})"
    if R9700_GFX not in text:
        return f"rocminfo names no {R9700_GFX}: this is not an R9700 pod"
    return None


def _rocminfo() -> str:
    program = find_tool("rocminfo")
    if program is None:
        raise ReadError("rocminfo is not installed")
    return run_command([program], timeout_s=30.0)


# --- is the sampler running -------------------------------------------------------------------------------------

def is_sampler_cmdline(cmdline: bytes) -> bool:
    """Whether a /proc/<pid>/cmdline (NUL-separated arguments) is pod_power.py serve, under any interpreter."""
    args = [part.decode("utf-8", errors="replace") for part in cmdline.split(b"\0")]
    return any(PurePosixPath(arg).name == SAMPLER_SCRIPT and following == SAMPLER_SERVE
               for arg, following in zip(args, args[1:]))


def alive_sampler_pid(pidfile: Path, proc: Path = PROC_ROOT) -> int | None:
    """The pid in `pidfile` when that process is a running `pod_power.py serve`, else None.

    None for a missing pid file, one that holds no positive whole number, a pid with no /proc entry, or a pid whose
    command line is any other program (the number was reused after a relaunch). The answer comes from
    proc/<pid>/cmdline only; the pid is never signalled. A pid file or /proc entry that exists but cannot be read
    raises OSError, so the caller can tell "cannot tell" from "not running"."""
    try:
        text = Path(pidfile).read_text(encoding="ascii", errors="replace").strip()
    except FileNotFoundError:
        return None
    if not text.isdigit():
        return None
    pid = int(text)
    if pid <= 0:
        return None
    try:
        cmdline = (Path(proc) / str(pid) / "cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError):  # no such process, or it exited while being read
        return None
    return pid if is_sampler_cmdline(cmdline) else None


def alive(pidfile: Path, proc: Path = PROC_ROOT) -> int:
    """The `alive` subcommand: print the sampler's pid and return ALIVE_EXIT_YES, or return ALIVE_EXIT_NO, or
    say on stderr what could not be read and return ALIVE_EXIT_UNKNOWN."""
    try:
        pid = alive_sampler_pid(pidfile, proc)
    except OSError as exc:
        print(f"cannot tell whether {pidfile} names a running sampler: {type(exc).__name__}: {exc}", file=sys.stderr)
        return ALIVE_EXIT_UNKNOWN
    if pid is None:
        return ALIVE_EXIT_NO
    print(pid)
    return ALIVE_EXIT_YES


# --- command line -----------------------------------------------------------------------------------------------

def check(gpu: int = 0, save: str | Path | None = None, run: Callable[[Sequence[str]], str] = run_command,
          which: Callable[[str], str | None] = shutil.which) -> dict[str, Any]:
    """One read with every reader, the one serve would choose, and (with save) each tool's raw output on disk."""
    folder = Path(save).expanduser() if save else None
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
    raw: dict[str, str] = {}

    def keep(argv: Sequence[str]) -> str:
        text = run(argv)
        raw[" ".join([Path(argv[0]).name, *argv[1:]])] = text
        return text

    attempts: list[dict[str, Any]] = []
    for reader in candidates(gpu, keep if folder else run, which):
        if isinstance(reader, dict):
            attempts.append(reader)
            continue
        try:
            reading = reader.read()
            attempts.append({**reader.describe(), "reading": {**reading.__dict__, "temps_c": dict(reading.temps_c)}})
        except (ReadError, OSError) as exc:
            attempts.append({**reader.describe(), "error": str(exc)})
    chosen = next((a for a in attempts if (a.get("reading") or {}).get("socket_w") is not None), None)
    saved = []
    if folder is not None:
        for command, text in raw.items():
            name = re.sub(r"[^A-Za-z0-9]+", "_", command).strip("_") + ".txt"
            (folder / name).write_text(text, encoding="utf-8")
            saved.append(name)
    return {"ok": chosen is not None, "chosen": None if chosen is None else {"tool": chosen["tool"],
                                                                             "options": chosen["options"]},
            "attempts": attempts, "saved": saved}


def serve(args: argparse.Namespace, *, rocminfo: Callable[[], str] = _rocminfo) -> int:
    refused = gate(args.node, rocminfo)
    if refused:
        print(json.dumps({"ok": False, "outcome": "refused", "error": refused}))
        return 2
    reader, attempts = choose_reader(args.gpu)
    if reader is None:
        print(json.dumps({"ok": False, "outcome": "no_reader", "error": "no tool read a power number",
                          "attempts": attempts}))
        return 1
    sampler = PodSampler(reader, args.node, period_ms=args.period_ms, stale_after_s=args.stale_after_s,
                         check_window_s=args.check_window_s, check_settle_s=args.check_settle_s,
                         tolerance=args.tolerance, sink=DailyJsonl(args.dir))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(sampler))
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    threading.Thread(target=server.serve_forever, name="pod-power-http", daemon=True).start()
    print(json.dumps({"ok": True, "outcome": "serving", "node": args.node, "port": args.port, "dir": str(args.dir),
                      "reader": reader.describe(), "period_ms": args.period_ms, "attempts": attempts}), flush=True)
    if args.minutes:
        threading.Timer(args.minutes * 60.0, stop.set).start()
    try:
        run_loop(sampler, stop)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        print(json.dumps({"ok": True, "outcome": "stopped", **{k: v for k, v in sampler.status().items()
                                                               if k not in ("ok", "energy_checks")}}), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pod_power.py", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("check", help="one read with each tool and option set; nothing is kept unless --save")
    one.add_argument("--gpu", type=int, default=0)
    one.add_argument("--save", help="folder for each tool's raw output (fixtures for the parsers)")
    run = sub.add_parser("serve", help="sample at the period and serve the samples on a local port")
    run.add_argument("--node", required=True, choices=NODES)
    run.add_argument("--gpu", type=int, default=0)
    run.add_argument("--period-ms", type=float, default=DEFAULT_PERIOD_MS)
    run.add_argument("--host", default=DEFAULT_HOST)
    run.add_argument("--port", type=int, default=DEFAULT_PORT)
    run.add_argument("--dir", default=DEFAULT_DIR, help="where the daily sample files go")
    run.add_argument("--stale-after-s", type=float, default=STALE_AFTER_S)
    run.add_argument("--check-window-s", type=float, default=CHECK_WINDOW_S)
    run.add_argument("--check-settle-s", type=float, default=CHECK_SETTLE_S)
    run.add_argument("--tolerance", type=float, default=CHECK_TOLERANCE)
    run.add_argument("--minutes", type=float, default=0.0, help="stop after this long (0: run until stopped)")
    up = sub.add_parser("alive", help="print the pid and exit 0 only if the pid file names a running pod_power.py "
                                      "serve; exit 1 if it names none, 2 if that cannot be read")
    up.add_argument("--pidfile", required=True, type=Path, help="the launcher's sampler.pid")
    up.add_argument("--proc", type=Path, default=PROC_ROOT, help="the proc filesystem to look in (tests)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check":
        result = check(args.gpu, args.save)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    if args.command == "alive":
        return alive(args.pidfile, args.proc)
    return serve(args)


if __name__ == "__main__":
    sys.exit(main())
