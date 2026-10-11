"""The benchmark harness: decode tokens per second and joules per generated token, one cold server spawn per run.

On a pod (R20 runs it on pod B; the repo clone and `pip install --user httpx` are all it needs):

    python lab/bench.py run --engine vllm --checkpoint ~/models/Qwen3.8-27B-INT4 --engine-dir ~/vllm \\
        --patch lab/vllm-gfx1201.diff --node pod-b [--batches 1,4,8,16] [--repeats 5] [--metrics-file]
    python lab/bench.py run --engine llamacpp --checkpoint ~/models/student.gguf --node pod-b ...

From the box, against a server already running (no spawn, no process change on the pod):

    python lab/bench.py attach --account C [--port 8000] --batches 1 --requests-per-stream 4 --warmup 0 --no-probe
    python lab/bench.py attach --url http://127.0.0.1:8000 [--pid N --serve-log PATH]

`python -m lab bench ...` is the same. Every call prints one JSON line and exits 0 when it worked.

What a run is. One cold server spawn (`run`): the harness reads the idle watts with no server, starts the engine with
the launch command (vLLM's from lab/pod_a_teacher.sh, or llama-server's, or --serve-cmd), waits for /v1/models,
reads the serve log and the running process, reads the loaded-idle watts, sends the warm-up requests (one per prompt,
then one round at the batch size; discarded from every number and listed in the row under `warmup`), sends one probe
request alone (exactly one request decoding: the single-stream rate and mem_busy_percent come from it), then the
measured requests, and stops the server. Repeats are spawns: `--repeats 5` at batch sizes 1, 4, 8 and 16 is twenty
spawns, each written as one JSON row (`--share-spawn` measures every batch size in one spawn per repeat instead,
and its rows say so). Batch sizes default to 1, 4, 8 and 16 (kernels rubric K1), plus 64 when it fits in memory
(owner ruling 5 in lab/refine/kernels/OWNER_RULINGS.md): the 64 arm runs only when the KV-cache capacity the serve
log reports holds 64 requests of the workload (the longest prompt the warm-up saw plus max_tokens), and is otherwise
a `skipped` row with the reason. `--batches` asks for any others.

The prompt set (fixtures/lab/bench/prompts.json) is text and drawn fixture images only, never camera frames; every
request is greedy (temperature 0, top_p 1, seed 0), has a fixed max_tokens with ignore_eos so each one generates the
same number of tokens, asks the Qwen chat template for no thinking (`--thinking`), and starts with a per-request nonce
whenever prefix caching may be on (vLLM V1 and llama-server both cache prefixes by default; `--prefix-caching off`
drops it), so time to first token never measures a cache hit. Batch size B means B concurrent streams, each sending
--requests-per-stream requests one after another (default: one per prompt), stream s starting at prompt s.

What is measured, from the client's own token stream (never the server's latency histogram): time to first token
(request sent to first content chunk) and the decode rate of each request ((tokens - 1) / (last chunk - first chunk),
tokens from the usage chunk). A stream whose chunks arrived all at once (a proxy that held the reply back) gives no
decode rate and is counted as buffered. A row's decode_tok_s is the median per-request rate, decode_tok_s_total the
tokens generated over the decode window (first token of the first request to last token of the last), in_flight_mean
how many requests were decoding on average over it. Watts are the pod sampler's amd-smi SOCKET_POWER (lab/pod_power.py:
the same reader choice and parser as T3, polled here every --power-period-s; or the T6 sampler on port 8790 through
the hub proxy in `attach`), integrated over the decode window with its edges interpolated: average and peak watts,
energy, and joules per generated token, with the idle readings (no server; server loaded) beside it and the joules per
token above the loaded idle. Whether amd-smi reported throttling is the window's THROTTLE_STATUS (null when the chip
reads N/A, as the Strix Halo does).

What every row carries (the kernels rubric test 1 conditions, K1 b): engine, engine commit (git in --engine-dir, or
llama-server's build line) and the sha256 of each patch, checkpoint, quantization, max model length, batch size, node,
ROCm, torch and Triton versions (asked of --python on the pod), the environment read back from the running server's
/proc/<pid>/environ (engine, ROCm and HIP variables only, anything named like a secret left out, GPU_MAX_HW_QUEUES
always present and null when unset; `env_meant` is what the launch asked for), the linear and attention kernel
classes and the KV-cache capacity from the serve log; the host as far as the pod can read it (hub and account,
hostname, kernel release, amdgpu driver from `amd-smi static --driver`, firmware from `amd-smi firmware`, ASIC serial
and the chip from `amd-smi static --asic`; amd-smi is found under /opt/rocm/bin when it is off PATH); the KFD queue
count from /sys/class/kfd/kfd/proc/<pid>/queues where the pod exposes it; the time since spawn; mem_busy_percent
(/sys/class/drm/card*/device) sampled while the probe request decodes alone; and the decode settings and prompt-set
hash. Anything that could not be read is null with the reason under `unavailable`. The harness is chip-agnostic: it
reads the chip from amd-smi and never checks it.

Repeats and bands. Each configuration (engine and batch size) reports its per-spawn values with median, min, max,
mean and sd. The spawns are put in bands by their own probe decode rates: sorted, split at the largest gap when that
gap is at least --band-gap (10 %) of the median rate, recursively, with at least 4 spawns to split at all; between
two bands the rate threshold and the mem_busy_percent threshold (midway between the bands' median readings) are
worked out from those clusters, never fixed in advance, with whether that threshold separates them. Each
configuration then reports the share of spawns in each band and the statistics per band.

Where rows go. <out>/<session>/rows.jsonl (one row per run, written as each run ends) and summary.json; default out
lab/state/bench. Each run is a `bench.run` ledger row (lab/machine/record.py's record(): the metrics, node, units,
conditions, low_confidence true for a single spawn), the whole call is the `bench` row that @recorded writes, and
with --metrics-file each configuration's medians go to lab/refine/metrics.jsonl (or the file named) with their
spread and bands in the conditions, low_confidence while a configuration has fewer than verdict()'s 5 spawns.

Spawning on pod C would collide with the teacher it serves (port 8000, and most of the 64 GB): the harness refuses
a port that something already listens on, defaults to 8100, and the spawn-per-run measurements are R20's on pod B.
Tested against lab/bench_fake.py (a fake server spawned as a real process, and a fake pod).
Every wait is bounded: readiness by the ready timeout; each read of a request by --request-timeout-s; a batch's
streams by a grace period plus --request-timeout-s per request each stream sends (a stream that has not finished by
then fails its run, the row keeps the warm-ups and the probe that ran, and the runs left on the same server, a shared
spawn or an attached one, are written as not measured, since the stuck streams may still be decoding there; a
spawned server is still stopped). With --port 0 the launch command must carry
{port_file}, where the server writes the port it bound (the fake does; vLLM and llama-server take a fixed port).
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

if not __package__:  # run by path on a pod: the repo root, one folder up, holds lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import pod_power  # noqa: E402
from lab.machine import metrics as metrics_file  # noqa: E402
from lab.machine.record import record, recorded  # noqa: E402
from lab.machine.verdict import MIN_RUNS_PER_ARM  # noqa: E402

HARNESS = "lab/bench.py 1"
ROW_SCHEMA = "lab.bench.row/1"
REPO = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = REPO / "fixtures" / "lab" / "bench" / "prompts.json"
DEFAULT_OUT = REPO / "lab" / "state" / "bench"
DEFAULT_BATCHES = (1, 4, 8, 16)
EXTRA_BATCH = 64            # owner ruling 5: added when it fits in memory
DEFAULT_PORT = 8100         # not the teacher's 8000
DEFAULT_MAX_TOKENS = 128
DEFAULT_REPEATS = MIN_RUNS_PER_ARM
DEFAULT_IDLE_S = 10.0
DEFAULT_POWER_PERIOD_S = 0.5
DEFAULT_READY_TIMEOUT_S = 900.0
DEFAULT_REQUEST_TIMEOUT_S = 900.0   # one read of a streamed request; a batch's streams get this per request
DEFAULT_STREAM_GRACE_S = 60.0       # added once to a batch's bound, for the streams to start and join
MODELS_TIMEOUT_S = 10.0             # GET /v1/models while waiting for readiness
DEFAULT_BAND_GAP = 0.10
MIN_SPAWNS_TO_BAND = 4
SAMPLER_PORT = pod_power.DEFAULT_PORT   # the T6 sampler's port (8790)
BUFFERED_GAP_S = 0.001      # chunk gaps shorter than this count as arriving together
ENGINES = ("vllm", "llamacpp")
SAMPLER_NODE = {"pod-a": "gpu-a", "pod-c": "gpu-a", "pod-b": "gpu-b", "pod-d": "gpu-b"}
ENV_PREFIXES = ("GPU_", "HIP_", "HSA_", "ROC", "AMD_", "VLLM_", "TORCH", "PYTORCH_", "TRITON_", "MIOPEN_", "OMP_",
                "LLAMA_", "GGML_", "HF_", "CUDA_", "NCCL_", "RCCL_", "MAX_JOBS", "LD_LIBRARY_PATH")
SECRET_WORDS = ("TOKEN", "SECRET", "PASS", "KEY", "AUTH", "COOKIE", "CREDENTIAL")
ALWAYS_ENV = ("GPU_MAX_HW_QUEUES",)
METRIC_UNITS = {"decode_tok_s": "tok/s", "decode_tok_s_total": "tok/s", "ttft_s": "s", "joules_per_token": "J/token",
                "watts_avg": "W", "watts_peak": "W"}
VERSION_SCRIPT = ("import json\nout = {}\nfor name in ('torch', 'triton'):\n    try:\n        m = __import__(name)\n"
                  "        out[name] = m.__version__\n    except Exception as exc:\n        out[name] = None\n"
                  "try:\n    import torch\n    out['hip'] = torch.version.hip\nexcept Exception:\n    out['hip'] = None\n"
                  "import platform\nout['python'] = platform.python_version()\nprint(json.dumps(out))\n")


class BenchError(Exception):
    """A run that cannot go on: bad arguments, a port in use, a server that never came up."""


class StuckRun(BenchError):
    """A run whose streams did not finish within their bound; `row` is its fail row with what had already run."""

    def __init__(self, message: str, row: dict[str, Any]) -> None:
        super().__init__(message)
        self.row = row


class HostError(Exception):
    """A command on the host did not answer."""


def log(message: str) -> None:
    print(f"[bench {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def utc(t: float | None = None) -> str:
    return pod_power.iso(time.time() if t is None else t)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- the prompt set ---------------------------------------------------------------------------------------------

@dataclass
class Settings:
    """The decode settings every request carries. prefix_caching None means it may be on (the engines' default)."""

    model: str
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.0
    ignore_eos: bool = True
    thinking: bool | None = False
    prefix_caching: bool | None = None

    @property
    def nonce_on(self) -> bool:
        return self.prefix_caching is not False


@dataclass
class PromptSet:
    id: str
    path: Path
    prompts: list[dict[str, Any]]
    images: dict[str, bytes]
    sha256: str

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PROMPTS) -> "PromptSet":
        path = Path(path).resolve()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BenchError(f"cannot read the prompt set {path}: {exc}") from exc
        folder = path.parent
        prompts = data.get("prompts")
        if not isinstance(prompts, list) or not prompts:
            raise BenchError(f"{path.name} has no prompts")
        digest = hashlib.sha256(path.read_bytes())
        images: dict[str, bytes] = {}
        for p in prompts:
            kind = p.get("kind")
            if kind not in ("text", "image"):
                raise BenchError(f"prompt {p.get('id')!r} is a {kind!r}: the set is text and fixture images only "
                                 "(raw camera frames never leave the Jetson)")
            if not isinstance(p.get("user"), str) or not p["user"].strip():
                raise BenchError(f"prompt {p.get('id')!r} has no user text")
            if kind == "image":
                image = (folder / str(p.get("image", ""))).resolve()
                if not image.is_relative_to(folder) or image.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    raise BenchError(f"prompt {p.get('id')!r}: {p.get('image')!r} is not a fixture image beside "
                                     f"{path.name}")
                try:
                    images[p["image"]] = image.read_bytes()
                except OSError as exc:
                    raise BenchError(f"prompt {p.get('id')!r}: cannot read {p.get('image')}: {exc}") from exc
                digest.update(images[p["image"]])
        return cls(str(data.get("id") or path.stem), path, prompts, images, digest.hexdigest())

    def describe(self) -> dict[str, Any]:
        return {"id": self.id, "sha256": self.sha256, "prompts": len(self.prompts),
                "images": sum(1 for p in self.prompts if p["kind"] == "image")}

    def messages(self, index: int, nonce: str | None) -> list[dict[str, Any]]:
        p = self.prompts[index % len(self.prompts)]
        system = "\n".join(part for part in (f"[run {nonce}]" if nonce else "", p.get("system") or "") if part)
        out: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
        if p["kind"] == "image":
            raw = self.images[p["image"]]
            mime = "image/png" if p["image"].lower().endswith(".png") else "image/jpeg"
            url = f"data:{mime};base64,{base64.b64encode(raw).decode()}"
            out.append({"role": "user", "content": [{"type": "text", "text": p["user"]},
                                                    {"type": "image_url", "image_url": {"url": url}}]})
        else:
            out.append({"role": "user", "content": p["user"]})
        return out

    def request_body(self, index: int, settings: Settings, nonce: str | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": settings.model, "messages": self.messages(index, nonce if settings.nonce_on else None),
            "max_tokens": settings.max_tokens, "temperature": settings.temperature, "top_p": 1.0, "seed": 0,
            "ignore_eos": settings.ignore_eos, "stream": True, "stream_options": {"include_usage": True}}
        if settings.thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": bool(settings.thinking)}
        return body

    def prompt_id(self, index: int) -> str:
        return str(self.prompts[index % len(self.prompts)].get("id"))


# --- batch sizes ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PlannedBatch:
    size: int
    when_fits: bool = False


def plan_batches(requested: Sequence[int] | None, *, add_64: bool) -> list[PlannedBatch]:
    """The batch sizes to run: the request, else 1, 4, 8 and 16; plus 64 (run only when it fits) when add_64."""
    sizes = list(DEFAULT_BATCHES if not requested else requested)
    for size in sizes:
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise BenchError(f"a batch size is a whole number of at least 1, not {size!r}")
    plan = [PlannedBatch(size) for size in sizes]
    if add_64 and EXTRA_BATCH not in sizes:
        plan.append(PlannedBatch(EXTRA_BATCH, when_fits=True))
    return plan


def fits(batch: int, *, kv_cache_tokens: int | None, tokens_per_request: int | None) -> bool | None:
    """Does the KV cache hold `batch` requests of the workload at once? None when either number is unknown."""
    if kv_cache_tokens is None or tokens_per_request is None:
        return None
    return kv_cache_tokens >= batch * tokens_per_request


# --- the client token stream ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Timing:
    ttft_s: float | None
    decode_tok_s: float | None
    buffered: bool


def stream_timing(*, sent: float, chunk_times: Sequence[float], tokens: int) -> Timing:
    """TTFT and decode rate of one request from when its content chunks arrived (any one clock)."""
    if not chunk_times:
        return Timing(None, None, False)
    ttft = chunk_times[0] - sent
    gaps = [b - a for a, b in zip(chunk_times, chunk_times[1:])]
    if not gaps:
        return Timing(ttft, None, False)
    spread = sum(1 for g in gaps if g > BUFFERED_GAP_S)
    span = chunk_times[-1] - chunk_times[0]
    if spread * 2 < len(gaps) or span <= BUFFERED_GAP_S:
        return Timing(ttft, None, True)
    return Timing(ttft, (tokens - 1) / span if tokens > 1 else None, False)


class WallClock:
    """perf_counter for durations, anchored once to the wall clock so power samples and requests line up."""

    def __init__(self) -> None:
        self.perf0 = time.perf_counter()
        self.wall0 = time.time()

    def wall(self, perf: float) -> float:
        return self.wall0 + (perf - self.perf0)


@dataclass
class RequestResult:
    role: str                      # warmup, probe or measure
    stream: int
    index: int
    prompt_id: str
    sent: float                    # wall-clock seconds
    first: float | None = None
    last: float | None = None
    chunks: int = 0
    tokens: int = 0
    prompt_tokens: int | None = None
    finish_reason: str | None = None
    ttft_s: float | None = None
    decode_tok_s: float | None = None
    buffered: bool = False
    error: str | None = None
    cached_tokens: int | None = None   # usage.prompt_tokens_details.cached_tokens: a prefix-cache hit when > 0

    def as_dict(self) -> dict[str, Any]:
        return {"role": self.role, "stream": self.stream, "index": self.index, "prompt": self.prompt_id,
                "sent": utc(self.sent), "first_token": None if self.first is None else utc(self.first),
                "last_token": None if self.last is None else utc(self.last), "chunks": self.chunks,
                "tokens": self.tokens, "prompt_tokens": self.prompt_tokens, "finish_reason": self.finish_reason,
                "ttft_s": _r(self.ttft_s), "decode_tok_s": _r(self.decode_tok_s), "buffered": self.buffered,
                "cached_prompt_tokens": self.cached_tokens, "error": self.error}


class DirectEndpoint:
    """A server reached directly (on the pod: 127.0.0.1:<port>)."""

    def __init__(self, base_url: str, *, timeout_s: float = 900.0) -> None:
        import httpx  # the pod needs `pip install --user httpx`; only the endpoints use it

        self.base_url = base_url.rstrip("/")
        self.label = self.base_url
        self._client = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(timeout_s, connect=10.0))

    def models(self) -> list[dict[str, Any]]:
        reply = self._client.get("/v1/models", timeout=MODELS_TIMEOUT_S)
        reply.raise_for_status()
        return list(reply.json().get("data") or [])

    def events(self, body: Mapping[str, Any]) -> Iterator[tuple[float, str]]:
        from lab.proxy import parse_sse

        headers = {"Accept": "text/event-stream", "Accept-Encoding": "identity"}
        with self._client.stream("POST", "/v1/chat/completions", json=dict(body), headers=headers) as response:
            if response.status_code != 200:
                response.read()
                raise BenchError(f"{self.label} answered {response.status_code}: {response.text[:200]}")
            for event in parse_sse(response.iter_lines(), time.perf_counter):
                yield event.t, event.data

    def close(self) -> None:
        self._client.close()


class ProxyEndpoint:
    """A server on a pod port reached through the hub proxy (lab/proxy.py's PodLink, token and streaming rules)."""

    def __init__(self, link: Any, port: int = 8000) -> None:
        self.link = link
        self.port = port
        self.label = f"pod {link.letter} port {port} through the hub proxy"

    def models(self) -> list[dict[str, Any]]:
        reply = self.link.server(self.port).get("v1/models")
        return list(reply.json().get("data") or [])

    def events(self, body: Mapping[str, Any]) -> Iterator[tuple[float, str]]:
        from lab.proxy import parse_sse

        with self.link.stream("POST", self.link.proxy_rel(self.port, "v1/chat/completions"), json=dict(body)) as h:
            for event in parse_sse(h.response.iter_lines(), time.perf_counter):
                yield event.t, event.data

    def close(self) -> None:
        return None


def run_request(endpoint: Any, body: Mapping[str, Any], *, role: str, stream: int, index: int, prompt_id: str,
                clock: WallClock, on_token: Callable[[], None] | None = None) -> RequestResult:
    """Send one streamed chat completion and time it from its chunks. on_token is called as each content chunk
    arrives (the probe reads mem_busy_percent there, so every reading falls inside the decode)."""
    sent = time.perf_counter()
    times: list[float] = []
    usage: dict[str, Any] = {}
    finish = None
    error = None
    try:
        for t, data in endpoint.events(body):
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                error = "a stream event that is not JSON"
                break
            if not isinstance(chunk, dict):
                continue
            if "error" in chunk and not chunk.get("choices"):
                error = f"the server streamed an error: {str(chunk['error'])[:200]}"
                break
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    times.append(t)
                    if on_token is not None:
                        on_token()
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    except Exception as exc:  # one failed request is a row's reason, never a crash of the session
        error = f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}"
    tokens = int(usage.get("completion_tokens") or len(times))
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    timing = stream_timing(sent=sent, chunk_times=times, tokens=tokens)
    if error is None and not times:
        error = "no token arrived"
    return RequestResult(role, stream, index, prompt_id, clock.wall(sent),
                         clock.wall(times[0]) if times else None, clock.wall(times[-1]) if times else None,
                         len(times), tokens, usage.get("prompt_tokens"), finish, timing.ttft_s, timing.decode_tok_s,
                         timing.buffered, error, cached)


def run_batch(endpoint: Any, prompts: PromptSet, settings: Settings, *, batch: int, per_stream: int, role: str,
              clock: WallClock, tag: str, timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
              grace_s: float = 60.0) -> list[RequestResult]:
    """batch concurrent streams, each sending per_stream requests one after another; stream s starts at prompt s.
    Every wait is bounded: streams not finished after grace_s plus per_stream x timeout_s raise BenchError."""
    results: list[RequestResult] = []
    lock = threading.Lock()
    barrier = threading.Barrier(batch)

    def stream(s: int) -> None:
        try:
            barrier.wait(timeout=grace_s + timeout_s)
        except threading.BrokenBarrierError:
            return
        for k in range(per_stream):
            index = s + k
            nonce = f"{tag}-{role}-{s}-{k}-{uuid.uuid4().hex[:8]}" if settings.nonce_on else None
            body = prompts.request_body(index, settings, nonce)
            out = run_request(endpoint, body, role=role, stream=s, index=index, prompt_id=prompts.prompt_id(index),
                              clock=clock)
            with lock:
                results.append(out)

    threads = [threading.Thread(target=stream, args=(s,), daemon=True) for s in range(batch)]
    for t in threads:
        t.start()
    limit = grace_s + per_stream * timeout_s
    deadline = time.monotonic() + limit
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    stuck = sum(1 for t in threads if t.is_alive())
    if stuck:
        raise BenchError(f"{stuck} of {batch} {role} streams had not finished after {limit:g} s")
    if len(results) != batch * per_stream:
        raise BenchError(f"{batch * per_stream - len(results)} {role} requests never started (a stream did not join)")
    return sorted(results, key=lambda r: (r.stream, r.index))


def run_sequential(endpoint: Any, prompts: PromptSet, settings: Settings, *, role: str, clock: WallClock,
                   tag: str) -> list[RequestResult]:
    """One request per prompt, one at a time."""
    out = []
    for i in range(len(prompts.prompts)):
        nonce = f"{tag}-{role}-seq-{i}-{uuid.uuid4().hex[:8]}" if settings.nonce_on else None
        out.append(run_request(endpoint, prompts.request_body(i, settings, nonce), role=role, stream=0, index=i,
                               prompt_id=prompts.prompt_id(i), clock=clock))
    return out


# --- watts ------------------------------------------------------------------------------------------------------

def watt_points(events: Iterable[Mapping[str, Any]]) -> list[tuple[float, float]]:
    """(wall seconds, socket watts) of telemetry.gpu.power samples; `t_local` (set by RemotePower) wins over ts."""
    out = []
    for e in events:
        w = e.get("socket_w")
        if w is None:
            continue
        t = e.get("t_local")
        if t is None:
            t = pod_power.parse_ts(str(e["ts"])).timestamp()
        out.append((float(t), float(w)))
    return sorted(out)


def power_window(points: Sequence[tuple[float, float]], start: float, end: float, *,
                 period_s: float) -> dict[str, Any]:
    """Watts over [start, end]: energy by the trapezoid rule (pod_power.trapezoid_j) with the edges interpolated,
    the time-weighted average and the peak. Unavailable when the samples do not reach both edges."""
    pts = sorted(points)
    tolerance = 2.0 * period_s + 0.25   # a late or skipped read near an edge still covers it
    if end < start:
        return {"available": False, "reason": "the decode window is empty"}
    if not pts or pts[0][0] > start + tolerance or pts[-1][0] < end - tolerance:
        return {"available": False, "reason": "the power samples do not cover the decode window"}

    def at(x: float) -> float:
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1]
        for (t1, w1), (t2, w2) in zip(pts, pts[1:]):
            if t1 <= x <= t2:
                return w1 if t2 == t1 else w1 + (w2 - w1) * (x - t1) / (t2 - t1)
        return pts[-1][1]

    inner = [(t, w) for t, w in pts if start < t < end]
    seq = [(start, at(start)), *inner, (end, at(end))]
    duration = end - start
    energy = pod_power.trapezoid_j(seq) or 0.0
    return {"available": True, "reason": None, "avg_w": energy / duration if duration > 0 else at(start),
            "peak_w": max(w for _, w in seq), "energy_j": energy, "duration_s": duration,
            "samples": len(inner)}


def throttled(events: Iterable[Mapping[str, Any]]) -> bool | None:
    """True when any sample read THROTTLED, False when they read UNTHROTTLED, None when none said (N/A)."""
    seen = [e.get("throttled") for e in events if e.get("throttled") is not None]
    return None if not seen else any(seen)


def _one(values: Iterable[Any]) -> Any:
    found = sorted({json.dumps(v, sort_keys=True) for v in values if v is not None})
    return None if not found else json.loads(found[0]) if len(found) == 1 else [json.loads(v) for v in found]


class NoPower:
    available = False

    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.period_s = DEFAULT_POWER_PERIOD_S

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def events(self) -> list[dict[str, Any]]:
        return []


class LocalPower:
    """amd-smi read on this host by lab/pod_power.py's reader and PodSampler, in a thread, for the whole session."""

    def __init__(self, reader: Any, *, node: str, period_s: float, reason: str | None = None) -> None:
        self.reader = reader
        self.period_s = period_s
        self.available = reader is not None
        self.reason = reason
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.sampler = None
        if reader is not None:
            self.sampler = pod_power.PodSampler(reader, SAMPLER_NODE.get(node, "gpu-b"), period_ms=period_s * 1000.0,
                                                sink=self._keep)

    @classmethod
    def for_host(cls, host: "Host", *, node: str, period_s: float) -> "LocalPower":
        reader, attempts = pod_power.choose_reader(run=host.run_text, which=host.which)
        reason = None if reader is not None else "no power reader on this host: " + json.dumps(attempts)[:300]
        return cls(reader, node=node, period_s=period_s, reason=reason)

    def _keep(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(event)

    def start(self) -> None:
        if self.sampler is None or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=pod_power.run_loop, args=(self.sampler, self._stop), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(10)
            self._thread = None

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)


class RemotePower:
    """The T6 sampler on a pod port through the hub proxy: GET /status at start, /samples?after=<seq> since then.
    Pod-clock times are moved to this clock by the offset measured from each answer's pod_sent."""

    def __init__(self, link: Any, port: int = SAMPLER_PORT, *, period_s: float = 1.0) -> None:
        self.link = link
        self.port = port
        self.period_s = period_s
        self.available = False
        self.reason: str | None = "not started"
        self._after = 0
        self._offset = 0.0
        self._events: list[dict[str, Any]] = []

    def _get(self, path: str, **params: Any) -> dict[str, Any]:
        before = time.time()
        reply = self.link.request("GET", self.link.proxy_rel(self.port, path), params=params or None)
        after = time.time()
        body = reply.json()
        if body.get("pod_sent"):
            self._offset = pod_power.parse_ts(str(body["pod_sent"])).timestamp() - (before + after) / 2.0
        return body

    def start(self) -> None:
        try:
            status = self._get("status")
        except Exception as exc:
            self.available = False
            self.reason = (f"the T6 sampler on port {self.port} did not answer through the proxy "
                           f"({type(exc).__name__}: {' '.join(str(exc).split())[:200]})")
            return
        self._after = int(status.get("last_seq") or 0)
        self.period_s = float(status.get("period_ms") or self.period_s * 1000.0) / 1000.0
        self.available, self.reason = True, None

    def stop(self) -> None:
        return None

    def events(self) -> list[dict[str, Any]]:
        if not self.available:
            return []
        try:
            while True:
                page = self._get("samples", after=self._after, limit=pod_power.MAX_LIMIT)
                items = page.get("samples") or []
                for item in items:
                    event = dict(item["event"])
                    event["t_local"] = pod_power.parse_ts(str(event["ts"])).timestamp() - self._offset
                    self._events.append(event)
                    self._after = max(self._after, int(item["seq"]))
                if len(items) < pod_power.MAX_LIMIT:
                    break
        except Exception as exc:
            self.reason = f"reading samples from port {self.port} failed: {type(exc).__name__}"
        return list(self._events)


def window_watts(power: Any, start: float, end: float) -> dict[str, Any]:
    """The watts entry of a row: power_window over the samples, plus what read them."""
    if not power.available:
        return {"available": False, "reason": power.reason, "avg_w": None, "peak_w": None, "energy_j": None,
                "samples": 0, "source": None, "power_field": None, "sampler": None, "period_s": None}
    events = power.events()
    out = power_window(watt_points(events), start, end, period_s=power.period_s)
    inside = [e for e in events if start - power.period_s <= _event_t(e) <= end + power.period_s]
    return {"available": out["available"], "reason": out.get("reason"), "avg_w": _r(out.get("avg_w")),
            "peak_w": _r(out.get("peak_w")), "energy_j": _r(out.get("energy_j")), "samples": out.get("samples", 0),
            "source": _one(e.get("source") for e in inside), "power_field": _one(e.get("power_field") for e in inside),
            "sampler": _one(e.get("sampler") for e in inside), "period_s": power.period_s,
            "throttled": throttled(inside)}


def _event_t(e: Mapping[str, Any]) -> float:
    t = e.get("t_local")
    return float(t) if t is not None else pod_power.parse_ts(str(e["ts"])).timestamp()


def idle_watts(power: Any, seconds: float) -> float | None:
    """Average watts over `seconds`, with nothing sent. The window opens one sample period after the call, so its
    first edge is interpolated from a sample taken after whatever ran before (a previous spawn's last decode, a
    server's start) had ended, never blended with it."""
    if not power.available or seconds <= 0:
        return None
    start = time.time() + power.period_s
    time.sleep(seconds + power.period_s)
    out = power_window(watt_points(power.events()), start, time.time(), period_s=power.period_s)
    return _r(out.get("avg_w")) if out["available"] else None


# --- the host: amd-smi, /proc, /sys ------------------------------------------------------------------------------

class Host:
    """The machine the server runs on, as the harness reads it: commands (run) and files under root."""

    def __init__(self, root: str | Path = "/", run: Callable[[Sequence[str]], str] | None = None,
                 which: Callable[[str], str | None] = shutil.which,
                 environ: Mapping[str, str] | None = None) -> None:
        self.root = Path(root)
        self.run = run or pod_power.run_command
        self.which = which
        self.environ = dict(os.environ if environ is None else environ)

    def run_text(self, argv: Sequence[str]) -> str:
        try:
            return self.run(list(argv))
        except (HostError, pod_power.ReadError, OSError) as exc:
            raise pod_power.ReadError(str(exc)) from exc

    def path(self, posix: str) -> Path:
        return self.root / PurePosixPath(posix).relative_to("/")

    def read_text(self, posix: str) -> str | None:
        try:
            return self.path(posix).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def read_bytes(self, posix: str) -> bytes | None:
        try:
            return self.path(posix).read_bytes()
        except OSError:
            return None

    def listdir(self, posix: str) -> list[str] | None:
        try:
            return sorted(p.name for p in self.path(posix).iterdir())
        except OSError:
            return None

    def tool(self, name: str) -> str | None:
        return pod_power.find_tool(name, self.which)

    def command(self, argv: Sequence[str]) -> str:
        try:
            return self.run(list(argv))
        except (pod_power.ReadError, OSError) as exc:
            raise HostError(str(exc)) from exc

    def amdsmi(self, *args: str) -> str:
        program = self.tool(pod_power.AMDSMI)
        if program is None:
            raise HostError("amd-smi is not on PATH or under /opt/rocm/bin")
        return self.command([program, *args])


def _device(text: str, gpu: int = 0) -> Mapping[str, Any]:
    return pod_power.amdsmi_tree(text).get(gpu, {})


def _section(tree: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = tree.get(name)
    return value if isinstance(value, Mapping) else {}


def _value(text: Any) -> str | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    return None if not text or text.upper() == "N/A" else text


def parse_amdsmi_driver(text: str) -> str | None:
    return _value(_section(_device(text), "DRIVER").get("VERSION"))


def parse_amdsmi_asic(text: str) -> dict[str, str | None]:
    asic = _section(_device(text), "ASIC")
    return {"asic_serial": _value(asic.get("ASIC_SERIAL")), "chip": _value(asic.get("TARGET_GRAPHICS_VERSION")),
            "market_name": _value(asic.get("MARKET_NAME"))}


def parse_amdsmi_firmware(text: str) -> dict[str, str]:
    out = {}
    for entry in _section(_device(text), "FW_LIST").values():
        if isinstance(entry, Mapping) and _value(entry.get("FW_ID")):
            out[str(entry["FW_ID"]).strip()] = _value(entry.get("FW_VERSION"))
    return out


def host_facts(host: Host, *, hub: str | None = None, account: str | None = None) -> dict[str, Any]:
    """Hub and account, hostname, kernel release, amdgpu driver, firmware, ASIC serial and chip, as far as readable."""
    unavailable: dict[str, str] = {}
    facts: dict[str, Any] = {
        "hub": hub or host.environ.get("LAB_HUB_URL") or host.environ.get("JUPYTERHUB_API_URL"),
        "account": account or host.environ.get("JUPYTERHUB_USER"),
        "hostname": (host.read_text("/proc/sys/kernel/hostname") or "").strip() or None,
        "kernel_release": (host.read_text("/proc/sys/kernel/osrelease") or "").strip() or None,
        "amdgpu_driver": None, "firmware": None, "asic_serial": None, "chip": None, "market_name": None}
    for key in ("hub", "account", "hostname", "kernel_release"):
        if facts[key] is None:
            unavailable[key] = "not readable on this host"
    reads = (("amdgpu_driver", ("static", "--driver"), parse_amdsmi_driver),
             ("asic", ("static", "--asic"), parse_amdsmi_asic),
             ("firmware", ("firmware",), parse_amdsmi_firmware))
    for key, args, parse in reads:
        try:
            parsed = parse(host.amdsmi(*args))
        except HostError as exc:
            for name in (("asic_serial", "chip", "market_name") if key == "asic" else (key,)):
                unavailable[name] = f"amd-smi {' '.join(args)}: {exc}"
            continue
        if key == "asic":
            facts.update(parsed)
        else:
            facts[key] = parsed or None
    for key in ("amdgpu_driver", "firmware", "asic_serial", "chip"):
        if facts[key] is None and key not in unavailable:
            unavailable[key] = "amd-smi printed N/A or nothing"
    facts["unavailable"] = unavailable
    return facts


def read_environ(host: Host, pid: int) -> tuple[dict[str, str | None] | None, str | None]:
    """The engine, ROCm and HIP variables of a running process, from /proc/<pid>/environ; no secrets."""
    where = f"/proc/{pid}/environ"
    raw = host.read_bytes(where)
    if raw is None:
        return None, f"{where} is not readable"
    env: dict[str, str | None] = {}
    for item in raw.split(b"\0"):
        name, sep, value = item.decode("utf-8", errors="replace").partition("=")
        if not sep or not name.startswith(ENV_PREFIXES) or any(word in name.upper() for word in SECRET_WORDS):
            continue
        env[name] = value
    for name in ALWAYS_ENV:
        env.setdefault(name, None)
    return dict(sorted(env.items())), None


def kfd_queues(host: Host, pid: int) -> tuple[int | None, str | None]:
    where = f"/sys/class/kfd/kfd/proc/{pid}/queues"
    names = host.listdir(where)
    if names is None:
        return None, f"{where} is not exposed"
    return sum(1 for n in names if n.isdigit()), None


def find_mem_busy(host: Host) -> str | None:
    """The mem_busy_percent file of the first AMD card under /sys/class/drm."""
    for name in host.listdir("/sys/class/drm") or []:
        if not re.fullmatch(r"card\d+", name):
            continue
        base = f"/sys/class/drm/{name}/device"
        vendor = (host.read_text(f"{base}/vendor") or "").strip().lower()
        if vendor == "0x1002" and host.read_text(f"{base}/mem_busy_percent") is not None:
            return f"{base}/mem_busy_percent"
    return None


def software_versions(host: Host, python: str) -> tuple[dict[str, Any], str | None]:
    try:
        return json.loads(host.command([python, "-c", VERSION_SCRIPT]).strip().splitlines()[-1]), None
    except (HostError, ValueError, IndexError) as exc:
        return {}, f"{python} -c (torch, triton): {exc}"


# --- the serve log ----------------------------------------------------------------------------------------------

_VLLM_LINEAR = re.compile(r"Using (\w+Kernel) for (\w+)")
_VLLM_ATTN = (re.compile(r"Using (\S+?) [Aa]ttention backend"), re.compile(r"Using (\w+) backend\b"),
              re.compile(r"Using (\w+) for (\w*Attention)\b"))
_VLLM_KV = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
_VLLM_CONC = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")
_VLLM_LEN = (re.compile(r"max_seq_len=(\d+)"), re.compile(r"Using max model len (\d+)"))
_VLLM_QUANT = re.compile(r"quantization=([\w.-]+)")
_VLLM_MODEL = re.compile(r"with config: model='([^']+)'")
_VLLM_PREFIX = re.compile(r"enable_prefix_caching=(True|False)")
_LLAMA_BUILD = re.compile(r"^build: (\d+ \([0-9a-f]+\))", re.M)
_LLAMA_KV = re.compile(r"llama_kv_cache\w*: size = .*?\(\s*(\d+) cells")
_LLAMA_CTX = re.compile(r"n_ctx\s*=\s*(\d+)")
_LLAMA_FA = re.compile(r"flash_attn\s*=\s*(\S+)")
_LLAMA_FTYPE = re.compile(r"file type\s*=\s*(.+?)\s*$", re.M)
_LLAMA_SEQ = re.compile(r"n_seq_max\s*=\s*(\d+)")


def _int(text: str) -> int:
    return int(text.replace(",", ""))


def _dedupe(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def parse_serve_log(text: str, engine: str) -> dict[str, Any]:
    """Kernel classes, attention backends, KV-cache capacity and engine config from a vLLM or llama-server log."""
    out: dict[str, Any] = {"linear_kernels": {}, "attention_backends": [], "kv_cache_tokens": None,
                           "max_concurrency": None, "concurrency_tokens": None, "max_model_len": None,
                           "quantization": None, "checkpoint": None, "engine_build": None, "prefix_caching": None}
    if engine == "llamacpp":
        if m := _LLAMA_BUILD.search(text):
            out["engine_build"] = m.group(1)
        kv = [int(x) for x in _LLAMA_KV.findall(text)]
        out["kv_cache_tokens"] = kv[-1] if kv else None
        ctx = [int(x) for x in _LLAMA_CTX.findall(text)]
        out["max_model_len"] = ctx[-1] if ctx else None
        out["attention_backends"] = _dedupe(f"flash_attn {v}" for v in _LLAMA_FA.findall(text))
        if m := _LLAMA_FTYPE.search(text):
            out["quantization"] = m.group(1)
        if (seq := _LLAMA_SEQ.findall(text)) and out["kv_cache_tokens"]:
            out["max_concurrency"] = float(seq[-1])
        return out
    out["linear_kernels"] = {layer: kernel for kernel, layer in _VLLM_LINEAR.findall(text)}
    found: list[tuple[int, str]] = []
    for pattern in _VLLM_ATTN:
        for m in pattern.finditer(text):
            if pattern.groups == 2:
                found.append((m.start(), f"{m.group(1)} ({m.group(2)})"))
            elif not m.group(1).endswith("Kernel"):
                found.append((m.start(), m.group(1)))
    out["attention_backends"] = _dedupe(name for _, name in sorted(found))
    if kv := _VLLM_KV.findall(text):
        out["kv_cache_tokens"] = _int(kv[-1])
    if conc := _VLLM_CONC.findall(text):
        out["concurrency_tokens"], out["max_concurrency"] = _int(conc[-1][0]), float(conc[-1][1])
    for pattern in _VLLM_LEN:
        if m := pattern.search(text):
            out["max_model_len"] = int(m.group(1))
            break
    if m := _VLLM_QUANT.search(text):
        out["quantization"] = None if m.group(1) == "None" else m.group(1)
    if m := _VLLM_MODEL.search(text):
        out["checkpoint"] = m.group(1)
    if m := _VLLM_PREFIX.search(text):
        out["prefix_caching"] = m.group(1) == "True"
    return out


# --- spawning ---------------------------------------------------------------------------------------------------

_PLACEHOLDER = re.compile(r"\{(port|port_file|log|batch|max_len|model|checkpoint)\}")


def default_command(engine: str, chip: str | None) -> list[str]:
    """The launch command: lab/pod_a_teacher.sh's vLLM serve line, or llama-server with one slot per stream."""
    if engine == "llamacpp":
        return ["llama-server", "-m", "{checkpoint}", "--alias", "{model}", "--host", "127.0.0.1", "--port", "{port}",
                "-c", "{max_len}", "-np", "{batch}", "-ngl", "999", "--jinja"]
    argv = ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "{checkpoint}", "--served-model-name",
            "{model}", "--port", "{port}", "--host", "127.0.0.1", "--max-model-len", "{max_len}",
            "--limit-mm-per-prompt", '{"image":1}', "--gpu-memory-utilization", "0.90", "--dtype", "bfloat16"]
    if chip and chip.startswith("gfx115"):
        argv += ["--mm-encoder-attn-backend", "TRITON_ATTN"]   # as pod_a_teacher.sh does on gfx115x
    return argv


def fill_command(argv: Sequence[str], values: Mapping[str, Any]) -> list[str]:
    return [_PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), str(a)) for a in argv]


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


@dataclass
class Spawned:
    proc: subprocess.Popen
    pid: int
    port: int
    log_path: Path
    spawned: float
    argv: list[str]
    ready: float | None = None
    stopped: bool = False
    out_handle: Any = None

    def stop(self, timeout_s: float = 30.0) -> bool:
        if self.proc.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(self.proc.pid, signal.SIGTERM)   # vLLM's engine core is a child process
                else:
                    self.proc.terminate()
                self.proc.wait(timeout_s)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(self.proc.pid, signal.SIGKILL)
                else:
                    self.proc.kill()
                try:
                    self.proc.wait(timeout_s)
                except subprocess.TimeoutExpired:
                    pass
            except ProcessLookupError:
                pass
        if self.out_handle is not None:
            self.out_handle.close()
        deadline = time.time() + timeout_s
        while self.port and port_open(self.port) and time.time() < deadline:
            time.sleep(0.05)
        self.stopped = self.proc.poll() is not None
        return self.stopped


def spawn_server(argv: list[str], *, port: int, log_path: Path, env: Mapping[str, str], cwd: Path,
                 own_log: bool) -> Spawned:
    """Start the server; its stdout and stderr go to the log (or beside it when the command writes its own).
    Port 0: the server picks a free port itself and writes it to its {port_file} (see wait_port)."""
    if port and port_open(port):
        raise BenchError(f"something already listens on 127.0.0.1:{port}; a server spawned there would collide "
                         "with it (on pod C that is the running teacher)")
    out_path = log_path.with_suffix(".out") if own_log else log_path
    handle = open(out_path, "wb")
    try:
        proc = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env={**os.environ, **env}, cwd=str(cwd), start_new_session=os.name == "posix")
    except OSError as exc:
        handle.close()
        raise BenchError(f"could not start {argv[0]}: {exc}") from exc
    return Spawned(proc, proc.pid, port, log_path, time.time(), argv, out_handle=handle)


def wait_port(spawned: Spawned, port_file: Path, timeout_s: float) -> int:
    """The port a server started with port 0 wrote to its port file; bounded by timeout_s."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        code = spawned.proc.poll()
        if code is not None:
            raise BenchError(f"the server exited with code {code} before it named its port; see {spawned.log_path}")
        try:
            port = int(port_file.read_text().strip())
        except (OSError, ValueError):
            time.sleep(0.02)
            continue
        spawned.port = port
        return port
    raise BenchError(f"the server named no port in {port_file.name} within {timeout_s:g} s")


def wait_ready(spawned: Spawned, endpoint: Any, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    pause = 0.05
    while time.time() < deadline:
        code = spawned.proc.poll()
        if code is not None:
            raise BenchError(f"the server exited with code {code} before it was ready; see {spawned.log_path}")
        if port_open(spawned.port):
            try:
                endpoint.models()
                spawned.ready = time.time()
                return
            except Exception:
                pass
        time.sleep(pause)
        pause = min(1.0, pause * 1.5)
    raise BenchError(f"the server was not ready after {timeout_s:g} s; see {spawned.log_path}")


# --- statistics and bands ---------------------------------------------------------------------------------------

def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), digits)


def stats(values: Iterable[float | None]) -> dict[str, Any]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return {"n": 0, "median": None, "min": None, "max": None, "mean": None, "sd": None}
    return {"n": len(xs), "median": statistics.median(xs), "min": min(xs), "max": max(xs),
            "mean": statistics.fmean(xs), "sd": statistics.stdev(xs) if len(xs) > 1 else None}


def classify_bands(rates: Mapping[str, float], *, rel_gap: float = DEFAULT_BAND_GAP,
                   min_spawns: int = MIN_SPAWNS_TO_BAND) -> list[list[str]]:
    """Spawns grouped by their own decode rates, fastest band first (see the module docstring)."""
    ordered = sorted(rates, key=lambda k: -rates[k])
    if len(ordered) < min_spawns:
        return [ordered] if ordered else []
    threshold = rel_gap * statistics.median(rates.values())

    def split(group: list[str]) -> list[list[str]]:
        if len(group) < 2:
            return [group]
        gaps = [(rates[a] - rates[b], i) for i, (a, b) in enumerate(zip(group, group[1:]))]
        gap, at = max(gaps)
        if gap < threshold or gap <= 0:
            return [group]
        return split(group[:at + 1]) + split(group[at + 1:])

    return split(ordered)


def band_rule(bands: Sequence[Sequence[str]], rates: Mapping[str, float],
              mem_busy: Mapping[str, float | None], *, rel_gap: float = DEFAULT_BAND_GAP) -> dict[str, Any]:
    """The clusters and, between neighbours, the decode-rate and mem_busy_percent thresholds they imply."""
    clusters = []
    for i, members in enumerate(bands):
        busy = [mem_busy[m] for m in members if mem_busy.get(m) is not None]
        clusters.append({"band": f"b{i + 1}", "spawns": list(members),
                         "rate_min": min(rates[m] for m in members), "rate_max": max(rates[m] for m in members),
                         "mem_busy_median": statistics.median(busy) if busy else None})
    boundaries = []
    for upper, lower in zip(clusters, clusters[1:]):
        entry: dict[str, Any] = {"between": [upper["band"], lower["band"]],
                                 "rate_threshold_tok_s": (upper["rate_min"] + lower["rate_max"]) / 2.0,
                                 "mem_busy_threshold_pct": None, "mem_busy_separates": None}
        a, b = upper["mem_busy_median"], lower["mem_busy_median"]
        if a is not None and b is not None and a != b:
            cut = (a + b) / 2.0
            side = (lambda v: v < cut) if a < b else (lambda v: v > cut)
            up = [mem_busy[m] for m in upper["spawns"] if mem_busy.get(m) is not None]
            low = [mem_busy[m] for m in lower["spawns"] if mem_busy.get(m) is not None]
            entry["mem_busy_threshold_pct"] = cut
            entry["mem_busy_separates"] = all(side(v) for v in up) and not any(side(v) for v in low)
        boundaries.append(entry)
    return {"method": f"largest-gap split of each spawn's probe decode rate, gap >= {rel_gap:.0%} of the "
                      f"median, at least {MIN_SPAWNS_TO_BAND} spawns",
            "clusters": clusters, "boundaries": boundaries}


# --- one session ------------------------------------------------------------------------------------------------

@dataclass
class Config:
    engine: str = "vllm"
    command: list[str] | None = None
    checkpoint: str = ""
    model: str = "teacher"
    max_len: int = 16384
    quantization: str | None = None
    engine_dir: str | None = None
    patches: list[str] = field(default_factory=list)
    launch_env: dict[str, str] = field(default_factory=dict)
    node: str = "pod-b"
    port: int = DEFAULT_PORT
    batches: list[int] | None = None
    add_64: bool = True
    repeats: int = DEFAULT_REPEATS
    share_spawn: bool = False
    max_tokens: int = DEFAULT_MAX_TOKENS
    thinking: bool | None = False
    prefix_caching: bool | None = None
    requests_per_stream: int | None = None
    warmup_rounds: int = 1
    probe: bool = True
    idle_s: float = DEFAULT_IDLE_S
    power_period_s: float = DEFAULT_POWER_PERIOD_S
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    stream_grace_s: float = DEFAULT_STREAM_GRACE_S
    ready_timeout_s: float = DEFAULT_READY_TIMEOUT_S
    out_dir: Path = DEFAULT_OUT
    prompts: Path = DEFAULT_PROMPTS
    metric_prefix: str = "teacher"
    python: str = "python3"
    hub: str | None = None
    account: str | None = None
    pid: int | None = None              # attach: the running server's pid, when it is on this host
    serve_log: str | None = None        # attach: its log, when it is on this host
    band_gap: float = DEFAULT_BAND_GAP
    record: bool = True
    labels: dict[str, str] = field(default_factory=dict)   # --condition NAME=VALUE: told, not measured

    def settings(self) -> Settings:
        return Settings(model=self.model, max_tokens=self.max_tokens, thinking=self.thinking,
                        prefix_caching=self.prefix_caching)


ROW_KEYS = ("schema", "harness", "run_id", "session", "repeat", "batch_size", "engine", "node", "outcome", "reason",
            "spawn", "time_since_spawn_s", "decode_tok_s", "decode_tok_s_total", "ttft_s", "tokens_generated",
            "window_s", "in_flight_mean", "buffered_requests", "watts", "joules_per_token",
            "joules_per_token_above_idle", "idle_w", "throttled", "probe", "warmup", "fits", "requests", "conditions")


class Session:
    def __init__(self, config: Config, *, host: Host | None, power: Any) -> None:
        if config.engine not in ENGINES:
            raise BenchError(f"engine is one of {', '.join(ENGINES)}, not {config.engine!r}")
        self.config = config
        self.host = host
        self.power = power
        self.prompts = PromptSet.load(config.prompts)
        self.settings = config.settings()
        self.plan = plan_batches(config.batches, add_64=config.add_64)
        self.per_stream = config.requests_per_stream or len(self.prompts.prompts)
        self.id = "bench-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:4]
        self.dir = Path(config.out_dir) / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rows_path = self.dir / "rows.jsonl"
        self.clock = WallClock()
        self.rows: list[dict[str, Any]] = []
        self.spawns: dict[str, dict[str, Any]] = {}
        self.base: dict[str, Any] = {}

    # conditions

    def static_conditions(self, chip_hint: str | None = None) -> dict[str, Any]:
        """What holds for the whole session: versions, engine commit, patches, host, decode settings."""
        c = self.config
        unavailable: dict[str, str] = {}
        facts: dict[str, Any] = {"hub": c.hub, "account": c.account}
        versions: dict[str, Any] = {}
        rocm = None
        commit = None
        if self.host is not None:
            facts = host_facts(self.host, hub=c.hub, account=c.account)
            versions, why = software_versions(self.host, c.python)
            if why:
                unavailable["torch"] = unavailable["triton"] = why
            rocm = (self.host.read_text("/opt/rocm/.info/version") or "").strip() or None
            if rocm is None:
                unavailable["rocm"] = "/opt/rocm/.info/version is not readable"
            if c.engine_dir:
                try:
                    commit = self.host.command(["git", "-C", c.engine_dir, "rev-parse", "HEAD"]).strip() or None
                except HostError as exc:
                    unavailable["engine_commit"] = f"git in {c.engine_dir}: {exc}"
        else:
            why = "attached through the hub proxy: the pod's host is not readable from the box"
            for key in ("torch", "triton", "rocm", "engine_commit"):
                unavailable[key] = why
            for key in ("hostname", "kernel_release", "amdgpu_driver", "firmware", "asic_serial", "chip"):
                facts[key] = None
                facts.setdefault("unavailable", {})[key] = why
        patches = {}
        for p in c.patches:
            try:
                patches[Path(p).name] = sha256_file(p)
            except OSError as exc:
                unavailable["patch_hash"] = f"{p}: {exc}"
        return {"engine": c.engine, "engine_commit": commit, "engine_build": None, "patch_hash": patches,
                "checkpoint": c.checkpoint or None, "served_model": c.model, "quantization": c.quantization,
                "max_model_len": c.max_len, "node": c.node, "rocm": rocm, "torch": versions.get("torch"),
                "triton": versions.get("triton"), "hip": versions.get("hip"), "python": versions.get("python"),
                "host": facts, "prompt_set": self.prompts.describe(), "max_tokens": c.max_tokens, "temperature": 0,
                "top_p": 1.0, "seed": 0, "ignore_eos": True, "nonce": self.settings.nonce_on,
                "thinking": c.thinking, "prefix_caching_setting": c.prefix_caching,
                "requests_per_stream": self.per_stream, "warmup_rounds": c.warmup_rounds, "harness": HARNESS,
                "unavailable": unavailable, **c.labels}

    def server_conditions(self, base: dict[str, Any], *, pid: int | None, log_text: str | None,
                          attached: bool) -> dict[str, Any]:
        c = self.config
        cond = json.loads(json.dumps(base))
        unavailable = cond["unavailable"]
        found = parse_serve_log(log_text, c.engine) if log_text is not None else None
        if found is None:
            why = "attached: the serve log is not readable from here" if attached else "the serve log was not written"
            for key in ("linear_kernels", "attention_backends", "kv_cache_tokens"):
                unavailable[key] = why
            found = parse_serve_log("", c.engine)
            found["linear_kernels"] = found["attention_backends"] = None
        cond.update({"linear_kernels": found["linear_kernels"], "attention_backends": found["attention_backends"],
                     "kv_cache_tokens": found["kv_cache_tokens"], "max_concurrency": found["max_concurrency"],
                     "concurrency_tokens": found["concurrency_tokens"], "prefix_caching_log": found["prefix_caching"],
                     "engine_build": found["engine_build"]})
        cond["quantization"] = c.quantization or found["quantization"]
        cond["checkpoint"] = c.checkpoint or found["checkpoint"] or cond["checkpoint"]
        cond["max_model_len"] = found["max_model_len"] or c.max_len
        if cond["engine_commit"] is None and found["engine_build"]:
            cond["engine_commit"] = found["engine_build"]
            unavailable.pop("engine_commit", None)
        if log_text is not None and found["linear_kernels"] == {} and c.engine == "llamacpp":
            unavailable["linear_kernels"] = "llama-server logs no linear kernel choice"
        if self.host is not None and pid is not None:
            env, why = read_environ(self.host, pid)
            queues, qwhy = kfd_queues(self.host, pid)
        else:
            env, queues = None, None
            why = qwhy = "attached: the server's process is not readable from here"
        cond.update({"env": env, "env_meant": dict(sorted(c.launch_env.items())) if not attached else None,
                     "env_source": f"/proc/{pid}/environ" if env is not None else None, "kfd_queues": queues})
        if why:
            unavailable["env"] = why
        if qwhy:
            unavailable["kfd_queues"] = qwhy
        return cond

    def served_model(self, endpoint: Any, conditions: dict[str, Any]) -> None:
        """Fill the checkpoint and max model length the server itself reports (GET /v1/models: vLLM's root and
        max_model_len) where nothing else named them. A read, never a change."""
        try:
            entries = endpoint.models()
        except Exception as exc:
            conditions["unavailable"]["served_model"] = f"GET /v1/models: {type(exc).__name__}"
            return
        entry = next((e for e in entries if e.get("id") == self.config.model), None)
        if entry is None:
            raise BenchError(f"the server does not serve {self.config.model!r} (it serves "
                             f"{', '.join(str(e.get('id')) for e in entries) or 'nothing'})")
        if not self.config.checkpoint and entry.get("root"):
            conditions["checkpoint"] = entry["root"]
        if entry.get("max_model_len") and "kv_cache_tokens" in conditions["unavailable"]:   # no serve log read
            conditions["max_model_len"] = entry["max_model_len"]

    # rows

    def blank_row(self, *, repeat: int, batch: int, spawn: dict[str, Any], conditions: dict[str, Any]) -> dict:
        row: dict[str, Any] = dict.fromkeys(ROW_KEYS)
        cond = dict(conditions, batch_size=batch)
        row.update({"schema": ROW_SCHEMA, "harness": HARNESS, "run_id": f"{self.id}-r{repeat}-b{batch}",
                    "session": self.id, "repeat": repeat, "batch_size": batch, "engine": self.config.engine,
                    "node": self.config.node, "spawn": spawn, "requests": [], "buffered_requests": 0,
                    "tokens_generated": 0, "conditions": cond,
                    "warmup": {"requests": 0, "discarded": True, "items": []},
                    "probe": {"in_flight": 1, "decode_tok_s": None, "ttft_s": None, "mem_busy_pct": None,
                              "reason": "no probe in this run"},
                    "idle_w": {"no_server": None, "loaded": None}})
        return row

    def not_measured(self, todo: Sequence[tuple[int, PlannedBatch]], *, spawn: dict[str, Any],
                     conditions: dict[str, Any], after: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Fail rows for the runs left on a server whose earlier run had stuck streams, which may still decode there
        and would inflate every later number."""
        out = []
        for repeat, pb in todo:
            row = self.blank_row(repeat=repeat, batch=pb.size, spawn=spawn, conditions=conditions)
            row["outcome"] = "fail"
            row["reason"] = f"not measured: {after['run_id']} on the same server had stuck streams ({after['reason']})"
            out.append(row)
        return out

    def measure(self, endpoint: Any, pb: PlannedBatch, *, repeat: int, spawn: dict[str, Any],
                conditions: dict[str, Any], idle: dict[str, float | None], probe_now: bool) -> dict[str, Any]:
        c = self.config
        row = self.blank_row(repeat=repeat, batch=pb.size, spawn=spawn, conditions=conditions)
        row["idle_w"] = dict(idle)
        tag = row["run_id"]
        warm: list[RequestResult] = []

        def batch(size: int, per_stream: int, role: str) -> list[RequestResult]:
            try:
                return run_batch(endpoint, self.prompts, self.settings, batch=size, per_stream=per_stream, role=role,
                                 clock=self.clock, tag=tag, timeout_s=c.request_timeout_s, grace_s=c.stream_grace_s)
            except BenchError as exc:
                row["warmup"] = {"requests": len(warm), "discarded": True, "items": [r.as_dict() for r in warm]}
                row["outcome"], row["reason"] = "fail", str(exc)
                raise StuckRun(str(exc), row) from exc

        if c.warmup_rounds > 0:
            warm += run_sequential(endpoint, self.prompts, self.settings, role="warmup", clock=self.clock, tag=tag)
        prompt_tokens = [r.prompt_tokens for r in warm if r.prompt_tokens]
        per_request = max(prompt_tokens) + c.max_tokens if prompt_tokens else None
        verdict = fits(pb.size, kv_cache_tokens=conditions.get("kv_cache_tokens"), tokens_per_request=per_request)
        row["fits"] = {"checked": pb.when_fits, "fits": verdict, "kv_cache_tokens": conditions.get("kv_cache_tokens"),
                       "tokens_per_request": per_request}
        if pb.when_fits and verdict is not True:
            row["warmup"] = {"requests": len(warm), "discarded": True, "items": [r.as_dict() for r in warm]}
            row["outcome"] = "skipped"
            row["reason"] = (f"batch {pb.size} does not fit: the KV cache holds {conditions.get('kv_cache_tokens')} "
                             f"tokens and {pb.size} requests need {pb.size} x {per_request}"
                             if verdict is False else
                             f"cannot tell whether batch {pb.size} fits (KV-cache capacity "
                             f"{conditions.get('kv_cache_tokens')}, tokens per request {per_request})")
            return row
        for _ in range(c.warmup_rounds):
            warm += batch(pb.size, 1, "warmup")
        row["warmup"] = {"requests": len(warm), "discarded": True, "items": [r.as_dict() for r in warm]}
        if probe_now:
            row["probe"] = self.run_probe(endpoint, tag)
        measured = batch(pb.size, self.per_stream, "measure")
        row["requests"] = [r.as_dict() for r in measured]
        good = [r for r in measured if r.error is None and r.first is not None and r.last is not None]
        errors = [r.error for r in measured if r.error]
        row["buffered_requests"] = sum(1 for r in measured if r.buffered)
        row["tokens_generated"] = sum(r.tokens for r in good)
        if not good:
            row["outcome"], row["reason"] = "fail", errors[0] if errors else "no request completed"
            return row
        start, end = min(r.first for r in good), max(r.last for r in good)
        window = end - start
        row["window_s"] = _r(window)
        row["in_flight_mean"] = _r(sum(r.last - r.first for r in good) / window) if window > 0 else None
        row["decode_tok_s"] = _r(stats(r.decode_tok_s for r in good)["median"])
        row["decode_tok_s_total"] = _r(row["tokens_generated"] / window) if window > 0 else None
        row["ttft_s"] = _r(stats(r.ttft_s for r in good)["median"])
        spawned = spawn.get("spawned_t")
        row["time_since_spawn_s"] = _r(min(r.sent for r in measured) - spawned) if spawned else None
        watts = window_watts(self.power, start, end) if window > 0 else {
            **window_watts(NoPower("the decode window is empty"), start, end)}
        row["throttled"] = watts.pop("throttled", None)
        row["watts"] = watts
        if watts.get("energy_j") is not None and row["tokens_generated"]:
            row["joules_per_token"] = _r(watts["energy_j"] / row["tokens_generated"], 6)
            loaded = idle.get("loaded")
            if loaded is not None:
                row["joules_per_token_above_idle"] = _r((watts["energy_j"] - loaded * window)
                                                        / row["tokens_generated"], 6)
        row["outcome"] = "ok" if not errors else "fail"
        row["reason"] = None if not errors else f"{len(errors)} of {len(measured)} requests failed: {errors[0]}"
        return row

    def run_probe(self, endpoint: Any, tag: str) -> dict[str, Any]:
        """Exactly one request decoding: its rate, and mem_busy_percent sampled while it decodes."""
        path = find_mem_busy(self.host) if self.host is not None else None
        readings: list[float] = []
        host = self.host

        def read_busy() -> None:   # once per content chunk: the request is decoding, alone
            try:
                readings.append(float((host.read_text(path) or "").strip()))
            except ValueError:
                pass

        nonce = f"{tag}-probe-{uuid.uuid4().hex[:8]}" if self.settings.nonce_on else None
        result = run_request(endpoint, self.prompts.request_body(0, self.settings, nonce), role="probe", stream=0,
                             index=0, prompt_id=self.prompts.prompt_id(0), clock=self.clock,
                             on_token=read_busy if path is not None else None)
        if path is None:
            busy, why = None, ("attached: mem_busy_percent is not readable from here" if self.host is None
                               else "no AMD card exposes mem_busy_percent under /sys/class/drm")
        else:
            busy = {"median": statistics.median(readings), "max": max(readings), "samples": len(readings),
                    "path": path} if readings else None
            why = None if readings else "mem_busy_percent could not be read while the probe decoded"
        return {"in_flight": 1, "decode_tok_s": _r(result.decode_tok_s), "ttft_s": _r(result.ttft_s),
                "mem_busy_pct": busy, "reason": why or result.error, "request": result.as_dict()}

    def keep(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        with open(self.rows_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        log(f"{row['run_id']}: {row['outcome']} decode {row['decode_tok_s']} tok/s per stream, "
            f"{row['decode_tok_s_total']} total, ttft {row['ttft_s']} s, J/token {row['joules_per_token']}"
            + (f" ({row['reason']})" if row["reason"] else ""))
        if self.config.record:
            self.record_run(row)

    def record_run(self, row: dict[str, Any]) -> None:
        prefix = self.config.metric_prefix
        values = {f"{prefix}.decode_tok_s": row["decode_tok_s"], f"{prefix}.decode_tok_s_total": row["decode_tok_s_total"],
                  f"{prefix}.ttft_s": row["ttft_s"], f"{prefix}.joules_per_token": row["joules_per_token"],
                  f"{prefix}.watts_avg": (row["watts"] or {}).get("avg_w"),
                  f"{prefix}.watts_peak": (row["watts"] or {}).get("peak_w"),
                  "pod.idle_watts": (row["idle_w"] or {}).get("loaded")}
        found = {k: v for k, v in values.items() if v is not None}
        units = {k: ("W" if k == "pod.idle_watts" else METRIC_UNITS[k.split(".", 1)[1]]) for k in found}
        conditions = dict(row["conditions"], run_id=row["run_id"], spawn=row["spawn"].get("index"),
                          time_since_spawn_s=row["time_since_spawn_s"], throttled=row["throttled"],
                          mem_busy_pct=(row["probe"] or {}).get("mem_busy_pct"))
        try:
            record("bench.run", row["outcome"], found, {"run_id": row["run_id"]}, node=row["node"], units=units,
                   conditions=conditions, low_confidence=True, source=HARNESS, note=row["reason"])
        except Exception as exc:  # the row is on disk; a ledger that refuses it is said, not fatal
            log(f"no ledger row for {row['run_id']}: {type(exc).__name__}: {exc}")

    # the spawn loop

    def run(self) -> None:
        c = self.config
        if c.port and port_open(c.port):
            raise BenchError(f"something already listens on 127.0.0.1:{c.port}; a server spawned there would collide "
                             "with it (on pod C that is the running teacher)")
        if not c.port and not any("{port_file}" in str(a) for a in (c.command or [])):
            raise BenchError("port 0 needs a launch command with {port_file}, where the server writes the port it took")
        self.base = self.static_conditions()
        self.power.start()
        try:
            spawn_index = 0
            for repeat in range(c.repeats):
                groups = [self.plan] if c.share_spawn else [[pb] for pb in self.plan]
                for group in groups:
                    self.one_spawn(group, repeat=repeat, index=spawn_index)
                    spawn_index += 1
        finally:
            self.power.stop()

    def one_spawn(self, group: list[PlannedBatch], *, repeat: int, index: int) -> None:
        c = self.config
        idle_none = idle_watts(self.power, c.idle_s)
        log_path = self.dir / f"serve-{index:02d}.log"
        chip = self.base["host"].get("chip")
        template = c.command or default_command(c.engine, chip)
        port_file = self.dir / f"serve-{index:02d}.port"
        values = {"port": c.port, "port_file": port_file, "log": log_path, "batch": max(pb.size for pb in group),
                  "max_len": c.max_len, "model": c.model, "checkpoint": c.checkpoint}
        argv = fill_command(template, values)
        own_log = any("{log}" in str(a) for a in template)
        spawned = spawn_server(argv, port=c.port, log_path=log_path, env=c.launch_env, cwd=self.dir, own_log=own_log)
        spawn = {"index": index, "pid": spawned.pid, "cold": True, "attached": False, "spawned": utc(spawned.spawned),
                 "spawned_t": spawned.spawned, "ready_s": None, "stopped": False, "log": str(log_path),
                 "shared": c.share_spawn, "command": argv}
        endpoint = None
        rows: list[dict[str, Any]] = []
        log(f"spawn {index} (repeat {repeat}, batch {', '.join(str(pb.size) for pb in group)}): pid {spawned.pid}")
        try:
            try:
                port = c.port or wait_port(spawned, port_file, c.ready_timeout_s)
                endpoint = DirectEndpoint(f"http://127.0.0.1:{port}", timeout_s=c.request_timeout_s)
                wait_ready(spawned, endpoint, c.ready_timeout_s)
            except BenchError as exc:
                for pb in group:
                    row = self.blank_row(repeat=repeat, batch=pb.size, spawn=spawn, conditions=self.base)
                    row["outcome"], row["reason"] = "fail", str(exc)
                    rows.append(row)
                return
            spawn["ready_s"] = _r(spawned.ready - spawned.spawned)
            log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else None
            conditions = self.server_conditions(self.base, pid=spawned.pid, log_text=log_text, attached=False)
            self.served_model(endpoint, conditions)
            idle = {"no_server": idle_none, "loaded": idle_watts(self.power, c.idle_s)}
            probe_rate = None
            for k, pb in enumerate(group):
                try:
                    row = self.measure(endpoint, pb, repeat=repeat, spawn=spawn, conditions=conditions, idle=idle,
                                       probe_now=c.probe and probe_rate is None)
                except StuckRun as exc:   # the run fails; the rest of the group is not measured on this server
                    row = exc.row
                    rows.append(row)
                    probe_rate = probe_rate or (row["probe"] if row["probe"].get("decode_tok_s") is not None else None)
                    rows += self.not_measured([(repeat, rest) for rest in group[k + 1:]], spawn=spawn,
                                              conditions=conditions, after=row)
                    break
                if row["probe"].get("decode_tok_s") is not None:
                    probe_rate = row["probe"]
                rows.append(row)
            self.spawns[str(index)] = {"probe": probe_rate, "runs": [r["run_id"] for r in rows]}
        finally:
            if endpoint is not None:
                endpoint.close()
            spawn["stopped"] = spawned.stop()
            for row in rows:
                row["spawn"] = {k: v for k, v in spawn.items() if k != "spawned_t"}
                self.keep(row)

    def attach(self, endpoint: Any) -> None:
        c = self.config
        self.base = self.static_conditions()
        log_text = None
        if c.serve_log and self.host is not None:
            log_text = self.host.read_text(c.serve_log)
        conditions = self.server_conditions(self.base, pid=c.pid, log_text=log_text,
                                            attached=self.host is None or c.pid is None)
        self.served_model(endpoint, conditions)
        spawn = {"index": None, "pid": c.pid, "cold": False, "attached": True, "spawned": None, "ready_s": None,
                 "stopped": None, "log": c.serve_log, "shared": True, "command": None}
        self.power.start()
        try:
            idle = {"no_server": None, "loaded": idle_watts(self.power, c.idle_s)}
            probe_rate = None
            todo = [(repeat, pb) for repeat in range(c.repeats) for pb in self.plan]
            for k, (repeat, pb) in enumerate(todo):
                try:
                    row = self.measure(endpoint, pb, repeat=repeat, spawn=dict(spawn), conditions=conditions,
                                       idle=idle, probe_now=c.probe and probe_rate is None)
                except StuckRun as exc:   # the run fails; nothing more is measured on a server still decoding it
                    self.keep(exc.row)
                    for row in self.not_measured(todo[k + 1:], spawn=dict(spawn), conditions=conditions,
                                                 after=exc.row):
                        self.keep(row)
                    break
                if row["probe"].get("decode_tok_s") is not None:
                    probe_rate = row["probe"]
                self.keep(row)
            self.spawns["attached"] = {"probe": probe_rate, "runs": [r["run_id"] for r in self.rows]}
        finally:
            self.power.stop()

    # the summary

    def summarize(self) -> dict[str, Any]:
        return {"session": self.id, "harness": HARNESS, "engine": self.config.engine, "node": self.config.node,
                "prompt_set": self.prompts.describe(), "rows": len(self.rows),
                **summarize_runs(self.rows, self.spawns, band_gap=self.config.band_gap)}

    def write_metrics(self, summary: dict[str, Any], path: str | Path) -> int:
        """Each configuration's medians as metrics-file rows, their spread and bands in the conditions."""
        written = 0
        prefix = self.config.metric_prefix
        for key, entry in summary["configurations"].items():
            sample = next((r for r in self.rows if f"{r['engine']} b{r['batch_size']}" == key), None)
            base = {k: v for k, v in (sample or {}).get("conditions", {}).items()
                    if k in ("engine", "engine_commit", "patch_hash", "checkpoint", "quantization", "max_model_len",
                             "batch_size", "rocm", "torch", "triton", "prompt_set", "max_tokens", "ignore_eos",
                             "nonce", "thinking", "linear_kernels", "attention_backends", "kv_cache_tokens")
                    or k in self.config.labels}
            n = sum(1 for s in entry["spawns"] if s["outcome"] == "ok")
            low = n < MIN_RUNS_PER_ARM
            for name, unit in METRIC_UNITS.items():
                s = entry["stats"][name]
                if s["median"] is None:
                    continue
                conditions = dict(base, session=self.id, statistic=f"median of {n} spawns", stats=s,
                                  bands={b: {"share": v["share"], "stats": v["stats"][name]}
                                         for b, v in entry["bands"].items()})
                if low:
                    conditions["low_confidence_reason"] = (f"{n} spawn(s), under the {MIN_RUNS_PER_ARM} that "
                                                           "verdict() needs to call a difference")
                metrics_file.add({"metric": f"{prefix}.{name}", "value": round(s["median"], 4), "unit": unit,
                                  "node": self.config.node, "conditions": conditions, "source": f"{HARNESS} {self.id}",
                                  "ts": _metrics_ts(),
                                  "low_confidence": low}, path=path)
                written += 1
        return written


def summarize_runs(rows: Sequence[Mapping[str, Any]], spawns: Mapping[str, Mapping[str, Any]], *,
                   band_gap: float = DEFAULT_BAND_GAP) -> dict[str, Any]:
    """Bands and per-configuration statistics from the rows and the spawns ({key: {"probe", "runs"}}); pure."""
    rates: dict[str, float] = {}
    busy: dict[str, float | None] = {}
    run_spawn: dict[str, str] = {}
    for key, info in spawns.items():
        for run_id in info["runs"]:
            run_spawn[run_id] = key
        probe = info.get("probe")
        rate = (probe or {}).get("decode_tok_s")
        if rate is None:   # no probe: the spawn's first measured rate stands in
            first = next((r for r in rows if r["run_id"] in info["runs"] and r["decode_tok_s"]), None)
            rate = first["decode_tok_s"] if first else None
        if rate is not None:
            rates[key] = rate
            busy[key] = ((probe or {}).get("mem_busy_pct") or {}).get("median")
    bands = classify_bands(rates, rel_gap=band_gap)
    rule = band_rule(bands, rates, busy, rel_gap=band_gap) if bands else {"clusters": [], "boundaries": []}
    band_of = {spawn: f"b{i + 1}" for i, members in enumerate(bands) for spawn in members}
    configurations: dict[str, Any] = {}
    for row in rows:
        if row["outcome"] == "skipped":
            continue
        key = f"{row['engine']} b{row['batch_size']}"
        entry = configurations.setdefault(key, {"engine": row["engine"], "batch_size": row["batch_size"],
                                                "spawns": []})
        entry["spawns"].append({
            "run_id": row["run_id"], "spawn": run_spawn.get(row["run_id"]),
            "band": band_of.get(run_spawn.get(row["run_id"], "")), "outcome": row["outcome"],
            "decode_tok_s": row["decode_tok_s"], "decode_tok_s_total": row["decode_tok_s_total"],
            "ttft_s": row["ttft_s"], "joules_per_token": row["joules_per_token"],
            "watts_avg": (row["watts"] or {}).get("avg_w"), "watts_peak": (row["watts"] or {}).get("peak_w"),
            "time_since_spawn_s": row["time_since_spawn_s"]})
    for entry in configurations.values():
        ok = [s for s in entry["spawns"] if s["outcome"] == "ok"]
        entry["stats"] = {m: stats(s[m] for s in ok) for m in METRIC_UNITS}
        entry["bands"] = {}
        for band in sorted({s["band"] for s in ok if s["band"]}):
            members = [s for s in ok if s["band"] == band]
            entry["bands"][band] = {"spawns": len(members), "share": len(members) / len(ok),
                                    "stats": {m: stats(s[m] for s in members) for m in METRIC_UNITS}}
    return {"band_rule": rule, "configurations": configurations}


def _metrics_ts() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _finish(session: Session, metrics_path: str | Path | None) -> dict[str, Any]:
    summary = session.summarize()
    (session.dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if metrics_path:
        summary["metrics_rows"] = session.write_metrics(summary, metrics_path)
    return {"session": session.id, "rows": session.rows, "summary": summary, "rows_path": str(session.rows_path),
            "summary_path": str(session.dir / "summary.json")}


def run_session(config: Config, *, host: Host | None = None, power: Any = None,
                metrics_path: str | Path | None = None) -> dict[str, Any]:
    """Spawn, measure and stop, config.repeats times per batch size; the rows, the summary and where they are."""
    host = host or Host()
    power = power or LocalPower.for_host(host, node=config.node, period_s=config.power_period_s)
    session = Session(config, host=host, power=power)
    session.run()
    return _finish(session, metrics_path)


def attach_session(config: Config, *, endpoint: Any, power: Any = None, host: Host | None = None,
                   metrics_path: str | Path | None = None) -> dict[str, Any]:
    """Measure a server the harness did not start: no spawn, no process change; the rows say what is unknown."""
    session = Session(config, host=host, power=power or NoPower("no power sampler was given"))
    session.attach(endpoint)
    return _finish(session, metrics_path)


# --- the command line -------------------------------------------------------------------------------------------

# Condition names the harness fills itself; --condition may add others but never overwrite these.
RESERVED_CONDITIONS = frozenset({
    "engine", "engine_commit", "engine_build", "patch_hash", "checkpoint", "served_model", "quantization",
    "max_model_len", "node", "rocm", "torch", "triton", "hip", "python", "host", "prompt_set", "max_tokens",
    "temperature", "top_p", "seed", "ignore_eos", "nonce", "thinking", "prefix_caching_setting",
    "requests_per_stream", "warmup_rounds", "harness", "unavailable", "linear_kernels", "attention_backends",
    "kv_cache_tokens", "max_concurrency", "concurrency_tokens", "prefix_caching_log", "env", "env_meant",
    "env_source", "kfd_queues", "batch_size", "share_spawn", "time_since_spawn_s", "session", "statistic", "stats",
    "bands", "low_confidence_reason", "run_id", "spawn"})


def _batches(text: str) -> list[int]:
    try:
        return [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"batch sizes are whole numbers separated by commas, not {text!r}") from None


def _env_pair(text: str) -> tuple[str, str]:
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, not {text!r}")
    return key, value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lab/bench.py", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="spawn the server once per run on this host and measure it")
    attach = sub.add_parser("attach", help="measure a server already running (no spawn)")
    for s in (run, attach):
        s.add_argument("--engine", choices=ENGINES, default="vllm")
        s.add_argument("--model", default="teacher", help="the served model name")
        s.add_argument("--checkpoint", default="")
        s.add_argument("--quantization", default=None)
        s.add_argument("--max-len", type=int, default=16384)
        s.add_argument("--node", default=None, help="the ledger's name for the machine (pod-b, pod-c, ...)")
        s.add_argument("--batches", type=_batches, default=None, help="comma-separated; default 1,4,8,16 (+64)")
        s.add_argument("--no-64", action="store_true", help="do not add the 64 arm")
        s.add_argument("--repeats", type=int, default=None)
        s.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
        s.add_argument("--requests-per-stream", type=int, default=None)
        s.add_argument("--warmup", type=int, default=1, help="warm-up rounds at the batch size (0: none)")
        s.add_argument("--no-probe", action="store_true")
        s.add_argument("--thinking", choices=("off", "on", "unset"), default="off")
        s.add_argument("--prefix-caching", choices=("auto", "on", "off"), default="auto")
        s.add_argument("--idle-s", type=float, default=None)
        s.add_argument("--power-period-s", type=float, default=DEFAULT_POWER_PERIOD_S)
        s.add_argument("--band-gap", type=float, default=DEFAULT_BAND_GAP)
        s.add_argument("--request-timeout-s", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
        s.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
        s.add_argument("--out", default=str(DEFAULT_OUT))
        s.add_argument("--metrics-file", nargs="?", const=str(metrics_file.DEFAULT_PATH), default=None)
        s.add_argument("--metric-prefix", default="teacher")
        s.add_argument("--hub", default=None)
        s.add_argument("--python", default="python3", help="the interpreter whose torch and Triton the server uses")
        s.add_argument("--condition", action="append", type=_env_pair, default=[], metavar="NAME=VALUE",
                       help="a condition the harness cannot read, recorded on every row and metrics row "
                            "(e.g. backend=vulkan); repeatable; a name the harness itself records is refused")
    run.add_argument("--serve-cmd", default=None,
                     help="the launch command (JSON list or shell words) with {port} {log} {batch} {max_len} {model} "
                          "{checkpoint}; default: pod_a_teacher.sh's vLLM line or llama-server")
    run.add_argument("--engine-dir", default=None, help="the engine's git checkout, for its commit")
    run.add_argument("--patch", action="append", default=[], help="a patch applied to the engine (repeatable)")
    run.add_argument("--env", action="append", type=_env_pair, default=[], help="NAME=VALUE for the launch")
    run.add_argument("--port", type=int, default=DEFAULT_PORT)
    run.add_argument("--share-spawn", action="store_true", help="every batch size in one spawn per repeat")
    run.add_argument("--account", default=None, help="the hub account the pod runs under (default $JUPYTERHUB_USER)")
    attach.add_argument("--url", default=None, help="the server's base URL, reached directly")
    attach.add_argument("--account", default=None, help="hub account letter (C, D): reach the server through the proxy")
    attach.add_argument("--port", type=int, default=8000)
    attach.add_argument("--power-port", type=int, default=SAMPLER_PORT)
    attach.add_argument("--pid", type=int, default=None, help="the server's pid, when it runs on this host")
    attach.add_argument("--serve-log", default=None, help="the server's log, when it is on this host")
    return p


def _config(args: argparse.Namespace) -> Config:
    tristate = {"auto": None, "on": True, "off": False}
    attach = args.cmd == "attach"
    command = None
    if not attach and args.serve_cmd:
        text = args.serve_cmd.strip()
        command = json.loads(text) if text.startswith("[") else shlex.split(text)
    labels = dict(args.condition)
    if taken := sorted(set(labels) & RESERVED_CONDITIONS):
        raise BenchError(f"--condition cannot set {', '.join(taken)}: the harness records those itself")
    return Config(
        labels=labels, engine=args.engine, command=command, checkpoint=args.checkpoint, model=args.model, max_len=args.max_len,
        quantization=args.quantization, engine_dir=None if attach else args.engine_dir,
        patches=[] if attach else list(args.patch), launch_env={} if attach else dict(args.env),
        node=args.node or ("pod-c" if attach else "pod-b"), port=args.port, batches=args.batches,
        add_64=not args.no_64 and not attach, repeats=args.repeats or (1 if attach else DEFAULT_REPEATS),
        share_spawn=False if attach else args.share_spawn, max_tokens=args.max_tokens,
        thinking={"off": False, "on": True, "unset": None}[args.thinking],
        prefix_caching=tristate[args.prefix_caching], requests_per_stream=args.requests_per_stream,
        warmup_rounds=args.warmup, probe=not args.no_probe,
        idle_s=args.idle_s if args.idle_s is not None else (0.0 if attach else DEFAULT_IDLE_S),
        power_period_s=args.power_period_s, out_dir=Path(args.out), prompts=Path(args.prompts),
        metric_prefix=args.metric_prefix, python=args.python, hub=args.hub,
        account=getattr(args, "account", None) if not attach else None,
        pid=getattr(args, "pid", None), serve_log=getattr(args, "serve_log", None), band_gap=args.band_gap,
        request_timeout_s=args.request_timeout_s)


def _attach(config: Config, args: argparse.Namespace, host: Host | None) -> dict[str, Any]:
    if bool(args.url) == bool(args.account):
        raise BenchError("attach takes --url or --account, one of them")
    if args.url:
        endpoint = DirectEndpoint(args.url, timeout_s=config.request_timeout_s)
        local = host or (Host() if (config.pid or config.serve_log) else None)
        power = LocalPower.for_host(local, node=config.node, period_s=config.power_period_s) if local else None
        try:
            return attach_session(config, endpoint=endpoint, power=power, host=local, metrics_path=args.metrics_file)
        finally:
            endpoint.close()
    from lab import hub as hub_module
    from lab.proxy import PodLink

    account = hub_module.load_accounts([args.account.upper()])[args.account.upper()]
    with PodLink.for_account(account) as link:
        config.hub = config.hub or link.base_url
        config.account = account.user
        power = RemotePower(link, args.power_port)
        return attach_session(config, endpoint=ProxyEndpoint(link, args.port), power=power,
                              metrics_path=args.metrics_file)


@recorded("bench")
def main(argv: Sequence[str] | None = None, *, host: Host | None = None,
         power: Any = None) -> dict[str, Any]:
    """Benchmark a model server: decode tok/s, TTFT and joules per token, one cold spawn per run."""
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        config = _config(args)
        if args.cmd == "run":
            result = run_session(config, host=host, power=power, metrics_path=args.metrics_file)
        else:
            result = _attach(config, args, host)
    except BenchError as exc:
        return {"ok": False, "outcome": "fail", "error": str(exc)}
    prefix = config.metric_prefix
    found: dict[str, float] = {}
    for entry in result["summary"]["configurations"].values():
        for name in ("decode_tok_s", "decode_tok_s_total", "ttft_s", "joules_per_token"):
            median = entry["stats"][name]["median"]
            if median is not None:
                found[f"{prefix}.{name}.b{entry['batch_size']}"] = round(median, 4)
    outcomes = [r["outcome"] for r in result["rows"]]
    ok = bool(outcomes) and all(o in ("ok", "skipped") for o in outcomes)
    return {"ok": ok, "outcome": "ok" if ok else "fail", "session": result["session"],
            "rows": [r["run_id"] for r in result["rows"]], "outcomes": outcomes, "rows_path": result["rows_path"],
            "summary_path": result["summary_path"], "metrics": found, "refs": {"run_id": result["session"]},
            **({} if ok else {"error": next((r["reason"] for r in result["rows"] if r["reason"]), "no rows")})}


if __name__ == "__main__":
    answer = main()
    print(json.dumps(answer, default=str))
    sys.exit(0 if answer.get("ok") else 1)
