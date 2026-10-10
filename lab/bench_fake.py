"""A stand-in model server and pod for lab/bench.py: spawned like vLLM or llama-server, read like a Strix Halo pod.

    python lab/bench_fake.py serve --port 8100 --engine vllm --log serve.log --root FAKE_ROOT [options]

    from lab.bench_fake import FakeHost
    host = FakeHost(FAKE_ROOT)              # what lab/bench.py reads on a pod: amd-smi, /proc, /sys, versions

Nothing here touches a GPU or the network beyond 127.0.0.1. The tests spawn `serve` as a real process, so a run of the
harness against it is a real cold spawn, a real token stream and a real stop; only the GPU is pretend.

The server (`serve`). An OpenAI-compatible server with the routes the harness uses: GET /v1/models (the served name,
its checkpoint as root and max_model_len, as vLLM answers), GET /health, and POST /v1/chat/completions streamed as
server-sent events (one token per chunk, a finish chunk, a usage chunk when stream_options.include_usage is set, then
[DONE]). It honours max_tokens, ignore_eos (without it, an answer ends after a third of max_tokens with finish reason
stop) and temperature. Time is real and kept short: the first token after --ttft-s (plus --image-ttft-s per image),
then one token every 1 / min(--stream-tok-s, --capacity-tok-s / streams decoding) seconds, so more concurrent
streams make each one slower and the sum faster, as batching does. With --prefix-caching on, a request whose whole
prompt was seen before answers its first token after a tenth of the TTFT, as a prefix-cache hit would; the harness's
per-request nonce is what keeps that from happening.

At start-up it waits --startup-s and writes a serve log in the engine's own words (--engine vllm: the lines vLLM's
b1388b1 source logs for the engine config, the linear kernel, the attention backends and the KV cache; --engine
llamacpp: llama-server's build, n_ctx, flash_attn and llama_kv_cache lines). Those lines are made from the engines'
log statements, not recorded on a pod (fixtures/lab/bench/*_synthetic.log are the same text); R20 records the real
ones. Under --root it then writes what a pod exposes about a running server:

    <root>/proc/<pid>/environ                     its environment, NUL-separated, after any --set-env K=V (a launch
                                                  wrapper that changes what the launch meant to set)
    <root>/sys/class/kfd/kfd/proc/<pid>/queues/N  --queues queue folders
    <root>/sys/class/drm/card0/device/vendor      0x1002
    <root>/sys/class/drm/card0/device/mem_busy_percent   updated while it serves
    <root>/fake_load                              {"t", "pid", "port", "active", "band"}, refreshed every 50 ms
                                                  and whenever a stream starts or ends

Bands. <root>/band_plan.txt, when present, holds one word per spawn (fast or slow), taken in spawn order through
<root>/spawn_count. A slow spawn decodes at --slow-factor of the rate and reads a higher mem_busy_percent, which is
the pattern the harness's band classification has to find without being told.

The pod (`FakeHost`). lab.bench.Host over a fake file-system root and recorded command output: amd-smi metric prints
the Strix Halo output recorded on pod C (fixtures/telemetry/pod_power/strixhalo_amdsmi_metric_power.txt), its
SOCKET_POWER set from <root>/fake_load (8 W with no server, 9.2 W with one loaded and idle, 15 W more per stream
decoding, the morning's pod C figures in lab/README.md stretched to make the arithmetic visible); amd-smi static
--driver and --asic and amd-smi firmware print fixtures/lab/bench/amdsmi_*_synthetic.txt (made from amd-smi's output
layout, not recorded: the SJC image has amd-smi off PATH and nobody has run those subcommands there yet); python
prints the pod's torch and Triton versions; git prints a fixed commit. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

if not __package__:  # run by path (the tests spawn it so): the repo root, one folder up, holds lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import bench  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
POWER_FIXTURE = REPO / "fixtures" / "telemetry" / "pod_power" / "strixhalo_amdsmi_metric_power.txt"
BENCH_FIXTURES = REPO / "fixtures" / "lab" / "bench"
FAKE_COMMIT = "b1388b1fbfakefakefakefakefakefakefake0000"
FAKE_VERSIONS = {"torch": "2.9.1+rocm7.13.0", "triton": "3.5.1", "hip": "7.13.0", "python": "3.12.3"}
IDLE_NO_SERVER_W = 8.0
IDLE_LOADED_W = 9.2
WATTS_PER_STREAM = 15.0
HEARTBEAT_S = 0.05
MEM_BUSY = {"fast": 55.0, "slow": 85.0}


# --- the server -------------------------------------------------------------------------------------------------

class FakeEngine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = Path(args.root) if args.root else None
        self.active = 0
        self.lock = threading.Lock()
        self.files = threading.Lock()  # one heartbeat writes at a time, so a stale count never lands last
        self.seen: set[str] = set()
        self.band = self._take_band()
        self.factor = args.slow_factor if self.band == "slow" else 1.0
        self.stop = threading.Event()

    def _take_band(self) -> str:
        if self.root is None:
            return "fast"
        plan = self.root / "band_plan.txt"
        counter = self.root / "spawn_count"
        n = int(counter.read_text()) if counter.exists() else 0
        counter.write_text(str(n + 1))
        if not plan.exists():
            return "fast"
        words = plan.read_text().split()
        return words[n] if n < len(words) else "fast"

    # files a pod exposes

    def _write(self, rel: str, data: str | bytes) -> None:
        if self.root is None:
            return
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        tmp.write_bytes(data.encode() if isinstance(data, str) else data)
        for _ in range(20):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:  # Windows: a reader has it open
                time.sleep(0.005)

    def publish_process(self) -> None:
        pid = os.getpid()
        environ = b"".join(f"{k}={v}".encode() + b"\0" for k, v in os.environ.items())
        self._write(f"proc/{pid}/environ", environ)
        for q in range(self.args.queues):
            if self.root is not None:
                (self.root / f"sys/class/kfd/kfd/proc/{pid}/queues/{q}").mkdir(parents=True, exist_ok=True)
        self._write("sys/class/drm/card0/device/vendor", "0x1002\n")
        self.heartbeat()

    def mem_busy(self, active: int) -> float:
        if active == 0:
            return 0.0
        return min(100.0, MEM_BUSY[self.band] + 4.0 * (active - 1))

    def heartbeat(self) -> None:
        with self.files:
            with self.lock:
                active = self.active
            self._write("fake_load", json.dumps({"t": time.time(), "pid": os.getpid(), "port": self.args.port,
                                                 "active": active, "band": self.band}))
            self._write("sys/class/drm/card0/device/mem_busy_percent", f"{int(self.mem_busy(active))}\n")

    def beat_forever(self) -> None:
        while not self.stop.wait(HEARTBEAT_S):
            self.heartbeat()

    def serve_log(self) -> str:
        a = self.args
        if a.engine == "llamacpp":
            return "\n".join([
                "build: 6500 (1a2b3c4d) with cc (GCC) 13.2.0 for x86_64-linux-gnu",
                "ggml_cuda_init: found 1 ROCm devices:",
                "  Device 0: AMD Radeon 8060S, gfx1151 (0x1151), VMM: no, Wave Size: 32",
                "print_info: file type   = Q4_K - Medium",
                f"llama_context: n_seq_max     = {a.parallel}",
                f"llama_context: n_ctx         = {a.max_len}",
                "llama_context: flash_attn    = enabled",
                f"llama_kv_cache: size = 2048.00 MiB ( {a.kv_tokens} cells,  16 layers, {a.parallel}/{a.parallel} seqs), "
                "K (f16): 1024.00 MiB, V (f16): 1024.00 MiB",
                f"main: server is listening on http://127.0.0.1:{a.port} - starting the main loop", ""])
        conc = a.kv_tokens / a.max_len
        pre = "(EngineCore_DP0 pid=%d) INFO 10-10 19:30:20 " % os.getpid()
        return "\n".join([
            "INFO 10-10 19:30:01 [api_server.py:1885] vLLM API server version 0.19.2+gfx1151",
            f"{pre}[core.py:93] Initializing a V1 LLM engine (v0.19.2+gfx1151) with config: model='{a.checkpoint}', "
            f"dtype=torch.bfloat16, max_seq_len={a.max_len}, quantization=compressed-tensors, enforce_eager=False, "
            f"enable_prefix_caching={a.prefix_caching == 'on'}",
            f"{pre}[compressed_tensors_wNa16.py:95] Using ConchLinearKernel for CompressedTensorsWNA16",
            f"{pre}[rocm.py:300] Using Triton Attention backend.",
            f"{pre}[mm_encoder_attention.py:88] Using TRITON_ATTN for MMEncoderAttention.",
            f"{pre}[kv_cache_utils.py:1087] GPU KV cache size: {a.kv_tokens:,} tokens",
            f"{pre}[kv_cache_utils.py:1091] Maximum concurrency for {a.max_len:,} tokens per request: {conc:.2f}x",
            "INFO:     Application startup complete.", ""])

    # requests

    def chat(self, handler: BaseHTTPRequestHandler, body: dict[str, Any]) -> None:
        a = self.args
        messages = body.get("messages") or []
        prompt_text = json.dumps(messages, sort_keys=True)
        images = prompt_text.count('"image_url"')
        max_tokens = int(body.get("max_tokens") or 16)
        n = max_tokens if body.get("ignore_eos") else max(1, max_tokens // 3)
        finish = "length" if body.get("ignore_eos") else "stop"
        ttft = a.ttft_s + a.image_ttft_s * images
        with self.lock:
            hit = a.prefix_caching == "on" and prompt_text in self.seen
            self.seen.add(prompt_text)
        if hit:
            ttft /= 10.0
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        time.sleep(ttft)
        with self.lock:
            self.active += 1
        self.heartbeat()  # the load and mem_busy_percent change as the stream starts decoding, not 50 ms later
        try:
            for i in range(n):
                with self.lock:
                    streams = self.active
                rate = min(a.stream_tok_s, a.capacity_tok_s / max(1, streams)) * self.factor
                if i:
                    time.sleep(1.0 / rate)
                self._event(handler, {"object": "chat.completion.chunk", "model": a.model,
                                      "choices": [{"index": 0, "delta": {"content": f"t{i} "}, "finish_reason": None}]})
            self._event(handler, {"object": "chat.completion.chunk", "model": a.model,
                                  "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        finally:
            with self.lock:
                self.active -= 1
            self.heartbeat()
        if (body.get("stream_options") or {}).get("include_usage"):
            prompt_tokens = len(prompt_text) // 4 + 300 * images
            self._event(handler, {"object": "chat.completion.chunk", "model": a.model, "choices": [],
                                  "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n,
                                            "total_tokens": prompt_tokens + n}})
        handler.wfile.write(b"data: [DONE]\n\n")
        handler.wfile.flush()

    @staticmethod
    def _event(handler: BaseHTTPRequestHandler, chunk: dict[str, Any]) -> None:
        handler.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        handler.wfile.flush()


def make_handler(engine: FakeEngine) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _json(self, status: int, body: Any) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            a = engine.args
            if self.path.rstrip("/") == "/v1/models":
                return self._json(200, {"object": "list", "data": [
                    {"id": a.model, "object": "model", "root": a.checkpoint, "max_model_len": a.max_len}]})
            if self.path.rstrip("/") == "/health":
                return self._json(200, {"status": "ok"})
            return self._json(404, {"error": {"message": f"no route {self.path}"}})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._json(404, {"error": {"message": f"no route {self.path}"}})
            if body.get("model") not in (None, engine.args.model):
                return self._json(404, {"error": {"message": f"The model `{body.get('model')}` does not exist."}})
            engine.chat(self, body)

    return Handler


def serve(args: argparse.Namespace) -> int:
    for item in args.set_env or []:
        key, _, value = item.partition("=")
        os.environ[key] = value
    engine = FakeEngine(args)
    time.sleep(args.startup_s)
    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(engine.serve_log(), encoding="utf-8")
    engine.publish_process()
    threading.Thread(target=engine.beat_forever, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine))
    server.daemon_threads = True
    try:
        server.serve_forever(poll_interval=0.05)
    finally:
        engine.stop.set()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lab/bench_fake.py", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, required=True)
    s.add_argument("--engine", choices=("vllm", "llamacpp"), default="vllm")
    s.add_argument("--log", required=True)
    s.add_argument("--root", default=None)
    s.add_argument("--model", default="teacher")
    s.add_argument("--checkpoint", default="/home/jovyan/models/Qwen3.8-27B-INT4")
    s.add_argument("--max-len", type=int, default=16384)
    s.add_argument("--kv-tokens", type=int, default=163840)
    s.add_argument("--parallel", type=int, default=16)
    s.add_argument("--stream-tok-s", type=float, default=400.0)
    s.add_argument("--capacity-tok-s", type=float, default=1600.0)
    s.add_argument("--ttft-s", type=float, default=0.03)
    s.add_argument("--image-ttft-s", type=float, default=0.02)
    s.add_argument("--startup-s", type=float, default=0.1)
    s.add_argument("--slow-factor", type=float, default=0.6)
    s.add_argument("--queues", type=int, default=3)
    s.add_argument("--prefix-caching", choices=("on", "off"), default="on")
    s.add_argument("--set-env", action="append", metavar="K=V")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return serve(args)


def spawn_python() -> str:
    """The interpreter to spawn the fake with, so the spawned pid is the server's own: on Windows a venv's python.exe
    is a launcher that starts the real interpreter as a child, and the child's pid is the one that serves."""
    return getattr(sys, "_base_executable", None) or sys.executable


def serve_command(port_placeholder: str = "{port}", **options: Any) -> list[str]:
    """The argv the harness spawns for the fake, with {port} and {log} left for it to fill."""
    argv = [spawn_python(), str(Path(__file__).resolve()), "serve", "--port", port_placeholder, "--log", "{log}"]
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, (list, tuple)):
            for item in value:
                argv += [flag, str(item)]
        else:
            argv += [flag, str(value)]
    return argv


# --- the pod ----------------------------------------------------------------------------------------------------

def pid_alive(pid: int) -> bool:
    """Is that process still running? Asked without signalling it (on Windows os.kill would terminate it)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259                     # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class FakeHost(bench.Host):
    """A Strix Halo pod as lab/bench.py reads it, over a fake root (see the module docstring)."""

    def __init__(self, root: str | Path, *, environ: dict[str, str] | None = None, throttle: str = "N/A",
                 kernel_release: str = "6.8.0-fake-generic", hostname: str = "jupyter-roscon02",
                 rocm_version: str = "7.13.0-fake") -> None:
        root = Path(root)
        (root / "proc/sys/kernel").mkdir(parents=True, exist_ok=True)
        (root / "proc/sys/kernel/hostname").write_text(hostname + "\n")
        (root / "proc/sys/kernel/osrelease").write_text(kernel_release + "\n")
        (root / "opt/rocm/.info").mkdir(parents=True, exist_ok=True)
        (root / "opt/rocm/.info/version").write_text(rocm_version + "\n")
        super().__init__(root=root, run=self._run, which=self._which,
                         environ=environ if environ is not None else {"JUPYTERHUB_USER": "roscon02"})
        self.throttle = throttle
        self.calls: list[list[str]] = []

    @staticmethod
    def _which(name: str) -> str | None:
        return f"/opt/rocm/bin/{name}" if name == "amd-smi" else None

    def load(self) -> dict[str, Any] | None:
        try:
            state = json.loads((self.root / "fake_load").read_text())
        except (OSError, ValueError):
            return None
        # A server is up while its process lives: a stopped one (terminated, so it never cleared the file) is not.
        return state if pid_alive(int(state.get("pid", 0))) else None

    def watts(self) -> float:
        state = self.load()
        if state is None:
            return IDLE_NO_SERVER_W
        return IDLE_LOADED_W + WATTS_PER_STREAM * int(state.get("active", 0))

    def _run(self, argv: list[str] | tuple[str, ...]) -> str:
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        tool = Path(argv[0]).name
        rest = argv[1:]
        if tool == "amd-smi" and rest[:1] == ["metric"]:
            text = POWER_FIXTURE.read_text(encoding="utf-8")
            text = text.replace("SOCKET_POWER: 110 W", f"SOCKET_POWER: {self.watts():g} W")
            return text.replace("THROTTLE_STATUS: N/A", f"THROTTLE_STATUS: {self.throttle}")
        if tool == "amd-smi" and rest[:2] == ["static", "--driver"]:
            return (BENCH_FIXTURES / "amdsmi_static_driver_synthetic.txt").read_text(encoding="utf-8")
        if tool == "amd-smi" and rest[:2] == ["static", "--asic"]:
            return (BENCH_FIXTURES / "amdsmi_static_asic_synthetic.txt").read_text(encoding="utf-8")
        if tool == "amd-smi" and rest[:1] == ["firmware"]:
            return (BENCH_FIXTURES / "amdsmi_firmware_synthetic.txt").read_text(encoding="utf-8")
        if tool == "git" and rest[-2:] == ["rev-parse", "HEAD"]:
            return FAKE_COMMIT + "\n"
        if tool.startswith("python"):
            return json.dumps(FAKE_VERSIONS) + "\n"
        raise bench.HostError(f"{tool} is not on this fake pod")


if __name__ == "__main__":
    sys.exit(main())
