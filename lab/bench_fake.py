"""A stand-in model server and pod for lab/bench.py: spawned like vLLM or llama-server, read like a Strix Halo pod.

    python lab/bench_fake.py serve --port 0 --port-file serve.port --engine vllm --log serve.log --root FAKE_ROOT

    from lab.bench_fake import FakeHost, FakePower
    host = FakeHost(FAKE_ROOT)              # what lab/bench.py reads on a pod: amd-smi, /proc, /sys, versions
    power = FakePower(FAKE_ROOT)            # watts on a fixed grid, worked out from the fake server's own decodes

Nothing here touches a GPU or the network beyond 127.0.0.1. The tests spawn `serve` as a real process, so a run of the
harness against it is a real cold spawn, a real token stream and a real stop; only the GPU is pretend. What the tests
assert about it is structure and values the fake decides (counts, roles, conditions, cached tokens, mem_busy_percent,
watts bounds), never a rate measured on the test machine's clock.

The server (`serve`). An OpenAI-compatible server with the routes the harness uses: GET /v1/models (the served name,
its checkpoint as root and max_model_len, as vLLM answers), GET /health, and POST /v1/chat/completions streamed as
server-sent events (one token per chunk, a finish chunk, a usage chunk when stream_options.include_usage is set, then
[DONE]). It honours max_tokens and ignore_eos (without it, an answer ends after a third of max_tokens with finish
reason stop). Tokens come --stream-tok-s apart after --ttft-s, or --stall-tok-s apart for a request whose messages
contain the text --stall-match (the harness's nonce names the run and the role, so a test can slow one run only). With --prefix-caching on, a request whose whole prompt
was seen before is a prefix-cache hit and its usage says so in prompt_tokens_details.cached_tokens, as vLLM's does;
the harness's per-request nonce is what keeps measured requests from hitting. With --port 0 it binds a free port
itself and writes it to --port-file, so no test ever races another process for a port. With --profiler-config (vLLM's
JSON, profiler torch) it also answers POST /start_profile and /stop_profile: while profiling, every streamed token
is one decode step in vLLM's naming (an execute_context_0(0)_generation_<n>(<n>) range, a hipGraphLaunch and one
_gemm_kernel), and a stop writes them to torch_profiler_dir as one gzip'd Chrome trace (lab/profile_decode.py's
tests); without it those routes answer 404, as vLLM's do. `python lab/bench_fake.py rocprofv3 [rocprofv3's options]
-d DIR -o NAME -- <python> lab/bench_fake.py serve ...` stands in for rocprofv3: it runs the server in its own process
and appends one row of rocprofv3's kernel trace columns per streamed token to DIR/NAME_kernel_trace.csv (realtime
nanoseconds; %pid% in DIR is its pid).

At start-up it waits --startup-s and writes a serve log in the engine's own words (--engine vllm: the lines vLLM's
b1388b1 source logs for the engine config, the linear kernel, the attention backends and the KV cache; --engine
llamacpp: llama-server's build, n_ctx, flash_attn and llama_kv_cache lines). Those lines are made from the engines'
log statements, not recorded on a pod (fixtures/lab/bench/*_synthetic.log are the same text); R20 records the real
ones. Under --root it then writes what a pod exposes about a running server:

    <root>/proc/<pid>/environ                     its environment, NUL-separated, after any --set-env K=V (a launch
                                                  wrapper that changes what the launch meant to set)
    <root>/sys/class/kfd/kfd/proc/<pid>/queues/N  --queues queue folders
    <root>/sys/class/drm/card0/device/vendor      0x1002
    <root>/sys/class/drm/card0/device/mem_busy_percent   MEM_BUSY_ONE + MEM_BUSY_STEP per extra stream, set when a
                                                  stream starts decoding and held until the next one starts (so
                                                  a read during a lone request always sees MEM_BUSY_ONE)
    <root>/fake_load                              {"pid", "port", "active"}, rewritten whenever a stream starts or ends
    <root>/decodes.jsonl                          one {"pid", "start", "end"} per request, written before its [DONE]
    <root>/servers.jsonl                          one {"pid", "up"} per server, written before it names its port

The pod. `FakeHost` is lab.bench.Host over the fake root and recorded command output: amd-smi metric prints the Strix
Halo output recorded on pod C (fixtures/telemetry/pod_power/strixhalo_amdsmi_metric_power.txt), its SOCKET_POWER set
from <root>/fake_load (IDLE_NO_SERVER_W with no live server, IDLE_LOADED_W loaded, WATTS_PER_STREAM more per stream);
amd-smi static --driver and --asic and amd-smi firmware print fixtures/lab/bench/amdsmi_*_synthetic.txt (made from
amd-smi's output layout, not recorded); python prints the pod's torch and Triton versions; git prints a fixed commit.
`FakePower` is a power source for lab.bench sessions that never depends on a sampling thread being scheduled: its
samples sit on a fixed grid from start() to the moment they are asked for, each IDLE_LOADED_W while a server is up
(IDLE_NO_SERVER_W otherwise) plus WATTS_PER_STREAM per request decoding at that instant by <root>/decodes.jsonl. A
server is up from its line in <root>/servers.jsonl; one whose process has exited counts as up until the end of its
last logged decode (or its start, when it decoded nothing), because the fake keeps no record of the moment it was
stopped: the harness stops a server only after its last answer, and measures no idle window before the stop has
returned. Stdlib only.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

if not __package__:  # run by path (the tests spawn it so): the repo root, one folder up, holds lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import bench, pod_power  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
POWER_FIXTURE = REPO / "fixtures" / "telemetry" / "pod_power" / "strixhalo_amdsmi_metric_power.txt"
BENCH_FIXTURES = REPO / "fixtures" / "lab" / "bench"
FAKE_COMMIT = "b1388b1fbfakefakefakefakefakefakefake0000"
FAKE_VERSIONS = {"torch": "2.9.1+rocm7.13.0", "triton": "3.5.1", "hip": "7.13.0", "python": "3.12.3"}
ROCPROF_COLUMNS = ("Kind", "Agent_Id", "Queue_Id", "Stream_Id", "Thread_Id", "Dispatch_Id", "Kernel_Id", "Kernel_Name",
                   "Correlation_Id", "Start_Timestamp", "End_Timestamp", "LDS_Block_Size", "Scratch_Size", "VGPR_Count",
                   "Accum_VGPR_Count", "SGPR_Count", "Workgroup_Size_X", "Workgroup_Size_Y", "Workgroup_Size_Z",
                   "Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z")
IDLE_NO_SERVER_W = 8.0
IDLE_LOADED_W = 9.2
WATTS_PER_STREAM = 15.0
MEM_BUSY_ONE = 55.0
MEM_BUSY_STEP = 4.0


# --- the server -------------------------------------------------------------------------------------------------

class FakeEngine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = Path(args.root) if args.root else None
        self.port = args.port
        self.active = 0
        self.busy = 0.0
        self.lock = threading.Lock()        # the counters
        self.files = threading.Lock()       # one writer of the shared files at a time, so a stale value never lands last
        self.seen: set[str] = set()
        config = json.loads(args.profiler_config) if getattr(args, "profiler_config", None) else {}
        self.profiler_dir = config.get("torch_profiler_dir") if config.get("profiler") == "torch" else None
        self.profiling = False
        self.profile_events: list[dict[str, Any]] = []
        self.correlation = 0
        self.rocprof_csv: Path | None = None
        if getattr(args, "rocprof_dir", None):
            folder = Path(args.rocprof_dir)
            folder.mkdir(parents=True, exist_ok=True)
            self.rocprof_csv = folder / f"{args.rocprof_name}_kernel_trace.csv"
            self.rocprof_csv.write_text(",".join(f'"{c}"' for c in ROCPROF_COLUMNS) + "\n", encoding="utf-8")

    # vLLM's torch profiler, as far as lab/profile_decode.py drives it

    def profile(self, start: bool) -> int:
        """POST /start_profile and /stop_profile: 404 without --profiler-config (as vLLM, which then has no such
        routes); a stop writes the steps recorded since the start as one gzip'd Chrome trace in its folder."""
        if not self.profiler_dir:
            return 404
        with self.lock:
            if start:
                self.profiling, self.profile_events = True, []
                return 200
            events, self.profiling = self.profile_events, False
        trace = {"deviceProperties": [{"id": 0, "name": "fake gfx1151"}], "traceEvents": events}
        folder = Path(self.profiler_dir)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"fake_worker.{time.time_ns()}.pt.trace.json.gz"
        path.write_bytes(gzip.compress(json.dumps(trace).encode()))
        return 200

    def record_dispatch(self) -> None:
        """Under the fake rocprofv3: one kernel row per streamed token, appended at once (realtime nanoseconds), so
        a stop that kills the process loses nothing."""
        if self.rocprof_csv is None:
            return
        end = time.time_ns()
        with self.files:
            with self.lock:
                self.correlation += 1
                c = self.correlation
            row = ['"KERNEL_DISPATCH"', 2, 1, 0, os.getpid(), c, 17, '"_gemm_kernel.kd"', c, end - 200_000, end,
                   49152, 0, 128, 0, 64, 256, 1, 1, 20480, 1, 1]
            with open(self.rocprof_csv, "a", encoding="utf-8") as fh:
                fh.write(",".join(str(v) for v in row) + "\n")

    def record_step(self) -> None:
        """While profiling, one decode step per streamed token: vLLM's step range, a graph launch and one kernel."""
        with self.lock:
            if not self.profiling:
                return
            self.correlation += 1
            c, n, t = self.correlation, max(1, self.active), time.time() * 1e6
            self.profile_events += [
                {"ph": "X", "cat": "user_annotation", "name": f"execute_context_0(0)_generation_{n}({n})",
                 "pid": 1, "tid": 1, "ts": t, "dur": 100.0},
                {"ph": "X", "cat": "cuda_runtime", "name": "hipGraphLaunch", "pid": 1, "tid": 1, "ts": t + 10.0,
                 "dur": 5.0, "args": {"correlation": c}},
                {"ph": "X", "cat": "kernel", "name": "_gemm_kernel", "pid": 0, "tid": 7, "ts": t + 50.0, "dur": 200.0,
                 "args": {"correlation": c, "grid": [80, 1, 1], "block": [256, 1, 1]}}]

    # files a pod exposes

    def _write(self, rel: str, data: str | bytes) -> None:
        if self.root is None:
            return
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        tmp.write_bytes(data.encode() if isinstance(data, str) else data)
        for _ in range(200):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:  # Windows: a reader has it open for a moment
                time.sleep(0.005)
        raise OSError(f"could not replace {path}")

    def publish_process(self) -> None:
        pid = os.getpid()
        environ = b"".join(f"{k}={v}".encode() + b"\0" for k, v in os.environ.items())
        self._write(f"proc/{pid}/environ", environ)
        for q in range(self.args.queues):
            if self.root is not None:
                (self.root / f"sys/class/kfd/kfd/proc/{pid}/queues/{q}").mkdir(parents=True, exist_ok=True)
        self._write("sys/class/drm/card0/device/vendor", "0x1002\n")
        self.publish_load()
        if self.root is not None:
            with self.files:
                with open(self.root / "servers.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"pid": pid, "up": time.time()}) + "\n")

    def publish_load(self) -> None:
        with self.files:
            with self.lock:
                active, busy = self.active, self.busy
            self._write("fake_load", json.dumps({"pid": os.getpid(), "port": self.port, "active": active}))
            self._write("sys/class/drm/card0/device/mem_busy_percent", f"{busy:g}\n")

    def log_decode(self, start: float, end: float) -> None:
        if self.root is None:
            return
        with self.files:
            with open(self.root / "decodes.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"pid": os.getpid(), "start": start, "end": end}) + "\n")

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
                "main: server is listening - starting the main loop", ""])
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
        prompt_text = json.dumps(body.get("messages") or [], sort_keys=True)
        images = prompt_text.count('"image_url"')
        max_tokens = int(body.get("max_tokens") or 16)
        n = max_tokens if body.get("ignore_eos") else max(1, max_tokens // 3)
        finish = "length" if body.get("ignore_eos") else "stop"
        with self.lock:
            hit = a.prefix_caching == "on" and prompt_text in self.seen
            self.seen.add(prompt_text)
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        time.sleep(a.ttft_s)
        with self.lock:
            self.active += 1
            self.busy = min(100.0, MEM_BUSY_ONE + MEM_BUSY_STEP * (self.active - 1))
        self.publish_load()   # before the first token goes out, so a read at any token sees this stream decoding
        rate = a.stall_tok_s if a.stall_match and a.stall_match in prompt_text else a.stream_tok_s
        start = time.time()
        try:
            for i in range(n):
                if i:
                    time.sleep(1.0 / rate)
                text, token_id, logprob = self.token(prompt_text, i)
                choice: dict[str, Any] = {"index": 0, "delta": {"content": text}, "finish_reason": None}
                if body.get("logprobs"):   # as vLLM answers return_tokens_as_token_ids: the id stands for the token
                    name = f"token_id:{token_id}" if body.get("return_tokens_as_token_ids") else text
                    choice["logprobs"] = {"content": [{"token": name, "logprob": logprob,
                                                       "bytes": list(text.encode()), "top_logprobs": []}]}
                self._event(handler, {"object": "chat.completion.chunk", "model": a.model, "choices": [choice]})
                self.record_step()
                self.record_dispatch()
            self._event(handler, {"object": "chat.completion.chunk", "model": a.model,
                                  "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        finally:
            with self.lock:
                self.active -= 1
            self.publish_load()
            self.log_decode(start, time.time())
        if (body.get("stream_options") or {}).get("include_usage"):
            prompt_tokens = len(prompt_text) // 4 + 300 * images
            self._event(handler, {"object": "chat.completion.chunk", "model": a.model, "choices": [],
                                  "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n,
                                            "total_tokens": prompt_tokens + n,
                                            "prompt_tokens_details": {"cached_tokens": prompt_tokens if hit else 0}}})
        handler.wfile.write(b"data: [DONE]\n\n")
        handler.wfile.flush()

    def token(self, prompt_text: str, i: int) -> tuple[str, int, float]:
        """Token i of the answer: its text, a token id fixed by the prompt and the position, and a logprob. From
        --diverge-at (and --retoken-at) on, for the prompts that contain --diverge-match, the answer differs from the
        default one: the text and the id (diverge), or the id alone (retoken, the same text under another id)."""
        a = self.args
        text, token_id, logprob = f"t{i} ", 1000 + (zlib.crc32(prompt_text.encode()) + 7919 * i) % 50000, -0.01 * (i + 1)
        if a.diverge_match and a.diverge_match not in prompt_text:
            return text, token_id, logprob
        if a.diverge_at is not None and i >= a.diverge_at:
            return f"x{i} ", token_id + 1, logprob - 0.5
        if a.retoken_at is not None and i == a.retoken_at:
            return text, token_id + 1, logprob
        return text, token_id, logprob

    @staticmethod
    def _event(handler: BaseHTTPRequestHandler, chunk: dict[str, Any]) -> None:
        handler.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        handler.wfile.flush()


def make_handler(engine: FakeEngine) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"
        timeout = 60   # a client that stops reading cannot hold a handler thread for ever

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
            if self.path.rstrip("/") in ("/start_profile", "/stop_profile"):
                status = engine.profile(self.path.rstrip("/") == "/start_profile")
                return self._json(status, {} if status == 200 else {"detail": "Not Found"})
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
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine))
    server.daemon_threads = True
    engine.port = server.server_address[1]
    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(engine.serve_log(), encoding="utf-8")
    engine.publish_process()
    if args.port_file:   # written last and whole: once the harness can read it, the port is bound and listening
        tmp = Path(args.port_file + ".tmp")
        tmp.write_text(str(engine.port))
        os.replace(tmp, args.port_file)
    server.serve_forever(poll_interval=0.05)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lab/bench_fake.py", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, required=True, help="0: bind a free port and write it to --port-file")
    s.add_argument("--port-file", default=None)
    s.add_argument("--engine", choices=("vllm", "llamacpp"), default="vllm")
    s.add_argument("--log", required=True)
    s.add_argument("--root", default=None)
    s.add_argument("--model", default="teacher")
    s.add_argument("--checkpoint", default="/home/jovyan/models/Qwen3.8-27B-INT4")
    s.add_argument("--max-len", type=int, default=16384)
    s.add_argument("--kv-tokens", type=int, default=163840)
    s.add_argument("--parallel", type=int, default=16)
    s.add_argument("--stream-tok-s", type=float, default=400.0)
    s.add_argument("--ttft-s", type=float, default=0.01)
    s.add_argument("--stall-match", default=None, help="requests whose messages contain this text stream slowly")
    s.add_argument("--stall-tok-s", type=float, default=4.0)
    s.add_argument("--startup-s", type=float, default=0.05)
    s.add_argument("--queues", type=int, default=3)
    s.add_argument("--prefix-caching", choices=("on", "off"), default="on")
    s.add_argument("--diverge-at", type=int, default=None, help="from this position on, the answer's text and ids differ")
    s.add_argument("--retoken-at", type=int, default=None, help="at this position, the same text under another id")
    s.add_argument("--diverge-match", default=None, help="only prompts containing this text are changed by the two")
    s.add_argument("--set-env", action="append", metavar="K=V")
    s.add_argument("--profiler-config", default=None, help="vLLM's JSON: with profiler torch, the profile routes "
                                                            "answer and write traces to torch_profiler_dir")
    s.add_argument("--rocprof-dir", default=None, help=argparse.SUPPRESS)    # set by the fake rocprofv3
    s.add_argument("--rocprof-name", default="run", help=argparse.SUPPRESS)
    return p


def fake_rocprof(argv: list[str]) -> int:
    """`rocprofv3 [options] -d DIR -o NAME -- <python> lab/bench_fake.py serve ...` in one process: the server runs
    here (so stopping this process stops it, on Windows too) and writes rocprofv3's kernel trace CSV columns to
    DIR/NAME_kernel_trace.csv, DIR with %pid% replaced, one row per streamed token."""
    if "--" not in argv:
        raise SystemExit("fake rocprofv3: no -- before the command")
    tool, rest = argv[:argv.index("--")], argv[argv.index("--") + 1:]
    out_dir = tool[tool.index("-d") + 1] if "-d" in tool else "."
    name = tool[tool.index("-o") + 1] if "-o" in tool else "run"
    if "serve" not in rest:
        raise SystemExit("fake rocprofv3: it only runs lab/bench_fake.py serve")
    serve_args = rest[rest.index("serve") + 1:]
    args = build_parser().parse_args(["serve", *serve_args, "--rocprof-dir",
                                      out_dir.replace("%pid%", str(os.getpid())), "--rocprof-name", name])
    return serve(args)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["rocprofv3"]:
        return fake_rocprof(argv[1:])
    args = build_parser().parse_args(argv)
    return serve(args)


def rocprof_command() -> list[str]:
    """The command lab/profile_decode_pod.py puts in front of the server instead of rocprofv3 in tests."""
    return [spawn_python(), str(Path(__file__).resolve()), "rocprofv3"]


def spawn_python() -> str:
    """The interpreter to spawn the fake with, so the spawned pid is the server's own: on Windows a venv's python.exe
    is a launcher that starts the real interpreter as a child, and the child's pid is the one that serves."""
    return getattr(sys, "_base_executable", None) or sys.executable


def serve_command(**options: Any) -> list[str]:
    """The argv the harness spawns for the fake: port 0 with {port_file} and {log} left for the harness to fill."""
    argv = [spawn_python(), str(Path(__file__).resolve()), "serve", "--port", "{port}", "--port-file", "{port_file}",
            "--log", "{log}"]
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        for item in (value if isinstance(value, (list, tuple)) else [value]):
            argv += [flag, str(item)]
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


class FakePower:
    """Watts for a lab.bench session on a fixed grid, from the fake server's decodes (see the module docstring)."""

    available = True
    reason = None

    def __init__(self, root: str | Path, *, period_s: float = 0.01) -> None:
        self.root = Path(root)
        self.period_s = period_s
        self.t0: float | None = None

    def start(self) -> None:
        self.t0 = time.time() - self.period_s

    def stop(self) -> None:
        return None

    def _lines(self, name: str) -> list[dict[str, Any]]:
        try:
            lines = (self.root / name).read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except ValueError:   # a line still being written
                continue
        return out

    def decodes(self) -> list[tuple[float, float]]:
        return [(float(d["start"]), float(d["end"])) for d in self._lines("decodes.jsonl")]

    def servers(self) -> list[tuple[float, float | None]]:
        """(up, down) per server; down None while its process lives, else the end of its last decode."""
        ends: dict[int, float] = {}
        for d in self._lines("decodes.jsonl"):
            pid = int(d.get("pid", 0))
            ends[pid] = max(ends.get(pid, 0.0), float(d["end"]))
        out = []
        for s in self._lines("servers.jsonl"):
            pid, up = int(s["pid"]), float(s["up"])
            out.append((up, None if pid_alive(pid) else max(up, ends.get(pid, up))))
        return out

    def events(self) -> list[dict[str, Any]]:
        if self.t0 is None:
            return []
        servers = self.servers()
        decodes = self.decodes()
        end = time.time() + self.period_s
        out = []
        k = 0
        while self.t0 + k * self.period_s <= end:
            t = self.t0 + k * self.period_s
            busy = sum(1 for a, b in decodes if a <= t <= b)
            up = any(a <= t and (b is None or t <= b) for a, b in servers)
            out.append({"type": pod_power.GPU_POWER, "ts": pod_power.iso(t), "t_local": t,
                        "socket_w": (IDLE_LOADED_W if up else IDLE_NO_SERVER_W) + WATTS_PER_STREAM * busy, "source": "amd-smi",
                        "power_field": "socket_power", "sampler": "lab/bench_fake.py FakePower", "throttled": None})
            k += 1
        return out


if __name__ == "__main__":
    sys.exit(main())
