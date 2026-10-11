"""L29: the live learning loop, one command from this machine to two finished runs on the box's Learning lab.

    bash tooling/demo/live_loop.sh --slot <GPU Queue row>             # start now
    bash tooling/demo/live_loop.sh --slot <row> --on-approval          # start when the phone approves an export
    bash tooling/demo/live_loop.sh --check                             # what the run needs, nothing started

Two acts of the L10 group fine-tune on pod D (tooling/demo/live_loop_pod.sh), one after the other:
act 1 "solo" is one member's 100 examples, act 2 "duo" is two members' 200 pooled. Each act labels at the pod C
teacher (16 streams), trains a LoRA (8 short steps, a loss row per step), packages and offers. Pod C labels act 2
while pod D trains act 1; act 2 trains on pod D's GPU while act 1 packages on its CPU.

While it runs, this relay
  - reads ~/l29/<stamp>/stages.jsonl and each act's telemetry.jsonl from pod D every --every seconds through the hub
    (lab/lab_push.py's PodSource, never written) and posts new step rows to the box's import route (lab_push's Box
    and Relay: the box keeps a step by run and number, so a grown file adds only its new steps); when an act is done it
    posts the package manifest and the stand-in gate's scores, then checks GET /api/learning/lab for the run;
  - prints a stopwatch line per stage change;
  - records three power traces: pod C's and pod D's samplers (lab/pod_power.py serve, :8790 through the hub proxy,
    every second) and the box's own VDD_IN (telemetry.jetson.power on the gateway's /ws);
  - serves the comparison card at http://127.0.0.1:8788/ (tooling/demo/live_loop_card.html): the three traces with the
    stages drawn on them, the stopwatch, and once both acts are done the energy table, act 2 against act 1.

Energy is T4's rule (contexts/telemetry/application/joules.py): gross joules, the trace integrated over the window by
contexts/telemetry/domain/energy.py's integrate (bracketing samples, interpolated edges), nothing subtracted, the
idle beside it (the calibrated figure of config/telemetry_idle.yaml and the quiet reading measured just before the
start). Windows: labeling on pod C from the act's start to the trainer's start (the pool merge, under a second, is in
it); training on pod D from the trainer's start to its end (the model load is in it); packaging on pod D from the
trainer's end to the act's end (merge, GGUF, Q4_K_M, the stand-in gate and the offer, which take under a second). Each
pod's clock is mapped onto this machine's from its own samples (the newest sample against the time it arrived).

Output in --out (default ../L29-data/<stamp>): relay.log, stages.jsonl, power-pod-c.jsonl, power-pod-d.jsonl,
power-box.jsonl, comparison.json, the telemetry and manifests posted, and one ledger row per act and stage
(verb live-loop.energy) in the lab's ledger (LAB_STATE_DIR, default lab/state). Secrets: the pod C hub token rides to
pod D in a file beside the pod script that the script deletes at once; nothing prints it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import httpx  # noqa: E402

from contexts.telemetry.domain.energy import Window, attribute, integrate  # noqa: E402
from lab import controller  # noqa: E402
from lab.lab_push import Box, PodSource, Relay, complete_lines  # noqa: E402
from lab.machine import ready  # noqa: E402
from lab.proxy import PodLink  # noqa: E402

POD_SCRIPT = HERE / "live_loop_pod.sh"
CARD = HERE / "live_loop_card.html"
TASK = "desk-objects"
SAMPLER_PORT = 8790
IDLE_W = {"pod-c": 10.27, "pod-d": 10.27, "box": 15.76}   # config/telemetry_idle.yaml (pod-a's figure for both pods)
ACTS = ("solo", "duo")
COMMAND_T = time.time()   # the start command: the stopwatch's zero unless --on-approval waits for the phone


def now_s() -> float:
    return time.time()


def iso_s(text: str) -> float:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def env_value(key: str) -> str | None:
    found, _ = ready.collect_env([key], os.environ, ready.default_env_files(os.environ))
    return (found.get(key) or "").strip() or None


class Log:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.lock = threading.Lock()

    def __call__(self, line: str) -> None:
        with self.lock:
            print(line, flush=True)
            if self.path:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")


# --- power traces --------------------------------------------------------------------------------------

class Trace:
    """One node's power samples: (node time, watts), plus the node-minus-local clock offset seen so far."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.points: list[tuple[float, float]] = []
        self.offset: float | None = None   # node clock minus this machine's, the largest (ts - arrival) seen
        self.lock = threading.Lock()
        self.error: str | None = None
        self.raw: list[dict[str, Any]] = []

    def add(self, ts: float, watts: float, arrived: float, raw: dict[str, Any]) -> None:
        with self.lock:
            if not self.points or ts > self.points[-1][0]:
                self.points.append((ts, watts))
                self.raw.append(raw)
            seen = ts - arrived
            self.offset = seen if self.offset is None else max(self.offset, seen)

    def snapshot(self) -> tuple[list[tuple[float, float]], float | None]:
        with self.lock:
            return list(self.points), self.offset

    def to_local(self, node_t: float) -> float:
        return node_t - (self.offset or 0.0)

    def from_local(self, local_t: float) -> float:
        return local_t + (self.offset or 0.0)


class PodPoller(threading.Thread):
    """The pod sampler's /samples through the hub proxy, once a second; starts 90 s back for the idle floor."""

    def __init__(self, link: PodLink, trace: Trace, stop: threading.Event, every_s: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.link, self.trace, self.stop, self.every_s = link, trace, stop, every_s
        self.after: int | None = None

    def get(self, path: str, **params: str) -> Any:
        params["_"] = f"{time.time():.3f}"
        return self.link.request("GET", self.link.proxy_rel(SAMPLER_PORT, path), params=params, timeout=10).json()

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                if self.after is None:
                    self.after = max(0, int(self.get("status")["last_seq"]) - 90)
                body = self.get("samples", after=str(self.after), limit="1000")
                arrived = now_s()
                for item in body.get("samples", []):
                    self.after = max(self.after, int(item["seq"]))
                    e = item["event"]
                    w = e.get("socket_w")
                    if isinstance(w, (int, float)) and math.isfinite(w):
                        self.trace.add(iso_s(e["ts"]), float(w), arrived,
                                       {"seq": item["seq"], "ts": e["ts"], "w": w, "busy": e.get("gfx_activity_pct")})
                self.trace.error = None
            except Exception as exc:   # a missed second is drawn as a gap; the next poll catches up
                self.trace.error = f"{type(exc).__name__}"
            self.stop.wait(self.every_s)


class BoxFeed(threading.Thread):
    """The gateway's /ws: telemetry.jetson.power (VDD_IN, mW) into a trace, and learning.dataset.exported."""

    def __init__(self, box: str, trace: Trace, stop: threading.Event) -> None:
        super().__init__(daemon=True)
        self.url, self.trace, self.stop = f"ws://{box}:8000/ws", trace, stop
        self.exported = threading.Event()
        self.export_event: dict[str, Any] | None = None

    def run(self) -> None:
        import websocket   # websocket-client
        while not self.stop.is_set():
            try:
                ws = websocket.create_connection(self.url, timeout=5)
            except Exception as exc:
                self.trace.error = type(exc).__name__
                self.stop.wait(2)
                continue
            try:
                while not self.stop.is_set():
                    try:
                        msg = ws.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    arrived = now_s()
                    try:
                        e = json.loads(msg)
                    except ValueError:
                        continue
                    kind = e.get("type")
                    if kind == "telemetry.jetson.power":
                        mw = (e.get("rails") or {}).get("VDD_IN")
                        if isinstance(mw, (int, float)) and e.get("ts"):
                            self.trace.add(iso_s(e["ts"]), mw / 1000.0, arrived, {"ts": e["ts"], "w": mw / 1000.0})
                    elif kind == "learning.dataset.exported":
                        self.export_event = {k: e.get(k) for k in ("ts", "task", "dataset_id", "examples", "action_id")}
                        self.exported.set()
            except Exception as exc:
                self.trace.error = type(exc).__name__
            finally:
                try:
                    ws.close()
                except Exception:
                    pass


# --- the run state the card reads ----------------------------------------------------------------------

class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.t0: float | None = None          # local time of the start command (or the approval)
        self.stamp: str | None = None
        self.stages: list[dict[str, Any]] = []   # as the pod wrote them, plus local_s (seconds after t0)
        self.acts: dict[str, dict[str, Any]] = {a: {} for a in ACTS}
        self.lines: list[str] = []
        self.comparison: dict[str, Any] | None = None
        self.finished_s: float | None = None
        self.trigger: dict[str, Any] | None = None


def stage_seconds(stages: list[dict[str, Any]], act: str) -> dict[str, float]:
    """Pod-clock seconds of each stage of one act: labeling, waiting, training, packaging, total."""
    at = {s["stage"]: iso_s(s["ts"]) for s in stages if s.get("act") == act and s.get("ts")}
    out: dict[str, float] = {}
    if "start" in at and "labeled" in at:
        out["labeling"] = at["labeled"] - at["start"]
    if "labeled" in at and "train" in at:
        out["waiting"] = at["train"] - at["labeled"]
    if "train" in at and "trained" in at:
        out["training"] = at["trained"] - at["train"]
    if "trained" in at and "done" in at:
        out["packaging"] = at["done"] - at["trained"]
    if "start" in at and "done" in at:
        out["total"] = at["done"] - at["start"]
    return {k: round(v, 1) for k, v in out.items()}


def window(stages: list[dict[str, Any]], act: str, first: str, last: str) -> tuple[float, float] | None:
    at = {s["stage"]: iso_s(s["ts"]) for s in stages if s.get("act") == act and s.get("ts")}
    return (at[first], at[last]) if first in at and last in at else None


def energy(trace: Trace, pod_d: Trace, span: tuple[float, float] | None) -> dict[str, Any] | None:
    """Gross joules of `trace` over a window given on pod D's clock (moved through this machine's clock)."""
    if span is None:
        return None
    points, _ = trace.snapshot()
    a = trace.from_local(pod_d.to_local(span[0]))
    b = trace.from_local(pod_d.to_local(span[1]))
    j = integrate(points, a, b)
    inside = [w for t, w in points if a <= t <= b]
    secs = b - a
    return {"seconds": round(secs, 1), "joules": None if j is None else round(j, 1),
            "mean_w": None if j is None or secs <= 0 else round(j / secs, 2),
            "peak_w": max(inside) if inside else None, "samples": len(inside)}


def pod_d_shares(stages: list[dict[str, Any]], pod_d: Trace) -> tuple[dict[str, float | None], dict[str, float]]:
    """Pod D's joules per stage window, an overlapping stretch split evenly (T4's split, no token weights), and how
    many seconds of each window another stage shared."""
    spans = {}
    for act in ACTS:
        for name, first, last in (("training", "train", "trained"), ("packaging", "trained", "done")):
            w = window(stages, act, first, last)
            if w:
                spans[f"{act}.{name}"] = w
    points, _ = pod_d.snapshot()
    shared = attribute(points, [Window(k, a, b) for k, (a, b) in spans.items()]) if spans else {}
    overlaps = {}
    for k, (a, b) in spans.items():
        overlaps[k] = round(sum(max(0.0, min(b, d) - max(a, c)) for j, (c, d) in spans.items() if j != k), 1)
    return {k: None if v is None else round(v, 1) for k, v in shared.items()}, overlaps


def quiet_floor(trace: Trace, before_node_t: float, span_s: float = 30.0) -> float | None:
    points, _ = trace.snapshot()
    ws = [w for t, w in points if before_node_t - span_s <= t < before_node_t]
    return round(statistics.median(ws), 2) if ws else None


def ratio(a: Any, b: Any) -> float | None:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b:
        return round(a / b, 3)
    return None


def per(a: Any, b: Any, nd: int = 3) -> float | None:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b:
        return round(a / b, nd)
    return None


# --- the card server -----------------------------------------------------------------------------------

def serve_card(port: int, state: State, traces: dict[str, Trace], pod_d: Trace) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/card", "/index.html"):
                return self.send(200, CARD.read_bytes(), "text/html; charset=utf-8")
            if self.path.startswith("/state"):
                return self.send(200, json.dumps(card_state(state, traces, pod_d)).encode(), "application/json")
            return self.send(404, b"{}", "application/json")

    class Server(ThreadingHTTPServer):
        allow_reuse_address = False   # on Windows reuse lets a second relay bind the same port and never be reached

    srv = Server(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def card_state(state: State, traces: dict[str, Trace], pod_d: Trace) -> dict[str, Any]:
    with state.lock:
        t0 = state.t0
        out: dict[str, Any] = {"stamp": state.stamp, "t0": t0, "now": now_s() - t0 if t0 else None,
                               "lines": state.lines[-14:], "comparison": state.comparison,
                               "finished_s": state.finished_s, "trigger": state.trigger,
                               "acts": {a: dict(v) for a, v in state.acts.items()}, "idle": IDLE_W}
        stages = list(state.stages)
    out["stages"] = [{"act": s.get("act"), "stage": s.get("stage"),
                      "t": round(pod_d.to_local(iso_s(s["ts"])) - t0, 2) if t0 and s.get("ts") else None}
                     for s in stages]
    ref = t0 or now_s()
    series = {}
    for name, tr in traces.items():
        points, _ = tr.snapshot()
        series[name] = {"error": tr.error,
                        "points": [[round(tr.to_local(t) - ref, 2), round(w, 2)] for t, w in points
                                   if tr.to_local(t) - ref >= -30]}
    out["traces"] = series
    return out


# --- the run -------------------------------------------------------------------------------------------

def start_pod(stamp: str, slot: str, steps: int, log: Log) -> dict[str, Any]:
    token = env_value("AUP_HUB_TOKEN_C")
    if not token:
        return {"ok": False, "error": "AUP_HUB_TOKEN_C is not in the environment or the .env files"}
    tmp = Path(tempfile.mkdtemp(prefix="l29-"))
    try:
        tt = tmp / "l29_tt"
        tt.write_text(token, encoding="utf-8")
        answer, code = controller.run(["run-script", "--account", "D", "--timeout", "90", "--with", str(tt),
                                       str(POD_SCRIPT), "start", stamp, slot, str(steps)])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    tail = str(answer.get("output_tail") or "")
    return {"ok": code == 0 and "started" in tail, "seconds": answer.get("seconds"), "tail": tail[-400:],
            "outcome": answer.get("outcome"), "error": answer.get("error")}


def run_loop(args: argparse.Namespace) -> int:
    box_host = args.box or env_value("jetson")
    if not box_host:
        print("no box address: --box or the `jetson` key of .env", file=sys.stderr)
        return 2
    box_url = f"http://{box_host}:8000"
    stamp = args.stamp or datetime.now().strftime("%m%d-%H%M%S")
    out = Path(args.out or REPO.parent / "L29-data" / stamp).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "relay.log")
    accounts = controller.load_accounts(["C", "D"])
    link_c, link_d, link_files = (PodLink.for_account(accounts["C"]), PodLink.for_account(accounts["D"]),
                                  PodLink.for_account(accounts["D"]))
    stop = threading.Event()
    traces = {"pod-c": Trace("pod-c"), "pod-d": Trace("pod-d"), "box": Trace("box")}
    pod_d = traces["pod-d"]
    PodPoller(link_c, traces["pod-c"], stop).start()
    PodPoller(link_d, pod_d, stop).start()
    feed = BoxFeed(box_host, traces["box"], stop)
    feed.start()
    state = State()
    state.stamp = stamp
    serve_card(args.port, state, traces, pod_d)
    log(f"live loop {stamp}: card at http://127.0.0.1:{args.port}/   out {out}")

    def say(line: str) -> None:
        with state.lock:
            state.lines.append(line)
        log(line)

    def first_look() -> None:
        try:
            lab = httpx.get(f"{box_url}/api/learning/lab", timeout=60)
            log(f"box Learning lab answers {lab.status_code}: {lab.json().get('real_runs')} real runs at the start")
        except Exception as exc:
            log(f"box Learning lab does not answer ({type(exc).__name__}); going on, the posts will say more")

    threading.Thread(target=first_look, daemon=True).start()

    if args.on_approval:
        say("waiting for an approved export on the box (learning.dataset.exported on its /ws)")
        while not feed.exported.wait(1.0):
            pass
        with state.lock:
            state.trigger = feed.export_event
        say(f"approval landed: {feed.export_event}")

    t0 = now_s() if args.on_approval else COMMAND_T
    with state.lock:
        state.t0 = t0

    def clock() -> str:
        return f"[{now_s() - t0:6.1f} s]"

    started = start_pod(stamp, args.slot, args.steps, log)
    if not started["ok"]:
        say(f"{clock()} pod D did not start the run: {started}")
        stop.set()
        return 1
    say(f"{clock()} pod D started both acts (run-script {started['seconds']} s)")

    def listed(act: str, run_id: str) -> None:
        """Whether the box's Learning lab lists the run now (its own API, the same list the screen draws)."""
        try:
            lab = httpx.get(f"{box_url}/api/learning/lab", timeout=60).json()
            mine = [r for r in lab.get("runs", []) if r.get("run_id") == run_id]
            shown = mine[0] if mine else None
            with state.lock:
                state.acts[act]["on_box"] = bool(shown)
                state.acts[act]["box_status"] = shown.get("status") if shown else None
            say(f"{clock()} {act}: the box's Learning lab lists {run_id}: "
                f"{'yes, status ' + str(shown.get('status')) if shown else 'not yet'}")
        except Exception as exc:
            say(f"{clock()} {act}: the box's Learning lab did not answer ({type(exc).__name__})")

    source = PodSource(link_files)
    box = Box(box_url, env_value("ATLAS_TOKEN"))
    stages_path = f"l29/{stamp}/stages.jsonl"
    seen_stages = 0
    relays: dict[str, Relay] = {}
    lines_posted: dict[str, int] = {}
    finished: dict[str, bool] = {a: False for a in ACTS}
    failed = False
    deadline = t0 + args.max_seconds
    while now_s() < deadline:
        round_start = now_s()
        data = safe_read(source, stages_path)
        rows = [json.loads(x) for x in complete_lines(data or b"").decode().splitlines() if x.strip()]
        for s in rows[seen_stages:]:
            with state.lock:
                state.stages.append(s)
                if s.get("act") in ACTS and s.get("stage") == "start":
                    state.acts[s["act"]]["run_id"] = s.get("run_id")
            act = s.get("act")
            pod_rel = pod_d.to_local(iso_s(s["ts"])) - t0 if s.get("ts") else None
            took = stage_seconds(rows, act) if act in ACTS else {}
            what = {"labeled": f"labeling took {took.get('labeling')} s", "train": f"waited {took.get('waiting')} s",
                    "trained": f"training took {took.get('training')} s",
                    "done": f"packaging took {took.get('packaging')} s, act {took.get('total')} s, rc {s.get('rc')}"}
            when = f"{pod_rel:6.1f}" if pod_rel is not None else "     ?"
            say(f"{clock()} {act}: {s.get('stage')} at {when} s on the pod clock  {what.get(s.get('stage'), '')}")
            if s.get("stage") == "failed" or (s.get("stage") == "done" and s.get("rc") not in (0, None)):
                failed = True
        seen_stages = len(rows)
        with state.lock:
            acts = {a: dict(v) for a, v in state.acts.items()}
        for act in ACTS:
            run_id = acts[act].get("run_id")
            if not run_id or finished[act]:
                continue
            relay = relays.setdefault(act, Relay(source, box, run_id=run_id, save_dir=out / act))
            tel = f"group-runs/{run_id}/runs/{run_id}/telemetry.jsonl"
            body = complete_lines(safe_read(source, tel) or b"")
            n = body.count(b"\n")
            if n > lines_posted.get(act, 0):
                try:
                    answer = relay.steps(tel)
                except Exception as exc:
                    say(f"{clock()} {act}: steps read failed ({type(exc).__name__}), next round")
                    continue
                lines_posted[act] = n
                with state.lock:
                    state.acts[act]["steps"] = n
                say(f"{clock()} {act}: {n} step rows posted, {answer['box'].brief() if 'box' in answer else answer}")
            done = any(s.get("act") == act and s.get("stage") == "done" for s in rows)
            if done:
                try:
                    answer = relay.steps(tel)
                except Exception:
                    answer = {}
                if "box" in answer:
                    lines_posted[act] = answer.get("lines", lines_posted.get(act, 0))
                version = f"atlas-group-{TASK}-{run_id}"
                try:
                    man = relay.one_file("manifest", f"data/packages/{version}/manifest.json")
                except Exception as exc:
                    man = {"state": f"read failed ({type(exc).__name__})"}
                say(f"{clock()} {act}: manifest {man['box'].brief() if 'box' in man else man}")
                if not args.no_scores:
                    say(f"{clock()} {act}: stand-in scores {post_scores(source, box, run_id, out / act)}")
                finished[act] = True
                if all(finished.values()):
                    listed(act, run_id)
                else:   # GET /api/learning/lab can take 20 s on the box: never hold up the next read of the pod
                    threading.Thread(target=listed, args=(act, run_id), daemon=True).start()
        if all(finished.values()) or failed:
            break
        stop_wait = args.every - (now_s() - round_start)
        if stop_wait > 0:
            time.sleep(stop_wait)
    total = now_s() - t0
    with state.lock:
        state.finished_s = round(total, 1)
    say(f"{clock()} {'FAILED' if failed else 'both runs on the box' if all(finished.values()) else 'TIMED OUT'}: "
        f"total {total:.1f} s from the start command")

    time.sleep(3.0)   # one more second of samples past the end on every trace
    with state.lock:
        stages = list(state.stages)
        acts = {a: dict(v) for a, v in state.acts.items()}
    (out / "stages.jsonl").write_text("".join(json.dumps(s) + "\n" for s in stages), encoding="utf-8")
    comparison = compare(stages, acts, traces, source, out, t0, total)
    comparison["stamp"] = stamp
    with state.lock:
        state.comparison = comparison
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    for name, tr in traces.items():
        (out / f"power-{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tr.raw), encoding="utf-8")
    write_rows(comparison, stamp)
    log(summary_text(comparison))
    if args.hold_minutes > 0:
        log(f"card stays up at http://127.0.0.1:{args.port}/ for {args.hold_minutes} min (Ctrl-C ends it)")
        try:
            time.sleep(args.hold_minutes * 60)
        except KeyboardInterrupt:
            pass
    stop.set()
    return 0 if all(finished.values()) and not failed else 1


def post_scores(source: PodSource, box: Box, run_id: str, save: Path) -> str:
    """The stand-in gate's scores.json for the first member, its event moved onto the trainer's run id."""
    for member in ("ilan", "dom"):
        data = safe_read(source, f"group-runs/{run_id}/gate/{member}/scores.json")
        if not data:
            continue
        body = json.loads(data)
        if isinstance(body.get("event"), dict):
            body["event"]["run_id"] = run_id
        payload = json.dumps(body).encode()
        save.mkdir(parents=True, exist_ok=True)
        (save / "scores.json").write_bytes(payload)
        return box.post("eval", payload, run_id).brief()
    return "no scores.json on the pod"


def safe_read(source: PodSource, path: str) -> bytes | None:
    """A pod read that a hub hiccup turns into "nothing yet" (the next round reads again)."""
    try:
        return source.read(path)
    except Exception:
        return None


def act_facts(source: PodSource, run_id: str) -> dict[str, Any]:
    """Counts from the act's own files: labels and tokens from its ledger copy, the trainer's row, the run's answer."""
    facts: dict[str, Any] = {}
    ledger = safe_read(source, f"group-runs/{run_id}/ledger.jsonl") or b""
    prompt = completion = labeled = 0
    for line in ledger.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        refs = row.get("refs") or {}
        if row.get("verb") == "label-example" and refs.get("run_id") == f"{run_id}.label":
            m = row.get("metrics") or {}
            prompt += int(m.get("label.prompt_tokens") or 0)
            completion += int(m.get("label.completion_tokens") or 0)
            labeled += 1
        elif row.get("verb") == "train" and refs.get("run_id") == run_id:
            m = row.get("metrics") or {}
            facts["trainer"] = {k.split(".", 1)[1]: m.get(k) for k in (
                "trainer.steps", "trainer.examples_seen", "trainer.examples", "trainer.tokens", "trainer.train_s",
                "trainer.loss_first", "trainer.loss_last", "trainer.energy_j", "trainer.mean_socket_w",
                "trainer.steps_with_power")}
    facts["labels"] = {"examples": labeled, "prompt_tokens": prompt, "completion_tokens": completion,
                       "tokens": prompt + completion}
    log_text = (safe_read(source, f"group-runs/{run_id}/run.log") or b"").decode("utf-8", "replace")
    for line in reversed(log_text.splitlines()):
        if line.startswith("{") and '"outcome"' in line:
            try:
                answer = json.loads(line)
            except ValueError:
                continue
            facts["counts"] = answer.get("counts")
            facts["outcome"] = answer.get("outcome")
            break
    return facts


def compare(stages: list[dict[str, Any]], acts: dict[str, dict[str, Any]], traces: dict[str, Trace],
            source: PodSource, out: Path, t0: float, total: float) -> dict[str, Any]:
    pod_c, pod_d, box = traces["pod-c"], traces["pod-d"], traces["box"]
    go = [s for s in stages if s.get("stage") == "go"]
    go_t = iso_s(go[0]["ts"]) if go else None
    box_quiet = quiet_floor(box, box.from_local(t0)) or quiet_floor(box, box.from_local(t0 + 30.0))
    result: dict[str, Any] = {"stamp": None, "t0": t0, "total_s": round(total, 1), "acts": {},
                              "rule": "gross joules (T4), integrated by contexts/telemetry/domain/energy.integrate; "
                                      "idle beside: calibrated (config/telemetry_idle.yaml) and quiet (median of the "
                                      "30 s before the start)",
                              "idle": {"pod-c": {"calibrated_w": IDLE_W["pod-c"],
                                                 "quiet_w": quiet_floor(pod_c, pod_c.from_local(t0))},
                                       "pod-d": {"calibrated_w": IDLE_W["pod-d"],
                                                 "quiet_w": quiet_floor(pod_d, pod_d.from_local(t0))},
                                       "box": {"calibrated_w": IDLE_W["box"],
                                               "quiet_w": box_quiet,
                                               "note": "bimodal: about 5 W between captions, about 18 W while the "
                                                       "box's GPU captions (T13)"}},
                              "clock_offsets_s": {k: None if t.offset is None else round(t.offset, 2)
                                                  for k, t in traces.items()}}
    shared, overlaps = pod_d_shares(stages, pod_d)
    result["pod_d_split"] = ("pod D's training and packaging windows overlap when act 2 trains while act 1 packages; "
                             "each overlapping stretch is split evenly between them (energy.attribute, no tokens), so "
                             "the stages add up to the pod's joules; gross_window_j is the whole window's integral")
    for act in ACTS:
        run_id = acts.get(act, {}).get("run_id")
        if not run_id:
            continue
        facts = act_facts(source, run_id)
        secs = stage_seconds(stages, act)
        label = energy(pod_c, pod_d, window(stages, act, "start", "labeled"))
        train = energy(pod_d, pod_d, window(stages, act, "train", "trained"))
        pack = energy(pod_d, pod_d, window(stages, act, "trained", "done"))
        for part, key in ((train, f"{act}.training"), (pack, f"{act}.packaging")):
            if part is not None:
                part["gross_window_j"] = part["joules"]
                part["joules"] = shared.get(key)
                part["overlap_s"] = overlaps.get(key)
        whole = window(stages, act, "start", "done")
        box_e = energy(box, pod_d, whole)
        lab_t = facts.get("labels", {})
        tr = facts.get("trainer") or {}
        counts = facts.get("counts") or {}
        row: dict[str, Any] = {"run_id": run_id, "seconds": secs, "outcome": facts.get("outcome"),
                               "examples": counts.get("examples"), "train_rows": counts.get("train_rows"),
                               "held_out": counts.get("held_out"),
                               "labeling": {**(label or {}), "examples": lab_t.get("examples"),
                                            "tokens": lab_t.get("tokens"),
                                            "completion_tokens": lab_t.get("completion_tokens"),
                                            "j_per_token": per((label or {}).get("joules"), lab_t.get("tokens")),
                                            "j_per_example": per((label or {}).get("joules"), lab_t.get("examples"), 1),
                                            "idle_j": round(IDLE_W["pod-c"] * label["seconds"], 1) if label else None,
                                            "node": "pod C (teacher)"},
                               "training": {**(train or {}), "steps": tr.get("steps"),
                                            "examples_seen": tr.get("examples_seen"), "tokens": tr.get("tokens"),
                                            "train_s": tr.get("train_s"), "loss_first": tr.get("loss_first"),
                                            "loss_last": tr.get("loss_last"),
                                            "trainer_energy_j": tr.get("energy_j"),
                                            "j_per_step": per((train or {}).get("joules"), tr.get("steps"), 1),
                                            "j_per_example_seen": per((train or {}).get("joules"),
                                                                      tr.get("examples_seen"), 2),
                                            "j_per_train_row": per((train or {}).get("joules"),
                                                                   counts.get("train_rows"), 2),
                                            "idle_j": round(IDLE_W["pod-d"] * train["seconds"], 1) if train else None,
                                            "node": "pod D (trainer)"},
                               "packaging": {**(pack or {}),
                                             "idle_j": round(IDLE_W["pod-d"] * pack["seconds"], 1) if pack else None,
                                             "node": "pod D (trainer)"},
                               "box": {**(box_e or {}),
                                       "idle_j": round(IDLE_W["box"] * box_e["seconds"], 1) if box_e else None,
                                       "node": "the box (Jetson VDD_IN)"}}
        pods_j = [x.get("joules") for x in (label, train, pack) if x]
        row["pods_joules"] = round(sum(pods_j), 1) if pods_j and all(j is not None for j in pods_j) else None
        row["j_per_example"] = per(row["pods_joules"], row["examples"], 1)
        row["start_s"] = round(pod_d.to_local(whole[0]) - t0, 1) if whole else None
        row["end_s"] = round(pod_d.to_local(whole[1]) - t0, 1) if whole else None
        result["acts"][act] = row
    a1, a2 = result["acts"].get("solo"), result["acts"].get("duo")
    if a1 and a2:
        def g(row: dict[str, Any], *path: str) -> Any:
            for p in path:
                row = row.get(p) if isinstance(row, dict) else None
            return row
        keys = {"examples": ("examples",), "act_seconds": ("seconds", "total"),
                "labeling_j": ("labeling", "joules"), "labeling_j_per_token": ("labeling", "j_per_token"),
                "labeling_j_per_example": ("labeling", "j_per_example"), "training_j": ("training", "joules"),
                "training_j_per_step": ("training", "j_per_step"),
                "training_j_per_train_row": ("training", "j_per_train_row"),
                "packaging_j": ("packaging", "joules"), "pods_joules": ("pods_joules",),
                "j_per_example": ("j_per_example",), "box_j": ("box", "joules")}
        result["duo_over_solo"] = {k: ratio(g(a2, *p), g(a1, *p)) for k, p in keys.items()}
    if go_t is not None:
        result["go_s"] = round(pod_d.to_local(go_t) - t0, 1)
    return result


def write_rows(comparison: dict[str, Any], stamp: str) -> None:
    """One ledger row per act and stage, in the lab's row format (lab/machine/record.py)."""
    try:
        from lab.machine.record import record
    except Exception:
        return
    for act, row in comparison.get("acts", {}).items():
        for stage in ("labeling", "training", "packaging", "box"):
            part = row.get(stage) or {}
            if part.get("joules") is None:
                continue
            metrics = {"energy.joules_gross": part.get("joules"), "energy.seconds": part.get("seconds"),
                       "energy.mean_w": part.get("mean_w"), "energy.idle_j": part.get("idle_j")}
            for k in ("j_per_token", "j_per_example", "j_per_step", "j_per_example_seen", "j_per_train_row"):
                if part.get(k) is not None:
                    metrics[f"energy.{k}"] = part[k]
            try:
                record("live-loop.energy", "ok", metrics, {"run_id": row["run_id"]}, task="L29",
                       node={"labeling": "pod-c", "box": "jetson"}.get(stage, "pod-d"),
                       conditions={"act": act, "stage": stage, "stamp": stamp, "examples": row.get("examples"),
                                   "rule": "gross, idle beside (T4)", "sampler": "lab/pod_power.py serve :8790"
                                   if stage != "box" else "telemetry.jetson.power VDD_IN"})
            except Exception:
                return


def summary_text(c: dict[str, Any]) -> str:
    lines = [f"total {c.get('total_s')} s"]
    for act, r in c.get("acts", {}).items():
        lab, tr, pk = r.get("labeling", {}), r.get("training", {}), r.get("packaging", {})
        lines.append(f"{act}: {r.get('examples')} examples; seconds {r.get('seconds')}; labeling {lab.get('joules')} J "
                     f"({lab.get('j_per_token')} J/token, {lab.get('j_per_example')} J/example); training "
                     f"{tr.get('joules')} J ({tr.get('j_per_step')} J/step, {tr.get('j_per_train_row')} J/train row); "
                     f"packaging {pk.get('joules')} J; pods {r.get('pods_joules')} J = {r.get('j_per_example')} J/example; "
                     f"box {r.get('box', {}).get('joules')} J")
    if c.get("duo_over_solo"):
        lines.append(f"duo over solo: {c['duo_over_solo']}")
    return "\n".join(lines)


def fallback(args: argparse.Namespace) -> int:
    """Push a recorded live loop (a relay --out folder or its fixture copy) to the box as finished runs, and serve the
    card with its recorded traces and comparison. No pod is touched; no slot is needed."""
    from lab.lab_push import LocalSource
    folder = Path(args.fallback).expanduser()
    box_host = args.box or env_value("jetson")
    if not box_host:
        print("no box address: --box or the `jetson` key of .env", file=sys.stderr)
        return 2
    comparison = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
    stages = [json.loads(x) for x in (folder / "stages.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    offsets = comparison.get("clock_offsets_s") or {}
    traces = {}
    for name in ("pod-c", "pod-d", "box"):
        tr = Trace(name)
        tr.offset = offsets.get(name) or 0.0
        path = folder / f"power-{name}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
            r = json.loads(line)
            tr.points.append((iso_s(r["ts"]), float(r["w"])))
        traces[name] = tr
    pod_d = traces["pod-d"]
    t0 = comparison.get("t0")
    go = [s for s in stages if s.get("stage") == "go"]
    if t0 is None and go:
        t0 = pod_d.to_local(iso_s(go[0]["ts"])) - float(comparison.get("go_s") or 0.0)
    state = State()
    state.t0, state.stamp, state.stages = t0, comparison.get("stamp") or folder.name, stages
    state.comparison, state.finished_s = comparison, comparison.get("total_s")
    for act in ACTS:
        state.acts[act] = {"run_id": (comparison.get("acts", {}).get(act) or {}).get("run_id")}
    box = Box(f"http://{box_host}:8000", env_value("ATLAS_TOKEN"))
    for act in ACTS:
        run_id = state.acts[act]["run_id"]
        if not run_id or not (folder / act / "telemetry.jsonl").is_file():
            continue
        relay = Relay(LocalSource(), box, run_id=run_id)
        answers = [relay.steps(str(folder / act / "telemetry.jsonl")),
                   relay.one_file("manifest", str(folder / act / "manifest.json"))]
        if not args.no_scores and (folder / act / "scores.json").is_file():
            answers.append(relay.one_file("eval", str(folder / act / "scores.json")))
        briefs = [a["box"].brief() if "box" in a else str(a) for a in answers]
        state.acts[act]["steps"] = answers[0].get("lines")
        state.lines.append(f"{act}: {run_id}: " + "; ".join(briefs))
        print(state.lines[-1], flush=True)
    serve_card(args.port, state, traces, pod_d)
    print(f"recorded card at http://127.0.0.1:{args.port}/ (Ctrl-C ends it)", flush=True)
    try:
        time.sleep(args.hold_minutes * 60)
    except KeyboardInterrupt:
        pass
    return 0


def check(args: argparse.Namespace) -> int:
    box_host = args.box or env_value("jetson")
    print(f"box address set: {bool(box_host)}; AUP_HUB_TOKEN_C set: {bool(env_value('AUP_HUB_TOKEN_C'))}")
    if box_host:
        try:
            r = httpx.get(f"http://{box_host}:8000/api/learning/lab", timeout=20)
            print(f"box Learning lab: {r.status_code}, real runs {r.json().get('real_runs')}")
        except Exception as exc:
            print(f"box Learning lab: no answer ({type(exc).__name__})")
    try:   # the teacher takes requests only: read its queue, never restart it
        link = PodLink.for_account(controller.load_accounts(["C"])["C"])
        text = link.request("GET", link.proxy_rel(8000, "metrics"), params={"_": f"{time.time():.3f}"}, timeout=20).text
        busy = {}
        for line in text.splitlines():
            if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{")) and "reason" not in line:
                name = line.split("{", 1)[0].split(":", 1)[1]
                model = line.split('model_name="', 1)[1].split('"', 1)[0] if 'model_name="' in line else "?"
                busy[f"{model} {name}"] = float(line.rsplit(" ", 1)[1])
        print(f"pod C teacher queue: {busy} (labeling at 16 streams wants these near 0)")
    except Exception as exc:
        print(f"pod C teacher metrics: no answer ({type(exc).__name__})")
    answer, code = controller.run(["run-script", "--account", "D", "--timeout", "90", str(POD_SCRIPT), "check"])
    print(str(answer.get("output_tail") or answer.get("error")))
    return code


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--slot", help="the GPU Queue row id that is Running for pod D (python -m lab slot start)")
    p.add_argument("--steps", type=int, default=8, help="optimizer steps per act (one loss row each)")
    p.add_argument("--every", type=float, default=5.0, help="seconds between reads of the pod's run folder")
    p.add_argument("--box", help="the box's address (default: the `jetson` key of .env)")
    p.add_argument("--port", type=int, default=8788, help="the comparison card's port on 127.0.0.1")
    p.add_argument("--stamp", help="the run stamp (default: local MMDD-HHMMSS); run ids are live-<stamp>-solo|duo")
    p.add_argument("--out", help="where the relay keeps its copies (default ../L29-data/<stamp>)")
    p.add_argument("--on-approval", action="store_true",
                   help="start when the box publishes learning.dataset.exported (the phone approved an export)")
    p.add_argument("--no-scores", action="store_true", help="do not post the stand-in gate's scores to the box")
    p.add_argument("--max-seconds", type=float, default=600.0, help="give up after this long")
    p.add_argument("--hold-minutes", type=float, default=30.0, help="keep the card up this long after the end")
    p.add_argument("--check", action="store_true", help="check the pod and the box; start nothing")
    p.add_argument("--fallback", metavar="DIR",
                   help="no pod: push a recorded loop (fixtures/learning/live-loop-2026-10-11) to the box, serve its card")
    args = p.parse_args(argv)
    if args.check:
        return check(args)
    if args.fallback:
        return fallback(args)
    if not args.slot:
        p.error("--slot is required: GPU time on pod D only inside a GPU Queue slot")
    if args.every < 2:
        p.error("--every is at least 2 seconds")
    return run_loop(args)


if __name__ == "__main__":
    sys.exit(main())
