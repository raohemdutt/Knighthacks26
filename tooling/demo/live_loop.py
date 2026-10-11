"""L29: the live learning loop, one command from this machine to two finished runs on the box's Learning lab.

    bash tooling/demo/live_loop.sh --slot <GPU Queue row>             # start now
    bash tooling/demo/live_loop.sh --slot <row> --on-approval          # start when the phone approves an export
    bash tooling/demo/live_loop.sh --check                             # what the run needs, nothing started

Two acts of the L10 group fine-tune (tooling/demo/live_loop_pod.sh), one after the other: act 1 "solo" is one
member's examples, act 2 "duo" is two members' pooled (--solo-examples and --duo-examples per member, 12 each by
default, set from the pair's measured 2.34 s a label to keep the run under 4 minutes; 15 and 15 is the stretch; the
budget is in LIVE_LOOP_RUN_OF_SHOW.md). Each act labels at the teacher
(16 streams), trains a LoRA (8 short steps, a loss row per step), packages and offers.

The teacher is the 122B pair served across pods C and D (vLLM pipeline-parallel, its API on pod C's port 8000 as
"teacher" and "group-brain", the same URL and token as before): --teacher-expect Qwen3.5-122B is the checkpoint the
labeling checks for, and its energy is both pods'. The 27B on pod C alone is the fallback: --teacher-expect
Qwen3.8-27B (its energy is pod C's). Act 1 trains on pod D alone. Act 2 trains on both pods at once with R65's
two-pod LoRA (lab/r65_local_sgd.py: rank 0 on pod D, rank 1 on pod C, each on half the rows, int8 deltas swapped
through the hub proxy every 4 steps; --duo-pods 1 keeps it on pod D). Act 2 labels while act 1 trains, and trains
while act 1 packages on pod D's CPU.

While it runs, this relay
  - reads ~/l29/<stamp>/stages.jsonl and each act's telemetry.jsonl from pod D every --every seconds through the hub
    (lab/lab_push.py's PodSource, never written) and posts new step rows to the box's import route (lab_push's Box
    and Relay: the box keeps a step by run and number, so a grown file adds only its new steps); when an act is done it
    posts the package manifest and the stand-in gate's scores, then checks GET /api/learning/lab for the run;
  - posts each act's stage lines as they appear (kind=stage: the Training run tab's stage strip and "running"); act
    2's start line carries "2 members" and its pods, and pod C's rank 1 posts its own lines and steps under the part
    run id live-<stamp>-duo.pod-c, which the box shows inside act 2;
  - prints a stopwatch line per stage change;
  - records three power traces: pod C's and pod D's samplers (lab/pod_power.py serve, :8790 through the hub proxy,
    every second) and the box's own VDD_IN (telemetry.jetson.power on the gateway's /ws);
  - when an act finishes, posts its power windows (kind=power: labeling on pod C, training and packaging on pod D,
    and the whole act on all three nodes) and the comparison so far (kind=compare, the finished acts only); at the
    end it posts both again with the last seconds of samples and the comparison of both acts. The box's Compare tab
    (/lab/?tab=compare, GET /api/learning/lab/compare) draws them: that is where the audience looks;
  - with --card only (off by default), also serves the old local card at http://127.0.0.1:8788/
    (tooling/demo/live_loop_card.html).

Energy is T4's rule (contexts/telemetry/application/joules.py): gross joules, the trace integrated over the window by
contexts/telemetry/domain/energy.py's integrate (bracketing samples, interpolated edges), nothing subtracted, the
idle beside it (the calibrated figure of config/telemetry_idle.yaml and the quiet reading measured just before the
start). The arithmetic lives in contexts/learning/application/loop_energy.py, which the box's compare route uses
too, so the relay and the box compute the same numbers. Windows: labeling on pod C from the act's start to the
trainer's start (the pool merge, under a second, is in it); training on pod D from the trainer's start to its end (the model load is in it); packaging on pod D from the
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
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import httpx  # noqa: E402

from contexts.learning.application import loop_energy as le  # noqa: E402
from contexts.learning.application.loop_energy import (ACTS, IDLE_W, integrate, iso_s, stage_seconds,  # noqa: E402,F401
                                                       window)
from lab import controller  # noqa: E402
from lab.lab_push import Box, PodSource, Relay, complete_lines  # noqa: E402
from lab.machine import ready  # noqa: E402
from lab.proxy import PodLink  # noqa: E402

POD_SCRIPT = HERE / "live_loop_pod.sh"
CARD = HERE / "live_loop_card.html"
TASK = "desk-objects"
SAMPLER_PORT = 8790
TEACHER_PAIR = "Qwen3.5-122B"      # the two-pod teacher's checkpoint (pods C and D, API on pod C's port 8000)
TEACHER_SINGLE = "Qwen3.8-27B"     # the fallback: the 27B on pod C alone
PART = ".pod-c"                    # act 2's rank 1 on pod C posts under <duo run id>.pod-c
COMMAND_T = time.time()   # the start command: the stopwatch's zero unless --on-approval waits for the phone


def now_s() -> float:
    return time.time()


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
        self.stages_c: list[dict[str, Any]] = []   # pod C's rank 1 lines (act 2 on two pods), on pod C's clock


def pod_d_shares(stages: list[dict[str, Any]], pod_d: Trace) -> tuple[dict[str, float | None], dict[str, float]]:
    """Pod D's joules per stage window, an overlapping stretch split evenly (loop_energy.pod_d_shares)."""
    return le.pod_d_shares(stages, pod_d.snapshot()[0])


def node_traces(traces: dict[str, Trace]) -> dict[str, tuple[list[tuple[float, float]], float | None]]:
    """The traces as loop_energy takes them: (points on the node's clock, the node's clock minus this machine's)."""
    return {name: tr.snapshot() for name, tr in traces.items()}


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


# --- what the box keeps: stage lines, power windows, the comparison ------------------------------------

SAMPLERS = {"pod-c": "lab/pod_power.py serve :8790 (socket_w), polled through the hub proxy",
            "pod-d": "lab/pod_power.py serve :8790 (socket_w), polled through the hub proxy",
            "box": "telemetry.jetson.power VDD_IN on the gateway's /ws"}
EDGE_S = 5.0   # samples kept either side of a window, so the box can bracket its edges


def iso_z(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).isoformat().replace("+00:00", "Z")


def act_lines(rows: list[dict[str, Any]], act: str) -> list[dict[str, Any]]:
    """One act's stage lines with the task on each, as kind=stage takes them."""
    return [{**r, "task": TASK} for r in rows if r.get("act") == act]


def teacher_nodes_for(expect: str, given: str | None = None) -> tuple[str, ...]:
    """The pods the teacher runs on: --teacher-nodes, else both pods for the pair and pod C for the 27B."""
    if given:
        return tuple(n.strip() for n in given.split(",") if n.strip())
    return ("pod-c", "pod-d") if TEACHER_PAIR.lower() in expect.lower() else ("pod-c",)


def power_windows(stages: list[dict[str, Any]], act: str, run_id: str, traces: dict[str, Trace],
                  where: dict[str, dict[str, Any]], teacher_nodes: tuple[str, ...] = ("pod-c",)
                  ) -> list[tuple[str, dict[str, Any]]]:
    """(stage, window) for each of the act's windows the traces cover, in the box's kind=power shape: the samples on
    the node's own clock a few seconds either side, the node's clock offset, the sampler and its measured period, the
    idle floor, and the hub and account of a pod. Labeling on every teacher pod, training on every pod that trains
    (pod C's rank 1 on pod C's clock), packaging on pod D, and the whole act on all three nodes."""
    whole = le.window(stages, act, "start", "done")
    spans = [(stage, node, span, clock) for stage, node, span, clock in le.act_windows(stages, act, teacher_nodes)]
    spans += [("act", node, whole, "pod-d") for node in ("pod-c", "pod-d", "box")] if whole else []
    out = []
    for stage, node, span, clock in spans:
        tr = traces.get(node)
        if tr is None:
            continue
        points, off = tr.snapshot()
        a, b = le.moved(span, traces[clock].offset, off)
        inside = [(t, w) for t, w in points if a - EDGE_S <= t <= b + EDGE_S]
        if not inside:
            continue
        gaps = sorted(y - x for (x, _), (y, _) in zip(inside, inside[1:]))
        body = {"run_id": run_id, "stage": stage, "node": node, "sampler": SAMPLERS.get(node),
                "period_ms": round(gaps[len(gaps) // 2] * 1000, 1) if gaps else None,
                "idle_w": IDLE_W.get(node), "quiet_w": le.quiet_floor(points, a), "clock_offset_s": off,
                "start": iso_z(a), "end": iso_z(b), "error": tr.error,
                **{k: v for k, v in (where.get(node) or {}).items() if v is not None},
                "samples": [{"ts": iso_z(t), "w": w} for t, w in inside]}
        out.append((stage, {k: v for k, v in body.items() if v is not None}))
    return out


def post_windows(box: Box, stages: list[dict[str, Any]], act: str, run_id: str, traces: dict[str, Trace],
                 where: dict[str, dict[str, Any]], teacher_nodes: tuple[str, ...] = ("pod-c",)) -> str:
    """Post the act's power windows; one line of what the box said."""
    said = []
    for stage, body in power_windows(stages, act, run_id, traces, where, teacher_nodes):
        answer = box.post("power", json.dumps(body).encode(), run_id, {"stage": stage})
        said.append(f"{stage}.{body['node']} {answer.status or 'no answer'}")
    return ", ".join(said) or "no samples to post yet"


def post_comparison(box: Box, comparison: dict[str, Any]) -> str:
    return box.post("compare", json.dumps(comparison).encode(), None).brief()


def pod_where(link: PodLink) -> dict[str, Any]:
    """The measured-row fields of a pod's samples: its hub's host and its account letter (never a token)."""
    try:
        host = httpx.URL(link.base_url).host
    except Exception:
        host = None
    return {"hub": host, "account": link.letter}


# --- the run -------------------------------------------------------------------------------------------

def pod_settings(args: argparse.Namespace, c_user: str | None) -> str:
    """The l29_env file pod D reads (KEY=VALUE, only known keys; never a token)."""
    lines = {"SOLO_EXAMPLES": args.solo_examples, "DUO_EXAMPLES": args.duo_examples, "DUO_PODS": args.duo_pods,
             "TEACHER_EXPECT": args.teacher_expect, "C_USER": c_user or "", "L29_SERIAL": 1 if args.serial else 0}
    return "".join(f"{k}={v}\n" for k, v in lines.items())


def run_script(account: str, files: dict[str, str], args: list[str]) -> dict[str, Any]:
    """One pod script run with small files beside it (written to a temporary folder here, removed at once)."""
    tmp = Path(tempfile.mkdtemp(prefix="l29-"))
    try:
        withs = []
        for name, text in files.items():
            (tmp / name).write_text(text, encoding="utf-8")
            withs += ["--with", str(tmp / name)]
        answer, code = controller.run(["run-script", "--account", account, "--timeout", "90", *withs,
                                       str(POD_SCRIPT), *args])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    tail = str(answer.get("output_tail") or "")
    return {"ok": code == 0 and ("started" in tail or "waiting" in tail), "seconds": answer.get("seconds"),
            "tail": tail[-400:], "outcome": answer.get("outcome"), "error": answer.get("error")}


def start_pod(stamp: str, slot: str, steps: int, log: Log, settings: str = "") -> dict[str, Any]:
    """Pod D: both acts. The teacher's token (pod C's account: the API is on pod C's port 8000 for the pair and the
    27B alike) rides in l29_tt, deleted by the pod script at once; nothing prints it."""
    token = env_value("AUP_HUB_TOKEN_C")
    if not token:
        return {"ok": False, "error": "AUP_HUB_TOKEN_C is not in the environment or the .env files"}
    files = {"l29_tt": token, **({"l29_env": settings} if settings else {})}
    return run_script("D", files, ["start", stamp, slot, str(steps)])


def start_rank1(stamp: str, slot: str, d_user: str) -> dict[str, Any]:
    """Pod C: rank 1 of act 2, waiting for pod D's handoff. Pod D's account token (rank 1 reads pod D's handoff and
    mailbox through the hub proxy) rides in l29_td, moved off the upload path at once and deleted when rank 1 ends."""
    token = env_value("AUP_HUB_TOKEN_D")
    if not token:
        return {"ok": False, "error": "AUP_HUB_TOKEN_D is not in the environment or the .env files"}
    return run_script("C", {"l29_td": token}, ["rank1", stamp, slot, d_user])


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
    link_c_files = PodLink.for_account(accounts["C"])
    teachers = teacher_nodes_for(args.teacher_expect, args.teacher_nodes)
    two_pods = args.duo_pods == 2
    stop = threading.Event()
    traces = {"pod-c": Trace("pod-c"), "pod-d": Trace("pod-d"), "box": Trace("box")}
    pod_d = traces["pod-d"]
    PodPoller(link_c, traces["pod-c"], stop).start()
    PodPoller(link_d, pod_d, stop).start()
    feed = BoxFeed(box_host, traces["box"], stop)
    feed.start()
    state = State()
    state.stamp = stamp
    if args.card:
        serve_card(args.port, state, traces, pod_d)
    where = {"pod-c": pod_where(link_c), "pod-d": pod_where(link_d)}
    log(f"live loop {stamp}: the box's Compare tab is http://{box_host}:8000/lab/?tab=compare"
        + (f"   local card http://127.0.0.1:{args.port}/" if args.card else "") + f"   out {out}")

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

    if two_pods:   # rank 1 first, so it is waiting when act 2's handoff appears; its start overlaps pod D's
        rank1: dict[str, Any] = {}
        waiter = threading.Thread(target=lambda: rank1.update(start_rank1(stamp, args.slot_c, accounts["D"].user)))
        waiter.start()
    started = start_pod(stamp, args.slot, args.steps, log, pod_settings(args, accounts["C"].user))
    if two_pods:
        waiter.join()
        if not rank1.get("ok"):
            say(f"{clock()} pod C did not start rank 1: {rank1}")
            stop.set()
            return 1
        say(f"{clock()} pod C's rank 1 waits for act 2 (run-script {rank1.get('seconds')} s)")
    if not started["ok"]:
        say(f"{clock()} pod D did not start the run: {started}")
        stop.set()
        return 1
    say(f"{clock()} pod D started both acts (run-script {started['seconds']} s); teacher {args.teacher_expect} on "
        f"{' and '.join(teachers)}; act 2 on {args.duo_pods} pod(s)")

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
    source_c = PodSource(link_c_files)
    box = Box(box_url, env_value("ATLAS_TOKEN"))
    stages_path = f"l29/{stamp}/stages.jsonl"
    seen_stages = 0
    rank1_relay = Relay(source_c, box, run_id=f"live-{stamp}-duo{PART}", save_dir=out / "duo-pod-c")
    rank1_state = {"lines": 0, "stage_posted": 0, "steps": 0}

    def read_rank1(final: bool = False) -> list[dict[str, Any]]:
        """Pod C's rank 1: its stage lines (posted under the part run id, kept for the energy windows) and its steps."""
        if not two_pods:
            return []
        part = rank1_relay.run_id
        data = safe_read(source_c, f"l29/{stamp}/stages-c.jsonl")
        lines = [json.loads(x) for x in complete_lines(data or b"").decode().splitlines() if x.strip()]
        for s in lines[rank1_state["lines"]:]:
            say(f"{clock()} duo: pod C {s.get('stage')} {('rc ' + str(s.get('rc'))) if 'rc' in s else ''}")
        rank1_state["lines"] = len(lines)
        if len(lines) > rank1_state["stage_posted"]:
            answer = box.post("stage", "".join(json.dumps({**x, "task": TASK}) + "\n" for x in lines).encode(), part,
                              {"task": TASK})
            if answer.ok:
                rank1_state["stage_posted"] = len(lines)
        tel = f"l29/{stamp}/rank1/telemetry.jsonl"
        n = complete_lines(safe_read(source_c, tel) or b"").count(b"\n")
        if n > rank1_state["steps"] or (final and n):
            try:
                answer = rank1_relay.steps(tel)
                rank1_state["steps"] = n
                say(f"{clock()} duo: pod C {n} step rows posted, {answer['box'].brief() if 'box' in answer else answer}")
            except Exception as exc:
                say(f"{clock()} duo: pod C steps read failed ({type(exc).__name__}), next round")
        return lines
    stage_posted: dict[str, int] = {a: 0 for a in ACTS}
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
        rows_c = read_rank1()
        with state.lock:
            state.stages_c = rows_c
            acts = {a: dict(v) for a, v in state.acts.items()}
        for act in ACTS:   # the stage strip on the box: each act's lines as they appear (the same lines again add nothing)
            run_id, lines = acts[act].get("run_id"), act_lines(rows, act)
            if run_id and len(lines) > stage_posted[act]:
                answer = box.post("stage", "".join(json.dumps(x) + "\n" for x in lines).encode(), run_id, {"task": TASK})
                if answer.ok:
                    stage_posted[act] = len(lines)
                else:
                    say(f"{clock()} {act}: stage lines {answer.brief()}, next round")
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
                if act == "duo":
                    rows_c = read_rank1(final=True)
                every = rows + rows_c
                say(f"{clock()} {act}: power windows "
                    f"{post_windows(box, every, act, run_id, traces, where, teachers)}")
                done_ids = {a: acts[a].get("run_id") if finished[a] else None for a in ACTS}
                part = compare(every, {a: {"run_id": r} for a, r in done_ids.items()}, traces, source, out, t0,
                               now_s() - t0 if all(finished.values()) else None, stamp, teachers, source_c)
                say(f"{clock()} {act}: comparison so far {post_comparison(box, part)}")
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
    rows_c = read_rank1(final=True)
    with state.lock:
        stages = list(state.stages) + rows_c
        acts = {a: dict(v) for a, v in state.acts.items()}
    (out / "stages.jsonl").write_text("".join(json.dumps(s) + "\n" for s in stages), encoding="utf-8")
    comparison = compare(stages, acts, traces, source, out, t0, total, stamp, teachers, source_c)
    with state.lock:
        state.comparison = comparison
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    for act in ACTS:   # the windows again with the last seconds of samples, then the whole comparison
        if finished.get(act) and acts[act].get("run_id"):
            log(f"{act}: final power windows "
                f"{post_windows(box, stages, act, acts[act]['run_id'], traces, where, teachers)}")
    log(f"comparison on the box: {post_comparison(box, comparison)}")
    for name, tr in traces.items():
        (out / f"power-{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tr.raw), encoding="utf-8")
    write_rows(comparison, stamp)
    log(summary_text(comparison))
    if args.card and args.hold_minutes > 0:
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


def rank_facts(source: PodSource | None, path: str, run_id: str) -> dict[str, Any] | None:
    """A rank's own run.json (lab/train.py's answer): its steps and losses, for the act's per-pod training row."""
    data = safe_read(source, path) if source is not None else None
    try:
        answer = json.loads(data) if data else None
    except ValueError:
        answer = None
    if not isinstance(answer, dict):
        return None
    m = answer.get("metrics") or {}
    return {"run_id": run_id, "steps": m.get("trainer.steps"), "loss_first": m.get("trainer.loss_first"),
            "loss_last": m.get("trainer.loss_last"), "train_s": m.get("trainer.train_s"), "outcome": answer.get("outcome")}


def compare(stages: list[dict[str, Any]], acts: dict[str, dict[str, Any]], traces: dict[str, Trace],
            source: PodSource, out: Path, t0: float, total: float | None, stamp: str | None = None,
            teacher_nodes: tuple[str, ...] = ("pod-c",), source_c: PodSource | None = None) -> dict[str, Any]:
    """The comparison document (loop_energy.compare), with each act's counts read from its run folder on pod D, and
    for an act that trained on both pods, pod C's rank from its own run.json."""
    run_ids = {act: acts.get(act, {}).get("run_id") for act in ACTS}
    facts = {act: act_facts(source, run_id) for act, run_id in run_ids.items() if run_id}
    for act, run_id in run_ids.items():
        if run_id and "pod-c" in le.trains_on(stages, act):
            stamp_of = run_id.removeprefix("live-").rsplit("-", 1)[0]
            rank1 = rank_facts(source_c, f"l29/{stamp_of}/rank1/run.json", f"{run_id}{PART}")
            facts[act]["trainer_pods"] = {"pod-c": rank1 or {"run_id": f"{run_id}{PART}"}}
    return le.compare(stages, run_ids, facts, node_traces(traces), t0, total, stamp, teacher_nodes)


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
    """Push a recorded live loop (a relay --out folder or its fixture copy) to the box as finished runs, and with
    --card serve the card with its recorded traces and comparison. No pod is touched; no slot is needed.

    Kept for tests and for repairing a box after a failed post. The demo never uses it: the screens show only what
    the box records live, and this would put recorded runs on the box's Learning lab. It posts no comparison and no
    power windows, so the Compare tab never shows a recorded loop."""
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
    if not args.card:
        return 0
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
    print(f"teacher expected: {args.teacher_expect} on {' and '.join(teacher_nodes_for(args.teacher_expect, args.teacher_nodes))}"
          f"; act 2 on {args.duo_pods} pod(s); act sizes {args.solo_examples} and {args.duo_examples} per member")
    code = 0
    for letter in ("D", "C") if args.duo_pods == 2 else ("D",):
        answer, rc = controller.run(["run-script", "--account", letter, "--timeout", "90", str(POD_SCRIPT), "check"])
        print(f"pod {letter}: " + str(answer.get("output_tail") or answer.get("error")))
        code = code or rc
    return code


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--slot", help="the GPU Queue row id that is Running for pod D (python -m lab slot start)")
    p.add_argument("--slot-c", help="the GPU Queue row id that is Running for pod C (act 2's rank 1; --duo-pods 2)")
    p.add_argument("--teacher-expect", default=TEACHER_PAIR,
                   help=f"the teacher checkpoint labeling checks for (default {TEACHER_PAIR}, the pair on pods C and "
                        f"D; fallback {TEACHER_SINGLE}, the 27B on pod C; 'a|b' takes either)")
    p.add_argument("--teacher-nodes", help="the pods the teacher's energy is counted on, comma-separated (default: "
                                           "pod-c,pod-d for the pair, pod-c otherwise)")
    p.add_argument("--solo-examples", type=int, default=12,
                   help="act 1's examples (one member); 12 ends near 212 s at the pair's 2.34 s a label, 15 near 233 s")
    p.add_argument("--duo-examples", type=int, default=12, help="act 2's examples per member (two members); 12 or 15")
    p.add_argument("--duo-pods", type=int, choices=(1, 2), default=2,
                   help="act 2 trains on both pods (2, R65's two-pod LoRA) or on pod D alone (1)")
    p.add_argument("--serial", action="store_true", help="act 2 trains only after act 1 has packaged (slower, less "
                                                         "memory on pod D at once)")
    p.add_argument("--steps", type=int, default=8, help="optimizer steps per act (one loss row each)")
    p.add_argument("--every", type=float, default=5.0, help="seconds between reads of the pod's run folder")
    p.add_argument("--box", help="the box's address (default: the `jetson` key of .env)")
    p.add_argument("--card", action="store_true",
                   help="also serve the old local comparison card (off: the box's Compare tab is the screen)")
    p.add_argument("--port", type=int, default=8788, help="the local card's port on 127.0.0.1 (with --card)")
    p.add_argument("--stamp", help="the run stamp (default: local MMDD-HHMMSS); run ids are live-<stamp>-solo|duo")
    p.add_argument("--out", help="where the relay keeps its copies (default ../L29-data/<stamp>)")
    p.add_argument("--on-approval", action="store_true",
                   help="start when the box publishes learning.dataset.exported (the phone approved an export)")
    p.add_argument("--no-scores", action="store_true", help="do not post the stand-in gate's scores to the box")
    p.add_argument("--max-seconds", type=float, default=600.0, help="give up after this long")
    p.add_argument("--hold-minutes", type=float, default=30.0, help="with --card: keep it up this long after the end")
    p.add_argument("--check", action="store_true", help="check the pod and the box; start nothing")
    p.add_argument("--fallback", metavar="DIR",
                   help="tests and repairs only, never the demo: push a recorded loop's runs to the box")
    args = p.parse_args(argv)
    if args.check:
        return check(args)
    if args.fallback:
        return fallback(args)
    if not args.slot:
        p.error("--slot is required: GPU time on pod D only inside a GPU Queue slot")
    if args.duo_pods == 2 and not args.slot_c:
        p.error("--slot-c is required with --duo-pods 2: act 2's rank 1 trains on pod C, inside its own GPU Queue slot")
    if min(args.solo_examples, args.duo_examples) < 8:
        p.error("--solo-examples and --duo-examples are at least 8: a quarter is held out, the rest trains")
    if args.every < 2:
        p.error("--every is at least 2 seconds")
    return run_loop(args)


if __name__ == "__main__":
    sys.exit(main())
