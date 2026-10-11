"""The routing what-if on the Learning lab: what each routing policy would do with a stream of requests, and what
that would cost, worked out from the lab's recorded numbers. A simulation, and it says so.

    simulate({"preset": "study"}, rows)       # what POST /api/learning/lab/routing/simulate serves

It runs the real hook, not a copy of it. Every request is decided by `runtime.routing.route` with one of the two
policies in `contexts.telemetry.application.routing` (`capability`, `energy`), on a snapshot built by that module's
own `routing_snapshot`. The privacy line is therefore the hook's: a request about the user's own space, or from a
context that does not already send work to the lab, answers on the box whatever the energy says.

The model (every figure it uses comes back under `inputs` with the metrics-file line it was read from):

- Energy of an answer = joules per token on the node that answers x the answer's length in tokens.
  The teacher's joules per token is the recorded batching curve (`teacher_batching.joules_per_token` on pod-a), read
  at the batch size the teacher would run at, interpolated between recorded batch sizes on a log scale and held
  flat outside them. The student's is derived from its recorded watts, time per answer and decode rate, on one of
  two bases the caller picks: "request" (watts x seconds per answer / tokens: the whole request, as the teacher's
  figure is) or "decode" (watts / tokens per second: decoding only).
- Answer length per kind of work is an assumption, shown and editable (`tokens`).
- Latency = time to the first token + tokens / tokens per second, the teacher's two terms again read off its
  batching curve.
- The teacher's batch size is the number of requests in flight at once: arrivals per second x seconds per request
  (Little's law), solved as a fixed point because a bigger batch makes each request slower, and clamped to the
  recorded range 1..64.
- The snapshot the hook is handed is the steady state: the joules per answer each node would record at the load
  the policy itself creates. It is found by starting from "everything the hook lets out goes to the teacher" and
  repeating until the decisions stop changing. When a cold start (teacher idle) would settle somewhere else, the
  result says so (`cold_start_differs`).
- Gross energy of the answers only: neither node's idle power, the network, nor the cost of waiting is counted.

Deterministic: the request stream is drawn from `random.Random(seed)`, so the same inputs give the same output.

When the metrics file gains closed-loop rows (the same metric measured with the policy on and off over at least
MEASURED_MIN_REQUESTS requests), `measured` carries them and the simulation is relabelled "what-if".
TODO(T11): `closed_loop` assumes those rows carry `conditions.policy` naming the policy and a request count under
`conditions.requests` (or `n` / `samples`); adjust that one function when T11's rows exist.
"""

import math
import random
from types import SimpleNamespace
from typing import Any

from contexts.learning.application.lab_bench import _newest, _number, point, student, sweeps
from contexts.telemetry.application.routing import CAPABILITY, ENERGY, POLICIES, STUDENT_NODE, TEACHER_NODE, routing_snapshot
from runtime.routing import LAB_SOURCES, PRIVACY, RouteRequest, route

MODEL_STATEMENT = "Modelled from recorded energy per answer; not a live measurement"
NO_MEASUREMENT = "no closed-loop measurement recorded yet"
MEASURED_MIN_REQUESTS = 100
CALLS = ["runtime.routing.route", "contexts.telemetry.application.routing.capability",
         "contexts.telemetry.application.routing.energy", "contexts.telemetry.application.routing.routing_snapshot"]
MAX_REQUESTS = 1500
BASES = ("request", "decode")

# The kinds of work the hook names (runtime/routing.py RouteRequest.kind), each as the request the simulator sends.
# `source`, `own_space` and `student_can` are RouteRequest's own fields; `tokens` is the assumed answer length.
KINDS: list[dict[str, Any]] = [
    {"kind": "chat", "title": "A question about your room or your day", "source": "agent", "own_space": True,
     "student_can": True, "tokens": 80},
    {"kind": "caption", "title": "Describing what a camera sees", "source": "perception", "own_space": True,
     "student_can": True, "tokens": 80},
    {"kind": "label", "title": "Labelling training examples you approved", "source": "learning", "own_space": False,
     "student_can": False, "tokens": 128},
    {"kind": "eval", "title": "Scoring practice questions for the gate", "source": "learning", "own_space": False,
     "student_can": True, "tokens": 80},
]
PRESETS: list[dict[str, Any]] = [
    {"id": "quiet", "title": "Quiet evening", "rpm": 6, "mix": {"chat": 70, "caption": 25, "label": 0, "eval": 5}},
    {"id": "study", "title": "Study session", "rpm": 24, "mix": {"chat": 40, "caption": 20, "label": 20, "eval": 20}},
    {"id": "burst", "title": "Labelling burst", "rpm": 90, "mix": {"chat": 5, "caption": 5, "label": 45, "eval": 45}},
]
DEFAULT_PRESET = "burst"
ASSUMPTIONS = [
    "Energy of an answer is joules per token on the node that answers, times the answer's length in tokens.",
    "Answer length per kind of work is assumed (shown, and editable); the lab has not recorded it per kind.",
    "The teacher's batch size is the number of requests in flight at once (arrivals per second times seconds per request), "
    "kept inside the recorded range of 1 to 64.",
    "Between recorded batch sizes the teacher's numbers are interpolated on a log scale; beyond them they are held flat.",
    "The teacher's curve was measured on one prompt set (128-token answers) and is applied to every kind of work.",
    "The student's energy and delay come from one recorded caption run with a picture in the prompt; text-only work is likely cheaper and faster, and is not recorded.",
    "Only the energy of the answers is counted (gross). Neither node's idle power nor the network is included.",
]


class SimError(ValueError):
    pass


# --- the recorded inputs ---

def recorded_inputs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What the model stands on: the teacher's newest batching sweep on the routing hook's teacher node and the
    student's figures on the Jetson. `missing` says what the file does not hold; the simulator refuses to run then."""
    sweep = next((s for s in sweeps(rows) if s["node"] == TEACHER_NODE and all(
        len(s["metrics"].get(m, [])) >= 2 for m in ("joules_per_token", "decode_tok_s", "ttft_s"))), None)
    ref = student(rows)
    missing = []
    if sweep is None:
        missing.append(f"a batching sweep on {TEACHER_NODE} with joules per token, tokens per second and time to first token")
    if ref is None or ref["tok_s"] is None or ref["j_per_token"] is None:
        missing.append("the student's watts and decode rate on the Jetson")
    teacher = None
    if sweep is not None:
        teacher = {"node": sweep["node"], "model": sweep["model"], "session": sweep["session"], "chip": sweep["gpu"],
                   "max_tokens": sweep["max_tokens"], "confidence_note": sweep["confidence_note"],
                   "batches": sweep["batches"],
                   "j_per_token": sweep["metrics"]["joules_per_token"], "tok_s": sweep["metrics"]["decode_tok_s"],
                   "ttft_s": sweep["metrics"]["ttft_s"]}
    return {"teacher": teacher, "student": ref, "missing": missing}


def interp(points: list[dict[str, Any]], batch: float) -> float:
    """A recorded curve at `batch`: linear in log2(batch) between recorded points, flat outside them."""
    pts = sorted(((p["x"], p["value"]) for p in points if p["x"]), key=lambda xy: xy[0])
    if batch <= pts[0][0]:
        return pts[0][1]
    if batch >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= batch <= x1:
            u = (math.log2(batch) - math.log2(x0)) / (math.log2(x1) - math.log2(x0))
            return y0 + u * (y1 - y0)
    return pts[-1][1]


class Model:
    """The recorded curves and the caller's assumptions, as the functions the simulation needs."""

    def __init__(self, inputs: dict[str, Any], tokens: dict[str, int], basis: str) -> None:
        self.t, self.s, self.tokens = inputs["teacher"], inputs["student"], tokens
        self.lo, self.hi = min(self.t["batches"]), max(self.t["batches"])
        per_request = self.s["j_per_token_request"]
        # "request" needs the row's time per answer and token limit; without them the decode basis is what there is
        self.basis = "request" if basis == "request" and per_request else "decode"
        self.student_jpt_num = per_request if self.basis == "request" else self.s["j_per_token"]
        self.student_jpt = self.student_jpt_num["value"]
        self.student_tok_s = self.s["tok_s"]["value"]
        self.student_first = self.s["first_token_s"]["value"] if self.s["first_token_s"] else 0.0

    def clamp(self, batch: float) -> float:
        return min(self.hi, max(self.lo, batch))

    def teacher_jpt(self, batch: float) -> float:
        return interp(self.t["j_per_token"], self.clamp(batch))

    def joules(self, node: str, kind: str, batch: float) -> float:
        per_token = self.teacher_jpt(batch) if node == "teacher" else self.student_jpt
        return per_token * self.tokens[kind]

    def latency(self, node: str, kind: str, batch: float) -> float:
        if node == "teacher":
            b = self.clamp(batch)
            return interp(self.t["ttft_s"], b) + self.tokens[kind] / interp(self.t["tok_s"], b)
        return self.student_first + self.tokens[kind] / self.student_tok_s

    def steady_batch(self, per_second: float, mean_tokens: float) -> tuple[float, bool]:
        """Requests in flight on the teacher at `per_second` arrivals: b = rate x seconds per request at batch b.
        Returns the batch (clamped to the recorded range) and whether the teacher cannot keep up even at the top."""
        if per_second <= 0:
            return self.lo, False
        service = lambda b: interp(self.t["ttft_s"], b) + mean_tokens / interp(self.t["tok_s"], b)  # noqa: E731
        b = self.lo
        for _ in range(200):
            nxt = self.clamp(per_second * service(b))
            if abs(nxt - b) < 1e-6:
                break
            b = nxt
        return b, per_second * service(self.hi) > self.hi + 1e-9

    def snapshot(self, batch: float) -> dict[str, Any]:
        """The joules per answer each node would record per kind of work at this teacher batch, in the shape the
        policies read. Built by the telemetry context's own `routing_snapshot`. The student has a value only for
        work it can do, the teacher only for work the lab is ever sent."""
        values = []
        for k in KINDS:
            if k["student_can"]:
                values.append(SimpleNamespace(node=STUDENT_NODE, kind=k["kind"], joules=round(self.joules("student", k["kind"], batch), 3),
                                              request_id=f"modelled:{STUDENT_NODE}:{k['kind']}", low_confidence=True))
            if k["source"] in LAB_SOURCES:
                values.append(SimpleNamespace(node=TEACHER_NODE, kind=k["kind"], joules=round(self.joules("teacher", k["kind"], batch), 3),
                                              request_id=f"modelled:{TEACHER_NODE}:{k['kind']}@batch{batch:.2f}", low_confidence=True))
        return routing_snapshot(values)


def as_request(kind: dict[str, Any]) -> RouteRequest:
    return RouteRequest(kind=kind["kind"], source=kind["source"], own_space=kind["own_space"], student_can=kind["student_can"])


def privacy_line() -> dict[str, Any]:
    """Which kinds the hook keeps on the box whatever the numbers say, found by asking the hook itself: each kind is
    routed with the energy policy on a snapshot where the teacher costs next to nothing."""
    free = routing_snapshot([SimpleNamespace(node=n, kind=k["kind"], joules=j, request_id="privacy-check", low_confidence=True)
                             for k in KINDS for n, j in ((STUDENT_NODE, 1e9), (TEACHER_NODE, 1e-9))])
    locked = []
    for k in KINDS:
        d = route(as_request(k), free, POLICIES[ENERGY])
        if d.policy == PRIVACY:
            locked.append({"kind": k["kind"], "reason": d.reason})
    return {"rule": "Anything about your own space, and any work Learning or Circle does not already send to the lab, "
                    "is answered on the box. The hook checks this before any policy runs, so no energy figure can change it.",
            "locked": locked, "lab_sources": sorted(LAB_SOURCES), "policy_name": PRIVACY,
            "source": "runtime/routing.py route(), checked here with the teacher priced at 0.000000001 J per answer"}


# --- one run ---

def resolve(params: dict[str, Any] | None) -> dict[str, Any]:
    """The caller's inputs with the preset and the defaults filled in, each checked. SimError says what is wrong."""
    p = dict(params or {})
    preset_id = p.get("preset") if p.get("preset") is not None else (None if ("mix" in p or "rpm" in p) else DEFAULT_PRESET)
    preset = next((x for x in PRESETS if x["id"] == preset_id), None)
    if preset_id is not None and preset is None:
        raise SimError(f"preset must be one of {[x['id'] for x in PRESETS]}")
    base = preset or next(x for x in PRESETS if x["id"] == DEFAULT_PRESET)
    names = [k["kind"] for k in KINDS]
    mix = {**base["mix"], **{k: v for k, v in (p.get("mix") or {}).items() if k in names}} if isinstance(p.get("mix", {}), dict) else None
    if mix is None or any(not isinstance(v, int | float) or isinstance(v, bool) or v < 0 or not math.isfinite(v) for v in mix.values()):
        raise SimError("mix must give each kind a weight of 0 or more")
    if sum(mix.values()) <= 0:
        raise SimError("mix needs at least one kind with a weight above 0")
    rpm = p.get("rpm", base["rpm"])
    minutes = p.get("minutes", 5)
    for name, value, lo, hi in (("rpm", rpm, 1, 240), ("minutes", minutes, 1, 30)):
        if not isinstance(value, int | float) or isinstance(value, bool) or not lo <= value <= hi:
            raise SimError(f"{name} must be a number from {lo} to {hi}")
    tokens = {k["kind"]: k["tokens"] for k in KINDS}
    for kind, value in (p.get("tokens") or {}).items():
        if kind in tokens:
            if not isinstance(value, int) or isinstance(value, bool) or not 4 <= value <= 1024:
                raise SimError("tokens must be whole numbers from 4 to 1024")
            tokens[kind] = value
    policy = p.get("policy", ENERGY)
    if policy not in POLICIES:
        raise SimError(f"policy must be one of {sorted(POLICIES)}")
    basis = p.get("student_basis", "request")
    if basis not in BASES:
        raise SimError(f"student_basis must be one of {list(BASES)}")
    seed = p.get("seed", 7)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise SimError("seed must be a whole number")
    custom = preset is None or any(mix[k] != preset["mix"][k] for k in names) or rpm != preset["rpm"]
    return {"preset": None if custom else preset["id"], "mix": {k: mix[k] for k in names}, "rpm": rpm, "minutes": minutes,
            "tokens": tokens, "policy": policy, "student_basis": basis, "seed": seed}


def stream(params: dict[str, Any]) -> tuple[list[float], list[int]]:
    """The requests of one run: when each arrives (seconds, in order) and which kind it is. Seeded."""
    rng = random.Random(params["seed"])
    n = min(MAX_REQUESTS, max(1, round(params["rpm"] * params["minutes"])))
    window = params["minutes"] * 60.0
    times = sorted(round(rng.random() * window, 2) for _ in range(n))
    kinds = rng.choices(range(len(KINDS)), weights=[params["mix"][k["kind"]] for k in KINDS], k=n)
    return times, kinds


def _decide(model: Model, policy: str, batch: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """The hook's decision for each kind of work on the snapshot at `batch` (the decision depends on nothing else)."""
    snap = model.snapshot(batch)
    return snap, {k["kind"]: route(as_request(k), snap, POLICIES[policy]) for k in KINDS}


def _load(model: Model, decisions: dict[str, Any], kinds: list[int], window_s: float) -> tuple[float, bool, float]:
    sent = [KINDS[i]["kind"] for i in kinds if decisions[KINDS[i]["kind"]].target == "teacher"]
    if not sent:
        return model.lo, False, 0.0
    batch, saturated = model.steady_batch(len(sent) / window_s, sum(model.tokens[k] for k in sent) / len(sent))
    return batch, saturated, len(sent) / window_s * 60


def _settle(model: Model, policy: str, kinds: list[int], window_s: float, start: float) -> tuple[float, bool, float, dict, dict, int]:
    batch, saturated, rpm = start, False, 0.0
    snap, decisions = _decide(model, policy, batch)
    for n in range(1, 13):
        new, saturated, rpm = _load(model, decisions, kinds, window_s)
        if abs(new - batch) < 1e-6:
            return batch, saturated, rpm, snap, decisions, n
        batch = new
        snap, decisions = _decide(model, policy, batch)
    return batch, saturated, rpm, snap, decisions, 12


def run_policy(model: Model, policy: str, times: list[float], kinds: list[int], window_s: float) -> dict[str, Any]:
    # Start from the teacher as busy as the hook could ever make it; a cold start (teacher idle) is compared after.
    eligible = [KINDS[i]["kind"] for i in kinds if not KINDS[i]["own_space"] and KINDS[i]["source"] in LAB_SOURCES]
    busy = model.steady_batch(len(eligible) / window_s, sum(model.tokens[k] for k in eligible) / len(eligible))[0] if eligible else model.lo
    batch, saturated, rpm, snap, decisions, rounds = _settle(model, policy, kinds, window_s, busy)
    cold = _settle(model, policy, kinds, window_s, model.lo)
    cold_targets = {k: d.target for k, d in cold[4].items()}

    by_kind, targets, total, wait = {}, [], 0.0, 0.0
    for k in KINDS:
        d = decisions[k["kind"]]
        by_kind[k["kind"]] = {"target": d.target, "decided_by": d.policy, "reason": d.reason, "n": 0,
                              "joules": round(model.joules(d.target, k["kind"], batch), 3),
                              "latency_s": round(model.latency(d.target, k["kind"], batch), 3), "energy_j": 0.0}
    for i in kinds:
        row = by_kind[KINDS[i]["kind"]]
        row["n"] += 1
        row["energy_j"] += row["joules"]
        total += row["joules"]
        wait += row["latency_s"]
        targets.append("p" if row["decided_by"] == PRIVACY else row["target"][0])
    for row in by_kind.values():
        row["energy_j"] = round(row["energy_j"], 3)
    n = len(kinds)
    to_teacher = targets.count("t")
    return {
        "policy": policy, "targets": "".join(targets), "by_kind": by_kind,
        "to_teacher": to_teacher, "to_student": n - to_teacher, "locked": targets.count("p"),
        "energy_j": round(total, 3), "mean_latency_s": round(wait / n, 3),
        "teacher_batch": round(batch, 2) if to_teacher else None, "teacher_rpm": round(rpm, 2),
        "teacher_j_per_token": round(model.teacher_jpt(batch), 4) if to_teacher else None,
        "saturated": saturated, "snapshot": snap, "rounds": rounds,
        "cold_start_differs": cold_targets != {k: d.target for k, d in decisions.items()},
        "cold_start_note": None if cold_targets == {k: d.target for k, d in decisions.items()} else
        "This outcome needs the teacher already busy. Starting from an idle teacher, its first answers cost more per token "
        "and the policy would keep " + ", ".join(k for k, t in cold_targets.items() if t != decisions[k].target) + " work on the box.",
    }


def break_even(model: Model, runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per token, where the teacher's recorded curve crosses the student's line: the batch size the teacher must hold
    before sending it work saves energy, and the request rate that holds it."""
    grid = sorted({*model.t["batches"], *(round(2 ** (i / 4), 2) for i in range(int(math.log2(model.hi) * 4) + 1))})
    recorded = set(model.t["batches"])
    curve = [{"batch": b, "teacher_j_per_token": round(model.teacher_jpt(b), 4), "recorded": b in recorded} for b in grid]
    line = model.student_jpt
    batch = None
    pts = [(p["x"], p["value"]) for p in sorted(model.t["j_per_token"], key=lambda p: p["x"])]
    if pts[0][1] <= line:
        batch = pts[0][0]
    else:
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if y0 > line >= y1:
                batch = 2 ** (math.log2(x0) + (y0 - line) / (y0 - y1) * (math.log2(x1) - math.log2(x0)))
                break
    tokens = model.tokens["eval"]
    rpm = None
    if batch is not None:
        seconds = interp(model.t["ttft_s"], batch) + tokens / interp(model.t["tok_s"], batch)
        rpm = round(60 * batch / seconds, 1)
    return {
        "unit": "J/token", "student_j_per_token": round(line, 4), "curve": curve,
        "batch": round(batch, 1) if batch is not None else None, "rpm_to_teacher": rpm, "tokens": tokens,
        "reason": None if batch is not None else
        f"At every recorded batch size the box costs less per token ({line:.2f} J against {pts[-1][1]:.2f} J at batch {pts[-1][0]:g}), "
        "so on energy alone the teacher never wins.",
        "now": {name: run["teacher_batch"] for name, run in runs.items()},
    }


def _sentence(n: int, cap: dict[str, Any], eng: dict[str, Any]) -> str:
    slower = eng["mean_latency_s"] - cap["mean_latency_s"]
    if eng["targets"] == cap["targets"]:
        return (f"Energy-aware routing makes the same choices as capability-only here: {eng['to_teacher']} of {n} requests go to the "
                f"teacher and both use {_joules(eng['energy_j'])}.")
    saved = cap["energy_j"] - eng["energy_j"]
    pct = saved / cap["energy_j"] * 100 if cap["energy_j"] else 0.0
    pace = (f"{abs(slower):.1f} s {'slower' if slower > 0 else 'faster'} on average" if abs(slower) >= 0.05 else "no slower on average")
    return (f"Energy-aware routing sends {eng['to_teacher']} of {n} requests to the teacher and uses {_joules(eng['energy_j'])} instead of "
            f"{_joules(cap['energy_j'])}: {abs(pct):.1f}% {'less' if saved >= 0 else 'more'}, with answers {pace}.")


def _joules(j: float) -> str:
    return f"{j / 1000:,.1f} kJ" if j >= 10_000 else f"{j:,.0f} J"


def closed_loop(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The measured closed loop, when the metrics file holds one: the same metric on the same node with the capability
    policy and with the energy policy, each over at least MEASURED_MIN_REQUESTS requests. See the module's TODO(T11)."""
    groups: dict[tuple[str, str | None], dict[str, list[dict[str, Any]]]] = {}
    for r in rows:
        c = r["conditions"]
        policy = c.get("policy") if c.get("policy") in POLICIES else c.get("routing_policy")
        n = next((_number(c.get(k)) for k in ("requests", "n", "samples") if _number(c.get(k)) is not None), None)
        if policy in POLICIES and n is not None and n >= MEASURED_MIN_REQUESTS:
            groups.setdefault((r["metric"], r.get("node")), {}).setdefault(policy, []).append(r)
    found = []
    for (metric, node), by_policy in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        if CAPABILITY in by_policy and ENERGY in by_policy:
            off, on = point(_newest(by_policy[CAPABILITY])), point(_newest(by_policy[ENERGY]))
            change = (on["value"] - off["value"]) / off["value"] * 100 if off["value"] else None
            found.append({"metric": metric, "node": node, "unit": on["unit"], "capability": off, "energy": on,
                          "change_pct": round(change, 1) if change is not None else None,
                          "low_confidence": on["low_confidence"] or off["low_confidence"]})
    return {"rows": found, "min_requests": MEASURED_MIN_REQUESTS} if found else None


def simulate(params: dict[str, Any] | None, rows: list[dict[str, Any]]) -> dict[str, Any]:
    resolved = resolve(params)
    inputs = recorded_inputs(rows)
    measured = closed_loop(rows)
    head = {"model_statement": MODEL_STATEMENT, "label": "what-if" if measured else "simulation", "calls": CALLS,
            "measured": measured, "measured_reason": None if measured else NO_MEASUREMENT,
            "presets": PRESETS, "params": resolved, "assumptions": ASSUMPTIONS, "privacy": privacy_line()}
    if inputs["missing"]:
        return {**head, "available": False, "reason": "The metrics file does not hold " + " or ".join(inputs["missing"]) + ".",
                "inputs": inputs}
    model = Model(inputs, resolved["tokens"], resolved["student_basis"])
    resolved["student_basis"] = model.basis
    times, kinds = stream(resolved)
    window = resolved["minutes"] * 60.0
    runs = {name: run_policy(model, name, times, kinds, window) for name in (CAPABILITY, ENERGY)}
    cap, eng = runs[CAPABILITY], runs[ENERGY]
    locked = {x["kind"] for x in head["privacy"]["locked"]}
    return {
        **head, "available": True, "reason": None,
        "kinds": [{**k, "tokens": resolved["tokens"][k["kind"]], "locked": k["kind"] in locked,
                   "may_leave": k["kind"] not in locked, "share": round(kinds.count(i) / len(kinds), 4), "n": kinds.count(i)}
                  for i, k in enumerate(KINDS)],
        "inputs": {**inputs, "student_basis": model.basis, "student_j_per_token": model.student_jpt_num,
                   "student_bases": {"request": inputs["student"]["j_per_token_request"], "decode": inputs["student"]["j_per_token"]}},
        "requests": {"n": len(times), "window_s": window, "t": times, "kind": kinds},
        "policies": runs,
        "compare": {"moved": sum(a != b for a, b in zip(cap["targets"], eng["targets"])),
                    "saved_j": round(cap["energy_j"] - eng["energy_j"], 3),
                    "saved_pct": round((cap["energy_j"] - eng["energy_j"]) / cap["energy_j"] * 100, 2) if cap["energy_j"] else 0.0,
                    "slower_s": round(eng["mean_latency_s"] - cap["mean_latency_s"], 3),
                    "sentence": _sentence(len(times), cap, eng)},
        "break_even": break_even(model, runs),
    }
