"""The live learning loop's arithmetic, in one place (L29, L31): stage seconds, gross joules over a stage window, pod
D's split between overlapping stages, the idle floors, one act's row, the comparison of the two acts, and the ratios
of one run over another.

    rows = compare(stages, run_ids, facts, traces, t0, total)   # what tooling/demo/live_loop.py writes and posts
    ratios(row_a, row_b, RATIO_KEYS)                            # b over a, what GET /api/learning/lab/compare serves

The relay (tooling/demo/live_loop.py) builds the comparison with `compare` and posts it to the box; the box's compare
route (lab_compare.py) serves that document and works out every ratio with `ratios`. Both import this module, so the
relay's card and the box's Compare tab cannot disagree on a figure.

Pure functions: no threads, no network, no files. A trace is a list of (seconds on its node's own clock, watts), in
time order, and its offset is that node's clock minus the relay's (the largest `ts - arrival` the relay saw, so the
newest sample against the time it arrived). Stage lines are the pod's stages.jsonl rows ({act, stage, ts, ...}),
written on pod D's clock.

Energy is T4's rule (contexts/telemetry/application/joules.py): gross joules, the trace integrated over the window by
contexts/telemetry/domain/energy.py's integrate (bracketing samples, interpolated edges), nothing subtracted, the
idle beside it (the calibrated figure of config/telemetry_idle.yaml and the quiet reading measured just before the
start). Windows: labeling on pod C from the act's start to the trainer's start; training on pod D from the trainer's
start to its end; packaging on pod D from the trainer's end to the act's end; the box over the whole act.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from contexts.telemetry.domain.energy import Window, attribute, integrate

Points = Sequence[tuple[float, float]]

IDLE_W = {"pod-c": 10.27, "pod-d": 10.27, "box": 15.76}   # config/telemetry_idle.yaml (pod-a's figure for both pods)
ACTS = ("solo", "duo")
MEMBERS = {"solo": 1, "duo": 2}   # act 1 is one member's examples, act 2 two members' pooled
NODE_NAMES = {"pod-c": "Pod C (teacher, labeling)", "pod-d": "Pod D (trainer)", "box": "The box (Jetson)"}
RULE = ("gross joules (T4), integrated by contexts/telemetry/domain/energy.integrate; idle beside: calibrated "
        "(config/telemetry_idle.yaml) and quiet (median of the 30 s before the start)")
POD_D_SPLIT = ("pod D's training and packaging windows overlap when act 2 trains while act 1 packages; each "
               "overlapping stretch is split evenly between them (energy.attribute, no tokens), so the stages add up "
               "to the pod's joules; gross_window_j is the whole window's integral")
BOX_NOTE = "bimodal: about 5 W between captions, about 18 W while the box's GPU captions (T13)"

# The relay's own `duo_over_solo`: name -> path into an act row.
DUO_OVER_SOLO: dict[str, tuple[str, ...]] = {
    "examples": ("examples",), "act_seconds": ("seconds", "total"),
    "labeling_j": ("labeling", "joules"), "labeling_j_per_token": ("labeling", "j_per_token"),
    "labeling_j_per_example": ("labeling", "j_per_example"), "training_j": ("training", "joules"),
    "training_j_per_step": ("training", "j_per_step"), "training_j_per_train_row": ("training", "j_per_train_row"),
    "packaging_j": ("packaging", "joules"), "pods_joules": ("pods_joules",),
    "j_per_example": ("j_per_example",), "box_j": ("box", "joules"),
}

# What the Compare tab shows, b over a: the relay's keys plus members and each stage's seconds.
RATIO_KEYS: dict[str, tuple[str, ...]] = {
    "examples": ("examples",), "members": ("members",), "act_seconds": ("seconds", "total"),
    "labeling_s": ("seconds", "labeling"), "training_s": ("seconds", "training"),
    "packaging_s": ("seconds", "packaging"),
    **{k: v for k, v in DUO_OVER_SOLO.items() if k not in ("examples", "act_seconds")},
}


def iso_s(text: str) -> float:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def ratio(a: Any, b: Any) -> float | None:
    """a over b, three places; None when either is missing or b is zero."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b:
        return round(a / b, 3)
    return None


def per(a: Any, b: Any, nd: int = 3) -> float | None:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b:
        return round(a / b, nd)
    return None


def get(row: Any, path: Sequence[str]) -> Any:
    for p in path:
        row = row.get(p) if isinstance(row, dict) else None
    return row


def ratios(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None,
           keys: Mapping[str, Sequence[str]] = RATIO_KEYS) -> dict[str, float | None] | None:
    """Each figure of run b over run a; None when either run is missing."""
    if not a or not b:
        return None
    return {k: ratio(get(b, p), get(a, p)) for k, p in keys.items()}


# --- stages -----------------------------------------------------------------------------------------------

def stage_times(stages: Sequence[Mapping[str, Any]], act: str) -> dict[str, float]:
    """Each stage of one act at its pod-clock time."""
    return {s["stage"]: iso_s(s["ts"]) for s in stages if s.get("act") == act and s.get("ts")}


def stage_seconds(stages: Sequence[Mapping[str, Any]], act: str) -> dict[str, float]:
    """Pod-clock seconds of each stage of one act: labeling, waiting, training, packaging, total."""
    at = stage_times(stages, act)
    out: dict[str, float] = {}
    for name, first, last in (("labeling", "start", "labeled"), ("waiting", "labeled", "train"),
                              ("training", "train", "trained"), ("packaging", "trained", "done"),
                              ("total", "start", "done")):
        if first in at and last in at:
            out[name] = at[last] - at[first]
    return {k: round(v, 1) for k, v in out.items()}


def window(stages: Sequence[Mapping[str, Any]], act: str, first: str, last: str) -> tuple[float, float] | None:
    at = stage_times(stages, act)
    return (at[first], at[last]) if first in at and last in at else None


# The power windows the relay posts for each act: (stage, first stage line, last stage line, node). The stage windows
# are the ones the joules are worked out over; "act" is the whole act on every node, so the box can draw all three.
POWER_WINDOWS = (("labeling", "start", "labeled", "pod-c"), ("training", "train", "trained", "pod-d"),
                 ("packaging", "trained", "done", "pod-d"), ("act", "start", "done", "pod-c"),
                 ("act", "start", "done", "pod-d"), ("act", "start", "done", "box"))


# --- energy -----------------------------------------------------------------------------------------------

def moved(span: tuple[float, float], from_offset: float | None, to_offset: float | None) -> tuple[float, float]:
    """A window on one node's clock, moved onto another's through the relay's clock."""
    d = (to_offset or 0.0) - (from_offset or 0.0)
    return span[0] + d, span[1] + d


def energy(points: Points, span: tuple[float, float] | None, *, span_offset: float | None = 0.0,
           offset: float | None = 0.0) -> dict[str, Any] | None:
    """Gross joules of a trace over a window given on another clock (span_offset: that clock minus the relay's;
    offset: the trace's clock minus the relay's)."""
    if span is None:
        return None
    a, b = moved(span, span_offset, offset)
    j = integrate(points, a, b)
    inside = [w for t, w in points if a <= t <= b]
    secs = b - a
    return {"seconds": round(secs, 1), "joules": None if j is None else round(j, 1),
            "mean_w": None if j is None or secs <= 0 else round(j / secs, 2),
            "peak_w": max(inside) if inside else None, "samples": len(inside)}


def pod_d_shares(stages: Sequence[Mapping[str, Any]], points: Points
                 ) -> tuple[dict[str, float | None], dict[str, float]]:
    """Pod D's joules per stage window, an overlapping stretch split evenly (T4's split, no token weights), and how
    many seconds of each window another stage shared."""
    spans = {}
    for act in ACTS:
        for name, first, last in (("training", "train", "trained"), ("packaging", "trained", "done")):
            w = window(stages, act, first, last)
            if w:
                spans[f"{act}.{name}"] = w
    shared = attribute(points, [Window(k, a, b) for k, (a, b) in spans.items()]) if spans else {}
    overlaps = {}
    for k, (a, b) in spans.items():
        overlaps[k] = round(sum(max(0.0, min(b, d) - max(a, c)) for j, (c, d) in spans.items() if j != k), 1)
    return {k: None if v is None else round(v, 1) for k, v in shared.items()}, overlaps


def quiet_floor(points: Points, before_t: float, span_s: float = 30.0) -> float | None:
    """The median watts of the `span_s` seconds before `before_t` (on the trace's clock)."""
    ws = [w for t, w in points if before_t - span_s <= t < before_t]
    return round(statistics.median(ws), 2) if ws else None


def idle_floors(traces: Mapping[str, tuple[Points, float | None]], t0: float) -> dict[str, dict[str, Any]]:
    """Each node's calibrated idle and its quiet reading just before the start (t0 on the relay's clock)."""
    out: dict[str, dict[str, Any]] = {}
    for node in ("pod-c", "pod-d", "box"):
        points, off = traces.get(node, ([], None))
        quiet = quiet_floor(points, t0 + (off or 0.0))
        if node == "box" and quiet is None:
            quiet = quiet_floor(points, t0 + 30.0 + (off or 0.0))
        out[node] = {"calibrated_w": IDLE_W[node], "quiet_w": quiet}
        if node == "box":
            out[node]["note"] = BOX_NOTE
    return out


# --- one act and the comparison ---------------------------------------------------------------------------

def act_row(act: str, run_id: str, stages: Sequence[Mapping[str, Any]], facts: Mapping[str, Any],
            traces: Mapping[str, tuple[Points, float | None]], shared: Mapping[str, float | None],
            overlaps: Mapping[str, float], t0: float) -> dict[str, Any]:
    """One act's seconds and joules per stage, with the counts from its own files (`facts`: labels, trainer, counts,
    outcome, as the relay reads them from the act's run folder)."""
    (pc, pc_off), (pd, pd_off), (bx, bx_off) = (traces.get(n, ([], None)) for n in ("pod-c", "pod-d", "box"))
    secs = stage_seconds(stages, act)
    label = energy(pc, window(stages, act, "start", "labeled"), span_offset=pd_off, offset=pc_off)
    train = energy(pd, window(stages, act, "train", "trained"), span_offset=pd_off, offset=pd_off)
    pack = energy(pd, window(stages, act, "trained", "done"), span_offset=pd_off, offset=pd_off)
    for part, key in ((train, f"{act}.training"), (pack, f"{act}.packaging")):
        if part is not None:
            part["gross_window_j"] = part["joules"]
            part["joules"] = shared.get(key)
            part["overlap_s"] = overlaps.get(key)
    whole = window(stages, act, "start", "done")
    box_e = energy(bx, whole, span_offset=pd_off, offset=bx_off)
    lab_t = facts.get("labels") or {}
    tr = facts.get("trainer") or {}
    counts = facts.get("counts") or {}
    row: dict[str, Any] = {
        "run_id": run_id, "seconds": secs, "outcome": facts.get("outcome"),
        "examples": counts.get("examples"), "train_rows": counts.get("train_rows"), "held_out": counts.get("held_out"),
        "labeling": {**(label or {}), "examples": lab_t.get("examples"), "tokens": lab_t.get("tokens"),
                     "completion_tokens": lab_t.get("completion_tokens"),
                     "j_per_token": per((label or {}).get("joules"), lab_t.get("tokens")),
                     "j_per_example": per((label or {}).get("joules"), lab_t.get("examples"), 1),
                     "idle_j": round(IDLE_W["pod-c"] * label["seconds"], 1) if label else None,
                     "node": "pod C (teacher)"},
        "training": {**(train or {}), "steps": tr.get("steps"), "examples_seen": tr.get("examples_seen"),
                     "tokens": tr.get("tokens"), "train_s": tr.get("train_s"), "loss_first": tr.get("loss_first"),
                     "loss_last": tr.get("loss_last"), "trainer_energy_j": tr.get("energy_j"),
                     "j_per_step": per((train or {}).get("joules"), tr.get("steps"), 1),
                     "j_per_example_seen": per((train or {}).get("joules"), tr.get("examples_seen"), 2),
                     "j_per_train_row": per((train or {}).get("joules"), counts.get("train_rows"), 2),
                     "idle_j": round(IDLE_W["pod-d"] * train["seconds"], 1) if train else None,
                     "node": "pod D (trainer)"},
        "packaging": {**(pack or {}), "idle_j": round(IDLE_W["pod-d"] * pack["seconds"], 1) if pack else None,
                      "node": "pod D (trainer)"},
        "box": {**(box_e or {}), "idle_j": round(IDLE_W["box"] * box_e["seconds"], 1) if box_e else None,
                "node": "the box (Jetson VDD_IN)"},
    }
    pods_j = [x.get("joules") for x in (label, train, pack) if x]
    row["pods_joules"] = round(sum(pods_j), 1) if pods_j and all(j is not None for j in pods_j) else None
    row["j_per_example"] = per(row["pods_joules"], row["examples"], 1)
    row["start_s"] = round(whole[0] - (pd_off or 0.0) - t0, 1) if whole else None
    row["end_s"] = round(whole[1] - (pd_off or 0.0) - t0, 1) if whole else None
    return row


def compare(stages: Sequence[Mapping[str, Any]], run_ids: Mapping[str, str | None], facts: Mapping[str, Mapping[str, Any]],
            traces: Mapping[str, tuple[Points, float | None]], t0: float, total: float | None,
            stamp: str | None = None) -> dict[str, Any]:
    """The comparison document (fixtures/learning/live-loop-2026-10-11/comparison.json is one): each act that has a
    run id, the idle floors, the clock offsets, and act 2 over act 1 once both are in. `traces` maps pod-c, pod-d and
    box to (points, offset); `t0` is the stopwatch's zero on the relay's clock."""
    _, pd_off = traces.get("pod-d", ([], None))
    result: dict[str, Any] = {"stamp": stamp, "t0": t0, "total_s": None if total is None else round(total, 1),
                              "acts": {}, "rule": RULE, "idle": idle_floors(traces, t0),
                              "clock_offsets_s": {k: None if off is None else round(off, 2)
                                                  for k, (_, off) in traces.items()}}
    shared, overlaps = pod_d_shares(stages, traces.get("pod-d", ([], None))[0])
    result["pod_d_split"] = POD_D_SPLIT
    for act in ACTS:
        run_id = run_ids.get(act)
        if run_id:
            result["acts"][act] = act_row(act, run_id, stages, facts.get(act) or {}, traces, shared, overlaps, t0)
    a1, a2 = result["acts"].get("solo"), result["acts"].get("duo")
    if a1 and a2:
        result["duo_over_solo"] = ratios(a1, a2, DUO_OVER_SOLO)
    go = [s for s in stages if s.get("stage") == "go" and s.get("ts")]
    if go:
        result["go_s"] = round(iso_s(go[0]["ts"]) - (pd_off or 0.0) - t0, 1)
    return result
