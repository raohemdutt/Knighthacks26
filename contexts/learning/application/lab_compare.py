"""Two training runs side by side for the Learning lab's Compare tab (L31): alone against with a friend.

    compare_runs(loops, view, a, b)     what GET /api/learning/lab/compare?a=<run>&b=<run> serves; CompareMissing (404)
                                        when either run is not on this box yet
    load_loops(dirs)                    the live-loop records under those folders, newest first

Only what this box holds is served, never a fixture or a rehearsal: the folders are the box's own
`<data_dir>/lab_loops` (default_dirs), where POST /api/learning/lab/import?kind=compare and kind=power keep what the
relay posts (application/lab_loops.py). Where the numbers come from, for each run:

1. A live-loop record: a folder holding the relay's `comparison.json` (posted, or a relay's --out folder copied in by
   hand, with its `stages.jsonl` and `power-<node>.jsonl` beside it), plus the power windows posted for its runs. The
   joules in it are T4's rule, worked out by the relay with loop_energy.py: gross, nothing subtracted, the idle floor
   beside.
2. The box's own log (the run's story in lab_view.py): steps, seconds and examples for a run no sampler measured,
   such as L16. Its energy cells are empty with the reason, never a guess.

The ratios (b over a) are computed here with loop_energy.ratios, the function the relay's own `duo_over_solo` comes
from, so the card and the box agree on every figure.

The answer is the comparison document's shape (stamp, t0, total_s, acts, rule, idle, clock_offsets_s, pod_d_split,
duo_over_solo, go_s) with the two runs as its acts, plus:

    series      the three power traces (pod C, pod D, the box) on one time axis: {zero (ISO time of second 0, the
                relay's stopwatch zero), axis, nodes: {pod-c|pod-d|box: {name, idle_w, quiet_w, points: [[s, W]],
                windows: [{run_id, stage, sampler, period_ms, samples, error, hub, account}]}}, runs: [{run_id, act,
                start_s, end_s, stages: {labeling|training|packaging: [from_s, to_s]}}]}
    a, b        each run as the Compare tab draws it (members, examples, seconds, the stage rows, traces from the run's
                own start, source, idle)
    ratios      b over a for every RATIO_KEYS figure; ratio_keys, runs (every run the box can compare), asked, missing
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from contexts.learning.application import lab_loops
from contexts.learning.application.loop_energy import (
    DUO_OVER_SOLO,
    MEMBERS,
    NODE_NAMES,
    POD_D_SPLIT,
    RATIO_KEYS,
    RULE,
    iso_s,
    ratio,
    ratios,
)

NODES = ("pod-c", "pod-d", "box")
MAX_POINTS = 400        # per trace and run: enough for a phone-width chart
MAX_SERIES = 1500       # per node on the shared axis
AXIS = "seconds after the live loop's start (the relay's stopwatch zero; each node's clock moved onto the relay's)"

__all__ = ["RATIO_KEYS", "CompareMissing", "Loop", "compare_runs", "default_dirs", "load_loops", "ratio"]


class CompareMissing(LookupError):
    """A run the comparison needs is not on this box (yet): the route answers 404 with this message."""


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


class Loop:
    """One live-loop record: its comparison, the stage marks, and the samples of the three nodes on the relay's clock."""

    def __init__(self, folder: Path, root: Path) -> None:
        self.folder, self.root = folder, root
        self.source = f"{lab_loops.LOOPS_DIR}/{folder.relative_to(root).as_posix()}/{lab_loops.COMPARISON}"
        self.comparison: dict[str, Any] = json.loads((folder / lab_loops.COMPARISON).read_text(encoding="utf-8"))
        if not isinstance(self.comparison.get("acts"), dict):
            raise ValueError("no acts")
        if lab_loops.under_fixtures(folder) or lab_loops.marked_fixture(self.comparison):
            raise ValueError("a fixture: never served")
        self.stamp = str(self.comparison.get("stamp") or folder.name)
        self.stages = _jsonl(folder / "stages.jsonl")
        self.offsets = self.comparison.get("clock_offsets_s") or {}
        self.t0 = self._t0()

    def _t0(self) -> float | None:
        """The stopwatch's zero on the relay's clock, as the relay wrote it or as its --fallback works it out."""
        if isinstance(self.comparison.get("t0"), int | float):
            return float(self.comparison["t0"])
        go = [s for s in self.stages if s.get("stage") == "go" and s.get("ts")]
        if not go:
            return None
        return iso_s(go[0]["ts"]) - float(self.offsets.get("pod-d") or 0.0) - float(self.comparison.get("go_s") or 0.0)

    def acts(self) -> dict[str, dict[str, Any]]:
        """The acts with a run id, leaving out any run a committed rehearsal names (a fixture copied onto the box)."""
        return {act: row for act, row in self.comparison["acts"].items()
                if isinstance(row, dict) and row.get("run_id") and not lab_loops.is_fixture_run(row["run_id"])}

    def act_of(self, run_id: str) -> tuple[str, dict[str, Any]] | None:
        return next(((act, row) for act, row in self.acts().items() if row["run_id"] == run_id), None)

    def samples(self) -> dict[str, list[tuple[float, float]]]:
        """The relay's own power-<node>.jsonl beside the comparison (a copied --out folder), on the relay's clock."""
        out = {}
        for node in NODES:
            off = float(self.offsets.get(node) or 0.0)
            out[node] = [(iso_s(r["ts"]) - off, float(r["w"])) for r in _jsonl(self.folder / f"power-{node}.jsonl")
                         if r.get("ts") and isinstance(r.get("w"), int | float)]
        return out


def load_loops(dirs: list[Path]) -> list[Loop]:
    """Every folder (or folder one level down) under `dirs` that holds a comparison.json, newest stamp first."""
    seen: set[Path] = set()
    loops: list[Loop] = []
    for root in dirs:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/comparison.json")) + sorted(root.glob("*/*/comparison.json")):
            folder = path.parent.resolve()
            if folder in seen or path.parent.name.startswith("_") or path.parent.parent.name.startswith("_"):
                continue
            seen.add(folder)
            try:
                loop = Loop(path.parent, root)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if loop.acts():
                loops.append(loop)
    return sorted(loops, key=lambda lp: lp.stamp, reverse=True)


def default_dirs(data_dir: str | Path | None) -> list[Path]:
    """What this box holds: <data_dir>/lab_loops. Never the committed rehearsal (fixtures are for tests)."""
    return [lab_loops.loops_dir(data_dir)] if data_dir else []


def env_dirs() -> list[Path]:
    """The loops folder of a gateway whose app.py does not pass one: the data dir by the gateway's own rule
    (ATLAS_DATA_DIR, else ./data, as contexts/gateway/bootstrap.py Settings reads it)."""
    return default_dirs(lab_loops.env_data_dir())


# --- one run --------------------------------------------------------------------------------------------------

def _windows(loops: list[Loop], run_id: str) -> list[dict[str, Any]]:
    out, roots = [], []
    for loop in loops:
        if loop.root not in roots:
            roots.append(loop.root)
    for root in roots:
        out += lab_loops.power_windows(root, run_id)
    return out


def _from_loop(loop: Loop, act: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": row["run_id"], "act": act, "members": MEMBERS.get(act), "examples": row.get("examples"),
        "train_rows": row.get("train_rows"), "held_out": row.get("held_out"), "outcome": row.get("outcome"),
        "seconds": row.get("seconds") or {}, "labeling": row.get("labeling"), "training": row.get("training"),
        "packaging": row.get("packaging"), "box": row.get("box"), "pods_joules": row.get("pods_joules"),
        "j_per_example": row.get("j_per_example"), "traces": {},
        "source": {"kind": "loop", "ref": loop.source, "stamp": loop.stamp, "rule": loop.comparison.get("rule"),
                   "split": loop.comparison.get("pod_d_split")},
        "idle": loop.comparison.get("idle"), "reason": None,
    }


def _value(n: dict[str, Any] | None) -> Any:
    return n.get("value") if isinstance(n, dict) else None


def _from_story(story: dict[str, Any]) -> dict[str, Any]:
    """What the box's own log holds for a run that no sampler measured: counts and seconds, no joules."""
    tr, ds, power = story.get("training") or {}, story.get("dataset") or {}, story.get("power") or {}
    steps, train_s, energy = _value(tr.get("steps")), _value(tr.get("wall_s")), _value(power.get("energy_j"))
    why = "no power sampler ran for this run, so its joules are not in any record"
    return {
        "run_id": story["run_id"], "act": None, "members": None, "examples": _value(ds.get("examples")),
        "train_rows": _value(ds.get("train")), "held_out": _value(ds.get("held_out")), "outcome": story.get("status"),
        "seconds": {k: v for k, v in (("training", train_s), ("total", _value(story.get("wall_s")))) if v is not None},
        "labeling": None,
        "training": {"steps": steps, "seconds": train_s, "joules": energy,
                     "j_per_step": round(energy / steps, 1) if energy and steps else None,
                     "loss_first": _value(tr.get("loss_first")), "loss_last": _value(tr.get("loss_last")),
                     "node": tr.get("node")},
        "packaging": None, "box": None, "pods_joules": None, "j_per_example": None, "traces": {},
        "source": {"kind": "log", "ref": f"GET /api/learning/lab/{story['run_id']}", "stamp": None, "rule": None, "split": None},
        "idle": None, "reason": None if energy else why,
    }


def candidates(loops: list[Loop], view: Any | None) -> list[dict[str, Any]]:
    """Every run the card can compare, measured ones first: run id, act, where it comes from, newest first."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for loop in loops:
        for act, row in loop.acts().items():
            if row["run_id"] in seen:
                continue
            seen.add(row["run_id"])
            out.append({"run_id": row["run_id"], "act": act, "members": MEMBERS.get(act), "examples": row.get("examples"),
                        "measured": True, "stamp": loop.stamp, "source": loop.source})
    for r in (view.runs() if view is not None and getattr(view, "run_ids", None) else []):
        if r["run_id"] not in seen and not lab_loops.is_fixture_run(r["run_id"]):
            seen.add(r["run_id"])
            out.append({"run_id": r["run_id"], "act": None, "members": None, "examples": None, "measured": False,
                        "stamp": r.get("started"), "source": "the box's log"})
    return out


def _find(loops: list[Loop], view: Any | None, run_id: str) -> tuple[Loop | None, dict[str, Any] | None, dict[str, Any] | None]:
    """(its loop, its row in the loop's comparison, the run as the tab draws it), or Nones."""
    for loop in loops:
        found = loop.act_of(run_id)
        if found:
            return loop, found[1], _from_loop(loop, *found)
    real = view is not None and run_id in getattr(view, "run_ids", ()) and not lab_loops.is_fixture_run(run_id)
    story = view.story(run_id) if real else None
    return None, None, (_from_story(story) if story else None)


def defaults(loops: list[Loop]) -> tuple[str | None, str | None, str | None]:
    """The newest record's solo run and pooled run, and why not when it lacks one (act 2 still going, say)."""
    if not loops:
        return None, None, ("this box holds no live-loop comparison yet; the relay posts one as each act finishes "
                            "(POST /api/learning/lab/import?kind=compare)")
    newest = loops[0]
    acts = newest.acts()
    solo = acts.get("solo", {}).get("run_id")
    pooled = next((row["run_id"] for act, row in acts.items() if (MEMBERS.get(act) or 0) > 1), None)
    if solo and pooled:
        return solo, pooled, None
    lacking = "the pooled run (act 2)" if solo else "the solo run (act 1)"
    return solo, pooled, f"the newest live loop on this box ({newest.stamp}) does not have {lacking} yet"


# --- the time axis --------------------------------------------------------------------------------------------

def _thin(points: list[list[float]], cap: int) -> list[list[float]]:
    step = max(1, -(-len(points) // cap))
    return points[::step]


def _iso(t: float | None) -> str | None:
    return None if t is None else datetime.fromtimestamp(t, UTC).isoformat().replace("+00:00", "Z")


def _spans(row: dict[str, Any] | None, start: float) -> dict[str, list[float]]:
    """Each stage's [from, to] on the axis, from the act's start and its stage seconds."""
    secs = (row or {}).get("seconds") or {}
    out, at = {}, start
    for name, gap in (("labeling", None), ("training", "waiting"), ("packaging", None)):
        if gap and isinstance(secs.get(gap), int | float):
            at += secs[gap]
        if not isinstance(secs.get(name), int | float):
            break
        out[name] = [round(at, 1), round(at + secs[name], 1)]
        at += secs[name]
    return out


def series(found: list[tuple[Loop | None, dict[str, Any] | None, dict[str, Any]]], loops: list[Loop]) -> dict[str, Any]:
    """The three power traces of both runs on one axis, and each run's stage spans on it."""
    pts: dict[str, dict[float, float]] = {n: {} for n in NODES}
    windows: dict[str, list[dict[str, Any]]] = {n: [] for n in NODES}
    idle: dict[str, dict[str, Any]] = {}
    starts: list[tuple[str, float | None]] = []
    used: list[Loop] = []
    for loop, row, run in found:
        if loop is not None and loop not in used:
            used.append(loop)
            for node, samples in loop.samples().items():
                for t, w in samples:
                    pts[node].setdefault(round(t, 3), w)
            for node, floor in (loop.comparison.get("idle") or {}).items():
                if isinstance(floor, dict):
                    idle.setdefault(node, floor)
        offsets = loop.offsets if loop is not None else {}
        for win in _windows(loops, run["run_id"]):
            node = win.get("node")
            if node not in pts:
                pts[node], windows[node] = {}, []
            off = win.get("clock_offset_s")
            off = float(off if isinstance(off, int | float) else offsets.get(node) or 0.0)
            for s in win["samples"]:
                try:
                    pts[node].setdefault(round(iso_s(s["ts"]) - off, 3), float(s["w"]))
                except (KeyError, TypeError, ValueError):
                    continue
            windows[node].append({"run_id": run["run_id"], **{k: win.get(k) for k in (
                "stage", "sampler", "period_ms", "error", "hub", "account", "idle_w", "quiet_w")},
                "samples": len(win["samples"])})
            if isinstance(win.get("idle_w"), int | float):
                idle.setdefault(node, {"calibrated_w": win["idle_w"], "quiet_w": win.get("quiet_w")})
        start = row.get("start_s") if row else None
        starts.append((run["run_id"], loop.t0 + start if loop is not None and loop.t0 is not None
                       and isinstance(start, int | float) else None))
    zeros = [loop.t0 for loop in used if loop.t0 is not None]
    every = sorted(t for node in pts.values() for t in node)
    known = [t for _, t in starts if t is not None]
    zero = min(zeros) if zeros else min(known) if known else every[0] if every else None
    runs = []
    for (loop, row, run), (_, at) in zip(found, starts):
        start_s = None if at is None or zero is None else round(at - zero, 1)
        end_s = (round(start_s + run["seconds"]["total"], 1) if start_s is not None
                 and isinstance((run.get("seconds") or {}).get("total"), int | float) else None)
        runs.append({"run_id": run["run_id"], "act": run.get("act"), "start_s": start_s, "end_s": end_s,
                     "stages": _spans(row, start_s) if start_s is not None else {}})
    nodes = {}
    for node, by_t in pts.items():
        points = [[round(t - zero, 2), round(w, 2)] for t, w in sorted(by_t.items())] if zero is not None else []
        if runs and all(r["start_s"] is not None and r["end_s"] is not None for r in runs):
            lo, hi = min(r["start_s"] for r in runs) - 30, max(r["end_s"] for r in runs) + 10
            points = [p for p in points if lo <= p[0] <= hi]
        floor = idle.get(node) or {}
        nodes[node] = {"name": NODE_NAMES.get(node, node), "idle_w": floor.get("calibrated_w"),
                       "quiet_w": floor.get("quiet_w"), "points": _thin(points, MAX_SERIES), "windows": windows.get(node, [])}
    return {"zero": _iso(zero), "axis": AXIS, "nodes": nodes, "runs": runs}


def _run_traces(run: dict[str, Any], axis: dict[str, Any]) -> dict[str, Any]:
    """One run's three traces from its own start (the Compare tab's per-run charts), cut from the shared axis."""
    mine = next((r for r in axis["runs"] if r["run_id"] == run["run_id"]), None)
    if not mine or mine["start_s"] is None or mine["end_s"] is None:
        return {}
    s, e = mine["start_s"], mine["end_s"]
    out = {}
    for node, body in axis["nodes"].items():
        points = [[round(t - s, 2), w] for t, w in body["points"] if s - 5 <= t <= e + 5]
        if points:
            out[node] = {"name": body["name"], "points": _thin(points, MAX_POINTS)}
    return out


# --- the answer -----------------------------------------------------------------------------------------------

def compare_runs(loops: list[Loop], view: Any | None, a: str | None = None, b: str | None = None) -> dict[str, Any]:
    da, db, why = defaults(loops)
    a, b = a or da, b or db
    if not a or not b:
        raise CompareMissing(why or "give two run ids: ?a=<run_id>&b=<run_id>")
    found = [_find(loops, view, x) for x in (a, b)]
    missing = [x for x, (_, _, run) in zip((a, b), found) if run is None]
    if missing:
        raise CompareMissing(f"no run {' or '.join(repr(m) for m in missing)} on this box yet; "
                             "the relay posts each act's comparison when the act finishes")
    (loop_a, row_a, run_a), (loop_b, row_b, run_b) = found
    axis = series([(loop_a, row_a, run_a), (loop_b, row_b, run_b)], loops)
    for run in (run_a, run_b):
        run["traces"] = _run_traces(run, axis)
    key_a, key_b = run_a["act"] or "a", run_b["act"] or "b"
    if key_a == key_b:
        key_a, key_b = "a", "b"
    rows = {key_a: row_a or run_a, key_b: row_b or run_b}
    same = loop_a is not None and loop_a is loop_b
    base = dict(loop_a.comparison) if same else {
        "stamp": None, "t0": None, "total_s": None, "rule": RULE,
        "idle": (loop_a or loop_b).comparison.get("idle") if (loop_a or loop_b) else None,
        "clock_offsets_s": None, "pod_d_split": POD_D_SPLIT, "go_s": None}
    base.update(acts=rows, duo_over_solo=ratios(rows[key_a], rows[key_b], DUO_OVER_SOLO))
    return {**base, "series": axis, "a": run_a, "b": run_b, "ratios": ratios(run_a, run_b, RATIO_KEYS),
            "ratio_keys": {k: list(p) for k, p in RATIO_KEYS.items()}, "runs": candidates(loops, view),
            "asked": {"a": a, "b": b}, "missing": [], "same_loop": same}
