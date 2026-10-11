"""Two training runs side by side for the Learning lab's Compare tab (L31): alone against with a friend.

    compare_runs(loops, view, a, b)     what GET /api/learning/lab/compare?a=<run>&b=<run> serves
    load_loops(dirs)                    the live-loop records under those folders, newest first

Where the numbers come from, in this order for each run:

1. A live-loop record (L29): a folder holding the relay's `comparison.json`, its `stages.jsonl` and the three power
   traces `power-pod-c.jsonl`, `power-pod-d.jsonl`, `power-box.jsonl` (tooling/demo/live_loop.py writes them). The
   committed rehearsal is under fixtures/learning/live-loop-*/; a relay's own --out folder copied to
   `<data_dir>/lab_loops/<stamp>/` is read the same way, newest first. The joules in it are T4's rule, worked out by
   the relay with contexts/telemetry/domain/energy.integrate: gross, nothing subtracted, the idle floor beside.
2. The box's own log (the run's story in lab_view.py): steps, seconds and examples for a run that no sampler measured,
   such as L16. Its energy cells are empty with the reason, never a guess.

The ratios (b over a) are computed here, once, with the keys and the rounding of the relay's `duo_over_solo`, so the
card and the relay agree on every figure.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
COMMITTED = REPO / "fixtures" / "learning"
NODES = ("pod-c", "pod-d", "box")
NODE_NAMES = {"pod-c": "Pod C (teacher, labeling)", "pod-d": "Pod D (trainer)", "box": "The box (Jetson)"}
MEMBERS = {"solo": 1, "duo": 2}  # the relay's two acts: one member's examples, then two members' pooled
MAX_POINTS = 400  # per trace and run: enough for a phone-width chart

# What the card shows and the ratio of each, b over a: name -> path into an act row (the relay's `duo_over_solo` keys).
RATIO_KEYS: dict[str, tuple[str, ...]] = {
    "examples": ("examples",),
    "members": ("members",),
    "act_seconds": ("seconds", "total"),
    "labeling_s": ("seconds", "labeling"),
    "training_s": ("seconds", "training"),
    "packaging_s": ("seconds", "packaging"),
    "labeling_j": ("labeling", "joules"),
    "labeling_j_per_token": ("labeling", "j_per_token"),
    "labeling_j_per_example": ("labeling", "j_per_example"),
    "training_j": ("training", "joules"),
    "training_j_per_step": ("training", "j_per_step"),
    "training_j_per_train_row": ("training", "j_per_train_row"),
    "packaging_j": ("packaging", "joules"),
    "pods_joules": ("pods_joules",),
    "j_per_example": ("j_per_example",),
    "box_j": ("box", "joules"),
}


def ratio(a: Any, b: Any) -> float | None:
    """a over b, rounded like the relay's; None when either is missing or b is zero."""
    if isinstance(a, int | float) and isinstance(b, int | float) and b:
        return round(a / b, 3)
    return None


def _get(row: dict[str, Any] | None, path: tuple[str, ...]) -> Any:
    for p in path:
        row = row.get(p) if isinstance(row, dict) else None
    return row


def _iso_s(text: str) -> float:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.name if path.parent.name == "" else f"{path.parent.name}/{path.name}"


class Loop:
    """One live-loop record: its comparison, the stage marks and the three traces on the relay's clock."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.source = _rel(folder / "comparison.json")
        self.comparison: dict[str, Any] = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
        self.stamp = str(self.comparison.get("stamp") or folder.name)
        self.stages = _jsonl(folder / "stages.jsonl")
        self.offsets = self.comparison.get("clock_offsets_s") or {}
        self.traces = {n: [(_iso_s(r["ts"]), float(r["w"])) for r in _jsonl(folder / f"power-{n}.jsonl")
                           if r.get("ts") and isinstance(r.get("w"), int | float)] for n in NODES}
        self.t0 = self._t0()

    def _t0(self) -> float | None:
        """The stopwatch's zero on the relay's clock, as the relay's own --fallback works it out."""
        if isinstance(self.comparison.get("t0"), int | float):
            return float(self.comparison["t0"])
        go = [s for s in self.stages if s.get("stage") == "go" and s.get("ts")]
        if not go:
            return None
        return _iso_s(go[0]["ts"]) - float(self.offsets.get("pod-d") or 0.0) - float(self.comparison.get("go_s") or 0.0)

    def acts(self) -> dict[str, dict[str, Any]]:
        return {act: row for act, row in (self.comparison.get("acts") or {}).items() if isinstance(row, dict) and row.get("run_id")}

    def trace(self, node: str, start_s: float, end_s: float) -> list[list[float]]:
        """One node's watts over a run's window, as [seconds after the run's start, watts], thinned to MAX_POINTS."""
        if self.t0 is None:
            return []
        off = float(self.offsets.get(node) or 0.0)
        pts = [[round(t - off - self.t0 - start_s, 2), round(w, 2)] for t, w in self.traces.get(node, [])
               if start_s - 5 <= t - off - self.t0 <= end_s + 5]
        step = max(1, len(pts) // MAX_POINTS)
        return pts[::step]


def load_loops(dirs: list[Path]) -> list[Loop]:
    """Every folder (or folder one level down) under `dirs` that holds a comparison.json, newest stamp first."""
    seen: set[Path] = set()
    loops: list[Loop] = []
    for root in dirs:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/comparison.json")) + sorted(root.glob("*/*/comparison.json")):
            folder = path.parent.resolve()
            if folder in seen:
                continue
            seen.add(folder)
            try:
                loops.append(Loop(path.parent))
            except (OSError, ValueError, KeyError):
                continue
    return sorted(loops, key=lambda lp: lp.stamp, reverse=True)


def default_dirs(data_dir: str | Path | None) -> list[Path]:
    """The relay copies on this box first (newest wins a tie), then the committed rehearsal."""
    dirs = [Path(data_dir) / "lab_loops"] if data_dir else []
    return dirs + [COMMITTED]


def _from_loop(loop: Loop, act: str, row: dict[str, Any]) -> dict[str, Any]:
    start, end = row.get("start_s"), row.get("end_s")
    traces = ({n: {"name": NODE_NAMES[n], "points": loop.trace(n, float(start), float(end))} for n in NODES}
              if isinstance(start, int | float) and isinstance(end, int | float) else {})
    return {
        "run_id": row["run_id"], "act": act, "members": MEMBERS.get(act), "examples": row.get("examples"),
        "train_rows": row.get("train_rows"), "held_out": row.get("held_out"), "outcome": row.get("outcome"),
        "seconds": row.get("seconds") or {}, "labeling": row.get("labeling"), "training": row.get("training"),
        "packaging": row.get("packaging"), "box": row.get("box"), "pods_joules": row.get("pods_joules"),
        "j_per_example": row.get("j_per_example"), "traces": traces,
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
    examples = _value(ds.get("examples"))
    why = "no power sampler ran for this run, so its joules are not in any record"
    return {
        "run_id": story["run_id"], "act": None, "members": None, "examples": examples,
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
        if r["run_id"] not in seen:
            seen.add(r["run_id"])
            out.append({"run_id": r["run_id"], "act": None, "members": None, "examples": None, "measured": False,
                        "stamp": r.get("started"), "source": "the box's log"})
    return out


def one_run(loops: list[Loop], view: Any | None, run_id: str) -> dict[str, Any] | None:
    for loop in loops:
        for act, row in loop.acts().items():
            if row["run_id"] == run_id:
                return _from_loop(loop, act, row)
    story = view.story(run_id) if view is not None and run_id in getattr(view, "run_ids", ()) else None
    return _from_story(story) if story else None


def defaults(cands: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """The latest solo run and the latest pooled run; else the two newest measured runs."""
    solo = next((c["run_id"] for c in cands if c["act"] == "solo"), None)
    pooled = next((c["run_id"] for c in cands if (c["members"] or 0) > 1), None)
    if solo and pooled:
        return solo, pooled
    ids = [c["run_id"] for c in cands]
    return (ids[0] if ids else None), (ids[1] if len(ids) > 1 else None)


def compare_runs(loops: list[Loop], view: Any | None, a: str | None = None, b: str | None = None) -> dict[str, Any]:
    cands = candidates(loops, view)
    da, db = defaults(cands)
    a, b = a or da, b or db
    run_a = one_run(loops, view, a) if a else None
    run_b = one_run(loops, view, b) if b else None
    ratios = ({k: ratio(_get(run_b, p), _get(run_a, p)) for k, p in RATIO_KEYS.items()} if run_a and run_b else None)
    return {"a": run_a, "b": run_b, "ratios": ratios, "ratio_keys": {k: list(p) for k, p in RATIO_KEYS.items()},
            "runs": cands, "asked": {"a": a, "b": b},
            "missing": [x for x, run in ((a, run_a), (b, run_b)) if x and run is None]}
