"""What the live loop's relay pushes to the box besides events: its comparison document and its power windows (L31).
Kept as files under the box's data dir, read by the Compare tab (lab_compare.py) on its next request, no restart.

    <data_dir>/lab_loops/<stamp>/comparison.json          POST /api/learning/lab/import?kind=compare
    <data_dir>/lab_loops/_power/<run_id>/<stage>.<node>.json   POST /api/learning/lab/import?kind=power&run_id=..&stage=..

kind=compare   one JSON document per two-act run, the shape of tooling/demo/live_loop.py's comparison.json
    ({stamp, t0, total_s, acts: {solo|duo: {run_id, seconds, labeling, training, packaging, box, ...}}, rule, idle,
    clock_offsets_s, pod_d_split, duo_over_solo, go_s}). The relay posts it as each act finishes, so a later post of
    the same stamp replaces the earlier one. It needs a `stamp` that is a plain folder name and an `acts` object whose
    every row names its `run_id`.
kind=power     one sampler window attached to a run and a stage: {node, sampler, period_ms, samples: [{ts, w}],
    idle_w, quiet_w, clock_offset_s, start, end, error, hub, account}. `ts` is ISO time on the node's own clock and
    `w` (or `watts`) the reading; `clock_offset_s` is the node's clock minus the relay's, which puts every node on one
    time axis. The run id and the stage come from the query (or the body). Posting the same window again replaces it.

Both are written to a temporary file beside the copy and renamed over it; the same bytes again write nothing and say
`changed: false`. Nothing here is a fixture: only what a relay posted is on the box, and only that is served.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

LOOPS_DIR = "lab_loops"
POWER_DIR = "_power"            # under lab_loops; a stamp cannot start with "_", so it never names a run's folder
COMPARISON = "comparison.json"
MAX_SAMPLES = 20_000            # a 1 Hz sampler for 5 hours; the box's 3 Hz feed for nearly 2
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class Refused(ValueError):
    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


def env_data_dir() -> Path:
    """The gateway's data dir by its own rule (contexts/gateway/bootstrap.py Settings): ATLAS_DATA_DIR, else ./data."""
    return Path(os.environ.get("ATLAS_DATA_DIR", "data"))


def loops_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / LOOPS_DIR


def power_dir(root: str | Path) -> Path:
    """The power windows kept under one loops folder (`<data_dir>/lab_loops`)."""
    return Path(root) / POWER_DIR


def plain(name: Any, what: str) -> str:
    if not isinstance(name, str) or not NAME.fullmatch(name) or ".." in name:
        raise Refused([f"{what} must be a plain name (letters, digits, '.', '_', '-'; not starting with one of those)"])
    return name


def _write(path: Path, raw: bytes) -> bool:
    try:
        if path.read_bytes() == raw:
            return False
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=path.parent, prefix=".part-", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(raw)
            out.flush()
            os.fsync(out.fileno())
        os.chmod(temp, 0o644)   # the gateway may run as another user than the command
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise
    return True


def _json(raw: bytes) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise Refused(["the body is not a JSON document"]) from None


def check_compare(doc: Any) -> dict[str, Any]:
    if not isinstance(doc, dict):
        raise Refused(["a comparison is a JSON object (tooling/demo/live_loop.py's comparison.json)"])
    problems = []
    try:
        plain(doc.get("stamp"), "stamp")
    except Refused as err:
        problems += err.problems
    acts = doc.get("acts")
    if not isinstance(acts, dict) or not acts:
        problems.append("acts must be an object with at least one act")
    else:
        for act, row in acts.items():
            if not isinstance(row, dict):
                problems.append(f"acts.{act} is not an object")
                continue
            try:
                plain(row.get("run_id"), f"acts.{act}.run_id")
            except Refused as err:
                problems += err.problems
    if problems:
        raise Refused(problems)
    return doc


def store_compare(raw: bytes, data_dir: str | Path) -> dict[str, Any]:
    doc = check_compare(_json(raw))
    path = loops_dir(data_dir) / doc["stamp"] / COMPARISON
    changed = _write(path, raw)
    return {"kind": "compare", "changed": changed, "stamp": doc["stamp"],
            "runs": {act: row["run_id"] for act, row in doc["acts"].items()},
            "complete": bool(doc.get("duo_over_solo")), "stored": f"{LOOPS_DIR}/{doc['stamp']}/{COMPARISON}"}


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_power(doc: Any, run_id: str | None, stage: str | None) -> dict[str, Any]:
    if not isinstance(doc, dict):
        raise Refused(["a power window is a JSON object {node, sampler, period_ms, samples: [{ts, w}], idle_w}"])
    run = plain(run_id or doc.get("run_id"), "run_id")
    if run_id and doc.get("run_id") not in (None, run_id):
        raise Refused([f"the window is for run {doc.get('run_id')!r}, not {run_id!r}"])
    st = plain(stage or doc.get("stage"), "stage")
    node = plain(doc.get("node"), "node")
    samples = doc.get("samples")
    if not isinstance(samples, list) or not samples:
        raise Refused(["samples must be a list of {ts, w}"])
    if len(samples) > MAX_SAMPLES:
        raise Refused([f"more than {MAX_SAMPLES} samples in one window"])
    problems, clean = [], []
    for n, s in enumerate(samples, 1):
        w = s.get("w", s.get("watts")) if isinstance(s, dict) else None
        ts = s.get("ts") if isinstance(s, dict) else None
        try:
            datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ValueError:
            problems.append(f"sample {n}: ts is not an ISO time")
            continue
        if not _finite(w):
            problems.append(f"sample {n}: w is not a number")
            continue
        clean.append({"ts": ts, "w": float(w)})
    if problems:
        raise Refused(problems[:20] + ([f"and {len(problems) - 20} more"] if len(problems) > 20 else []))
    out = {k: doc.get(k) for k in ("sampler", "period_ms", "idle_w", "quiet_w", "clock_offset_s", "start", "end",
                                   "error", "hub", "account") if doc.get(k) is not None}
    for k in ("period_ms", "idle_w", "quiet_w", "clock_offset_s"):
        if k in out and not _finite(out[k]):
            raise Refused([f"{k} must be a number"])
    return {"run_id": run, "stage": st, "node": node, **out, "samples": clean}


def store_power(raw: bytes, data_dir: str | Path, run_id: str | None = None, stage: str | None = None) -> dict[str, Any]:
    win = check_power(_json(raw), run_id, stage)
    body = json.dumps(win, separators=(",", ":")).encode()
    path = power_dir(loops_dir(data_dir)) / win["run_id"] / f"{win['stage']}.{win['node']}.json"
    changed = _write(path, body)
    return {"kind": "power", "changed": changed, "run_id": win["run_id"], "stage": win["stage"], "node": win["node"],
            "samples": len(win["samples"]), "stored": f"{LOOPS_DIR}/{POWER_DIR}/{win['run_id']}/{path.name}"}


def power_windows(root: str | Path, run_id: str) -> list[dict[str, Any]]:
    """Every window stored for one run under a loops folder, by stage and node; [] when none (or the id is not plain)."""
    if not isinstance(run_id, str) or not NAME.fullmatch(run_id) or ".." in run_id:
        return []
    folder = power_dir(root) / run_id
    out = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            win = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(win, dict) and isinstance(win.get("samples"), list):
            out.append(win)
    return out
