"""The lab's "Under the hood" report pack, cut down for the Learning lab screen: read-only.

    under_the_hood(pack_file, metrics_file)     what GET /api/learning/lab/report serves

The pack is `lab/refine/report/under-the-hood.json` (`atlas.lab_report/1`, written by `python -m lab report`, described
in lab/docs/report.md). It is the lab lane's file and is only read here. Per section the screen gets the title, the
`reading` (the one sentence the lab wrote to be said aloud), the `state` (measured, partial, empty), the runs it is
still waiting for, and a few key numbers, each with the file and line it was read from. Nothing is recomputed: a
number here is a number object of the pack, copied.

The pack is a committed snapshot, so it can be older than the rows the other tabs read. `stale` says so from two facts
the pack itself records about lab/refine/metrics.jsonl (its sha256 and its line count) against the file on disk now;
the screen then says the report is older instead of letting two tabs disagree in silence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
PACK_FILE = REPO / "lab" / "refine" / "report" / "under-the-hood.json"
PACK_REL = "lab/refine/report/under-the-hood.json"
FORMAT = "atlas.lab_report/1"
KEEP = 6  # key numbers a section shows; the pack holds the rest


def _is_number(v: Any) -> bool:
    return isinstance(v, dict) and {"id", "label", "value", "kind"} <= v.keys()


def _number(n: dict[str, Any]) -> dict[str, Any]:
    """One number of the pack, as the screen shows it: value, unit, kind and where it was read."""
    src = n.get("source") or {}
    return {"id": n["id"], "label": n["label"], "value": n["value"], "unit": n.get("unit"), "kind": n["kind"],
            "file": src.get("file"), "line": src.get("line"), "formula": n.get("formula"), "ts": n.get("ts"),
            "low_confidence": bool(n.get("low_confidence")), "chip": n.get("chip_label"), "other_chip": bool(n.get("other_chip"))}


def _best(numbers: Any) -> dict[str, Any] | None:
    """The largest number of a list that was measured on the lab's own chip."""
    own = [n for n in (numbers or []) if _is_number(n) and not n.get("other_chip") and isinstance(n.get("value"), int | float)]
    return max(own, key=lambda n: n["value"]) if own else None


def _key_numbers(section: dict[str, Any]) -> list[dict[str, Any]]:
    """The few numbers a section's sentence stands on. Each is taken from where lab/docs/report.md says it lives; a
    section id this reader does not know gives none."""
    sid = section.get("id")
    picked: list[Any] = []
    if sid == "ceiling":
        best = _best(section.get("achieved"))
        share = next((s for s in section.get("shares") or [] if best and _is_number(s) and s["id"].endswith(best["id"])), None)
        picked = [section.get("ceiling"), best, share, *(section.get("inputs") or [])]
    elif sid == "frontier":
        picked = [*(section.get("ratios") or [])[:1], *(section.get("more_work") or [])[:1]]
        for sweep in (section.get("power") or [])[:1]:
            points = [p.get("number") for p in sweep.get("points") or []]
            picked += [points[0], points[-1]] if points else []
    elif sid == "engines":
        picked = [e.get("best") for e in section.get("engines") or []]
    elif sid == "finetune":
        loss = section.get("loss") or []
        picked = [loss[0] if loss else None, loss[-1] if len(loss) > 1 else None, section.get("median_step"), *(section.get("numbers") or [])]
    seen: set[str] = set()
    out = []
    for n in picked:
        if _is_number(n) and n["id"] not in seen and n.get("value") is not None:
            seen.add(n["id"])
            out.append(_number(n))
    return out[:KEEP]


def _stale(pack: dict[str, Any], metrics_file: Path | None) -> dict[str, Any] | None:
    """How far the pack is behind the metrics file on disk, or None when it is not (or cannot be told)."""
    rows = next((s for s in pack.get("sources") or [] if s.get("role") == "rows"), None)
    if rows is None or metrics_file is None or not metrics_file.is_file():
        return None
    data = metrics_file.read_bytes()
    if hashlib.sha256(data).hexdigest() == rows.get("sha256"):
        return None
    now = len(data.splitlines())
    then = rows.get("lines")
    return {"file": rows.get("file"), "lines_then": then, "lines_now": now,
            "added": now - then if isinstance(then, int) and now >= then else None}


def under_the_hood(pack_file: str | Path | None = PACK_FILE, metrics_file: str | Path | None = None) -> dict[str, Any]:
    path = Path(pack_file) if pack_file else None
    empty = {"available": False, "file": PACK_REL, "title": None, "data_as_of": None, "chip": None, "sections": [], "stale": None,
             "still_coming": []}
    if path is None or not path.is_file():
        return {**empty, "reason": "the lab's report pack is not on this box (python -m lab report writes it)"}
    try:
        pack = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {**empty, "reason": "the lab's report pack on this box is not readable JSON"}
    if not isinstance(pack, dict) or pack.get("format") != FORMAT:
        return {**empty, "reason": f"the lab's report pack is not in the format this screen reads ({FORMAT})"}
    awaited = {a.get("task"): a for a in pack.get("awaited") or [] if isinstance(a, dict)}
    sections = []
    for s in pack.get("sections") or []:
        if not isinstance(s, dict) or not s.get("reading"):
            continue  # a section with no sentence has nothing to say: never a card with only a title
        waiting = [{"task": t, "what": awaited[t].get("what"), "status": awaited[t].get("status")}
                   for t in s.get("awaiting") or [] if t in awaited and not awaited[t].get("landed")]
        sections.append({"id": s.get("id"), "title": s.get("title"), "reading": s["reading"],
                         "state": s.get("state") if s.get("state") in ("measured", "partial", "empty") else "partial",
                         "why": s.get("why"), "numbers": _key_numbers(s), "waiting": waiting,
                         "notes": [n for n in s.get("notes") or [] if isinstance(n, str)][:2]})
    chip = pack.get("chip") or {}
    return {"available": True, "reason": None, "file": PACK_REL, "title": pack.get("title"), "data_as_of": pack.get("data_as_of"),
            "chip": chip.get("name"), "sections": sections,
            "stale": _stale(pack, Path(metrics_file) if metrics_file else None),
            "still_coming": [{"task": a.get("task"), "what": a.get("what"), "status": a.get("status")}
                             for a in awaited.values() if not a.get("landed")]}
