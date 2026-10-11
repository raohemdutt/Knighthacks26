"""The lab's optimisation loop as a timeline for the Learning lab: what was tried, what it measured and what the
verdict was. Real records only, each card with the record it came from.

    timeline(rows)        # what GET /api/learning/lab/experiments serves; `rows` are the metrics file's rows

Three kinds of record, each read with the lab's own code (imported; nothing in lab/ is changed or written):

- experiments: the lab ledger's `experiment add / amend / result` rows, folded by `lab.machine.experiments.fold`
  into what each experiment is now, with the verdict `lab.machine.verdict` gave its arms (Welch's t test or the
  two-proportion z test, a floor of runs per arm). The ledger lives in LAB_STATE_DIR (else lab/state) on the lab's
  machine. Where this box has no ledger file, the cards are the experiments committed in that module's SEEDS (the
  queue the ledger starts from) and each says so: queued, never run here.
- refinement passes: one card per `## Pass N` and `## Merge N` entry of lab/refine/<track>/REFINE_LOG.md and
  MERGE_LOG.md, parsed by `lab.machine.refine_formats.parse_log`: the date, how many changes it made, how many
  questions it raised for the owner, and whether the track converged.
- benchmark sessions: the rows of lab/refine/metrics.jsonl that share a `conditions.session`, as one card per
  session (what was measured, on which node, how many rows, how many are low confidence).

Status is one of queued, running, won (an arm beat the control), lost (the control stayed ahead), inconclusive (no
difference, or too early to call), dropped, measured (a session with no verdict) and, for a pass, converged or
open. Nothing is ranked or scored here beyond what the records carry.
"""

from pathlib import Path
from typing import Any

from contexts.learning.application.lab_bench import REL, _text
from contexts.learning.application.lab_view import by_file

REFINE_DIR = Path(__file__).resolve().parents[3] / "lab" / "refine"
VERDICT_STATUS = {"b_ahead": "won", "a_ahead": "lost", "no_difference": "inconclusive", "too_early": "inconclusive"}
VERDICT_WORDS = {"b_ahead": "An arm beat the control", "a_ahead": "The control stayed ahead",
                 "no_difference": "No difference found", "too_early": "Too early to call"}
STATUSES = ("queued", "running", "won", "lost", "inconclusive", "dropped", "measured", "converged", "open")


def _num(label: str, value: Any, unit: str | None, source: dict[str, Any], low: bool = False, reason: str | None = None) -> dict[str, Any]:
    return {"label": label, "value": value, "unit": unit, "source": source, "low_confidence": low, "reason": reason if low else None}


def experiment_cards(state_dir: str | Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The experiments as cards, and where they were read from. Never creates the ledger."""
    try:
        from lab.machine import experiments as lab_experiments
        from lab.machine import ledger
    except ImportError as exc:  # the gateway run from outside the repo root
        return [], {"kind": "none", "ref": None, "note": f"the lab's code could not be imported here ({exc})"}
    path = ledger.ledger_path(state_dir)
    if path.is_file():
        folded, problems = lab_experiments.fold(ledger.read(state_dir=state_dir, verb=lambda v: v in lab_experiments.VERBS))
        origin = {"kind": "ledger", "ref": str(path.name), "note": None, "problems": problems}
        ranked = lab_experiments.rank(folded.values(), include_finished=True)
    else:
        folded = {}
        for fields in lab_experiments.SEEDS:
            definition = lab_experiments.validate_definition(fields)
            entry = {**definition, "status": lab_experiments.QUEUED, "status_note": None, "result": None,
                     "added": {"seq": len(folded) + 1, "ts": None, "actor": None}, "updated": {"seq": None, "ts": None}}
            entry["gain"] = lab_experiments.expected_gain(entry["expected_delta"], entry["better"])
            entry["score"] = lab_experiments.score_of(entry)
            folded[entry["id"]] = entry
        origin = {"kind": "seeds", "ref": "lab/machine/experiments.py SEEDS",
                  "note": "This box has no lab ledger, so these are the experiments the ledger starts from. "
                          "Whatever the lab has run or added since is in its own ledger, not here."}
        ranked = lab_experiments.rank(folded.values(), include_finished=True)
    return [_experiment_card(e, origin) for e in ranked], origin


def _experiment_card(e: dict[str, Any], origin: dict[str, Any]) -> dict[str, Any]:
    result = e.get("result")
    from_ledger = origin["kind"] == "ledger"
    source = ({"kind": "ledger", "ref": f"lab ledger seq {(result or e['updated'])['seq']}", "ids": []} if from_ledger
              else {"kind": "file", "ref": origin["ref"], "ids": []})
    numbers = []
    if e.get("expected_delta") is not None:
        numbers.append(_num(f"Expected change in {e['metric']}", round(e["expected_delta"] * 100, 1), "%", source))
    if e.get("confidence") is not None:
        numbers.append(_num("Chance of at least half that change", round(e["confidence"] * 100), "%", source))
    if e.get("score") is not None:
        numbers.append(_num("Score (expected gain x confidence)", round(e["score"], 3), None, source))
    status, verdict, detail = e["status"], None, e.get("status_note")
    if result:
        label = result.get("label")
        verdict = VERDICT_WORDS.get(label, label)
        status = VERDICT_STATUS.get(label, "inconclusive") if e["status"] == "done" else "running"
        early = label == "too_early"
        for name, arm in (result.get("arms") or {}).items():
            if arm.get("mean") is not None:
                numbers.append(_num(f"Arm {name}: mean of {arm.get('n')} runs", round(arm["mean"], 4), _unit_of(e["metric"]), source,
                                    low=early, reason="fewer runs than the verdict needs"))
            elif arm.get("x") is not None:
                numbers.append(_num(f"Arm {name}: successes of {arm.get('n')} trials", arm["x"], "trials", source,
                                    low=early, reason="fewer trials than the verdict needs"))
        notes = [c.get("note") for c in (result.get("comparisons") or {}).values() if c.get("note")]
        detail = " ".join(notes) or detail
        winner = result.get("winner")
        if label == "b_ahead" and winner:
            verdict = f"Arm {winner} beat the control"
    return {
        "id": e["id"], "kind": "experiment", "title": e.get("title") or e["lever"], "status": status,
        "ts": (result or {}).get("ts") or e["updated"].get("ts") or e["added"].get("ts"), "day_only": False,
        "track": e.get("track"), "node": e.get("node"), "lever": e["lever"], "arms": e["arms"], "metric": e["metric"],
        "better": e["better"], "hypothesis": e["hypothesis"], "rank": e.get("rank"), "needs": e.get("needs") or [],
        "verdict": verdict, "detail": detail, "numbers": numbers, "source": source, "citation": e.get("citation"),
    }


def _unit_of(metric: str) -> str | None:
    tail = metric.rsplit(".", 1)[-1]
    return {"joules_per_token": "J/token", "decode_tok_s": "tok/s", "samples_per_s": "samples/s", "ttft_s": "s"}.get(tail)


def pass_cards(refine_dir: str | Path = REFINE_DIR) -> list[dict[str, Any]]:
    """One card per refinement pass and merge pass the logs hold."""
    try:
        from lab.machine import refine_formats
    except ImportError:
        return []
    root = Path(refine_dir)
    logs = [(track, root / track / "REFINE_LOG.md") for track in refine_formats.TRACKS] + [("all tracks", root / "MERGE_LOG.md")]
    cards = []
    for track, path in logs:
        if not path.is_file():
            continue
        try:
            log = refine_formats.parse_log(path.read_text(encoding="utf-8"))
        except ValueError:
            continue  # a log the lab's own parser refuses is not shown rather than half-read
        rel = f"lab/refine/{path.relative_to(root).as_posix()}"
        for entry in log.entries:
            if not isinstance(entry, refine_formats.PassEntry):
                continue
            heading = f"## {'Merge' if entry.kind == 'merge' else 'Pass'} {entry.number}, {entry.date}"
            source = {"kind": "file", "ref": f"{rel}, \"{heading}\"", "ids": []}
            converged = entry.converged is True or entry.ending == "converged"
            cards.append({
                "id": f"{'merge' if entry.kind == 'merge' else track}-{entry.kind}-{entry.number}", "kind": entry.kind,
                "title": f"{'Merge' if entry.kind == 'merge' else track.capitalize() + ' pass'} {entry.number}",
                "status": "converged" if converged else "open", "ts": entry.date, "day_only": True, "track": track,
                "verdict": "Converged: the pass found nothing left to change" if converged else "Not converged: more passes to run",
                "detail": entry.summary.strip()[:600] or None, "source": source,
                "numbers": [_num("Changes made", len(entry.changes), "changes", source),
                            _num("Findings", len(entry.findings), "findings", source),
                            _num("Questions raised for the owner", len(entry.rulings_raised), "questions", source)],
            })
    return cards


def session_cards(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One card per benchmark session in the metrics file: the rows that share a `conditions.session`."""
    sessions: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        session = _text(r["conditions"].get("session"))
        if session:
            sessions.setdefault(session, []).append(r)
    cards = []
    for session, members in sessions.items():
        lines = sorted(r["_line"] for r in members)
        low = sum(bool(r.get("low_confidence")) for r in members)
        families = sorted({r["metric"].split(".")[0] for r in members})
        nodes = sorted({str(r.get("node")) for r in members})
        batches = sorted({b for b in (r["conditions"].get("batch_size") for r in members) if isinstance(b, int | float)})
        source = by_file(f"{REL}:{lines[0]}-{lines[-1]}")
        reason = _text(members[0]["conditions"].get("low_confidence_reason"))
        cards.append({
            "id": session, "kind": "bench", "title": f"Benchmark: {', '.join(families)} on {', '.join(nodes)}",
            "status": "measured", "ts": max(str(r.get("ts")) for r in members), "day_only": False, "track": None,
            "node": nodes[0] if len(nodes) == 1 else None,
            "lever": "batch size " + ", ".join(f"{b:g}" for b in batches) if len(batches) > 1 else None,
            "verdict": "Measured, no verdict: " + reason if low and reason else "Measured",
            "detail": _text(members[0]["conditions"].get("mode")), "source": source,
            "numbers": [_num("Rows recorded", len(members), "rows", source),
                        _num("Rows marked low confidence", low, "rows", source)],
        })
    return cards


def timeline(rows: list[dict[str, Any]], state_dir: str | Path | None = None, refine_dir: str | Path = REFINE_DIR) -> dict[str, Any]:
    experiments, origin = experiment_cards(state_dir)
    cards = experiments + pass_cards(refine_dir) + session_cards(rows)
    dated = sorted((c for c in cards if c["ts"]), key=lambda c: (str(c["ts"])[:10], str(c["ts"])), reverse=True)
    undated = [c for c in cards if not c["ts"]]  # seeded experiments: queued, with no time recorded; in queue order
    counts = {s: sum(c["status"] == s for c in cards) for s in STATUSES}
    exp = [c for c in cards if c["kind"] == "experiment"]
    run = sum(c["status"] in ("won", "lost", "inconclusive") for c in exp)
    return {
        "cards": dated + undated, "counts": {k: v for k, v in counts.items() if v},
        "experiments": {"total": len(exp), "queued": sum(c["status"] == "queued" for c in exp),
                        "running": sum(c["status"] == "running" for c in exp), "run": run,
                        "dropped": sum(c["status"] == "dropped" for c in exp), "source": origin},
        "passes": sum(c["kind"] in ("pass", "merge") for c in cards), "sessions": sum(c["kind"] == "bench" for c in cards),
        "summary": _summary(exp, run, origin),
    }


def _summary(exp: list[dict[str, Any]], run: int, origin: dict[str, Any]) -> str:
    if not exp:
        return "No experiment is recorded on this box."
    queued = sum(c["status"] == "queued" for c in exp)
    running = sum(c["status"] == "running" for c in exp)
    where = "in the lab ledger" if origin["kind"] == "ledger" else "seeded in the lab's code; this box has no lab ledger"
    return (f"{len(exp)} experiment{'s' if len(exp) != 1 else ''} {where}: {queued} queued, {running} running, "
            f"{run} with a verdict.")
