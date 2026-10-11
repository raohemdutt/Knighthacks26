"""The lab's benchmark and energy rows as the Learning lab's "Speed and energy" tab reads them.

    rows, problems = read_rows()          # lab/refine/metrics.jsonl, each row with its line number
    bench(rows, problems)                 # what GET /api/learning/lab/bench serves

Pure functions over the rows of the lab's metrics file: nothing here measures, calls a pod or writes a file.
Every value keeps the row it came from (`lab/refine/metrics.jsonl:<line>`), its time, its sampler, its
low-confidence flag and the row's own reason. A metric with no row is absent; nothing is filled in.

Why the file is parsed here and not through lab/machine/metrics.py: that reader opens the file for writing to
hold an exclusive lock (and creates it when it is missing), and refuses the whole file when one line is bad.
A screen must not take the lab's write lock or go blank over one line, so this reads the text, skips a line it
cannot use and reports it under `skipped`. The row shape is the one that module documents (FIELDS).

New rows show up without a code change:

- `series` groups every row generically, by metric, node, model, build and session, and gives a group an
  x-axis when its rows differ in batch size;
- a `sweep` is any metric family whose rows carry a batch size at two or more levels (today
  `teacher_batching.*` on pod-a; R20's pod-b matrix and a kernel or quantisation arm appear as more sweeps);
- `power` takes, per node, the newest `*.idle_watts`, `*.watts_avg` and `*.watts_peak` rows;
- `not_charted` lists every metric and node the screen draws nothing for, so no row is dropped silently.

The lab moved from the R9700s (gfx1201) to the Strix Halo pods (gfx1151) on Sat 2026-10-10 at 4:25 PM ET, and it also
records reference runs on machines that are not its own (R57: an NVIDIA A100 on Colab). Every row therefore carries
the chip it was measured on, `chip: {key, name, where, current, role, reason}`, worked out from the row by `chip_of`
(the rules, in order, are in its docstring). `other_chip` is kept beside it: true for anything that is not the
current lab. A row from another chip is never put on a Strix Halo curve: it forms sweeps of its own (one per chip),
is left out of `power`'s bars (listed under `power.other_chip` instead) and says so in `not_charted`. A current
sweep lists under `comparisons` the other chips' sweeps that recorded the same metrics at the same batch sizes.

A number is a small dict in the lab view's shape, `{label, value, unit, source, low_confidence, reason}`, plus
`kind`: "measured" (a row's own value) or "derived" (worked out here from rows, with the formula in `formula`).
On a present value `reason` is why it is low confidence, in the row's own words.
"""

import json
import re
from pathlib import Path
from typing import Any

from contexts.learning.application.lab_view import by_file

METRICS_FILE = Path(__file__).resolve().parents[3] / "lab" / "refine" / "metrics.jsonl"
REL = "lab/refine/metrics.jsonl"
NODES = ("jetson", "pod-a", "pod-b")  # the names lab/README.md and contracts/telemetry.py use
STUDENT_NODE = "jetson"
TRAINER_STRIP = ("trainer.samples_per_s", "trainer.smoke_loss_first", "trainer.smoke_loss_last", "trainer.peak_mem_gb")
SWEEP_METRICS = ("decode_tok_s", "decode_tok_s_total", "joules_per_token", "ttft_s", "watts_avg", "watts_peak")
NO_REASON = "the row is flagged low confidence and gives no reason"
OTHER_CHIP = "another chip, not the current lab"
LAB_MOVED_AT = "2026-10-10T20:25:00Z"
MOVED_WORDS = "the lab moved to the Strix Halo pods on Sat 2026-10-10 at 4:25 PM ET"
CURRENT_CHIP = {"key": "strix-halo", "name": "Radeon 8060S (Strix Halo)", "marks": ("gfx1151", "8060s", "strix"), "role": "current"}
EARLIER_CHIP = {"key": "r9700", "name": "Radeon R9700", "marks": ("gfx1201", "r9700"), "role": "earlier"}

try:  # the lab's own rule for which rows describe the hardware it no longer uses
    from lab.slide_numbers import _old_chip as _lab_old_chip
except ImportError:  # the gateway run where lab/ or its imports are missing: the same rule, restated
    def _lab_old_chip(rec: dict) -> bool:
        if (rec.get("conditions") or {}).get("other_chip") is True:
            return True
        gpu = str((rec.get("conditions") or {}).get("gpu", ""))
        if "gfx1151" in gpu or "Strix" in gpu:
            return False
        if "gfx1201" in gpu or "R9700" in gpu:
            return True
        return bool(rec.get("ts")) and str(rec["ts"]) < LAB_MOVED_AT


def _where(row: dict[str, Any]) -> str | None:
    """Where the row was measured, as far as it says: its hub ("colab" -> "Colab", "sjc" -> "SJC"), else its node."""
    hub = (row.get("conditions") or {}).get("hub")
    if isinstance(hub, str) and hub.strip():
        hub = hub.strip()
        return hub.upper() if len(hub) <= 3 else hub[0].upper() + hub[1:]
    node = row.get("node")
    return node if isinstance(node, str) and node.strip() else None


def chip_name(text: str) -> str:
    """A short name out of a GPU description: "NVIDIA A100-SXM4-40GB, 400 W power limit" -> "NVIDIA A100".
    The words up to the first comma or bracket, cut after the model number (the first word holding a digit), which
    loses its own suffix after a dash."""
    head = re.split(r"[,;(\[]", text.strip(), maxsplit=1)[0].split() or text.split()
    words = []
    for word in head:
        if any(ch.isdigit() for ch in word):
            words.append(word.split("-")[0] or word)
            break
        words.append(word)
    return " ".join(words)[:40]


def _chip(kind: dict[str, Any], row: dict[str, Any], reason: str) -> dict[str, Any]:
    return {"key": kind["key"], "name": kind["name"], "where": _where(row), "current": kind["role"] == "current",
            "role": kind["role"], "reason": reason}


def chip_of(row: dict[str, Any]) -> dict[str, Any]:
    """The chip a row was measured on: `{key, name, where, current, role, reason}`. `role` is "current" (the lab's
    Strix Halo pods, or the Jetson, which is the box and not a lab card), "earlier" (the R9700s the lab had before),
    "reference" (any other named chip) or "unnamed". The first rule that fits decides:

    1. a Jetson row is the box;
    2. the row names its chip (`conditions.gpu`, else `conditions.chip`): the Strix Halo (gfx1151, 8060S, Strix),
       the R9700 (gfx1201, R9700), or any other chip, which is named from that text and is never the current lab;
    3. the row names no chip but marks itself `other_chip: true`: another chip, not named;
    4. the row says nothing: lab/slide_numbers.py's `_old_chip` decides by the time it was measured (before the move
       it was the R9700s, after it the Strix Halo pods), and the reason says that this is by date only."""
    c = row.get("conditions") or {}
    if row.get("node") == STUDENT_NODE:
        return {"key": "jetson", "name": "Jetson", "where": "the box", "current": True, "role": "current",
                "reason": "a Jetson row: the box itself, not one of the lab's cards"}
    field = next((f for f in ("gpu", "chip") if _text(c.get(f))), None)
    if field:
        said = c[field].strip()
        low = said.lower()
        for kind in (CURRENT_CHIP, EARLIER_CHIP):
            if any(mark in low for mark in kind["marks"]):
                return _chip(kind, row, f"conditions.{field} says \"{said}\"")
        name = chip_name(said)
        return _chip({"key": re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "other", "name": name, "role": "reference"},
                     row, f"conditions.{field} says \"{said}\"")
    if c.get("other_chip") is True:
        return _chip({"key": "unnamed", "name": "Another chip, not named", "role": "unnamed"}, row,
                     "the row marks itself other_chip and names no chip")
    if _lab_old_chip(row):
        return _chip(EARLIER_CHIP, row, f"the row names no chip and was measured before {MOVED_WORDS}")
    if row.get("ts"):
        return _chip(CURRENT_CHIP, row, f"the row names no chip and was measured after {MOVED_WORDS}")
    return _chip(CURRENT_CHIP, row, "the row names no chip and has no time, so it is counted with the current lab")


def other_chip(row: dict[str, Any]) -> bool:
    """True for a row that was not measured on the current lab's chip (the R9700s before the move, a reference run
    on another machine). The Jetson is not one of the lab's cards, so its rows never are."""
    return not chip_of(row)["current"]


def read_rows(path: str | Path | None = METRICS_FILE) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The file's rows, each with `_line` (counted from 1, blank lines counted), and the lines that are not rows."""
    file = Path(path) if path else None
    if file is None or not file.is_file():
        return [], []
    rows, skipped = [], []
    for n, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            skipped.append({"line": n, "problem": "not JSON"})
            continue
        value = row.get("value") if isinstance(row, dict) else None
        if not isinstance(row, dict) or not isinstance(row.get("metric"), str) or isinstance(value, bool) \
                or not isinstance(value, int | float):
            skipped.append({"line": n, "problem": "no metric name or no numeric value"})
            continue
        if not isinstance(row.get("conditions"), dict):
            row["conditions"] = {}
        row["_line"] = n
        rows.append(row)
    return rows, skipped


# --- one row as a number ---

def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _number(value: Any) -> float | None:
    return value if isinstance(value, int | float) and not isinstance(value, bool) else None


def batch_of(row: dict[str, Any]) -> float | None:
    c = row["conditions"]
    return _number(c.get("batch_size")) if _number(c.get("batch_size")) is not None else _number(c.get("batch"))


def model_of(c: dict[str, Any]) -> str | None:
    name = _text(c.get("model")) or _text(c.get("checkpoint"))
    return name.rstrip("/").split("/")[-1] if name else None


def build_of(c: dict[str, Any]) -> str | None:
    engine = _text(c.get("engine"))
    commit = _text(c.get("engine_commit"))
    if engine and commit and re.fullmatch(r"[0-9a-f]{7,40}", commit):
        engine = f"{engine}@{commit[:7]}"
    quant = _text(c.get("quantization"))
    return " ".join(x for x in (engine, quant) if x) or None


def sampler_of(c: dict[str, Any]) -> dict[str, Any] | None:
    """How the reading was taken, as far as the row says: the program, its period, how many samples."""
    out = {"sampler": _text(c.get("sampler")),
           "tool": _text(c.get("reader")) or _text(c.get("sensor")) or _text(c.get("power_field")) or _text(c.get("rail")) or _text(c.get("power_source")),
           "period_ms": _number(c.get("period_ms")), "samples": _number(c.get("samples")),
           "error": _text(c.get("error"))}
    return out if any(v is not None for v in out.values()) else None


def state_of(row: dict[str, Any]) -> str | None:
    """What the node was doing, in the row's own words when it gives them."""
    c = row["conditions"]
    for key in ("state", "load", "server", "while"):
        if _text(c.get(key)):
            return c[key]
    b = batch_of(row)
    return f"batch {b:g}" if b is not None else None


def point(row: dict[str, Any], label: str | None = None) -> dict[str, Any]:
    c = row["conditions"]
    low = bool(row.get("low_confidence"))
    chip = chip_of(row)
    stats = c.get("stats") if isinstance(c.get("stats"), dict) else {}
    return {
        "label": label or row["metric"], "value": row["value"], "unit": row.get("unit"),
        "source": by_file(REL, row["_line"]), "low_confidence": low,
        "reason": (_text(c.get("low_confidence_reason")) or NO_REASON) if low else None,
        "kind": "derived" if c.get("derived") is True else "measured",
        "formula": _text(c.get("formula")) if c.get("derived") is True else None,
        "metric": row["metric"], "node": row.get("node"), "ts": row.get("ts"), "line": row["_line"],
        "recorded_by": _text(row.get("source")), "sampler": sampler_of(c), "x": batch_of(row),
        "runs": _number(stats.get("n")) if _number(stats.get("n")) is not None else _runs_listed(c), "statistic": _text(c.get("statistic")),
        "spread": {k: stats[k] for k in ("min", "max", "sd") if _number(stats.get(k)) is not None} or None,
        "model": model_of(c), "build": build_of(c), "session": _text(c.get("session")), "state": state_of(row),
        "other_chip": not chip["current"], "chip": chip,
    }


def _runs_listed(c: dict[str, Any]) -> int | None:
    """How many runs a row lists one by one when it gives no `stats.n` (R57's `per_spawn`)."""
    for key in ("per_spawn", "per_run", "per_repeat"):
        if isinstance(c.get(key), list) and c[key]:
            return len(c[key])
    return None


def derived(label: str, value: float, unit: str, formula: str, *inputs: dict[str, Any], reason: str | None = None,
            digits: int = 4) -> dict[str, Any]:
    """A number worked out from rows: the formula with its values, the lines it used, and low confidence when any
    input is (or when `reason` names an assumption of its own)."""
    reasons = [p["reason"] for p in inputs if p.get("low_confidence") and p.get("reason")]
    if reason:
        reasons.insert(0, reason)
    lines = [p["source"]["ref"] for p in inputs if p.get("source")]
    return {"label": label, "value": round(value, digits), "unit": unit,
            "source": {"kind": "derived", "ref": f"{formula}; from {', '.join(dict.fromkeys(lines))}", "ids": []},
            "low_confidence": bool(reasons), "reason": "; ".join(dict.fromkeys(reasons)) or None,
            "kind": "derived", "formula": formula, "inputs": list(dict.fromkeys(lines))}


def _newest(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Latest ts; the later line wins a tie (the order lab/machine/metrics.py gives)."""
    return max(rows, key=lambda r: (str(r.get("ts") or ""), r["_line"])) if rows else None


# --- the read model ---

def series(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every row, grouped by metric, node, model, build and session; newest group first within a metric."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        c = r["conditions"]
        groups.setdefault((r["metric"], r.get("node"), model_of(c), build_of(c), _text(c.get("session"))), []).append(r)
    out = []
    for (metric, node, model, build, session), members in groups.items():
        xs = sorted({b for b in map(batch_of, members) if b is not None})
        points = sorted((point(r) for r in members), key=lambda p: (p["x"] is None, p["x"] or 0, p["ts"] or "", p["line"]))
        out.append({"metric": metric, "family": metric.split(".")[0], "node": node, "model": model, "build": build,
                    "session": session, "unit": members[0].get("unit"), "x": "batch size" if len(xs) > 1 else None,
                    "points": points, "low_confidence": sum(p["low_confidence"] for p in points)})
    out.sort(key=lambda s: (s["metric"], s["node"] or "", -max(p["line"] for p in s["points"])))
    return out


def sweeps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each run of one metric family over two or more batch sizes on one node, newest first."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        if batch_of(r) is None:
            continue
        c = r["conditions"]
        family = r["metric"].split(".")[0]
        groups.setdefault((family, r.get("node"), _text(c.get("session")) or _text(r.get("source")) or "", chip_of(r)["key"]), []).append(r)
    out = []
    for (family, node, run, _key), members in groups.items():
        batches = sorted({batch_of(r) for r in members})
        if len(batches) < 2:
            continue
        metrics: dict[str, list[dict[str, Any]]] = {}
        superseded = 0
        for name in sorted({r["metric"].split(".", 1)[1] for r in members}):
            pts = []
            for b in batches:
                same = [r for r in members if r["metric"].split(".", 1)[1] == name and batch_of(r) == b]
                if same:
                    pts.append(point(_newest(same)))
                    superseded += len(same) - 1
            metrics[name] = pts
        newest = _newest(members)
        c = newest["conditions"]
        chip = chip_of(newest)
        every = [p for pts in metrics.values() for p in pts]
        out.append({
            "id": f"{family}:{node}:{run}" + ("" if chip["current"] else f":{chip['key']}"), "other_chip": not chip["current"],
            "chip": chip, "chip_note": _text(c.get("other_chip_note")),
            "power_source": next((t for t in (_text(r["conditions"].get("power_source")) for r in members) if t), None), "gpu": _text(c.get("chip")) or _text(c.get("gpu")),
            "family": family, "node": node, "run": run, "session": _text(c.get("session")),
            "model": model_of(c), "build": build_of(c),
            "mode": _text(c.get("mode")), "prompt_set": (c.get("prompt_set") or {}).get("id") if isinstance(c.get("prompt_set"), dict) else None,
            "max_tokens": _number(c.get("max_tokens")), "ts": newest.get("ts"), "recorded_by": _text(newest.get("source")),
            "batches": batches, "metrics": metrics, "superseded": superseded,
            "other_metrics": sorted(set(metrics) - set(SWEEP_METRICS)),
            "points": len(every), "low_confidence": sum(p["low_confidence"] for p in every),
            "confidence_note": _confidence_note(every),
        })
    out.sort(key=lambda s: str(s["ts"] or ""), reverse=True)
    out.sort(key=lambda s: s["other_chip"])  # the current lab's sweeps first
    for s in out:
        s["comparisons"] = [] if s["other_chip"] else [x for x in (_comparison(s, o) for o in out if o["other_chip"]) if x]
    return out


def _comparison(mine: dict[str, Any], other: dict[str, Any]) -> dict[str, Any] | None:
    """Another chip's sweep that can be read beside this one: the metrics both recorded, at the batch sizes both ran."""
    batches = [b for b in mine["batches"] if b in other["batches"]]
    names = [n for n in SWEEP_METRICS if any(p["x"] in batches for p in mine["metrics"].get(n, []))
             and any(p["x"] in batches for p in other["metrics"].get(n, []))]
    if not batches or not names:
        return None
    return {"id": other["id"], "chip": other["chip"], "metrics": names, "batches": batches,
            "same_model": mine["model"] is not None and mine["model"] == other["model"]}


def _confidence_note(points: list[dict[str, Any]]) -> str:
    """What the rows themselves say about how far to trust the sweep: how many runs stand behind a point."""
    runs = {p["runs"] for p in points}
    sessions = {p["session"] for p in points if p["session"]}
    stat = next((p["statistic"] for p in points if p["statistic"]), None)
    if runs <= {None, 1} and not stat:
        head = "Each point is a single run."
    else:
        n = "/".join(f"{r:g}" for r in sorted(r for r in runs if r))
        head = f"Each point is the {stat}" if stat else f"Each point stands on {n} runs"
        head += f", all from one session ({next(iter(sessions))})." if len(sessions) == 1 else "."
    low = [p for p in points if p["low_confidence"]]
    if not low:
        return head
    reasons = list(dict.fromkeys(p["reason"] for p in low if p["reason"]))
    share = "Every point" if len(low) == len(points) else f"{len(low)} of {len(points)} points"
    return f"{head} {share} is marked low confidence: {reasons[0]}." if len(reasons) == 1 else \
        f"{head} {share} is marked low confidence, for {len(reasons)} different reasons (tap a point)."


def student(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The Jetson student as a reference: its watts while answering, its tokens per second, and the joules per token
    those two give. A `student.decode_tok_s` row is used when the file has one; today the decode rate is a field of
    the watts row's conditions (`decode_tok_s_median`), so it is cited as that row."""
    on_box = [r for r in rows if r.get("node") == STUDENT_NODE]
    watts_row = _newest([r for r in on_box if r["metric"] == "student.watts_avg"])
    if watts_row is None:
        return None
    c = watts_row["conditions"]
    watts = point(watts_row, "Student power while answering")
    tok_row = _newest([r for r in on_box if r["metric"] == "student.decode_tok_s"])
    if tok_row is not None:
        tok = point(tok_row, "Student decode speed")
    elif _number(c.get("decode_tok_s_median")) is not None:
        tok = {**point(watts_row, "Student decode speed"), "value": c["decode_tok_s_median"], "unit": "tok/s",
               "metric": "student.watts_avg (conditions.decode_tok_s_median)", "statistic": "median over the run's requests",
               "source": by_file(f"{REL}:{watts_row['_line']} conditions.decode_tok_s_median")}
    else:
        tok = None
    out: dict[str, Any] = {"node": STUDENT_NODE, "watts": watts, "tok_s": tok, "workload": state_of(watts_row),
                           "j_per_token": None, "request_s": None, "j_per_answer": None, "answer_tokens": None,
                           "j_per_token_request": None, "first_token_s": None}
    if tok is not None and tok["value"] > 0:
        out["j_per_token"] = derived("Student energy per token while decoding", watts["value"] / tok["value"], "J/token",
                                     f"{watts['value']:g} W ÷ {tok['value']:g} tok/s", watts, tok)
    secs = _number(c.get("request_mean_s"))
    if secs is not None:
        req = {**point(watts_row, "Student time per answer"), "value": secs, "unit": "s",
               "metric": "student.watts_avg (conditions.request_mean_s)",
               "source": by_file(f"{REL}:{watts_row['_line']} conditions.request_mean_s")}
        out["request_s"] = req
        out["j_per_answer"] = derived("Student energy per answer, whole request", watts["value"] * secs, "J",
                                      f"{watts['value']:g} W × {secs:g} s", watts, req, digits=1)
        limit = re.search(r"max_tokens (\d+)", _text(c.get("load")) or "")
        if limit:
            n = int(limit.group(1))
            why = f"assumes every answer ran to its {n}-token limit; the row records the limit, not the tokens written"
            out["answer_tokens"] = {**point(watts_row, "Student answer length limit"), "value": n, "unit": "tokens",
                                    "metric": "student.watts_avg (conditions.load: max_tokens)",
                                    "source": by_file(f"{REL}:{watts_row['_line']} conditions.load")}
            out["j_per_token_request"] = derived("Student energy per token, whole request", watts["value"] * secs / n, "J/token",
                                                 f"{watts['value']:g} W × {secs:g} s ÷ {n} tokens", watts, req, reason=why)
            if tok is not None and tok["value"] > 0 and secs > n / tok["value"]:
                out["first_token_s"] = derived("Student time before the first token", secs - n / tok["value"], "s",
                                               f"{secs:g} s − {n} tokens ÷ {tok['value']:g} tok/s", req, tok, reason=why, digits=2)
    return out


def power(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per node, the newest idle, serving and peak watts the file holds: what it costs to be on."""
    nodes = []
    names = list(dict.fromkeys([*NODES, *sorted({str(r.get("node")) for r in rows if r.get("node")})]))
    watts = (".idle_watts", ".watts_avg", ".watts_peak")
    old = [point(r) for r in rows if other_chip(r) and r["metric"].endswith(watts)]
    for node in names:
        mine = [r for r in rows if r.get("node") == node and not other_chip(r)]
        idle = [r for r in mine if r["metric"].endswith(".idle_watts")]
        serving = [r for r in mine if r["metric"].endswith(".watts_avg")]
        peak = [r for r in mine if r["metric"].endswith(".watts_peak")]
        if not (idle or serving):
            continue
        pick = lambda found, label: point(_newest(found), label) if found else None  # noqa: E731
        nodes.append({"node": node, "idle": pick(idle, f"{node} idle power"), "serving": pick(serving, f"{node} power while serving"),
                      "peak": pick(peak, f"{node} peak power"), "idle_rows": len(idle), "serving_rows": len(serving)})
    return {"nodes": nodes, "absent": [n for n in NODES if n not in {x["node"] for x in nodes}], "other_chip": old,
            "other_chips": _by_chip(old)}


def _by_chip(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Points grouped by the chip they were measured on, in the order the chips first appear, each with its lines.
    A reference chip is told apart by where it ran as well: the same card somewhere else is another group."""
    groups: dict[tuple, dict[str, Any]] = {}
    for p in points:
        chip = p["chip"]
        where = chip["where"] if chip["role"] == "reference" else None
        g = groups.setdefault((chip["key"], where), {"key": chip["key"], "name": chip["name"], "role": chip["role"],
                                                     "current": chip["current"], "where": where, "rows": 0, "lines": []})
        g["rows"] += 1
        g["lines"].append(p["line"])
    return list(groups.values())



def bench(rows: list[dict[str, Any]], skipped: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    all_series = series(rows)
    all_sweeps = sweeps(rows)
    ref = student(rows)
    watts = power(rows)
    drawn: set[int] = set()
    for s in all_sweeps:
        drawn.update(p["line"] for name in SWEEP_METRICS for p in s["metrics"].get(name, []))
    for n in watts["nodes"]:
        drawn.update(p["line"] for p in (n["idle"], n["serving"], n["peak"]) if p)
    if ref:
        drawn.add(ref["watts"]["line"])
        if ref["tok_s"]:
            drawn.add(ref["tok_s"]["line"])
    drawn.update(r["_line"] for r in rows if r["metric"] in TRAINER_STRIP)  # the Training tab's measured strip

    groups: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["metric"], r.get("node")), []).append(r)
    not_charted = []
    for (metric, node), members in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        left = [r for r in members if r["_line"] not in drawn]
        if left:
            newest = _newest(left)
            not_charted.append({"metric": metric, "node": node, "rows": len(left), "of": len(members), "unit": newest.get("unit"),
                                "other_chip": sum(other_chip(r) for r in left),
                                "other_chips": _by_chip([point(r) for r in left if other_chip(r)]),
                                "latest": point(newest), "lines": [r["_line"] for r in left]})
    families: dict[str, dict[str, Any]] = {}
    for r in rows:
        f = families.setdefault(r["metric"].split(".")[0], {"family": r["metric"].split(".")[0], "metrics": set(), "nodes": set(),
                                                             "rows": 0, "charted": 0, "low_confidence": 0})
        f["metrics"].add(r["metric"])
        f["nodes"].add(str(r.get("node")))
        f["rows"] += 1
        f["charted"] += r["_line"] in drawn
        f["low_confidence"] += bool(r.get("low_confidence"))
    return {
        "file": REL, "rows": len(rows), "skipped": skipped or [],
        "low_confidence_rows": sum(bool(r.get("low_confidence")) for r in rows),
        "other_chip_rows": sum(other_chip(r) for r in rows), "other_chip_label": OTHER_CHIP, "lab_moved_at": LAB_MOVED_AT,
        "chips": _by_chip([p for p in (point(r) for r in rows) if p["node"] != STUDENT_NODE]),
        "newest": max((str(r.get("ts")) for r in rows if r.get("ts")), default=None),
        "families": [{**f, "metrics": sorted(f["metrics"]), "nodes": sorted(f["nodes"])} for f in sorted(families.values(), key=lambda f: f["family"])],
        "sweeps": all_sweeps, "student": ref, "power": watts, "series": all_series,
        "charted_rows": len(drawn), "not_charted": not_charted,
        "empty_reason": None if rows else "the lab's metrics file has no rows on this box",
    }
