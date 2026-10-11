"""The architecture slide's numbers, from where they are measured (D7).

    python lab/slide_numbers.py --gateway http://<jetson-lan-ip>:8000 \
        --eval-before out/base.json --eval-after out/candidate.json \
        --stats http://<jetson-lan-ip>:8000/runtime/stats --telemetry data/telemetry/*.jsonl

Prints a markdown table that pastes into the Architecture Slide (D3) and the
Devpost (D4). Every number carries the row it came from (`metrics.jsonl:<line>`,
`ledger seq <n>`, `<file>:<line>` or the endpoint), when that row was MEASURED (not
when this script ran; an endpoint that gives no time prints "unknown"), and the row's
low-confidence flag, so a flagged number never reaches the slide looking like a
measured fact (telemetry rubric TT7, Q15). A source that isn't there yet prints as
`[TBD: ...]` instead of a made-up number.

Sources:
- gateway: /health (model version), /memory/stats (hours recorded, episodes, captions),
  /api/impact (Waste Watch)
- eval JSON from `python -m contexts.agent.eval --json` (before/after on the golden questions)
- stats endpoint (R5) and telemetry JSONL (T1 to T4): FPS, tok/s, watts, joules per answer,
  frames filtered, training minutes. Field names follow the Telemetry context page; adjust
  KEYS below if the endpoint names them differently. An endpoint value may be a plain number or
  {value, ts, source, low_confidence}.
- the lab's metrics file (lab/refine/metrics.jsonl) and ledger (lab/state/ledger.jsonl): the
  measured lab numbers (teacher tok/s, watts, joules, trainer), newest measurement wins.
"""

import argparse
import glob
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

# number on the slide -> keys to look for in the stats endpoint / telemetry records (first match wins)
KEYS = {
    "Detector FPS (Jetson)": ["detector_fps", "fps"],
    "Student tok/s (Jetson)": ["student_tok_s", "atlas_tok_s", "tok_s"],
    "Teacher tok/s (Strix Halo)": ["teacher_tok_s"],
    "Watts, Jetson": ["jetson_watts", "watts_jetson"],
    "Watts, Strix Halo pod": ["gpu_watts", "watts_gpu"],
    "Joules per answer": ["joules_per_answer"],
    "Frames filtered at the edge": ["frames_filtered_pct", "motion_filtered_pct"],
    "Training minutes per run": ["train_minutes", "training_minutes"],
    "Teacher joules per token": ["teacher_joules_per_token"],
    "Trainer samples/s (Strix Halo)": ["trainer_samples_per_s"],
}
# the same numbers as the lab's metrics file and ledger name them
LAB_METRICS = {
    "Teacher tok/s (Strix Halo)": "teacher.decode_tok_s",
    "Watts, Strix Halo pod": "teacher.watts_avg",
    "Teacher joules per token": "teacher.joules_per_token",
    "Trainer samples/s (Strix Halo)": "trainer.samples_per_s",
}
METRICS_FILE = Path(__file__).resolve().parent / "refine" / "metrics.jsonl"
UNKNOWN = "unknown"


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")


class Row:
    def __init__(self, name: str, value: str | None, source: str, when: str | None = None, missing: str | None = None,
                 low: bool = False):
        self.name, self.value, self.source, self.missing, self.low = name, value, source, missing, low
        self.when = when or UNKNOWN  # when the number was measured; never the time this script ran

    def cells(self) -> list[str]:
        if self.value is None:
            return [self.name, f"[TBD: {self.missing or self.source + ' not available'}]", self.source, "-", "-"]
        return [self.name, self.value, self.source, self.when, "LOW" if self.low else "no"]


def get_json(client: httpx.Client, url: str) -> dict | None:
    try:
        r = client.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def gateway_rows(client: httpx.Client, base: str | None) -> list[Row]:
    if not base:
        return [Row(n, None, "gateway", missing="pass --gateway") for n in ("Atlas model", "Hours of memory", "Diversion rate")]
    base = base.rstrip("/")
    health = get_json(client, f"{base}/health")
    stats = get_json(client, f"{base}/memory/stats")
    impact = get_json(client, f"{base}/api/impact")
    read = f"read {now()}"  # these endpoints describe the box as it is now
    rows = [Row("Atlas model", health.get("model") if health else None, f"{base}/health", read)]
    if stats:
        rows += [Row("Hours of memory", f"{stats['hours_recorded']:.1f} h", f"{base}/memory/stats", read),
                 Row("Episodes / captions", f"{stats['episodes']} / {stats['captions']}", f"{base}/memory/stats", read)]
    else:
        rows.append(Row("Hours of memory", None, f"{base}/memory/stats"))
    total = (impact or {}).get("total")
    if total:
        rows.append(Row("Waste Watch: items seen, diversion rate",
                        f"{total['items_seen']} items, {100 * total['diversion_rate']:.0f}% recycled", f"{base}/api/impact", read))
        ba = impact.get("before_after_nudges")
        if ba:
            rows.append(Row("Diversion before vs after nudges",
                            f"{100 * ba['before']['diversion_rate']:.0f}% -> {100 * ba['after']['diversion_rate']:.0f}%",
                            f"{base}/api/impact", read))
    else:
        rows.append(Row("Diversion rate", None, f"{base}/api/impact"))
    return rows


def eval_rows(before: str | None, after: str | None) -> list[Row]:
    def load(path):
        if not path or not Path(path).exists():
            return None
        data = json.loads(Path(path).read_text())
        when = datetime.fromtimestamp(Path(path).stat().st_mtime, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")  # when the eval ran
        return data, when

    b, a = load(before), load(after)
    if not (b and a):
        return [Row("Golden questions: before -> after", None, "contexts.agent.eval --json",
                    missing="run the eval on the base and the candidate model")]
    (bd, _), (ad, when) = b, a
    return [Row("Golden questions: before -> after", f"{bd['score']:.2f} ({bd['model']}) -> {ad['score']:.2f} ({ad['model']})",
                f"{before}, {after}", when)]


def _ts(text: str | None) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else None


# The lab moved from the R9700s (gfx1201, DUB) to the Strix Halo pods (gfx1151, SJC) on Sat 2026-10-10 at 4:25 PM ET.
# A row tagged with the old chip, or an untagged row measured before the move, describes hardware we no longer use.
LAB_MOVED_AT = "2026-10-10T20:25:00Z"


def _old_chip(rec: dict) -> bool:
    if (rec.get("conditions") or {}).get("other_chip") is True:  # says so itself: a reference on another chip (R57's A100)
        return True
    gpu = str((rec.get("conditions") or {}).get("gpu", ""))
    if "gfx1151" in gpu or "Strix" in gpu:
        return False
    if "gfx1201" in gpu or "R9700" in gpu:
        return True
    moment = _ts(rec.get("ts"))
    return moment is not None and moment < _ts(LAB_MOVED_AT)


def lab_metrics(path: str | Path | None = METRICS_FILE) -> dict[str, tuple]:
    """Newest row per metric in the lab's metrics file: metric -> (value, unit, "metrics.jsonl:<line>", ts, low).

    A row from the old chip (R9700) only fills a gap: any current-chip row beats it, and it prints as low confidence
    with "(R9700, not the current lab)" after its source, so an old number never reaches the slide as a Strix Halo one."""
    old_rows: set[str] = set()
    out: dict[str, tuple] = {}
    if not path or not Path(path).exists():
        return out
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        name, moment = rec.get("metric"), _ts(rec.get("ts"))
        if name is None or rec.get("value") is None or moment is None:
            continue
        if (rec.get("conditions") or {}).get("other_chip") is True:
            continue  # a reference measured on another chip (R57's A100 on Colab): never a number for the lab's slide
        old = _old_chip(rec)
        if name in out and old and name not in old_rows:
            continue  # a current-chip row is already here
        if name not in out or (name in old_rows and not old) or moment >= _ts(out[name][3]):  # newest; a later line wins a tie
            source = f"{Path(path).name}:{number}" + (" (R9700, not the current lab)" if old else "")
            out[name] = (rec["value"], rec.get("unit"), source, rec["ts"], old or bool(rec.get("low_confidence")))
            (old_rows.add if old else old_rows.discard)(name)
    return out


def ledger_metric(name: str, state_dir: str | None = None) -> tuple | None:
    """The ledger's newest measurement of a metric, as (value, unit, "ledger seq <n>", ts, low), or None."""
    try:
        from lab.machine.ledger import latest_metric
    except ImportError:
        return None
    row = latest_metric(name, state_dir=state_dir)
    if row is None:
        return None
    return (row["value"], row.get("unit"), f"ledger seq {row['seq']}", row["ts"], bool(row.get("low_confidence")))


def measured_rows(client: httpx.Client, stats_url: str | None, telemetry: list[str],
                  metrics_file: str | Path | None = METRICS_FILE, ledger_dir: str | None = None) -> list[Row]:
    found: dict[str, Row] = {}

    def offer(row: Row) -> None:
        """Keep the newest measurement of each number; one with no known time only fills a gap."""
        have = found.get(row.name)
        new, old = _ts(row.when), _ts(have.when) if have else None
        if have is None or (new is not None and (old is None or new >= old)):
            found[row.name] = row

    if stats_url and (stats := get_json(client, stats_url)):
        for name, keys in KEYS.items():
            for k in keys:
                v = stats.get(k)
                if isinstance(v, dict):  # {value, ts, source, low_confidence}; a null value comes with only its reason
                    if v.get("value") is not None:
                        offer(Row(name, str(v["value"]), v.get("source") or stats_url, v.get("ts"), low=bool(v.get("low_confidence"))))
                    break
                if v is not None:
                    offer(Row(name, str(v), stats_url))  # a bare number: when it was measured is unknown
                    break
    for pattern in telemetry:
        for path in sorted(glob.glob(pattern)):
            for number, line in enumerate(Path(path).read_text().splitlines(), 1):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for name, keys in KEYS.items():
                    for k in keys:
                        if rec.get(k) is not None:
                            offer(Row(name, str(rec[k]), f"{path}:{number}", rec.get("ts"), low=bool(rec.get("low_confidence"))))
                            break
    lab = lab_metrics(metrics_file)
    for name, metric in LAB_METRICS.items():
        for got in (lab.get(metric), ledger_metric(metric, ledger_dir)):
            if got:
                value, unit, source, ts, low = got
                offer(Row(name, f"{value} {unit}" if unit else str(value), source, ts, low=low))
    return [found.get(name) or Row(name, None, "stats endpoint / telemetry / lab metrics",
                                   missing="R5 stats, T1-T4 telemetry or a lab metrics row not recorded yet")
            for name in KEYS]


def table(rows: list[Row]) -> str:
    out = ["| Number | Value | Source | Measured at | Low confidence |", "| --- | --- | --- | --- | --- |"]
    out += ["| " + " | ".join(c.replace("|", "/") for c in r.cells()) + " |" for r in rows]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gateway", help="the box's gateway, e.g. http://192.168.1.20:8000")
    p.add_argument("--eval-before", help="eval JSON for the current/base model")
    p.add_argument("--eval-after", help="eval JSON for the candidate model")
    p.add_argument("--stats", help="stats endpoint URL (R5)")
    p.add_argument("--telemetry", action="append", default=[], help="telemetry JSONL glob (T1-T4); repeatable")
    args = p.parse_args(argv)
    with httpx.Client() as client:
        rows = gateway_rows(client, args.gateway) + eval_rows(args.eval_before, args.eval_after)
        rows += measured_rows(client, args.stats, args.telemetry)
    print(table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
