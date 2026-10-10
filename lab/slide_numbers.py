"""The architecture slide's numbers, from where they are measured (D7).

    python lab/slide_numbers.py --gateway http://<jetson-lan-ip>:8000 \
        --eval-before out/base.json --eval-after out/candidate.json \
        --stats http://<jetson-lan-ip>:8000/api/stats --telemetry data/telemetry/*.jsonl

Prints a markdown table that pastes into the Architecture Slide (D3) and the
Devpost (D4). Every number carries its source and when it was read; a source
that isn't there yet prints as `[TBD: ...]` instead of a made-up number.

Sources:
- gateway: /health (model version), /memory/stats (hours recorded, episodes, captions),
  /api/impact (Waste Watch)
- eval JSON from `python -m contexts.agent.eval --json` (before/after on the golden questions)
- stats endpoint (R5) and telemetry JSONL (T1 to T4): FPS, tok/s, watts, joules per answer,
  frames filtered, training minutes. Field names follow the Telemetry context page; adjust
  KEYS below if the endpoint names them differently.
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
    "Teacher tok/s (R9700)": ["teacher_tok_s"],
    "Watts, Jetson": ["jetson_watts", "watts_jetson"],
    "Watts, R9700": ["gpu_watts", "watts_gpu"],
    "Joules per answer": ["joules_per_answer"],
    "Frames filtered at the edge": ["frames_filtered_pct", "motion_filtered_pct"],
    "Training minutes per run": ["train_minutes", "training_minutes"],
}


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")


class Row:
    def __init__(self, name: str, value: str | None, source: str, when: str | None = None, missing: str | None = None):
        self.name, self.value, self.source, self.when, self.missing = name, value, source, when or now(), missing

    def cells(self) -> list[str]:
        value = self.value if self.value is not None else f"[TBD: {self.missing or self.source + ' not available'}]"
        return [self.name, value, self.source, self.when if self.value is not None else "-"]


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
    rows = [Row("Atlas model", health.get("model") if health else None, f"{base}/health")]
    if stats:
        rows += [Row("Hours of memory", f"{stats['hours_recorded']:.1f} h", f"{base}/memory/stats"),
                 Row("Episodes / captions", f"{stats['episodes']} / {stats['captions']}", f"{base}/memory/stats")]
    else:
        rows.append(Row("Hours of memory", None, f"{base}/memory/stats"))
    total = (impact or {}).get("total")
    if total:
        rows.append(Row("Waste Watch: items seen, diversion rate",
                        f"{total['items_seen']} items, {100 * total['diversion_rate']:.0f}% recycled", f"{base}/api/impact"))
        ba = impact.get("before_after_nudges")
        if ba:
            rows.append(Row("Diversion before vs after nudges",
                            f"{100 * ba['before']['diversion_rate']:.0f}% -> {100 * ba['after']['diversion_rate']:.0f}%",
                            f"{base}/api/impact"))
    else:
        rows.append(Row("Diversion rate", None, f"{base}/api/impact"))
    return rows


def eval_rows(before: str | None, after: str | None) -> list[Row]:
    def load(path):
        if not path or not Path(path).exists():
            return None
        data = json.loads(Path(path).read_text())
        when = datetime.fromtimestamp(Path(path).stat().st_mtime, UTC).strftime("%Y-%m-%d %H:%M UTC")
        return data, when

    b, a = load(before), load(after)
    if not (b and a):
        return [Row("Golden questions: before -> after", None, "contexts.agent.eval --json",
                    missing="run the eval on the base and the candidate model")]
    (bd, _), (ad, when) = b, a
    return [Row("Golden questions: before -> after", f"{bd['score']:.2f} ({bd['model']}) -> {ad['score']:.2f} ({ad['model']})",
                f"{before}, {after}", when)]


def measured_rows(client: httpx.Client, stats_url: str | None, telemetry: list[str]) -> list[Row]:
    found: dict[str, tuple[str, str, str]] = {}  # name -> (value, source, when)
    if stats_url and (stats := get_json(client, stats_url)):
        for name, keys in KEYS.items():
            for k in keys:
                if stats.get(k) is not None:
                    found[name] = (str(stats[k]), stats_url, now())
                    break
    for pattern in telemetry:
        for path in sorted(glob.glob(pattern)):
            for line in Path(path).read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for name, keys in KEYS.items():
                    for k in keys:
                        if rec.get(k) is not None:  # later records win: the newest measurement
                            found[name] = (str(rec[k]), path, str(rec.get("ts", now())))
                            break
    return [Row(name, *found[name]) if name in found else
            Row(name, None, "stats endpoint / telemetry", missing="R5 stats or T1-T4 telemetry not recorded yet")
            for name in KEYS]


def table(rows: list[Row]) -> str:
    out = ["| Number | Value | Source | Read at |", "| --- | --- | --- | --- |"]
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
