"""File a chip sheet measured on a pod and write its ledger rows: ingest | self-check.

    python -m lab chipsheet ingest --sheet FILE [--data-dir lab/refine/research/kernels/data] [--dry-run]
    python -m lab chipsheet self-check [--out DIR]

ingest reads a sheet that lab/kernels/chipsheet/chipsheet.py wrote on a pod (pulled back with
`python -m lab.transfer pull-file --account D chipsheet/<name>.json <local>`), checks it against the format
(chipsheet_format.validate), writes <data-dir>/chipsheet-<gfx>-<date>.json and the Markdown summary beside it, and
appends one ledger row per section on the sheet's node (pod-b for pod D): verb "chipsheet", outcome ok, the section's
measured values as metrics named chip.<entry id> with their units, and in conditions the gfx, mode, date, section,
the file, and for every metric its spread (n, min, max, rel_spread) and loaded watts. refs.run_id is the file stem,
so a sheet already in the ledger is not written twice (outcome already_recorded). A quick sheet's rows are
low_confidence. A self-check sheet is refused: it holds invented timings. --dry-run writes nothing and answers with
the rows it would write.

self-check runs chipsheet.py self-check (no GPU, recorded tool output) and answers with its result.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lab.kernels.chipsheet import chipsheet_format as fmt
from lab.machine import ledger
from lab.machine.record import record

VERB = "chipsheet"
REPO = Path(__file__).resolve().parents[3]
DATA_DIR = REPO / "lab" / "refine" / "research" / "kernels" / "data"


def metric_name(entry_id: str) -> str:
    """The ledger metric of a sheet entry: chip.<id>."""
    return f"chip.{entry_id}"


def rows_for(sheet: Mapping[str, Any], source: str) -> list[dict[str, Any]]:
    """One ledger row (as record() keywords) per section with at least one measured value, in section order."""
    stem = fmt.file_stem(sheet)
    rows = []
    for section in fmt.sections_of(sheet["measurements"]):
        items = [m for m in sheet["measurements"] if m["section"] == section and m["status"] == "ok"]
        if not items:
            continue
        metrics = {metric_name(m["id"]): m["value"] for m in items}
        units = {metric_name(m["id"]): m["unit"] for m in items}
        spread = {metric_name(m["id"]): {k: m["spread"].get(k) for k in ("n", "min", "max", "rel_spread")}
                  for m in items}
        watts = {metric_name(m["id"]): (m.get("watts") or {}).get("loaded_mean_w") for m in items
                 if (m.get("watts") or {}).get("loaded_mean_w") is not None}
        skipped = [m["id"] for m in sheet["measurements"] if m["section"] == section and m["status"] != "ok"]
        rows.append({
            "metrics": metrics, "units": units, "node": sheet["node"], "refs": {"run_id": stem},
            "low_confidence": sheet["mode"] != "full", "source": source,
            "conditions": {"gfx": sheet["gfx"], "mode": sheet["mode"], "date": sheet["date"], "section": section,
                           "file": source, "spread": spread, "loaded_watts": watts, "not_measured": skipped,
                           "versions": sheet["versions"]},
        })
    return rows


def ingest(sheet_path: str | Path, data_dir: str | Path = DATA_DIR, *, dry_run: bool = False,
           state_dir: str | Path | None = None) -> dict[str, Any]:
    try:
        sheet = fmt.read(sheet_path)
    except (OSError, ValueError) as exc:
        return {"ok": False, "outcome": "invalid", "error": str(exc)}
    if sheet["mode"] == "self-check":
        return {"ok": False, "outcome": "refused",
                "error": "a self-check sheet holds invented timings; only a pod run is filed"}
    stem = fmt.file_stem(sheet)
    folder = Path(data_dir)
    try:
        source = (folder / f"{stem}.json").resolve().relative_to(REPO).as_posix()
    except ValueError:
        source = (folder / f"{stem}.json").as_posix()
    rows = rows_for(sheet, source)
    if dry_run:
        return {"ok": True, "outcome": "dry_run", "files": [f"{stem}.json", f"{stem}.md"], "rows": rows}
    already = ledger.read(state_dir=state_dir, verb=VERB, refs={"run_id": stem})
    json_path, md_path = fmt.write(sheet, folder)
    if already:
        return {"ok": True, "outcome": "already_recorded", "json": str(json_path), "markdown": str(md_path),
                "seqs": [row["seq"] for row in already]}
    seqs = []
    for row in rows:
        try:
            seqs.append(record(VERB, "ok", row["metrics"], row["refs"], node=row["node"], units=row["units"],
                               conditions=row["conditions"], low_confidence=row["low_confidence"],
                               source=row["source"], state_dir=state_dir))
        except (ledger.LockTimeout, ledger.RowError) as exc:
            return {"ok": False, "outcome": "error", "error": f"the ledger refused a row: {exc}",
                    "json": str(json_path), "markdown": str(md_path), "seqs": seqs}
    return {"ok": True, "outcome": "recorded", "json": str(json_path), "markdown": str(md_path), "seqs": seqs,
            "node": sheet["node"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lab chipsheet", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("ingest", help="file a pulled sheet and write its ledger rows")
    cmd.add_argument("--sheet", required=True, help="the sheet JSON pulled from the pod")
    cmd.add_argument("--data-dir", default=str(DATA_DIR), help="where the files go")
    cmd.add_argument("--dry-run", action="store_true", help="write nothing, answer with the rows")
    check = sub.add_parser("self-check", help="no GPU: parse recorded output, write and re-read a sheet")
    check.add_argument("--out", help="where to write it (default a new temporary folder)")
    return parser


def main(argv: Sequence[str]) -> dict[str, Any] | int:
    """File a chip sheet measured on a pod and write its ledger rows: ingest | self-check."""
    args = build_parser().parse_args(list(argv))
    if args.command == "self-check":
        from lab.kernels.chipsheet import chipsheet

        return chipsheet.main(["self-check", *(["--out", args.out] if args.out else [])])
    return ingest(args.sheet, args.data_dir, dry_run=args.dry_run)

