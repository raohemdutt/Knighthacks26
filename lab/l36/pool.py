"""L36: pool tool-calling-v3 with tool-calling-v2 into one dataset folder the trainer accepts.

    python lab/l36/pool.py --v2 <folder>/tool-calling-v2 --v3 fixtures/learning/tool-calling-v3 --out <folder>

Writes <folder>/tool-calling-v3v2-pooled/ (manifest.json, examples.jsonl, split.json) and, beside it,
rows-train.jsonl (v3's 721 train rows, then v2's 840) and rows-validation.jsonl (v3's 126, then v2's 126). The two
sets share one row format (checked here: the same keys, the same three tools, the same instruction), so no row is
converted. The pooled split's held-out side is both held-out sides (v3's 240 and v2's 228), so the trainer refuses
any of them as a row. Nothing is read from a clock or a random source; the same inputs give the same bytes.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

POOLED_ID = "tool-calling-v3v2-pooled"
ROW_KEYS = {"id", "frame_ref", "messages", "tools", "text", "spans"}


def lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", required=True)
    parser.add_argument("--v3", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    v2, v3, out = Path(args.v2), Path(args.v3), Path(args.out)
    m2 = json.loads((v2 / "manifest.json").read_text(encoding="utf-8"))
    m3 = json.loads((v3 / "manifest.json").read_text(encoding="utf-8"))
    if m2["spec"] != m3["spec"]:
        print("the two manifests carry different specs", file=sys.stderr)
        return 1
    s2 = json.loads((v2 / "split.json").read_text(encoding="utf-8"))
    s3 = json.loads((v3 / "split.json").read_text(encoding="utf-8"))
    ex3, ex2 = lines(v3 / "examples.jsonl"), lines(v2 / "examples.jsonl")
    ids3 = {json.loads(line)["id"] for line in ex3}
    ids2 = {json.loads(line)["id"] for line in ex2}
    if ids2 & ids3:
        print(f"{len(ids2 & ids3)} example ids are in both sets", file=sys.stderr)
        return 1
    rows = {}
    tools = set()
    for name in ("rows-train.jsonl", "rows-validation.jsonl"):
        both = lines(v3 / name) + lines(v2 / name)
        for line in both:
            raw = json.loads(line)
            if set(raw) - {"images"} != ROW_KEYS:
                print(f"{name}: row {raw.get('id')} has keys {sorted(raw)}", file=sys.stderr)
                return 1
            tools.add(json.dumps(raw["tools"], sort_keys=True))
        rows[name] = both
    if len(tools) != 1:
        print(f"the rows offer {len(tools)} different tool lists", file=sys.stderr)
        return 1
    folder = out / POOLED_ID
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset_id": POOLED_ID, "spec": m3["spec"], "created": m3["created"], "examples": len(ex3) + len(ex2),
                "action_id": "fixture:" + POOLED_ID}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    (folder / "examples.jsonl").write_text("\n".join(ex3 + ex2) + "\n", encoding="utf-8", newline="\n")
    (folder / "images").mkdir(exist_ok=True)
    split = {"version": 1, "dataset_id": POOLED_ID,
             "rule": ("L36: the union of tool-calling-v3's split and tool-calling-v2's split, each unchanged. Train: "
                      "both train sides (validation rows are on them and are never trained on). Held out: both held-out "
                      "sides, v3's 240 then v2's 228."),
             "train": s3["train"] + s2["train"], "held_out": s3["held_out"] + s2["held_out"]}
    (folder / "split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8", newline="\n")
    for name, both in rows.items():
        (out / name).write_text("\n".join(both) + "\n", encoding="utf-8", newline="\n")
    report = {"pooled": POOLED_ID, "examples": manifest["examples"], "train_ids": len(split["train"]),
              "held_out_ids": len(split["held_out"]), "rows_train": len(rows["rows-train.jsonl"]),
              "rows_validation": len(rows["rows-validation.jsonl"]),
              "inputs": {"v3_rows_train": sha(v3 / "rows-train.jsonl"), "v2_rows_train": sha(v2 / "rows-train.jsonl"),
                         "v3_rows_validation": sha(v3 / "rows-validation.jsonl"),
                         "v2_rows_validation": sha(v2 / "rows-validation.jsonl"),
                         "v3_split": sha(v3 / "split.json"), "v2_split": sha(v2 / "split.json")},
              "outputs": {"rows_train": sha(out / "rows-train.jsonl"), "rows_validation": sha(out / "rows-validation.jsonl"),
                          "split": sha(folder / "split.json"), "examples": sha(folder / "examples.jsonl")}}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
