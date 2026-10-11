"""Offline pairwise compares of correctness fixtures with lab/correctness.py's own compare_outputs and summarize."""
import json
import sys
from pathlib import Path

from lab import correctness as c

NEW = "fixtures/lab/correctness/vllm-gfx1151-v0.31.0-2026-10-11.json"
OLD = "fixtures/lab/correctness/vllm-gfx1151-2026-10-11.json"
A100 = "fixtures/lab/bench/a100-2026-10-10/vllm-a100-2026-10-10.json"
R60 = sys.argv[1]  # R60's compare result of 04:42 UTC (its outputs are the new build's)
OUT = Path("fixtures/lab/new_teacher_quality/correctness/three_way.json")

docs = {"new": c.load_baseline(NEW), "old": c.load_baseline(OLD), "a100": c.load_baseline(A100)}
r60 = json.loads(Path(R60).read_text(encoding="utf-8"))
docs["new_r60_0442"] = {"conditions": r60["conditions"], "outputs": r60["outputs"], "recorded_at": r60["at"]}
paths = {"new": NEW, "old": OLD, "a100": A100,
         "new_r60_0442": "fixtures/lab/correctness/r60-2026-10-11/compare-with-april-baseline.json "
                         "(branch ctx/runtime/vllm-upgrade), its outputs"}

pairs = [("new", "old", "new build against the old Strix baseline"),
         ("new", "a100", "new build against the A100 record"),
         ("old", "a100", "old Strix baseline against the A100 record"),
         ("new", "new_r60_0442", "new build now against the same build at 04:42 UTC (R60's compare run)")]
result = {"schema": "r64.correctness.three_way/1", "fixtures": paths, "pairs": []}
for run, base, label in pairs:
    rows = c.compare_outputs(docs[base]["outputs"], docs[run]["outputs"])
    summary = c.summarize(rows)
    result["pairs"].append({"run": run, "baseline": base, "label": label, "summary": summary, "prompts": rows})
    per = ", ".join(f"{r['prompt']} {'match' if r['tokens_match'] else 'at ' + str(r['first_divergence'])}"
                    f"{'' if r['tokens_match'] else ' (lp diff ' + str(r['logprob_difference']) + ')'}" for r in rows)
    print(f"{label}: {summary['matches']}/{summary['prompts']} = {summary['match_rate']}; {per}")
OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", OUT)
