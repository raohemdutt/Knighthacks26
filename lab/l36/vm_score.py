"""L36 scoring on the Colab A100: tool-call agreement (lab/tool_call_agreement.py, unchanged) in bf16, generated the
way the v2 search scored (lab/l16v2/sweep.py's Generator: transformers greedy on the bf16 weights with the adapter
attached; its Replay completer; prompts_of() with each example's own clock through system_for()).

    python3 lab/l36/vm_score.py gen <arm> <adapter folder | base> [<arm> <adapter> ...]
    python3 lab/l36/vm_score.py score <arm> [<arm> ...]

gen writes, per arm, the outputs on four sets: v3's 240 held out, v2's 228 held out, v3's 126 validation and v2's
126 validation, as /content/l36/gen/<arm>/<set>.json ({prompt: output}), then /content/l36/gen/<arm>.DONE.
score marks every arm on every set (counts, and counts by expected tool), and for the held-out sets runs tca.run()
for each pair (base against each arm, f2r4 against each new arm) with its exact McNemar and sign-flip p. Everything
goes under /content/l36/scores/, packed as /content/l36/score-l36.tar.gz with a .sha256; SCORE_DONE marks the end.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tarfile
import time
import traceback
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab" / "l16v2"))
ROOT = Path("/content/l36")
MODEL = "/content/models/Qwen3.5-4B"
SETS = {"v3_held_out": (REPO / "fixtures" / "learning" / "tool-calling-v3", "held_out"),
        "v2_held_out": (ROOT / "data" / "tool-calling-v2", "held_out"),
        "v3_validation": (REPO / "fixtures" / "learning" / "tool-calling-v3", "validation"),
        "v2_validation": (ROOT / "data" / "tool-calling-v2", "validation")}


def examples_of(name: str):
    import sweep
    from lab.dataset import load_split

    folder, side = SETS[name]
    if side == "held_out":
        split = load_split(folder, require_images=False)
        return split, list(split.held_out)
    return sweep.validation_examples(folder)


def gen(pairs: list[tuple[str, str]]) -> None:
    import sweep
    from lab import tool_call_agreement as tca

    generator = sweep.Generator(MODEL)
    status = ROOT / "gen" / "status.txt"
    for arm, adapter in pairs:
        (ROOT / "gen" / arm).mkdir(parents=True, exist_ok=True)
        generator.attach(None if adapter == "base" else Path(adapter))
        for name in SETS:
            split, examples = examples_of(name)
            prompts = sweep.prompts_of(split.data.manifest, examples)
            began = time.time()

            def tick(done: int, total: int, arm: str = arm, name: str = name, began: float = began) -> None:
                status.write_text(f"gen arm={arm} set={name} {done}/{total} elapsed_s={int(time.time() - began)}\n",
                                  encoding="ascii")

            outputs = generator.generate(prompts, tca.MAX_TOKENS, tick)
            (ROOT / "gen" / arm / f"{name}.json").write_text(json.dumps(outputs, ensure_ascii=False), encoding="utf-8")
            with (ROOT / "gen" / "times.txt").open("a", encoding="ascii") as handle:
                handle.write(f"{arm} {name} n={len(prompts)} seconds={time.time() - began:.1f} adapter={adapter}\n")
        (ROOT / "gen" / f"{arm}.DONE").write_text(adapter + "\n", encoding="utf-8")
    generator.attach(None)


def tool_of(expected) -> str:
    calls = expected.label.tool_calls
    return calls[0].name if calls else "none"


def score(arms: list[str]) -> None:
    import sweep
    from lab import tool_call_agreement as tca
    from lab.fixture_dataset import load_expected
    from lab.fixture_dataset_v2 import system_for

    out = ROOT / "scores"
    out.mkdir(parents=True, exist_ok=True)
    table: dict = {"rule_sha256": hashlib.sha256(tca.RULE.encode("utf-8")).hexdigest(), "arms": arms, "sets": {},
                   "pairs": {}}
    for name, (folder, side) in SETS.items():
        split, examples = examples_of(name)
        expected = load_expected(folder)
        manifest = split.data.manifest
        entry = {"dataset_id": split.dataset_id, "split_sha256": split.sha256, "n": len(examples),
                 "groups": len({example.frame_ref for example in examples}),
                 "by_tool_n": dict(Counter(tool_of(expected[example.id]) for example in examples)), "arms": {}}
        for arm in arms:
            outputs = json.loads((ROOT / "gen" / arm / f"{name}.json").read_text(encoding="utf-8"))
            with (out / f"{name}.{arm}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
                rows = tca.run_arm(arm, sweep.Replay(outputs), examples, expected,
                                   lambda example: system_for(manifest, example),
                                   lambda row: handle.write(json.dumps(row, ensure_ascii=False) + "\n"))
            by_tool = Counter(tool_of(expected[i]) for i, row in rows.items() if row["correct"])
            entry["arms"][arm] = {"correct": sum(bool(r["correct"]) for r in rows.values()),
                                  "names": sum(bool(r["names"]) for r in rows.values()),
                                  "parsed": sum(bool(r["parsed"]) for r in rows.values()), "by_tool": dict(by_tool)}
        table["sets"][name] = entry
        if side != "held_out":
            continue
        pairs = [("base", arm) for arm in arms if arm != "base"]
        if "f2r4" in arms:
            pairs += [("f2r4", arm) for arm in arms if arm not in ("base", "f2r4")]
        for before, after in pairs:
            completers = {role: sweep.Replay(json.loads((ROOT / "gen" / arm / f"{name}.json").read_text(encoding="utf-8")))
                          for role, arm in (("before", before), ("after", after))}
            summary = tca.run(dataset=folder, completers=completers, out=out / "pairs" / f"{name}.{before}-vs-{after}",
                              run_id=f"l36-{name}-{before}-vs-{after}", side="held_out", before_runs=1, after_runs=1,
                              servers={"before": {"model": f"Qwen3.5-4B bf16 transformers A100, arm {before}"},
                                       "after": {"model": f"Qwen3.5-4B bf16 transformers A100, arm {after}"}})
            table["pairs"][f"{name}.{before}-vs-{after}"] = {
                "before": summary["correct"]["current"], "after": summary["correct"]["candidate"],
                "gained_0_to_1": summary["mcnemar"]["gained_0_to_1"], "lost_1_to_0": summary["mcnemar"]["lost_1_to_0"],
                "mcnemar_p": summary["mcnemar"]["p_value"], "mcnemar_p_exact": summary["mcnemar"]["p_exact"],
                "sign_flip_p": summary["sign_flip"]["p_value"], "n_groups": summary["n_groups"]}
    (out / "table.json").write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    path = ROOT / "score-l36.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        tar.add(out, arcname="scores")
        tar.add(ROOT / "gen", arcname="gen")
    (ROOT / "score-l36.tar.gz.sha256").write_text(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()} {path.stat().st_size} score-l36.tar.gz\n", encoding="ascii")
    (ROOT / "SCORE_DONE").write_text("done\n", encoding="ascii")


def main() -> int:
    what, rest = sys.argv[1], sys.argv[2:]
    try:
        if what == "gen":
            gen(list(zip(rest[0::2], rest[1::2])))
        else:
            score(rest)
    except Exception:
        traceback.print_exc()
        (ROOT / f"{what}.FAILED").write_text(traceback.format_exc()[-1500:], encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
