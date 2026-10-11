"""The search of the L16 retrain as tables: every trial in order of proposal, what the search learned, the choice.

    python lab/l16v2/report.py <folder holding bo.json and chosen.json> [--line]

Prints Markdown for lab/L16V2_RESULT.md; with --line, one short paragraph for the channel. It reads the two files
lab/l16v2/bo_local.py wrote on the workstation (work/L16v2-out/bo/) and computes nothing new beyond sums and means.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

FIDELITY_STEPS = 24


def num(value: object, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"


def trials_table(trials: list[dict]) -> str:
    lines = ["| n | seen | kind | worker | learning rate | rank | alpha | dropout | warmup | outcome | val loss, step 12 | "
             "val loss, step 24 | value | best so far |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    best = None
    for t in sorted(trials, key=lambda t: t["number"]):
        if t.get("value") is not None and t.get("outcome") != "pruned":
            best = t["value"] if best is None else min(best, t["value"])
        outcome = "observation (grid run 1)" if t["kind"] == "seed" and t.get("max_steps") != FIDELITY_STEPS else t.get("outcome")
        lines.append(f"| {t['number']} | {t.get('seen')} | {t['kind']} | {t.get('worker')} | {t['lr']:.3g} | {t['rank']} | "
                     f"{t['alpha']} | {t['dropout']:.3f} | {t['warmup_ratio']:.3f} | {outcome} | {num(t.get('val_at_12'))} | "
                     f"{num(t.get('val_at_24'))} | {num(t.get('value'))} | {num(best)} |")
    return "\n".join(lines)


def by_kind(trials: list[dict]) -> str:
    lines = ["| kind | trials run | mean val loss at step 12 (all) | completed | mean value (completed) | best value | pruned |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for kind in ("seed", "init", "model"):
        ran = [t for t in trials if t["kind"] == kind and t.get("max_steps") == FIDELITY_STEPS]
        at_12 = [t["val_at_12"] for t in ran if t.get("val_at_12") is not None]
        done = [t["value"] for t in ran if t.get("value") is not None and t.get("outcome") != "pruned"]
        lines.append(f"| {kind} | {len(ran)} | {num(sum(at_12) / len(at_12) if at_12 else None)} | {len(done)} | "
                     f"{num(sum(done) / len(done) if done else None)} | {num(min(done) if done else None)} | "
                     f"{sum(t.get('outcome') == 'pruned' for t in ran)} |")
    return "\n".join(lines)


def full_runs(chosen: dict) -> str:
    picked = chosen["chosen"]["name"]
    lines = ["| full-length run | from | learning rate | rank | alpha | dropout | warmup | steps run | example passes "
             "(run / at best) | best val loss (step) | validation agreement | worker |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for f in chosen["full_length_runs"]:
        origin = (f"trial {f['from_trial']} ({f['trial_kind']}, value {num(f.get('trial_value'))})"
                  if f.get("from_trial") else "the grid's run 1")
        score = f.get("val_agreement")
        lines.append(f"| {f['name']}{' (chosen)' if f['name'] == picked else ''} | {origin} | {f['lr']:.3g} | {f['rank']} | "
                     f"{f['alpha']} | {f['dropout']:.3f} | {f['warmup_ratio']:.3f} | {f.get('steps')} | "
                     f"{f.get('passes')} / {f.get('best_passes')} | {num(f.get('best_val'))} ({f.get('best_step')}) | "
                     f"{str(score['correct']) + '/' + str(score['n']) if score else 'not scored'} | {f.get('worker')} |")
    return "\n".join(lines)


def line(trials: list[dict], chosen: dict) -> str:
    ran = [t for t in trials if t.get("max_steps") == FIDELITY_STEPS]
    done = [t for t in ran if t.get("value") is not None and t.get("outcome") != "pruned"]
    base, picked = chosen.get("base_validation_agreement") or {}, chosen["chosen"]
    runs = "; ".join(f"{f['name'].split('-')[0]} lr {f['lr']:.3g} r{f['rank']} a{f['alpha']} d{f['dropout']:.2f} w{f['warmup_ratio']:.2f}: "
                     f"val {num(f.get('best_val'))} at {f.get('best_passes')} passes"
                     + (f", agreement {f['val_agreement']['correct']}/{f['val_agreement']['n']}" if f.get("val_agreement") else "")
                     for f in chosen["full_length_runs"])
    return (f"Search: {len(ran)} trials of 24 steps ({len(done)} completed, {len(ran) - len(done)} pruned or failed) after the "
            f"seed observation; kinds: " + ", ".join(f"{k} {sum(t['kind'] == k for t in ran)}" for k in ("seed", "init", "model"))
            + f". Full-length runs: {runs}. Validation tool-call agreement, base {base.get('correct')}/{base.get('n')}. "
              f"Chosen by the frozen rule: {picked['name']} (step {picked['best_step']}, {picked['best_passes']} example passes).")


def main() -> int:
    folder = Path(sys.argv[1])
    trials = json.loads((folder / "bo.json").read_text(encoding="utf-8"))["trials"]
    chosen = json.loads((folder / "chosen.json").read_text(encoding="utf-8")) if (folder / "chosen.json").exists() else None
    if "--line" in sys.argv and chosen:
        print(line(trials, chosen))
        return 0
    print(trials_table(trials) + "\n\n" + by_kind(trials))
    if chosen:
        base = chosen.get("base_validation_agreement") or {}
        print("\n" + full_runs(chosen) + f"\n\nBase model, validation agreement: {base.get('correct')}/{base.get('n')}.\n\n"
              "Recipe flags for pod D: `" + " ".join(chosen["recipe_flags"]) + "`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
