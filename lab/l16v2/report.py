"""The sweep of the L16 retrain as a table: every run with its example passes and validation loss, then the choice.

    python lab/l16v2/report.py <folder holding sweep.json and chosen.json> [--line]

Prints Markdown for lab/L16V2_RESULT.md; with --line, one short paragraph for the channel. It reads the two files
lab/l16v2/sweep.py wrote on the A100 (they come back inside best-adapter.tar.gz) and computes nothing new.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def num(value: object, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"


def table(runs: list[dict], chosen: dict) -> str:
    picked = chosen.get("chosen", {}).get("name")
    lines = ["| run | learning rate | rank | dropout | outcome | steps run | example passes (run / at best) | "
             "best validation loss (step) | last validation loss | last training loss | wall s | examples/s |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run in runs:
        mark = " (chosen)" if run["name"] == picked else ""
        lines.append(
            f"| {run['index']}{mark} | {run['lr']:g} | {run['rank']} | {run['dropout']:g} | {run['outcome']} | "
            f"{run.get('steps', 'n/a')} | {run.get('passes', 'n/a')} / {run.get('best_passes', 'n/a')} | "
            f"{num(run.get('best_val'))} ({run.get('best_step', 'n/a')}) | {num(run.get('final_val'))} | "
            f"{num(run.get('loss_last'))} | {num(run.get('wall_s'), 0)} | {num(run.get('samples_per_s'), 2)} |")
    return "\n".join(lines)


def choice(chosen: dict) -> str:
    base = chosen["base_validation_agreement"]
    lines = [f"Validation tool-call agreement (A100, bf16, transformers greedy; {chosen['validation_examples']} examples "
             f"in {chosen['validation_groups']} groups): base {base['correct']}/{base['n']} = {base['rate']:.3f}.", "",
             "| run (the 3 lowest validation losses) | best validation loss | validation agreement | tool name | parsed |",
             "| --- | --- | --- | --- | --- |"]
    for run in chosen["top"]:
        score = run["val_agreement"]
        lines.append(f"| {run['name']} | {num(run['best_val'])} | {score['correct']}/{score['n']} = {score['rate']:.3f} | "
                     f"{score['names']}/{score['n']} | {score['parsed']}/{score['n']} |")
    picked = chosen["chosen"]
    lines += ["", f"Chosen: `{picked['name']}` (learning rate {picked['lr']:g}, rank {picked['rank']}, alpha "
                  f"{picked['alpha']}, dropout {picked['dropout']:g}, warmup {picked['warmup_ratio']}), the checkpoint of "
                  f"optimizer step {picked['best_step']} = {picked['best_passes']} example passes, validation loss "
                  f"{num(picked['best_val'])}, validation agreement {picked['val_agreement']['correct']}/"
                  f"{picked['val_agreement']['n']}.", "",
              "Recipe flags for pod D: `" + " ".join(chosen["recipe_flags"]) + "`"]
    return "\n".join(lines)


def line(runs: list[dict], chosen: dict) -> str:
    parts = []
    for run in runs:
        if run.get("ok"):
            parts.append(f"{run['index']}: lr {run['lr']:g} r{run['rank']} d{run['dropout']:g} val {num(run.get('best_val'))} "
                         f"at {run.get('best_passes')} of {run.get('passes')} passes")
        else:
            parts.append(f"{run['index']}: lr {run['lr']:g} r{run['rank']} d{run['dropout']:g} {run['outcome']}")
    picked, base = chosen["chosen"], chosen["base_validation_agreement"]
    tops = "; ".join(f"{run['name']} {run['val_agreement']['correct']}/{run['val_agreement']['n']}" for run in chosen["top"])
    return (f"Sweep ({len(runs)} runs; best validation loss and example passes): " + " | ".join(parts) +
            f". Validation tool-call agreement, base {base['correct']}/{base['n']}; top 3 by validation loss: {tops}. "
            f"Chosen by the rule fixed before the sweep: {picked['name']} at step {picked['best_step']} "
            f"({picked['best_passes']} example passes).")


def main() -> int:
    folder = Path(sys.argv[1])
    runs = json.loads((folder / "sweep.json").read_text(encoding="utf-8"))["runs"]
    chosen = json.loads((folder / "chosen.json").read_text(encoding="utf-8"))
    if "--line" in sys.argv:
        print(line(runs, chosen))
    else:
        print(table(runs, chosen) + "\n\n" + choice(chosen))
    return 0


if __name__ == "__main__":
    sys.exit(main())
