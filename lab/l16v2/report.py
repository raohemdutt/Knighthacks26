"""The search of the L16 retrain as tables: every trial in order of proposal, what the search learned, the choice.

    python lab/l16v2/report.py <folder holding bo.json and chosen.json> [--line] [--edge] [--ext] [--downloads <colab folder>]

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
    first = (chosen.get("extended_search") or {}).get("choice", {}).get("first")     # the choice before the extended search
    picked = first or chosen["chosen"]["name"]
    mark = " (the choice at 5:13 AM)" if first else " (chosen)"
    lines = ["| full-length run | from | learning rate | rank | alpha | dropout | warmup | steps run | example passes "
             "(run / at best) | best val loss (step) | validation agreement | worker |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for f in chosen["full_length_runs"]:
        origin = (f"trial {f['from_trial']} ({f['trial_kind']}, value {num(f.get('trial_value'))})"
                  if f.get("from_trial") else "the grid's run 1")
        score = f.get("val_agreement")
        lines.append(f"| {f['name']}{mark if f['name'] == picked else ''} | {origin} | {f['lr']:.3g} | {f['rank']} | "
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




# --- the edge round and the products (appended tables) ----------------------------------------------------------------

def edge_tables(folder: Path) -> str:
    """The edge round's trials, its noise repeats, and best value per rank and per learning-rate band."""
    edge = json.loads((folder / "bo-edge.json").read_text(encoding="utf-8"))
    first = [t for t in json.loads((folder / "bo.json").read_text(encoding="utf-8"))["trials"] if t.get("max_steps") == FIDELITY_STEPS]
    lines = ["| n | seen | kind | worker | learning rate | rank | alpha | dropout | warmup | seed | outcome | val loss, step 12 | "
             "val loss, step 24 | value |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    rows = sorted([*edge["trials"], *edge.get("noise", {}).values()], key=lambda t: t["number"])
    for t in rows:
        lines.append(f"| {t['number']} | {t.get('seen')} | {t['kind']} | {t.get('worker')} | {t['lr']:.3g} | {t['rank']} | {t['alpha']} | "
                     f"{t['dropout']:.2f} | {t['warmup_ratio']:.2f} | {t.get('seed', 42)} | {t.get('outcome')} | "
                     f"{num(t.get('val_at_12'))} | {num(t.get('val_at_24'))} | {num(t.get('value'))} |")
    everything = [t for t in [*first, *edge["trials"]] if t.get("value") is not None]
    out = ["\n".join(lines), "", "| rank | trials (both rounds) | completed | best value | best val loss at step 12 |", "| --- | --- | --- | --- | --- |"]
    for rank in edge["space"]["rank"]:
        of = [t for t in everything if t["rank"] == rank]
        done = [t["value"] for t in of if t.get("outcome") != "pruned"]
        at_12 = [t["val_at_12"] for t in of if t.get("val_at_12") is not None]
        out.append(f"| {rank} | {len(of)} | {len(done)} | {num(min(done) if done else None)} | {num(min(at_12) if at_12 else None)} |")
    out += ["", "| learning rate | trials (both rounds) | completed | best value |", "| --- | --- | --- | --- |"]
    bands = (("below 8e-5", 0.0, 8e-5), ("8e-5 to 1e-4", 8e-5, 1e-4), ("1e-4 to 2e-4", 1e-4, 2e-4), ("2e-4 to 4e-4", 2e-4, 4e-4),
             ("4e-4 to 1e-3", 4e-4, 1.0))
    for label, low, high in bands:
        of = [t for t in everything if low <= t["lr"] < high]
        done = [t["value"] for t in of if t.get("outcome") != "pruned"]
        out.append(f"| {label} | {len(of)} | {len(done)} | {num(min(done) if done else None)} |")
    return "\n".join(out)


def config_of(f: dict) -> str:
    return f"lr {f['lr']:.3g}, rank {f['rank']}, alpha {f['alpha']}, dropout {f['dropout']:.2f}, warmup {f['warmup_ratio']:.2f}"


def ext_tables(folder: Path) -> str:
    """The extended search (bo-ext.json): every full-length run, the configurations run with two seeds, the best per
    rank and per learning-rate band, the kinds, and each change of the choice."""
    ext = json.loads((folder / "bo-ext.json").read_text(encoding="utf-8"))
    runs = sorted(ext["runs"], key=lambda f: (f.get("number", 0), f["name"]))
    done = [f for f in runs if f.get("ok") and f.get("best_val") is not None]
    key = lambda f: (round(f["lr"], 9), f["rank"], f["alpha"], round(f["dropout"], 4), round(f["warmup_ratio"], 4))  # noqa: E731
    lines = ["| n | kind | run | seed | learning rate | rank | alpha | dropout | warmup | steps run | best step (example passes) | "
             "best val loss | validation agreement | worker | seconds |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for f in runs:
        agree = f"{f['agree']}/126" if f.get("agree") is not None else "not scored"
        lines.append(f"| {f.get('number', 0) or ''} | {f.get('kind')} | {f['name'][:44]} | {f.get('seed', 42)} | {f['lr']:.3g} | {f['rank']} | "
                     f"{f['alpha']} | {f['dropout']:.2f} | {f['warmup_ratio']:.2f} | {f.get('steps') if f.get('ok') else f.get('outcome')} | "
                     f"{f.get('best_step')} ({f.get('best_passes')}) | {num(f.get('best_val'))} | {agree} | {f.get('worker')} | "
                     f"{f.get('wall_s')} |")
    out = ["\n".join(lines), "", "| configuration run with two seeds | validation agreement, seed 42 / 43 | mean | difference | "
           "best val loss, seed 42 / 43 | difference |", "| --- | --- | --- | --- | --- | --- |"]
    pairs = []
    for one in sorted({key(f) for f in done}):
        rows = sorted([f for f in done if key(f) == one and f.get("agree") is not None], key=lambda f: f.get("seed", 42))
        if len(rows) >= 2:
            pairs.append(rows)
            out.append(f"| {config_of(rows[0])} | {' / '.join(str(f['agree']) for f in rows)} | "
                       f"{sum(f['agree'] for f in rows) / len(rows):.1f} | {abs(rows[0]['agree'] - rows[1]['agree'])} | "
                       f"{' / '.join(num(f['best_val']) for f in rows)} | {abs(rows[0]['best_val'] - rows[1]['best_val']):.4f} |")
    if pairs:
        gaps = [abs(rows[0]["agree"] - rows[1]["agree"]) for rows in pairs]
        out += ["", f"Noise of one run's validation agreement, from {len(pairs)} configurations run with two seeds: the two runs "
                    f"differ by {min(gaps)} to {max(gaps)} examples of 126 (mean {sum(gaps) / len(gaps):.1f})."]
    out += ["", "| rank | full-length runs | best val loss | best validation agreement | mean validation agreement |", "| --- | --- | --- | --- | --- |"]
    for rank in (2, 4, 8, 16, 32, 64):
        of = [f for f in done if f["rank"] == rank]
        scored = [f["agree"] for f in of if f.get("agree") is not None]
        out.append(f"| {rank} | {len(of)} | {num(min((f['best_val'] for f in of), default=None))} | "
                   f"{str(max(scored)) + '/126' if scored else 'n/a'} | {f'{sum(scored) / len(scored):.1f}' if scored else 'n/a'} |")
    out += ["", "| learning rate | full-length runs | best val loss | best validation agreement | mean validation agreement |", "| --- | --- | --- | --- | --- |"]
    for label, low, high in (("8e-5 to 1e-4", 8e-5, 1e-4), ("1e-4 to 2e-4", 1e-4, 2e-4), ("2e-4 to 4e-4", 2e-4, 4e-4), ("4e-4 to 1e-3", 4e-4, 1.1e-3)):
        of = [f for f in done if low <= f["lr"] < high]
        scored = [f["agree"] for f in of if f.get("agree") is not None]
        out.append(f"| {label} | {len(of)} | {num(min((f['best_val'] for f in of), default=None))} | "
                   f"{str(max(scored)) + '/126' if scored else 'n/a'} | {f'{sum(scored) / len(scored):.1f}' if scored else 'n/a'} |")
    out += ["", "| kind | runs | mean best val loss | mean validation agreement |", "| --- | --- | --- | --- |"]
    for kind, label in (("earlier", "earlier (the three runs before it)"), ("edge", "edge (forced)"), ("model", "model (the sampler's)"), ("repeat", "repeat (seed 43)")):
        of = [f for f in done if f.get("kind") == kind]
        scored = [f["agree"] for f in of if f.get("agree") is not None]
        out.append(f"| {label} | {len(of)} | {num(sum(f['best_val'] for f in of) / len(of) if of else None)} | "
                   f"{f'{sum(scored) / len(scored):.1f}' if scored else 'n/a'} |")
    failed = [f for f in runs if not (f.get("ok") and f.get("best_val") is not None)]
    if failed:
        out += ["", "Runs that did not finish: " + "; ".join(f"{f['name']} ({f.get('outcome')})" for f in failed) + "."]
    changes = ext["choice"].get("replaced") or []
    out += ["", "Changes of the choice, in order: " + ("; ".join(f"{c['key']} (mean {c['mean']:.1f}/126) replaced by {c['by']} (mean {c['mean_by']:.1f}/126)"
                                                                 for c in changes) if changes else "none") + "."]
    return "\n".join(out)


def downloads_table(colab: Path) -> str:
    """Every product that came back, once each (the last verified line of colab/downloads.txt per name)."""
    seen: dict[str, tuple[str, str]] = {}
    for line in (colab / "downloads.txt").read_text(encoding="ascii").splitlines():
        part = line.split()
        if len(part) >= 4 and part[3] == "verified":
            seen[part[0]] = (part[1], part[2])
    lines = ["| product (under work/L16v2-out/colab/) | bytes | sha256 |", "| --- | --- | --- |"]
    lines += [f"| `{name}` | {int(size):,} | `{digest}` |" for name, (size, digest) in sorted(seen.items())]
    return "\n".join(lines)


if __name__ == "__main__":
    code = main()
    if "--edge" in sys.argv:
        print()
        print(edge_tables(Path(sys.argv[1])))
    if "--ext" in sys.argv:
        print()
        print(ext_tables(Path(sys.argv[1])))
    if "--downloads" in sys.argv:
        print()
        print(downloads_table(Path(sys.argv[sys.argv.index("--downloads") + 1])))
    sys.exit(code)
