"""The extended search of the L16 retrain: full-length runs on both cards until a wall-clock time, noise taken out.

    <a Python with optuna> lab/l16v2/bo_ext.py --sessions l16v2-a100,l16v2-a100-b --until-utc 2026-10-11T11:00:00Z

Run from the repo root after lab/l16v2/bo_local.py has written chosen.json. The rules are lab/L16V2_RULES.md
amendment 6: every trial is a full-length run (at most 62 optimizer steps, validation every 8, the best checkpoint
kept, stop after 2 evaluations without a 3 percent gain; no step-12 pruning), its value is its best validation
loss, and every finished run gets its validation tool-call agreement on the 126 validation rows, generated on its
card. The space is the edge round's (lab/l16v2/bo_edge.py), the sampler and the asynchronous ask-and-tell from the
workstation are the first search's (lab/l16v2/bo_local.py); the study is `l16v2-full` in the same database, started
from the full-length runs that already exist.

    forced   FORCED, at full length, first (kind edge); then the sampler's points (kind model)
    repeats  the current choice is run again with seed 43; a configuration whose run has a higher validation
             agreement than the choice's mean so far is run again with seed 43 (kind repeat, ahead of everything
             else); it replaces the choice only when the mean agreement of its two runs is higher than the mean of
             the choice's two runs
    pulled   every run's metrics when it ends; the adapter of every run that is a candidate to replace the choice
    budget   no run starts that could not finish, scoring included, by --until-utc

Each finished run is one line: stage=ext trial=k kind=edge|model|repeat seen=k worker=.. lr=.. rank=.. drop=..
warm=.. steps=.. val=.. agree=../126 best=.. At the end: ext_table lines for every full-length run, the best per
rank and per learning-rate band, the noise from the repeats, and either "the choice stands" or the new choice with
its adapter's path. Only then is the FINAL_CHOICE file left on the chosen adapter's VM and the held-out side read.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bo_edge as be  # noqa: E402  (the space, the rounding to it, the held-out reading after the final choice)
import bo_local as bl  # noqa: E402  (worker calls, downloads, progress lines)

bl.LAUNCH = bl.COLAB / "launch-ext"
STUDY_NAME = "l16v2-full"
FORCED = [(32, 2e-4), (16, 4e-4), (32, 1e-4), (64, 1e-4), (4, 8e-4), (2, 4e-4), (8, 2e-4)]   # (rank, learning rate)
REPEAT_SEED = 43
RUN_S = 840.0          # a full-length run with its validation scoring, on a card of its own
EXT_JSON = "bo-ext.json"
CHOICE_REPEAT = "ext01-repeat-choice-seed43"


def config_key(fact: dict[str, Any]) -> str:
    return f"lr{fact['lr']:.6g}-rank{fact['rank']}-a{fact['alpha']}-drop{fact['dropout']:.4f}-warm{fact['warmup_ratio']:.4f}"


def flags_of(fact: dict[str, Any]) -> list[str]:
    return ["--max-steps", str(bl.FULL_STEPS), "--eval-steps", str(bl.FULL_EVAL), "--early-stop", str(bl.FULL_PATIENCE),
            "--early-stop-min-rel", str(bl.FULL_MIN_REL), "--lr", f"{fact['lr']:.6g}", "--rank", str(fact["rank"]),
            "--alpha", str(fact["alpha"]), "--dropout", f"{fact['dropout']:.4f}", "--warmup-ratio", f"{fact['warmup_ratio']:.4f}",
            *(["--seed", str(fact["seed"])] if fact.get("seed", 42) != 42 else [])]


def pull_adapter(session: str, name: str) -> dict[str, Any] | None:
    """A candidate's adapter, packed on its VM with its sha256, pulled, verified and unpacked here."""
    local = bl.COLAB / "runs" / name / "adapter" / "adapter_model.safetensors"
    if not local.exists():
        product = f"adapter-{name}.tar.gz"
        made = bl.shell(session, f"cd /content/l16v2 && tar -czf products/{product} -C runs {name}/adapter && cd products && "
                                 f"echo \"$(sha256sum {product} | cut -d' ' -f1) $(stat -c %s {product}) {product}\" > {product}.sha256 && echo packed",
                        "pack-adapter", timeout=300)
        if made is None or "packed" not in made:
            bl.plog(f"PROBLEM worker={session} could not pack the adapter of {name}")
            return None
        tarball = bl.get(session, product)
        if tarball is None:
            return None
        bl.extract(tarball, bl.COLAB / "runs")
    return {"adapter_path": str(local.parent).replace("\\", "/"), "adapter_sha256": bl.sha256_of(local)}


def main() -> int:
    import optuna
    from optuna.distributions import CategoricalDistribution, FloatDistribution
    from optuna.trial import TrialState

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--until-utc", required=True, help="no run starts that could not finish by this time")
    parser.add_argument("--skip-held-out", action="store_true")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sessions = [name for name in args.sessions.split(",") if name]
    until = dt.datetime.fromisoformat(args.until_utc.replace("Z", "+00:00")).timestamp()
    chosen = json.loads((bl.BO / "chosen.json").read_text(encoding="utf-8"))
    path = bl.BO / EXT_JSON
    if path.exists():       # a restarted driver: what it had finished comes back
        saved = json.loads(path.read_text(encoding="utf-8"))
        runs, repeats, choice, state = saved["runs"], saved["repeats"], saved["choice"], saved["state"]
    else:
        runs = [{**f, "seed": 42, "kind": "earlier", "agree": (f.get("val_agreement") or {}).get("correct"), "number": 0,
                 "ok": True} for f in chosen["full_length_runs"]]
        first = next(f for f in runs if f["name"] == chosen["chosen"]["name"])
        choice = {"key": config_key(first), "first": first["name"], "replaced": []}
        repeats = [{"name": CHOICE_REPEAT, "of": first["name"], "lr": first["lr"], "rank": first["rank"], "alpha": first["alpha"],
                    "dropout": first["dropout"], "warmup_ratio": first["warmup_ratio"], "why": "the current choice"}]
        state = {"numbers": 1, "asked_repeat": [], "decided": [], "stopped": False}
    storage = bl.build_storage(bl.BO / "study.db")
    try:
        existing = len(optuna.load_study(study_name=STUDY_NAME, storage=storage).trials)
    except KeyError:
        existing = 0
    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, constant_liar=True, seed=bl.SEED + 100 + existing,
                                         n_startup_trials=bl.RANDOM_STARTUP)
    study = optuna.create_study(direction="minimize", study_name=STUDY_NAME, sampler=sampler, storage=storage, load_if_exists=True)
    if not existing:
        distributions = {"lr": FloatDistribution(be.LR[0], be.LR[1], log=True), "rank": CategoricalDistribution(list(be.RANKS)),
                         "alpha_mult": CategoricalDistribution(list(be.ALPHA_MULTS)),
                         "dropout": CategoricalDistribution(list(be.DROPOUTS)), "warmup": CategoricalDistribution(list(be.WARMUPS))}
        for f in runs:
            study.add_trial(optuna.trial.create_trial(
                params={"lr": float(f["lr"]), "rank": int(f["rank"]), "alpha_mult": int(f["alpha"] // f["rank"]),
                        "dropout": be.nearest(f["dropout"], be.DROPOUTS), "warmup": be.nearest(f["warmup_ratio"], be.WARMUPS)},
                distributions=distributions, value=float(f["best_val"]), user_attrs={"name": f["name"], "kind": "earlier"}))
        for rank, lr in FORCED:
            study.enqueue_trial({"lr": lr, "rank": rank, "alpha_mult": 2, "dropout": 0.05, "warmup": 0.05}, skip_if_exists=True)
        bl.plog(f"decision extended search: study {STUDY_NAME}, started from the {len(runs)} full-length runs that exist; "
                f"{len(FORCED)} forced configurations enqueued; the current choice is {choice['first']} "
                f"({chosen['chosen']['val_agreement']['correct']}/126) and is repeated with seed {REPEAT_SEED} first; "
                f"no run starts after {dt.datetime.fromtimestamp(until - RUN_S, dt.timezone.utc):%H:%M:%S} UTC")
    lock = threading.RLock()

    def save() -> None:
        path.write_text(json.dumps({"rule": "lab/L16V2_RULES.md amendment 6", "study": STUDY_NAME, "until": until,
                                    "forced": FORCED, "runs": runs, "repeats": repeats, "choice": choice, "state": state},
                                   indent=2) + "\n", encoding="utf-8")

    def of_config(key: str) -> list[dict[str, Any]]:
        return [f for f in runs if f.get("ok") and f.get("agree") is not None and config_key(f) == key]

    def mean_agree(key: str) -> float:
        rows = of_config(key)
        return sum(f["agree"] for f in rows) / len(rows) if rows else -1.0

    def best_text() -> str:
        return f"{choice['key']}:{mean_agree(choice['key']):.1f}/126x{len(of_config(choice['key']))}"

    def settle() -> list[tuple[str, str]]:
        """Amendment 6's rule after a run finishes: queue a second seed for a challenger, replace the choice on the
        mean of two seeds. Returns the (session, run) adapters to pull as candidates."""
        to_pull: list[tuple[str, str]] = []
        target = mean_agree(choice["key"])
        for key in sorted({config_key(f) for f in runs if f.get("ok") and f.get("agree") is not None}):
            if key == choice["key"]:
                continue
            rows = of_config(key)
            if len(rows) == 1 and rows[0]["agree"] > target and key not in state["asked_repeat"]:
                state["asked_repeat"].append(key)
                one = rows[0]
                repeats.insert(0, {"of": one["name"], "lr": one["lr"], "rank": one["rank"], "alpha": one["alpha"],
                                   "dropout": one["dropout"], "warmup_ratio": one["warmup_ratio"], "why": "a challenger"})
                to_pull.append((one["worker"].split("#")[0], one["name"]))
                bl.plog(f"decision ext: {one['name']} has validation agreement {one['agree']}/126, above the choice's mean "
                        f"{target:.1f}: it is run again with seed {REPEAT_SEED} before it can replace anything")
            elif len(rows) >= 2 and key not in state["decided"] and len(of_config(choice["key"])) >= 2:
                state["decided"].append(key)
                mean = mean_agree(key)
                if mean > target:
                    choice["replaced"].append({"key": choice["key"], "mean": target, "by": key, "mean_by": mean})
                    bl.plog(f"decision ext: the choice is replaced: {key} has mean validation agreement {mean:.1f}/126 over "
                            f"{len(rows)} seeds against {target:.1f} for {choice['key']}")
                    choice["key"] = key
                    target = mean
                    to_pull += [(f["worker"].split("#")[0], f["name"]) for f in rows]
                else:
                    bl.plog(f"decision ext: {key} does not replace the choice: mean validation agreement {mean:.1f}/126 over "
                            f"{len(rows)} seeds against {target:.1f}")
        return to_pull

    def worker(session: str) -> None:
        while True:
            with lock:
                if time.time() + RUN_S > until:
                    if not state["stopped"]:
                        state["stopped"] = True
                        bl.plog("decision extended search stops starting runs: the next one could not finish by the end time")
                    return
                seen = sum(t.state == TrialState.COMPLETE for t in study.get_trials(deepcopy=False))
                trial = None
                if repeats:
                    task = repeats.pop(0)
                    if task.get("name"):
                        name = task["name"]
                    else:
                        state["numbers"] += 1
                        name = f"x{state['numbers']:02d}-repeat-seed{REPEAT_SEED}-of-{task['of'][:28]}"
                    fact = {"name": name, "kind": "repeat", "seed": REPEAT_SEED, "repeat_of": task["of"], "why": task["why"],
                            "lr": task["lr"], "rank": task["rank"], "alpha": task["alpha"], "dropout": task["dropout"],
                            "warmup_ratio": task["warmup_ratio"]}
                else:
                    waiting = [t.system_attrs.get("fixed_params") for t in study.get_trials(deepcopy=False, states=(TrialState.WAITING,))]
                    trial = study.ask()
                    params = be.suggest(trial)
                    forced = any(fixed and all(params.get(k) == v for k, v in fixed.items()) for fixed in waiting)
                    state["numbers"] += 1
                    fact = {"kind": "edge" if forced else "model", "seed": 42, "lr": params["lr"], "rank": params["rank"],
                            "alpha": params["rank"] * params["alpha_mult"], "dropout": params["dropout"],
                            "warmup_ratio": params["warmup"]}
                    fact["name"] = (f"x{state['numbers']:02d}-lr{fact['lr']:.3g}-rank{fact['rank']}-a{fact['alpha']}"
                                    f"-drop{fact['dropout']:.2f}-warm{fact['warmup_ratio']:.2f}")
                    for key, value in (("kind", fact["kind"]), ("seen", seen), ("worker", session), ("name", fact["name"])):
                        trial.set_user_attr(key, value)
                fact.update(number=state["numbers"], seen=seen, worker=session, max_steps=bl.FULL_STEPS, full_length=True)
                save()

            def line(status: dict[str, Any], fact: dict[str, Any] = fact) -> str:
                latest = status["val"][max(status["val"])] if status["val"] else None
                return (f"stage=ext trial={fact['number']} kind={fact['kind']} seen={fact['seen']} worker={fact['worker']} "
                        f"lr={fact['lr']:.3g} rank={fact['rank']} drop={fact['dropout']:.2f} warm={fact['warmup_ratio']:.2f} "
                        f"seed={fact['seed']} step={status['step']}/{bl.FULL_STEPS} val={bl.fmt(latest)} "
                        f"best_val={bl.fmt(min(status['val'].values()) if status['val'] else None)} best={best_text()}")

            fact.update(bl.run_on(session, fact["name"], flags_of(fact), False, line))
            fact["worker"] = session
            if fact.get("ok") and fact.get("best_val") is not None:
                scores = bl.score_on(session, fact["name"], [fact["name"]])
                fact["val_agreement"] = (scores or {}).get(fact["name"])
                fact["agree"] = (fact["val_agreement"] or {}).get("correct")
            with lock:
                if trial is not None:
                    if fact.get("ok") and fact.get("best_val") is not None:
                        study.tell(trial, float(fact["best_val"]))
                    else:
                        study.tell(trial, state=TrialState.FAIL)
                runs.append(fact)
                to_pull = settle() if fact.get("agree") is not None else []
                save()
                bl.plog(f"RESULT stage=ext trial={fact['number']} kind={fact['kind']} seen={fact['seen']} worker={session} "
                        f"name={fact['name']} lr={fact['lr']:.3g} rank={fact['rank']} alpha={fact['alpha']} drop={fact['dropout']:.2f} "
                        f"warm={fact['warmup_ratio']:.2f} seed={fact['seed']} outcome={fact.get('outcome')} steps={fact.get('steps')} "
                        f"best_step={fact.get('best_step')} passes_at_best={fact.get('best_passes')} val={bl.fmt(fact.get('best_val'))} "
                        f"agree={fact.get('agree')}/126 best={best_text()} wall_s={fact.get('wall_s')}"
                        + (f" error={str(fact.get('error'))[:100]}" if not fact.get("ok") else ""))
            for where, name in to_pull:
                if any(f["name"] == name and f.get("adapter_path") for f in runs):
                    continue        # already on this machine (an earlier finalist)
                pulled = pull_adapter(where, name)
                with lock:
                    for f in runs:
                        if f["name"] == name and pulled:
                            f.update(pulled)
                    save()
            if fact.get("outcome") == "lost":
                bl.plog(f"PROBLEM worker={session} lost during {fact['name']}; this worker leaves the extended search")
                return

    threads = [threading.Thread(target=worker, args=(session,), daemon=True) for session in sessions]
    for thread in threads:
        thread.start()
        time.sleep(2)
    for thread in threads:
        thread.join()

    # --- the end: every full-length run, the best per rank and band, the noise, the choice ----------------------------
    done = [f for f in runs if f.get("ok") and f.get("best_val") is not None]
    for f in sorted(done, key=lambda f: (f["number"], f["name"])):
        bl.plog(f"RESULT ext_table name={f['name']} kind={f['kind']} seed={f.get('seed', 42)} lr={f['lr']:.3g} rank={f['rank']} "
                f"alpha={f['alpha']} drop={f['dropout']:.3f} warm={f['warmup_ratio']:.3f} steps={f.get('steps')} "
                f"best_step={f.get('best_step')} passes_at_best={f.get('best_passes')} val={bl.fmt(f.get('best_val'))} "
                f"agree={f.get('agree')}/126 worker={f.get('worker')}")
    for rank in be.RANKS:
        of = [f for f in done if f["rank"] == rank]
        scored = [f for f in of if f.get("agree") is not None]
        bl.plog(f"RESULT ext_table rank={rank} runs={len(of)} best_val={bl.fmt(min((f['best_val'] for f in of), default=None))} "
                f"best_agree={max((f['agree'] for f in scored), default=None)}/126")
    for label, low, high in (("8e-5 to 1e-4", 8e-5, 1e-4), ("1e-4 to 2e-4", 1e-4, 2e-4), ("2e-4 to 4e-4", 2e-4, 4e-4), ("4e-4 to 1e-3", 4e-4, 1.1e-3)):
        of = [f for f in done if low <= f["lr"] < high]
        scored = [f for f in of if f.get("agree") is not None]
        bl.plog(f"RESULT ext_table learning_rate={label.replace(' ', '_')} runs={len(of)} "
                f"best_val={bl.fmt(min((f['best_val'] for f in of), default=None))} best_agree={max((f['agree'] for f in scored), default=None)}/126")
    for key in sorted({config_key(f) for f in done}):
        rows = [f for f in done if config_key(f) == key and f.get("agree") is not None]
        if len(rows) >= 2:
            bl.plog(f"RESULT ext_table noise config={key} seeds={'/'.join(str(f.get('seed', 42)) for f in rows)} "
                    f"val={'/'.join(bl.fmt(f['best_val']) for f in rows)} agree={'/'.join(str(f['agree']) for f in rows)} "
                    f"val_difference={abs(rows[0]['best_val'] - rows[1]['best_val']):.4f} agree_difference={abs(rows[0]['agree'] - rows[1]['agree'])}")
    final_rows = of_config(choice["key"])
    best = sorted(final_rows, key=lambda f: (-f["agree"], f["best_val"]))[0]
    if not best.get("adapter_path"):
        pulled = pull_adapter(best["worker"].split("#")[0], best["name"])
        if pulled:
            best.update(pulled)
    stop_step = best.get("best_step") or bl.FULL_STEPS
    changed = bool(choice["replaced"])
    chosen["previous_choice"] = chosen["chosen"]["name"] if changed else None
    chosen["extended_search"] = {"runs": runs, "choice": choice, "until_utc": args.until_utc}
    chosen["chosen"] = {**chosen["chosen"], **{k: best.get(k) for k in ("name", "lr", "rank", "alpha", "dropout", "warmup_ratio", "best_val",
                                                                        "final_val", "passes", "wall_s", "worker", "adapter_path",
                                                                        "adapter_sha256", "seed")},
                        "max_steps": bl.FULL_STEPS, "best_step": stop_step,
                        "best_passes": best.get("best_passes") or stop_step * bl.EFFECTIVE_BATCH,
                        "val_agreement": best.get("val_agreement") or {"correct": best.get("agree"), "n": 126},
                        "mean_val_agreement_over_seeds": mean_agree(choice["key"]), "seeds_run": len(final_rows)}
    chosen["recipe_flags"] = ["--max-steps", str(bl.FULL_STEPS), "--stop-step", str(stop_step), "--lr", f"{best['lr']:.6g}",
                              "--rank", str(best["rank"]), "--alpha", str(best["alpha"]), "--dropout", f"{best['dropout']:.4f}",
                              "--warmup-ratio", f"{best['warmup_ratio']:.4f}", "--batch", "4", "--grad-accum", "4",
                              "--seed", str(best.get("seed", 42))]
    chosen["made_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (bl.BO / "chosen.json").write_text(json.dumps(chosen, indent=2) + "\n", encoding="utf-8")
    save()
    if changed:
        bl.plog(f"RESULT extended search: NEW CHOICE {choice['key']} (mean validation agreement {mean_agree(choice['key']):.1f}/126 over "
                f"{len(final_rows)} seeds); it replaces {choice['replaced'][0]['key']} ({choice['replaced'][0]['mean']:.1f}); adapter handed on: "
                f"{best['name']} at {best.get('adapter_path')}")
    else:
        bl.plog(f"RESULT extended search: the choice stands: {choice['key']} (mean validation agreement {mean_agree(choice['key']):.1f}/126 "
                f"over {len(final_rows)} seeds); {len(done) - 3} more full-length runs were scored and none replaced it")
    bl.plog(bl.final_adapter_line({**best, "product_sha256": best.get("product_sha256"), "best_step": stop_step,
                                   "val_agreement": chosen["chosen"]["val_agreement"]}, chosen=True))
    bl.plog(f"RESULT chosen recipe (final): {' '.join(chosen['recipe_flags'])}")
    if not args.skip_held_out:
        be.read_held_out(best["worker"].split("#")[0], best["name"])
    bl.plog("extended-search driver done: every run's metrics, every candidate adapter and the final choice are on this machine")
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        if code != 0:
            bl.plog(f"PROBLEM extended-search driver exited with code {code}; sessions are left up")
    sys.exit(code)
