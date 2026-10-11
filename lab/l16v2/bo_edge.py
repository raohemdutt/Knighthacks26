"""The edge round of the L16 retrain's search: wider and coarser where the first twelve trials showed it was needed.

    <a Python with optuna> lab/l16v2/bo_edge.py --sessions l16v2-a100,l16v2-a100-b [--budget-s 1200]

Run from the repo root after lab/l16v2/bo_local.py's search (it reads that search's bo.json and, when it exists,
chosen.json under <workspace>/work/L16v2-out/bo/). Same fidelity (24 optimizer steps, validation at 12 and 24),
same step-12 rule (the median of all earlier step-12 losses, the first search's included), same asynchronous
ask-and-tell from the workstation, one Colab session per worker. The rules are lab/L16V2_RULES.md amendment 5.

What changes, and why (the first search's table): rank 32 was never tried and rank 16 took 8 of 12 trials; the best
24-step trial (12) sat near the top of the learning-rate range, the bottom of the ranks, the top of dropout and the
bottom of warmup at once; every learning rate below 8e-5 lost; and a 24-step trial cannot resolve a warmup
difference of 0.04 (under one optimizer step).

    space      learning rate 8e-5 to 1e-3 (log); rank in {2, 4, 8, 16, 32, 64}; alpha = rank x {1, 2}; dropout in
               {0, 0.05, 0.10, 0.15}; warmup in {0, 0.05, 0.10}
    study      Optuna cannot change a categorical's choices or a parameter's type inside one study, so the round
               continues in study `l16v2-edge` in the same database file, into which the first search's trials are
               added first as finished trials (completed or pruned, as they were), their dropout and warmup rounded
               to the nearest of the new values; the four whose learning rate is below the new floor cannot be
               stored in the new space and are left out (they all lost)
    forced     FORCED, enqueued first (kind edge): the corners the sampler skipped
    noise      NOISE (kind noise, not study trials, never pruned): trial 9's configuration with another seed, and
               the seed configuration as a proper 24-step trial; the two differences are the noise floor of a trial
    then       the sampler chooses (kind model) until the budget, 20 minutes from the first edge trial, is used
    a session  starts taking edge trials the moment its finalist's adapter is on the workstation and verified

Afterwards: if an edge-round trial's value beats the best 24-step value of the first search by more than the noise
floor, the best such configuration trains once at full length as a fourth finalist and the frozen choice rule is
applied again to all full-length runs; if none does, one RESULT line says so and the choice stands. One RESULT
table: best value per rank and per learning-rate decade, the noise floor, the edges now covered. Only then is the
FINAL_CHOICE file left on the chosen adapter's VM and the held-out side read there (lab/l16v2/vm_score.py).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bo_local as bl  # noqa: E402  (the same worker calls, downloads, progress lines and choice helpers)

bl.LAUNCH = bl.COLAB / "launch-edge"      # its own folder for the small files it runs on the VMs (the first driver may still be polling)
STUDY_NAME = "l16v2-edge"
LR = (8e-5, 1e-3)
RANKS = (2, 4, 8, 16, 32, 64)
ALPHA_MULTS = (1, 2)
DROPOUTS = (0.0, 0.05, 0.10, 0.15)
WARMUPS = (0.0, 0.05, 0.10)
FORCED = [{"lr": 2e-4, "rank": 32}, {"lr": 4e-4, "rank": 16}, {"lr": 4e-4, "rank": 32}, {"lr": 8e-4, "rank": 4},
          {"lr": 4e-4, "rank": 2}]      # each with alpha twice the rank, dropout 0.05, warmup 0.05
NOISE_SEED = 43
TRIAL_ALONE_S = 280.0     # a 24-step trial alone on a card, with its two validation passes and the model load
EDGE_JSON = "bo-edge.json"


def nearest(value: float, choices: tuple[float, ...]) -> float:
    return min(choices, key=lambda choice: (abs(choice - value), choice))


def suggest(trial: Any) -> dict[str, Any]:
    return {"lr": trial.suggest_float("lr", LR[0], LR[1], log=True),
            "rank": trial.suggest_categorical("rank", list(RANKS)),
            "alpha_mult": trial.suggest_categorical("alpha_mult", list(ALPHA_MULTS)),
            "dropout": trial.suggest_categorical("dropout", list(DROPOUTS)),
            "warmup": trial.suggest_categorical("warmup", list(WARMUPS))}


def flags_of(params: dict[str, Any], seed: int | None = None) -> list[str]:
    return ["--max-steps", str(bl.FIDELITY_STEPS), "--eval-steps", str(bl.EVAL_AT), "--lr", f"{params['lr']:.6g}",
            "--rank", str(params["rank"]), "--alpha", str(params["rank"] * params["alpha_mult"]),
            "--dropout", f"{params['dropout']:.4f}", "--warmup-ratio", f"{params['warmup']:.4f}",
            *(["--seed", str(seed)] if seed is not None else [])]


def decade(lr: float) -> str:
    return "8e-5 to 1e-4" if lr < 1e-4 else "1e-4 to 2e-4" if lr < 2e-4 else "2e-4 to 4e-4" if lr < 4e-4 else "4e-4 to 1e-3"


def read_held_out(session: str, name: str) -> int:
    """After the FINAL choice: leave the FINAL_CHOICE file on the chosen adapter's VM and read the held-out side
    there (base against chosen, transformers bf16, both datasets), then pull the product."""
    bl.plog(f"decision the final choice is written ({name}); the A100 held-out reading starts on {session} (a secondary "
            "reading: the reported numbers are pod D's, on the packaged GGUF)")
    command = (f"touch /content/l16v2/FINAL_CHOICE; rm -f /content/l16v2/scores/held_out.FAILED /content/l16v2/markers/HELD_OUT_DONE; "
               f"cd /content/Knighthacks26 && setsid nohup python3 /content/vm_score.py held_out {name} "
               f"> /content/l16v2/score-held-out.log 2>&1 &")
    code = "import subprocess" + chr(10) + f"subprocess.Popen({command!r}, shell=True)" + chr(10) + "print('LAUNCHED')" + chr(10)
    if bl.remote(session, code, "LAUNCHED", "held-out") is None:
        return 1
    began = time.time()
    while time.time() - began < 2400:
        time.sleep(30)
        answer = bl.shell(session, "cat /content/l16v2/status.txt 2>/dev/null; echo; ls /content/l16v2/markers/HELD_OUT_DONE "
                                   "/content/l16v2/scores/held_out.FAILED 2>/dev/null; grep -a held_out_bf16 /content/l16v2/finished.txt 2>/dev/null; "
                                   "tail -n 1 /content/l16v2/score-held-out.log | cut -c1-160", "held-out-wait")
        if answer is None:
            continue
        first = next((line for line in answer.splitlines() if line.startswith("stage=held_out")), "stage=held_out_bf16 starting")
        bl.plog(f"{first} worker={session}")
        if "held_out.FAILED" in answer:
            bl.plog(f"PROBLEM worker={session} held-out bf16 scoring failed: {answer.strip()[-200:]}")
            return 1
        if "HELD_OUT_DONE" in answer:
            for row in answer.splitlines():
                if row.startswith("held_out_bf16 "):
                    bl.plog("RESULT " + row + " engine=transformers_bf16_on_the_A100")
            return 0 if bl.get(session, "held-out-bf16.tar.gz") else 1
    return 1


def main() -> int:
    if "--held-out" in sys.argv:      # bo_edge.py --held-out <session> <chosen run name>
        at = sys.argv.index("--held-out")
        return read_held_out(sys.argv[at + 1], sys.argv[at + 2])
    import optuna
    from optuna.distributions import CategoricalDistribution, FloatDistribution
    from optuna.trial import TrialState

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--budget-s", type=float, default=1200.0)
    parser.add_argument("--no-wait", action="store_true", help="do not wait for each session's finalist adapter line")
    parser.add_argument("--corners-only", action="store_true",
                        help="after the budget: run only the forced corners still waiting in the study, for the "
                             "coverage table; they are marked as run after the budget and cannot name a finalist")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sessions = [name for name in args.sessions.split(",") if name]
    first = json.loads((bl.BO / "bo.json").read_text(encoding="utf-8"))["trials"]
    ran = [t for t in first if t.get("max_steps") == bl.FIDELITY_STEPS]
    best24 = min(t["value"] for t in ran if t.get("value") is not None and t.get("outcome") != "pruned")
    storage = bl.build_storage(bl.BO / "study.db")
    try:
        existing = len(optuna.load_study(study_name=STUDY_NAME, storage=storage).trials)
    except KeyError:
        existing = 0
    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, constant_liar=True, seed=bl.SEED + len(first) + existing,
                                         n_startup_trials=bl.RANDOM_STARTUP)
    study = optuna.create_study(direction="minimize", study_name=STUDY_NAME, sampler=sampler, storage=storage,
                                load_if_exists=True)
    distributions = {"lr": FloatDistribution(LR[0], LR[1], log=True), "rank": CategoricalDistribution(list(RANKS)),
                     "alpha_mult": CategoricalDistribution(list(ALPHA_MULTS)),
                     "dropout": CategoricalDistribution(list(DROPOUTS)), "warmup": CategoricalDistribution(list(WARMUPS))}
    lock = threading.RLock()
    trials: list[dict[str, Any]] = []
    noise: dict[str, dict[str, Any]] = {}
    mid = {t["number"]: float(t["val_at_12"]) for t in first if t.get("val_at_12") is not None}
    state: dict[str, Any] = {"began": None, "stopped": False, "numbers": max(t["number"] for t in first), "longest": 277.0}
    if (bl.BO / EDGE_JSON).exists():       # a restarted round: what it had already finished comes back
        saved = json.loads((bl.BO / EDGE_JSON).read_text(encoding="utf-8"))
        trials, noise = saved.get("trials", []), saved.get("noise", {})
        state["began"] = saved.get("began")
        state["numbers"] = max([state["numbers"], *(t["number"] for t in trials)])
        mid.update({t["number"]: float(t["val_at_12"]) for t in trials if t.get("val_at_12") is not None})

    def save() -> None:
        (bl.BO / EDGE_JSON).write_text(json.dumps({
            "rule": "lab/L16V2_RULES.md amendment 5", "study": STUDY_NAME, "began": state["began"], "budget_s": args.budget_s,
            "space": {"lr": LR, "rank": RANKS, "alpha_mult": ALPHA_MULTS, "dropout": DROPOUTS, "warmup": WARMUPS},
            "forced": FORCED, "best_24_step_value_before": best24, "noise": noise,
            "trials": sorted(trials, key=lambda t: t["number"])}, indent=2) + "\n", encoding="utf-8")

    left_out = []
    if not existing:
        for t in sorted(first, key=lambda t: t["number"]):
            if not LR[0] <= t["lr"] <= LR[1] or t["rank"] not in RANKS or t.get("value") is None:
                left_out.append(t["number"])
                continue
            params = {"lr": float(t["lr"]), "rank": int(t["rank"]), "alpha_mult": int(t["alpha_mult"]),
                      "dropout": nearest(t["dropout"], DROPOUTS), "warmup": nearest(t["warmup_ratio"], WARMUPS)}
            pruned = t.get("outcome") == "pruned"
            study.add_trial(optuna.trial.create_trial(
                params=params, distributions=distributions, state=TrialState.PRUNED if pruned else TrialState.COMPLETE,
                value=None if pruned else float(t["value"]),
                intermediate_values={bl.EVAL_AT: float(t["val_at_12"])} if t.get("val_at_12") is not None else {},
                user_attrs={"kind": t["kind"], "from_first_search_trial": t["number"], "name": t["name"]}))
        for corner in FORCED:
            study.enqueue_trial({"lr": corner["lr"], "rank": corner["rank"], "alpha_mult": 2, "dropout": 0.05, "warmup": 0.05},
                                skip_if_exists=True)
        bl.plog(f"decision edge round: study {STUDY_NAME} in the same database; {len(first) - len(left_out)} of the first search's "
                f"{len(first)} trials added as finished trials (dropout and warmup rounded to the new coarse values); left out, "
                f"below the new learning-rate floor: trials {left_out}; {len(FORCED)} forced corners enqueued; best 24-step value "
                f"so far {best24:.4f}")
    save()
    # the order of work: two forced corners, the two noise repeats, the other three corners, then the sampler
    plan = ["ask", "ask", "noise:trial9", "noise:seed24", "ask", "ask", "ask"]
    done_plan = {"asked": sum(t["kind"] in ("edge", "model") for t in trials), "noise": set(noise)}
    plan = [item for item in plan if not (item.startswith("noise:") and item.split(":")[1] in done_plan["noise"])]
    for _ in range(min(done_plan["asked"], plan.count("ask"))):
        plan.remove("ask")
    trial9 = next(t for t in first if t["number"] == 9)
    seed1 = next(t for t in first if t.get("full_length"))

    def finalist_here(session: str) -> bool:
        text = (bl.OUT / "progress.log").read_text(encoding="ascii", errors="replace")
        return any("RESULT final adapter name=final" in line and f"worker={session}" in line for line in text.splitlines())

    def worker(session: str) -> None:
        while not args.no_wait and not finalist_here(session):     # the card is still training its finalist
            time.sleep(5)
        bl.plog(f"decision worker={session} starts taking edge-round trials (its finalist's adapter is here and verified)")
        while True:
            with lock:
                if state["stopped"]:
                    return
                if state["began"] is None:
                    state["began"] = time.time()
                elapsed = time.time() - state["began"]
                still_waiting = len(study.get_trials(deepcopy=False, states=(TrialState.WAITING,)))
                if args.corners_only:
                    if not still_waiting:
                        return
                elif elapsed + TRIAL_ALONE_S > args.budget_s:
                    state["stopped"] = True
                    bl.plog(f"decision edge round stops proposing: budget ({int(elapsed)} s of {int(args.budget_s)} s used)")
                    return
                item = "ask" if args.corners_only else plan.pop(0) if plan else "ask"
                seen = sum(t.state in (TrialState.COMPLETE, TrialState.PRUNED) for t in study.get_trials(deepcopy=False))
                state["numbers"] += 1
                number = state["numbers"]
                trial = None
                if item == "noise:trial9":
                    params = {"lr": trial9["lr"], "rank": trial9["rank"], "alpha_mult": trial9["alpha_mult"],
                              "dropout": trial9["dropout"], "warmup": trial9["warmup_ratio"]}
                    kind, seed, name = "noise", NOISE_SEED, f"n{number:02d}-trial9-seed{NOISE_SEED}"
                elif item == "noise:seed24":
                    params = {"lr": seed1["lr"], "rank": seed1["rank"], "alpha_mult": seed1["alpha_mult"],
                              "dropout": seed1["dropout"], "warmup": seed1["warmup_ratio"]}
                    kind, seed, name = "noise", None, f"n{number:02d}-seedconfig-24steps"
                else:
                    waiting = [t.system_attrs.get("fixed_params") for t in study.get_trials(deepcopy=False, states=(TrialState.WAITING,))]
                    trial = study.ask()
                    params = suggest(trial)
                    forced = any(fixed and all(params.get(k) == v for k, v in fixed.items()) for fixed in waiting)
                    kind, seed = ("edge" if forced else "model"), None
                    if args.corners_only:
                        kind = "edge-after-budget"
                    name = "e" + bl.trial_name(number, params)[1:]
                    for key, value in (("kind", kind), ("seen", seen), ("worker", session), ("name", name)):
                        trial.set_user_attr(key, value)
                fact = {"number": number, "kind": kind, "seen": seen, "worker": session, "name": name, "lr": params["lr"],
                        "rank": params["rank"], "alpha": params["rank"] * params["alpha_mult"], "alpha_mult": params["alpha_mult"],
                        "dropout": params["dropout"], "warmup_ratio": params["warmup"], "max_steps": bl.FIDELITY_STEPS,
                        "seed": seed or 42, "what": item}

            def line(status: dict[str, Any], fact: dict[str, Any] = fact) -> str:
                latest = status["val"][max(status["val"])] if status["val"] else None
                return (f"stage=edge trial={fact['number']} kind={fact['kind']} seen={fact['seen']} worker={fact['worker']} "
                        f"lr={fact['lr']:.3g} rank={fact['rank']} alpha={fact['alpha']} drop={fact['dropout']:.3f} "
                        f"warm={fact['warmup_ratio']:.3f} step={status['step']}/{bl.FIDELITY_STEPS} loss={bl.fmt(status['loss'])} "
                        f"val={bl.fmt(latest)} best24={best24:.4f} step_s={bl.fmt(status['step_s'], 2)}")

            def at_mid(loss: float, fact: dict[str, Any] = fact, trial: Any = trial) -> bool:
                with lock:
                    earlier = [value for number, value in mid.items() if number != fact["number"]]
                    mid[fact["number"]] = float(loss)
                    if trial is None:              # a noise repeat is never pruned: its step-24 value is the point
                        return False
                    trial.report(float(loss), bl.EVAL_AT)
                    fact["median_at_12"] = statistics.median(earlier)
                    bl.plog(f"decision trial {fact['number']} step-12 loss {loss:.4f} against the median {fact['median_at_12']:.4f} "
                            f"of {len(earlier)} earlier step-12 losses: {'prune' if loss > fact['median_at_12'] else 'keep'}")
                    return loss > fact["median_at_12"]

            fact.update(bl.run_on(session, name, flags_of(params, seed), False, line, at_mid))
            with lock:
                if fact["outcome"] == "pruned" and fact.get("val_at_12") is not None:
                    fact["value"] = fact["val_at_12"]
                elif fact.get("ok") and fact.get("best_val") is not None:
                    fact["value"] = fact["best_val"]
                else:
                    fact["value"] = None
                if trial is not None:
                    if fact["outcome"] == "pruned":
                        study.tell(trial, state=TrialState.PRUNED)
                    elif fact["value"] is not None:
                        study.tell(trial, float(fact["value"]))
                    else:
                        study.tell(trial, state=TrialState.FAIL)
                    trials.append(fact)
                else:
                    noise[item.split(":")[1]] = fact
                if fact.get("wall_s") and fact["outcome"] != "lost":
                    state["longest"] = max(state["longest"], fact["wall_s"])
                save()
                bl.plog(f"RESULT edge trial={fact['number']} kind={fact['kind']} seen={fact['seen']} worker={fact['worker']} "
                        f"name={fact['name']} lr={fact['lr']:.3g} rank={fact['rank']} alpha={fact['alpha']} "
                        f"drop={fact['dropout']:.3f} warm={fact['warmup_ratio']:.3f} seed={fact['seed']} outcome={fact['outcome']} "
                        f"val12={bl.fmt(fact.get('val_at_12'))} val24={bl.fmt(fact.get('val_at_24'))} value={bl.fmt(fact.get('value'))} "
                        f"best24_before={best24:.4f} wall_s={fact.get('wall_s')}")
                if fact["outcome"] == "lost":
                    return

    threads = [threading.Thread(target=worker, args=(session,), daemon=True) for session in sessions]
    for thread in threads:
        thread.start()
        time.sleep(2)
    for thread in threads:
        thread.join()
    if args.corners_only:      # the round's own end (noise floor, tables, the fourth-finalist question) is already written
        return 0

    # --- the noise floor, the table, and whether there is a fourth finalist ----------------------------------------
    differences = {}
    if noise.get("trial9", {}).get("value") is not None:
        differences["trial 9's configuration, seed 43 against seed 42"] = abs(noise["trial9"]["value"] - trial9["value"])
    if noise.get("seed24", {}).get("value") is not None:
        differences["the seed configuration, a 24-step trial against step 24 of its 62-step run"] = abs(noise["seed24"]["value"] - seed1["value"])
    floor = max(differences.values()) if differences else None
    bl.plog("RESULT edge noise floor of a 24-step trial: " + ("; ".join(f"{what}: {value:.4f}" for what, value in differences.items())
            + f"; the floor used is the larger, {floor:.4f}" if differences else "not measured (no noise repeat finished)"))
    everything = [t for t in [*ran, *trials] if t.get("value") is not None]
    done = [t for t in everything if t.get("outcome") != "pruned"]
    for rank in RANKS:
        of = [t for t in everything if t["rank"] == rank]
        finished = [t["value"] for t in of if t.get("outcome") != "pruned"]
        bl.plog(f"RESULT edge_table rank={rank} trials={len(of)} completed={len(finished)} best_value={bl.fmt(min(finished) if finished else None)} "
                f"best_val12={bl.fmt(min((t['val_at_12'] for t in of if t.get('val_at_12') is not None), default=None))}")
    for label in ("8e-5 to 1e-4", "1e-4 to 2e-4", "2e-4 to 4e-4", "4e-4 to 1e-3"):
        of = [t for t in everything if t["lr"] >= LR[0] and decade(t["lr"]) == label]
        finished = [t["value"] for t in of if t.get("outcome") != "pruned"]
        bl.plog(f"RESULT edge_table learning_rate={label.replace(' ', '_')} trials={len(of)} completed={len(finished)} "
                f"best_value={bl.fmt(min(finished) if finished else None)}")
    below = [t for t in everything if t["lr"] < LR[0]]
    bl.plog(f"RESULT edge_table learning_rate=below_8e-5 trials={len(below)} completed={sum(t.get('outcome') != 'pruned' for t in below)} (all from the first search)")
    covered = sorted({(t["rank"], f"{t['lr']:.0e}") for t in trials if t["kind"] == "edge"})
    bl.plog(f"RESULT edge_table edges now covered by a trial: {covered}; highest learning rate tried {max(t['lr'] for t in everything):.3g}; "
            f"ranks tried {sorted({t['rank'] for t in everything})}")
    winners = sorted((t for t in trials if t.get("value") is not None and t.get("outcome") != "pruned"
                      and t["kind"] in ("edge", "model") and t["value"] < best24 - (floor or 0.0)),
                     key=lambda t: (t["value"], t["number"]))
    save()
    chosen_path = bl.BO / "chosen.json"
    if not winners or floor is None:
        best_edge = min((t["value"] for t in trials if t.get("value") is not None and t.get("outcome") != "pruned"), default=None)
        bl.plog(f"RESULT edge round: no edge-round trial beats the best 24-step value ({best24:.4f}) by more than the noise floor "
                f"({bl.fmt(floor)}); best edge-round value {bl.fmt(best_edge)}; there is no fourth finalist and the choice stands")
        fourth = None
    else:
        fourth = winners[0]
        bl.plog(f"decision edge round: trial {fourth['number']} ({fourth['kind']}) value {fourth['value']:.4f} beats {best24:.4f} by more "
                f"than the noise floor {floor:.4f}: it trains at full length as a fourth finalist on {fourth['worker']}")
    result = {"fourth": fourth, "floor": floor, "differences": differences, "best24": best24}
    (bl.BO / "edge-result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if fourth is None:
        return 0
    session = fourth["worker"].split("#")[0]
    name = f"final4-{fourth['name']}"
    flags = ["--max-steps", str(bl.FULL_STEPS), "--eval-steps", str(bl.FULL_EVAL), "--early-stop", str(bl.FULL_PATIENCE),
             "--early-stop-min-rel", str(bl.FULL_MIN_REL), "--lr", f"{fourth['lr']:.6g}", "--rank", str(fourth["rank"]),
             "--alpha", str(fourth["alpha"]), "--dropout", f"{fourth['dropout']:.4f}", "--warmup-ratio", f"{fourth['warmup_ratio']:.4f}"]
    fact = bl.run_on(session, name, flags, True, lambda status: (
        f"stage=finalists final=4 from_trial={fourth['number']} worker={session} lr={fourth['lr']:.3g} rank={fourth['rank']} "
        f"alpha={fourth['alpha']} drop={fourth['dropout']:.3f} warm={fourth['warmup_ratio']:.3f} step={status['step']}/{bl.FULL_STEPS} "
        f"val={bl.fmt(status['val'][max(status['val'])] if status['val'] else None)}"))
    fact.update(place=4, from_trial=fourth["number"], trial_kind=fourth["kind"], trial_value=fourth["value"], lr=fourth["lr"],
                rank=fourth["rank"], alpha=fourth["alpha"], alpha_mult=fourth["alpha_mult"], dropout=fourth["dropout"],
                warmup_ratio=fourth["warmup_ratio"], max_steps=bl.FULL_STEPS, full_length=True, worker=session)
    adapter = bl.COLAB / "runs" / name / "adapter" / "adapter_model.safetensors"
    if not (fact.get("ok") and adapter.exists()):
        bl.plog(f"PROBLEM the fourth finalist ({name}) did not give an adapter: outcome={fact.get('outcome')}; the choice stands")
        return 0
    fact["adapter_path"] = str(adapter.parent).replace("\\", "/")
    fact["adapter_sha256"] = bl.sha256_of(adapter)
    bl.plog(bl.final_adapter_line(fact))
    # the frozen choice rule again, over all full-length runs
    chosen = json.loads(chosen_path.read_text(encoding="utf-8"))
    runs = [*chosen["full_length_runs"], {k: fact.get(k) for k in chosen["full_length_runs"][0]}]
    ranked = sorted(runs, key=lambda f: (f["best_val"], f.get("best_passes") or 0, f.get("place") or 9))
    scored = ranked[:bl.TOP]
    if any(f["name"] == name for f in scored):
        answer = bl.score_on(session, "val-final4", [name])
        for f in runs:
            if f["name"] == name and answer:
                f["val_agreement"] = answer.get(name)
    with_score = [f for f in scored if f.get("val_agreement")]
    best = sorted(with_score, key=lambda f: (-f["val_agreement"]["correct"], f["best_val"], f.get("best_passes") or 0))[0] if with_score else None
    if best is None:
        bl.plog("PROBLEM the choice could not be applied again (no validation agreement); the earlier choice stands")
        return 0
    stop_step = best.get("best_step") or bl.FULL_STEPS
    chosen.update(full_length_runs=ranked, top=[{"name": f["name"], "best_val": f["best_val"], "val_agreement": f.get("val_agreement")} for f in scored],
                  previous_choice=chosen["chosen"]["name"], made_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    chosen["chosen"] = {**{k: best.get(k) for k in ("name", "lr", "rank", "alpha", "dropout", "warmup_ratio", "best_val", "final_val",
                                                    "passes", "wall_s", "val_agreement", "worker", "from_trial", "trial_kind",
                                                    "adapter_path", "adapter_sha256")},
                        "max_steps": bl.FULL_STEPS, "best_step": stop_step,
                        "best_passes": best.get("best_passes") or stop_step * bl.EFFECTIVE_BATCH}
    chosen["recipe_flags"] = ["--max-steps", str(bl.FULL_STEPS), "--stop-step", str(stop_step), "--lr", f"{best['lr']:.6g}",
                              "--rank", str(best["rank"]), "--alpha", str(best["alpha"]), "--dropout", f"{best['dropout']:.4f}",
                              "--warmup-ratio", f"{best['warmup_ratio']:.4f}", "--batch", "4", "--grad-accum", "4", "--seed", "42"]
    chosen_path.write_text(json.dumps(chosen, indent=2) + "\n", encoding="utf-8")
    bl.plog(f"RESULT the frozen choice rule applied again with the fourth finalist: chosen {best['name']} "
            f"(validation agreement {best['val_agreement']['correct']}/{best['val_agreement']['n']}, validation loss {bl.fmt(best['best_val'])}); "
            f"before the edge round it was {chosen['previous_choice']}")
    bl.plog(bl.final_adapter_line({**best, "best_step": stop_step, "product_sha256": best.get("product_sha256")}, chosen=True))
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        if code != 0:
            bl.plog(f"PROBLEM edge-round driver exited with code {code}; sessions are left up")
    sys.exit(code)
