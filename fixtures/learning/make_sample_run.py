"""Writes fixtures/learning/sample_run/: the Learning lab's SAMPLE run. Nothing in it was measured.

    python fixtures/learning/make_sample_run.py          # rewrites the folder (deterministic: same bytes every time)

The Learning lab screen shows it only when the box's own log has no training run, or with `?sample=1`,
always under a "Sample run" banner. Every line and file written here says `"sample": true`.

What is real in it: the questions and the expected tool calls, taken from the committed no-camera dataset
(`tool-calling-fixture/`, built from the golden questions). What is made up: every model answer, score,
loss, watt, joule, time and the model package. The made-up numbers obey the lab's own rules, so the screen
shows how a real run will read:

- the split keeps whole question groups on one side (60 examples in 8 groups train, 52 in 8 groups held
  out: at least the 50 items the gate needs, lab/refine/learning/OWNER_RULINGS.md ruling 1);
- scores are counts over n, the p-value is the exact two-sided sign-flip test over the 8 held-out groups,
  noise is the spread of three repeated scorings of the current model, and the decision is "deploy" only
  when n >= 50, the gain is larger than the noise and p < 0.05;
- the gate scores the packaged model, so packaging comes before the eval (ruling 5);
- energy is each step's mean watts times its length, and the totals are their sum.

Two runs of one task: the first (learning rate too low) is not proven and is not deployed; the second is
deployed. `contexts/learning/tests/test_lab_sample.py` checks all of the above against the written files.
"""

import hashlib
import json
import math
import random
import sys
from datetime import datetime, timedelta
from fractions import Fraction
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from contracts.agent import ActionProposed  # noqa: E402
from contracts.learning import DatasetExample, DatasetExported, DatasetManifest, ModelDeployed, RunEvaluated, RunProgress, \
    TaskCreated, TaskSpec  # noqa: E402
from contracts.telemetry import JoulesPerAnswer, RequestWindow, TrainStep  # noqa: E402
from contracts.workspace import ActionCompleted  # noqa: E402

FIXTURE = HERE / "tool-calling-fixture"
OUT = HERE / "sample_run"
DATASET_ID = "tool-calling-sample"
TASK = "tool-calling"
BASE = "Qwen/Qwen3.5-4B"
CURRENT = "atlas-base"
ACTION_ID = "sample-export-approval"
NODE = "pod-b"
BATCH = 4
EPOCHS = 4
ALPHA = 0.05
MIN_HELD_OUT = 50

# group -> (wordings held out, right before, then per run: (still right, broke, fixed, still wrong))
HELD_OUT = {
    "episodes_between-3fa0ffba": (7, 3, {"v1": (2, 1, 2, 2), "v2": (2, 1, 4, 0)}),
    "last_seen-23377f1f": (7, 4, {"v1": (3, 1, 1, 2), "v2": (4, 0, 2, 1)}),
    "episodes_between-dbc6dbec": (7, 2, {"v1": (2, 0, 1, 4), "v2": (2, 0, 3, 2)}),
    "last_seen-13a6314c": (7, 3, {"v1": (2, 1, 0, 4), "v2": (3, 0, 3, 1)}),
    "last_seen-d557cbb3": (6, 2, {"v1": (2, 0, 1, 3), "v2": (2, 0, 3, 1)}),
    "last_seen-a9bb595a": (6, 3, {"v1": (2, 1, 0, 3), "v2": (1, 2, 1, 2)}),
    "episodes_between-ce366c7c": (6, 2, {"v1": (2, 0, 1, 3), "v2": (2, 0, 3, 1)}),
    "episodes_between-e2d8a484": (6, 2, {"v1": (2, 0, 0, 4), "v2": (2, 0, 4, 0)}),
}
TRAIN = ["last_seen-8ee9a95f", "last_seen-d8f50a0a", "episodes_between-bc28f108", "episodes_between-2b15472a",
         "episodes_between-09f9a909", "episodes_between-8314bd12", "episodes_between-e311a374", "episodes_between-9b7c9695"]
NOISE_RUNS = [21, 21, 22]  # the current model scored three times on the 52 held-out questions
TEACHER_RIGHT = 47

RUNS = {
    "v1": {"run_id": "sample-run-1-not-proven", "version": "sample-atlas-v1-tool-calling",
           "start": datetime.fromisoformat("2026-10-10T11:20:00-04:00"), "labeling": True, "seed": 11,
           "loss": lambda k: 1.32 + 0.99 * math.exp(-k / 26), "note": "learning rate 2e-5"},
    "v2": {"run_id": "sample-run-2-deployed", "version": "sample-atlas-v2-tool-calling",
           "start": datetime.fromisoformat("2026-10-10T14:05:00-04:00"), "labeling": False, "seed": 7,
           "loss": lambda k: 0.31 + 2.0 * math.exp(-k / 11), "note": "learning rate 2e-4"},
}
TAUGHT = datetime.fromisoformat("2026-10-10T11:10:00-04:00")


def call_text(call: dict) -> str:
    """A tool call as one short line: last_seen(label="keys")."""
    args = ", ".join(f'{k}="{str(v)[:16] if k in ("start", "end") else v}"' for k, v in call["arguments"].items())
    return f'{call["name"]}({args})'


def wrong_answer(expected: dict, question: str, i: int) -> str:
    """A plausible miss: the wrong tool, a wrong argument, or no tool call at all."""
    call = expected["tool_calls"][0]
    if call["name"] == "last_seen":
        label = call["arguments"]["label"]
        options = [f'search_memory(q="{question.rstrip("?").lower()}")', f'last_seen(label="my {label}")',
                   'no tool call: "I didn\'t see that."']
    else:
        end = call["arguments"]["end"]
        options = [f'search_memory(q="{question.rstrip("?").lower()}")',
                   f'episodes_between(start="{end[:10]}T00:00", end="{end[:16]}")',
                   'no tool call: "I didn\'t see that."',
                   f'episodes_between(start="{call["arguments"]["start"][:16]}")']
    options = [o for o in options if o != call_text(call)]  # a miss must differ from the expected call
    return options[i % len(options)]


def sign_flip_p(diffs: list[Fraction]) -> float:
    """Exact two-sided sign-flip test: the share of sign patterns whose summed difference is at least as far from zero."""
    seen = abs(sum(diffs))
    hits = sum(abs(sum(s * d for s, d in zip(signs, diffs))) >= seen for signs in product((1, -1), repeat=len(diffs)))
    return hits / 2 ** len(diffs)


def build() -> tuple[list[dict], dict[str, str]]:
    expected = [json.loads(line) for line in (FIXTURE / "expected.jsonl").read_text().splitlines()]
    examples = {e["id"]: e for e in map(json.loads, (FIXTURE / "examples.jsonl").read_text().splitlines())}
    by_group: dict[str, list[dict]] = {}
    for e in expected:
        by_group.setdefault(e["group"], []).append(e)
    held = [e for g, (n, _, _) in HELD_OUT.items() for e in by_group[g][:n]]
    train = [e for g in TRAIN for e in by_group[g]]
    spec = TaskSpec.model_validate(json.loads((FIXTURE / "manifest.json").read_text())["spec"])

    # --- the dataset folder the box would have written, plus the lab's frozen split ---
    exported_at = TAUGHT + timedelta(seconds=70)
    rows = []
    for e in train + held:
        ex = DatasetExample.model_validate(examples[e["id"]])
        target = json.dumps({"tool_calls": e["label"]["tool_calls"]}, separators=(",", ":")) if e in train else None
        rows.append({**ex.model_copy(update={"target": target}).model_dump(mode="json"), "sample": True})
    manifest = DatasetManifest(dataset_id=DATASET_ID, spec=spec, created=exported_at, examples=len(rows), action_id=ACTION_ID)
    split = {"version": 1, "dataset_id": DATASET_ID, "sample": True,
             "rule": "Sample split: whole question groups on one side. Held out: 8 groups (the 4 golden-question groups, 7 "
                     "wordings each, and 4 variant groups). Train: the other 8 variant groups.",
             "train": [e["id"] for e in train], "held_out": [e["id"] for e in held]}
    files = {
        f"dataset/{DATASET_ID}/manifest.json": json.dumps({**manifest.model_dump(mode="json"), "sample": True}, indent=2) + "\n",
        f"dataset/{DATASET_ID}/examples.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        f"dataset/{DATASET_ID}/split.json": json.dumps(split, indent=2) + "\n",
    }

    events: list[dict] = []

    def emit(e) -> None:
        raw = e if isinstance(e, dict) else e.model_dump(mode="json")
        raw.setdefault("version", 1)
        events.append({**raw, "event_id": f"sample-e{len(events) + 1:04d}", "sample": True})

    emit(TaskCreated(ts=TAUGHT, task=TASK, instruction="Pick the right memory tool, with the right arguments, for a question "
                                                       "about what the cameras saw.", output_schema=spec.output_schema))
    emit(ActionProposed(ts=TAUGHT, action_id=ACTION_ID, kind="export_training_data", target=TASK,
                        reason=f"Send {len(rows)} questions and the tool calls that answer them to teach '{TASK}'",
                        params={"dataset": DATASET_ID}))
    emit(ActionCompleted(ts=exported_at, action_id=ACTION_ID, status="done", result=f"exported {len(rows)} examples as {DATASET_ID}"))
    emit(DatasetExported(ts=exported_at, task=TASK, dataset_id=DATASET_ID, examples=len(rows), action_id=ACTION_ID))

    # --- what the box measured for the model that was serving before any swap ---
    box_rng = random.Random(3)

    def answers(model: str, start: datetime, n: int, watts: float) -> None:
        for i in range(n):
            t0 = start + timedelta(minutes=4 * i, seconds=box_rng.randint(0, 50))
            tokens = box_rng.randint(38, 66)
            first = t0 + timedelta(milliseconds=box_rng.randint(1700, 2200))
            ended = first + timedelta(milliseconds=round(tokens / box_rng.uniform(12.4, 13.6) * 1000))
            w = round(watts + box_rng.uniform(-0.6, 0.6), 2)
            joules = round(w * (ended - t0).total_seconds(), 1)
            rid = f"sample-{model}-{i + 1}"
            emit(RequestWindow(ts=ended, node="jetson", request_id=rid, started=t0, ended=ended, tokens_in=box_rng.randint(820, 960),
                               tokens_out=tokens, target="student", first_token=first, model=model, engine="llama.cpp (sample)"))
            emit(JoulesPerAnswer(ts=ended, node="jetson", request_id=rid, target="student", model=model, started=t0, ended=ended,
                                 clock="jetson", joules=joules, node_energy_j=joules, tokens_out=tokens, idle_w=5.9,
                                 idle_ts=t0 - timedelta(seconds=30), power_field="VDD_IN", boundary="module input",
                                 accelerator="Jetson Orin Nano (sample)", sampler="sample", period_ms=1000.0,
                                 samples=int((ended - t0).total_seconds()) + 1, error=None, low_confidence=True,
                                 reasons=["sample run: not measured"]))

    answers(CURRENT, TAUGHT - timedelta(minutes=40), 8, 13.6)

    held_ids = hashlib.sha256(files[f"dataset/{DATASET_ID}/split.json"].encode()).hexdigest()
    for key, run in RUNS.items():
        rid, version, rng = run["run_id"], run["version"], random.Random(run["seed"])
        t = run["start"]

        def progress(stage: str, p: float, at: datetime, loss: float | None = None, message: str | None = None) -> None:
            emit(RunProgress(ts=at, task=TASK, run_id=rid, stage=stage, progress=p, loss=loss, message=message))

        if run["labeling"]:
            for p, minutes in ((0.0, 0), (0.5, 3), (1.0, 6)):
                progress("labeling", p, t + timedelta(minutes=minutes), message=f"teacher labeled {round(p * len(rows))} of {len(rows)}")
            t += timedelta(minutes=7)

        steps = len(train) * EPOCHS // BATCH
        emit({"type": "learning.training.started", "ts": t.isoformat(), "task": TASK, "run_id": rid, "dataset_id": DATASET_ID,
              "node": NODE, "base": BASE, "steps": steps, "batch": BATCH, "train_examples": len(train), "epochs": EPOCHS,
              "lora_rank": 16, "note": run["note"]})
        progress("training", 0.0, t, message=f"LoRA rank 16 on {BASE}, {run['note']}")
        at = t
        for k in range(1, steps + 1):
            step_ms = 7900 + rng.randint(-160, 160)
            at += timedelta(milliseconds=step_ms)
            loss = round(run["loss"](k - 1) + rng.uniform(-0.05, 0.05) * (0.4 + 1.2 * math.exp(-k / 15)), 3)
            watts = float(round(min(246.0, 168.0 + 30.0 * k) + rng.uniform(-6, 6))) if k < 4 else float(round(244 + rng.uniform(-7, 7)))
            emit(TrainStep(ts=at, node=NODE, run_id=rid, step=k, loss=loss, step_ms=float(step_ms), socket_w=watts,
                           energy_j=round(watts * step_ms / 1000, 1)))
            if k % 10 == 0:
                progress("training", k / steps, at, loss=loss, message=f"step {k} of {steps}")
        t = at

        # packaging first: the gate scores the packaged model (owner ruling 5)
        progress("packaging", 0.0, t + timedelta(seconds=20), message="merging the adapter, converting to GGUF, quantizing to Q4_K_M")
        t += timedelta(minutes=3, seconds=10)
        gguf = f"{version}-Q4_K_M.gguf"
        emit({"type": "learning.package.built", "ts": t.isoformat(), "task": TASK, "run_id": rid, "model_version": version, "base": BASE,
              "gguf": gguf, "bytes": 2_712_446_976 + (0 if key == "v2" else 4096), "quantization": "Q4_K_M",
              "sha256": hashlib.sha256(f"atlas sample run: no model file exists for {version}".encode()).hexdigest()})
        progress("packaging", 1.0, t, message=gguf)

        progress("eval", 0.0, t + timedelta(seconds=15), message=f"scoring {len(held)} held-out questions: current, candidate, teacher")
        t += timedelta(minutes=5, seconds=30)
        items, diffs, right = [], [], {"current": 0, "candidate": 0}
        for group, (n, before_right, plans) in HELD_OUT.items():
            still_right, broke, fixed, still_wrong = plans[key]
            assert still_right + broke == before_right and sum(plans[key]) == n, group
            after_flags = [True] * still_right + [False] * broke + [True] * fixed + [False] * still_wrong
            for i, e in enumerate(by_group[group][:n]):
                good = call_text(e["label"]["tool_calls"][0])
                b, a = i < before_right, after_flags[i]
                items.append({"id": e["id"], "group": group, "question": e["question"], "expected": good,
                              "before": {"answer": good if b else wrong_answer(e["label"], e["question"], i), "correct": b},
                              "after": {"answer": good if a else wrong_answer(e["label"], e["question"], i + 1), "correct": a}})
                right["current"] += b
                right["candidate"] += a
            diffs.append(Fraction(still_right + fixed - before_right, n))
        n = len(items)
        p_value = sign_flip_p(diffs)
        noise = round((max(NOISE_RUNS) - min(NOISE_RUNS)) / n, 4)
        scores = {"current": round(right["current"] / n, 4), "candidate": round(right["candidate"] / n, 4),
                  "teacher": round(TEACHER_RIGHT / n, 4)}
        deploy = n >= MIN_HELD_OUT and scores["candidate"] - scores["current"] > noise and p_value < ALPHA
        emit({"type": "learning.eval.reported", "ts": t.isoformat(), "task": TASK, "run_id": rid, "n_items": n, "n_groups": len(HELD_OUT),
              "correct": {**right, "teacher": TEACHER_RIGHT}, "noise": noise, "noise_runs": NOISE_RUNS, "p_value": p_value,
              "alpha": ALPHA, "test": "exact two-sided sign-flip test over question groups",
              "decode": "thinking off, greedy, max 256 output tokens", "split_sha256": held_ids, "low_confidence": False,
              "items": items})
        emit(RunEvaluated(ts=t, task=TASK, run_id=rid, scores=scores, decision="deploy" if deploy else "reject", model_version=version))
        progress("eval", 1.0, t, message=f"{right['candidate']} of {n} right (was {right['current']})")
        if not deploy:
            progress("done", 1.0, t + timedelta(seconds=5), message="not proven: not deployed")
            continue

        swap_start = t + timedelta(seconds=40)
        progress("deploying", 0.0, swap_start, message="switching the model on the box")
        swap_s = 41.6
        live = swap_start + timedelta(seconds=swap_s)
        emit({"type": "learning.swap.completed", "ts": live.isoformat(), "model_version": version, "previous_version": CURRENT,
              "started": swap_start.isoformat(), "swap_s": swap_s, "outcome": "serving", "reason": None})
        emit(ModelDeployed(ts=live, model_version=version, previous_version=CURRENT))
        progress("done", 1.0, live, message=f"{version} is serving")
        answers(version, live + timedelta(minutes=2), 8, 13.5)

    files["events.jsonl"] = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events)
    return events, files


def main() -> None:
    _, files = build()
    for name, text in files.items():
        path = OUT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(f"wrote {len(files)} files under {OUT}")


if __name__ == "__main__":
    main()
