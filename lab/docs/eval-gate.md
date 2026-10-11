# Eval gate: `python -m lab eval-gate` (L4)

Moved here from lab/README.md (O70); the verbs list there links every page. Arguments: `python -m lab eval-gate -h`.

Scores the current Jetson model, the packaged Q4_K_M candidate (from L5) and the teacher on the held-out side of a
dataset's frozen `split.json`, with Dominick's scorer (`python -m contexts.agent.eval --runtime llama --base-url
<server> --json out.json`), one scorer commit and one decode setting (thinking off, greedy, 512 output tokens),
and decides deploy or reject. The current model is scored at least twice; the spread is the run-to-run noise.

```
python -m lab eval-gate --dataset fixtures/learning/tool-calling-fixture --run-id run-3 \
    --current-url http://127.0.0.1:8080 --candidate-url http://127.0.0.1:8081 --teacher-url http://127.0.0.1:8000 \
    --candidate-manifest data/models/<version>/manifest.json --decode-attested [--preview-url URL] [--events-dir data/events]
```

The paired test is an exact sign-flip test over the question groups' mean differences (group = frame_ref, or the
episode for image tasks via `--episodes`), which is exact McNemar when each group holds one item. It deploys only
with at least 50 scored items, a gain above the noise and p below 0.05; a smaller set is reported, never deployed.
It writes one ledger row per score (`eval-gate.score`) and one for the decision (`eval-gate`), each with the
decode setting, scorer commit and split sha256; `lab/state/eval-gate/<run id>/scores.json` for the slide, a
`<role>.json` per score that `lab/slide_numbers.py --eval-before/--eval-after` reads, and the
`learning.run.evaluated` event. Tests run on `lab/eval_gate_fake.py`; the real run is `lab/L4_REAL_RUN.md`. The
committed tool-calling fixture holds 58 held-out items in only 4 groups, so its smallest possible p is 0.125 and it
can never deploy: a deploy needs at least 6 independent groups, all in favour. The scorer also asks only the golden
questions, so of those 58 it scores the 24 golden wordings (the 34 variants are reported as unscored).
