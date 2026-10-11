# Settings optimizer: `python -m lab optimize` (R45)

Arguments: `python -m lab optimize -h` and `python -m lab optimize run -h`. The full reference is the docstring at
the top of `lab/optimize_settings.py`; the design it implements is R49's note
`lab/refine/research/kernels/fast_optimization.md`.

Drives Ax 1.3.1 with BoTorch over the two committed search spaces, to the night configuration
`lab/refine/research/kernels/data/ax-config-night.json`: the inference lane (R38) over
`searchspace-inference-gfx1151.json` (R35's format) through `lab/bench.py`, the training lane (R39) over
`searchspace-training-gfx1151.json` (R36's format) through a trainer process. Where the configuration and R45's task
differ, the configuration wins: one server start per configuration with its requests as the repeats, warm start
from the baseline and the named prior arms instead of Sobol points, asynchronous trials, one per free worker.

```
python -m lab optimize --dry-run                                  # the inference lane's plan; runs and writes nothing
python -m lab optimize --dry-run --lane training --minutes 120
python -m lab optimize run --lane inference --fake --workers 2 --budget 8          # offline, on the fakes
python -m lab optimize install [--dry-run]                        # pip install -r lab/requirements-optimize.txt
python -m lab optimize run --lane inference --node pod-d --minutes 60 --slot <row> --commit \
    --run-dir lab/refine/research/kernels/data/ax-inference-<date> \
    --bench "--engine vllm --checkpoint ~/models/Qwen3.8-27B-INT4 --engine-dir ~/vllm"
python -m lab optimize confirm --lane inference --node pod-d --minutes 70 --slot <row> --commit --run-dir <the same> --bench "<the same>"
python -m lab optimize summary --run-dir <run dir>                # best arm, posterior, Markdown table, verdict
```

- **Objective.** The inference lane minimizes `joules_per_token.b1` (the harness's gross field at batch 1) with
  `decode_tok_s.b1 >= 2.88` and `ttft_s.b1 <= 2.0` as outcome constraints, read from the lane's
  `outcome_constraints`; the training lane maximizes `samples_per_s`.
- **The search.** The lane's initial arms run first (the baseline, then the prior arms), then each free worker asks
  the strategy for one trial while the others still run. Every second ask of a worker fixes its last compile class,
  so a class is started cold once and reused warm once. A trial whose first measured request is under half the
  baseline's median rate stops early; three failed trials in a row stop a worker; a suggestion that hits an invalid
  combination of the space is recorded as invalid and never run.
- **Stops.** `--budget` trials, this call's `--minutes` (a GPU slot) and the run's minute budget in the
  configuration: no trial starts whose estimate would pass either. The dry run on the committed configuration fits
  17 inference trials in 180 minutes (10 cold and 7 warm starts) and 10 training trials in 120.
- **Files.** `<run-dir>/run.json`, one `trials/trial-NNNN.json` per trial (parameters, launch args and env, worker,
  restart class, fidelity, every request or step, the observation the strategy got, versions, host, git sha,
  timestamps), `confirm/confirm-NN.json`, `summary.json` and `summary.md`. Each finished trial is also an
  `optimize.trial` ledger row. A run resumes from its files: finished trials are replayed into a fresh strategy and
  never run again, a trial in flight at the stop is run again. `--commit` commits each finished file.
- **Confirmation and summary.** After the search, spawns of the best feasible arm and the baseline alternate, then
  `lab/machine/verdict.py`'s Welch test decides (low confidence under 5 spawns an arm). The summary names the best
  arm, the change against the baseline, the model's posterior mean and SEM at every completed trial, and what is
  still owed: the R21 correctness gate on the best arm is not run by this verb.
- **Ax.** `lab/requirements-optimize.txt` pins `ax-platform==1.3.1` and `botorch==0.18.1`, outside `pyproject.toml`;
  `optimize install` installs them and holds the pod's torch where it is. Ax is imported only when a run uses it:
  `--strategy stub` runs everything without it, which is what the tests and CI do.
- **Not built.** Workers whose runner is `hub` (pods E and F) need `lab.controller run-script` from a driver host;
  a real run uses the local worker it runs on. The training lane runs the fake trainer until `lab/train.py` takes
  the trainer contract (a settings file in, `steps.jsonl` and `summary.json` out). Not yet run on a pod.

Tests: `tests/test_lab_optimize_space.py`, `tests/test_lab_optimize_plan.py`, `tests/test_lab_optimize_runners.py`,
`tests/test_lab_optimize_settings.py` (the fakes only; the two Ax tests run where Ax is installed).
