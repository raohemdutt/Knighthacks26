# Routing hook closed loop: `python -m lab routing-closed-loop` (T11)

The same labeling requests routed under T5's two policies, arms interleaved request by request (A, B, A, B): A is
policy one, capability (the control), B is policy two, energy, which reads the telemetry snapshot. The requests are
at least 100 examples of the L11 fixture (`fixtures/learning/tool-calling-fixture`) drawn with a fixed seed, never a
golden question. The snapshot is T4's joules projection, fed the power samples and one window per answer of either
arm, so policy two decides on what the run has just measured; one probe request per node fills it first. Per
request: arm, target, joules per answer, latency (`label.seconds`) and label agreement (`learning.label_agreement`,
the tool calls against `expected.jsonl`); per arm their mean and spread; `lab/machine/verdict.py` says whether B
changed energy or quality beyond the spread.

```
python -m lab routing-closed-loop --fake                  # the teacher and student fakes, on a fake clock
python -m lab routing-closed-loop --account C --power-log data/events/<UTC day>.jsonl --window-clock jetson
```

- Writes `<out>/<run id>/requests.jsonl`, `report.json` and `t0_result.txt` (the T0 experiment result text with its
  citation, for the dispatcher to post), one ledger row per arm (verb `routing-closed-loop`, task T11) and two T0
  experiments with their results. A `--fake` run keeps its ledger in `<out>/<run id>/state`.
- The two experiments are added under ids of the test's own, `T11-energy` and `T11-quality`: the experiments ledger
  hands out an E number only when it already holds one beyond T0's seeds, which a fresh state dir does not.
  `--experiment-ids E<n> E<m>` uses the two numbers T0 gave instead.
- `--metrics-file` also appends the run to `lab/refine/metrics.jsonl`: per arm, the mean of each of the three metrics,
  with `conditions.policy` (`capability` or `energy`) and `conditions.requests`, which is what the Learning lab's
  Routing tab reads as the measured closed loop (`closed_loop` in `contexts/learning/application/lab_routing.py`).
  `--row-condition NAME=VALUE` adds the told conditions of those rows (`gpu`, the hub, the teacher's build). Each
  row names its run under `run_id` (a shared `session` would make it a benchmark session on the Experiments tab) and
  carries the measured-row fields: `sampler`, `samples` (the arm's requests), `error` unknown, and `period_ms`, null
  unless `--row-condition power_period_ms=<ms>` gives the power samples' period for the joules rows.
- Whether telemetry rubric test 6 is met is a finding in the report and the T0 text; nothing under `lab/refine/` is
  written. The real run's commands and what stays deferred: `lab/T11_REAL_RUN.md`.

## The real run of 2026-10-11

Run `t11-20261011T053204Z-3bd0`, 05:32 to 06:28 UTC, from a workstation (`lab/t11_offbox_run.py`, power through
`lab/t11_power_bridge.py`; the commands are in `lab/T11_REAL_RUN.md`): the teacher on pod A (account C, SJC) serving
vLLM v0.31.0 with the RDNA hybrid W4A16 kernel (R60), the student `atlas-base` on the Jetson in MAXN_SUPER, 100
labeling requests of the fixture (seed 11) under each policy, one probe per node first, no pause, every answer
measured. The per-request records, the report, the T0 text and the run's ledger rows are in
`fixtures/telemetry/t11-closed-loop-2026-10-11/`; the six rows are `lab/refine/metrics.jsonl:211-216`.

| Policy | Answered by | Joules per answer (sd) | Seconds per answer (sd) | Label agreement |
|---|---|---|---|---|
| capability (the control) | student 100 | 235.3 (81.1) | 12.78 (4.11) | 73 of 100 |
| energy (reads the snapshot) | student 100 | 247.5 (83.6) | 13.37 (4.24) | 72 of 100 |

- **Verdict.** No difference in joules per answer (Welch t 1.05, p 0.30), in latency (p 0.32) or in label agreement
  (z -0.16, p 0.87). The energy policy changed nothing.
- **Why.** Policy two read the snapshot on all 100 decisions and kept every request on the student: the teacher's
  probe answer cost 1,045 J (11.7 s) against 233 J for the student's, and the student's answers in the two arms ran
  from 52 to 426 J. The first, interrupted attempt of the evening read 4,441 J for the same probe on the April build,
  so the upgrade cut the teacher's cost per answer about four times and it is still about four times the student's on
  this work at one request at a time.
- **TT6, as a finding for the merge pass.** Met as a test: the policy read a value on every request, the control is
  the all-student baseline and all three metrics have a spread. By its keep rule policy two is not kept.
- **Trust.** Every joules value is low confidence (no measured sensor error, telemetry ruling 7); the Jetson's
  samples are tegrastats lines at 350 ms stamped on arrival at the workstation.
