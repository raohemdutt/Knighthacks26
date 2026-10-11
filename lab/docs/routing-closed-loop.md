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
  `--row-condition NAME=VALUE` adds the told conditions of those rows (`gpu`, the hub, the teacher's build).
- Whether telemetry rubric test 6 is met is a finding in the report and the T0 text; nothing under `lab/refine/` is
  written. The real run's commands and what stays deferred: `lab/T11_REAL_RUN.md`.
