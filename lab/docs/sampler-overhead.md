# Pod sampler overhead: `python -m lab sampler-overhead` (T9)

Moved here from lab/README.md (O70); the verbs list there links every page. Arguments: `python -m lab sampler-overhead -h`.

Does the pod power sampler (`lab/pod_power.py`) slow the work it watches? Rubric test TT5 says decode tok/s and time to
first token with the sampler on must stay within the run-to-run spread of the same runs with it off. This verb runs that
experiment on pod B (account D) inside one GPU Queue slot: one discarded warm-up with the sampler off, then off, on, off,
on, off, on (three repeats an arm, interleaved). A repeat is one pass of the bench harness's fixed prompt set
(`fixtures/lab/bench/prompts.json`, batch 4, so every prompt once) against the server on the pod through the hub proxy,
with the harness's own power reading and ledger rows left off. CPU percent is read on the pod in the middle of the repeat
(two `/proc/stat` snapshots 12 s apart through run-script, plus the sampler's own `/proc/<pid>/stat`).

```
python -m lab sampler-overhead run --account D --dry-run             # the schedule and the steps; touches nothing
python -m lab sampler-overhead run --account D                       # request the slot, wait (every 2 min, up to 90), run, release
python -m lab sampler-overhead run --account D --row <id>            # use a slot that is already Running
python -m lab sampler-overhead analyze --rows fixtures/telemetry/sampler_overhead_rows.jsonl
```

- The slot goes through `lab/machine/slots.py` (`slots.run`), so every request, start, extension and release is a ledger
  row and a queue row as `python -m lab slot` makes them. A slot that never starts is cancelled and nothing runs on the pod.
- A model server is served with `pod_a_teacher.sh serve` only when none answers on the proxy, and stopped at the end;
  the sampler is put back as it was found; the slot is extended once, with a note, if the repeats will not fit.
- Per metric: mean and sd of each arm, on minus off (and as a percent of off), the spread (the larger sd), whether the
  difference is inside it, and Welch's test from `lab/machine/verdict.py` beside it. Verdicts: `no_interference`,
  `interference`, `inconclusive`. Three repeats an arm is low confidence (the verdict module wants five).
- Rows: `<out>/rows.jsonl`, `<out>/analysis.json` and ledger rows `sampler-overhead.repeat` and
  `sampler-overhead.summary` on node pod-b. The state of the measurement (not yet taken: the card was busy with an
  unqueued job), what the slot met and the proposed rubric line are in `lab/T9_REPORT.md`.

Tests: `tests/test_lab_sampler_overhead.py` drives the experiment against fakes (the slot queue over `slots_fake.py`,
the bench harness over `teacher_fake.py`, a fake pod and a fake `/proc`) on a fake clock.
