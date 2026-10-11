# T9: sampler overhead on pod B

Status: the harness is built and tested offline; the measurement has not been taken. The one slot that was taken on pod D
(account D, pod-b) ended in its first step because another job was already using the card outside the queue. Nothing here
is a measured number, and the rubric's bound is neither confirmed nor corrected yet.

## What is ready

`python -m lab sampler-overhead run --account D` (lab/sampler_overhead.py, tests/test_lab_sampler_overhead.py, 35 tests,
README section "Pod sampler overhead"). It takes a 15-minute slot through the slot verbs' own rules, serves the model
only if none answers, runs one discarded warm-up with the sampler off and then off, on, off, on, off, on (three repeats
each) of the bench harness's fixed prompt set at batch 4, reads CPU percent on the pod during each repeat, and writes
`rows.jsonl`, `analysis.json`, ledger rows and the slot's Result line. It puts the sampler, the server and the slot back
as found when a step fails. `python -m lab sampler-overhead analyze --rows <rows.jsonl>` recomputes the numbers.

Checked on the real pod (read-only, no GPU, no slot needed): the CPU script uploaded and run through the controller on
account D printed a valid reading (32 cores, `/proc/stat` deltas of 9600 ticks over a 3 s window, 0.15 percent busy on an
idle pod; the round trip was 6.4 s for the 3 s window, so the 12 s window costs about 15 s per repeat). `pod_power.sh
status` on pod D answered "not running" at 2026-10-11 00:19 UTC, so the sampler's state at the start is off.

## What happened to the slot

- Queue (GPU Queue, GPU B exclusive) was empty from 8:20 PM ET; no row from R33, R52 or R47 appeared in 14 minutes of
  polling, so a 15-minute row for T9 was requested and started at about 8:35 PM ET (00:35 UTC).
- The serve step (`pod_a_teacher.sh serve`, no server was up on D) failed after 32 s with vLLM's "Engine core
  initialization failed". At that moment `python3 -m lab bench run --engine vllm ...` (R20's benchmark matrix) was
  running on pod D, having started 00:33 UTC, and its own vLLM server came up at 00:37. R20's queue row was still Queued
  and waiting for the referee's critique, so the job was running without a started slot. The likely cause is the two
  servers contending for the card's memory; the log's root-cause line was not read to the end, so this is not proven.
- The experiment released the slot and stopped what it had started. It wrote no rows and no measurement.
- `pod_a_teacher.sh serve` begins with `pkill -f "vllm.entrypoints"`. That may have ended a server of R20's started in
  the same minute. Whoever owns R20 should check whether its first spawn at about 00:35 UTC is a failed run.
- Afterwards the harness polled pod D every two minutes (processes, used VRAM, queue) from 8:37 to 10:02 PM ET, 85 minutes.
  R20's spawns used 57 to 59 GB of the 64 GB the whole time (a cold start and a run per spawn, back to back), and the
  queue never showed a Running row. The card was never free for two polls in a row, so no second slot was taken.
  The hand-off: when the card is free, `python -m lab sampler-overhead run --account D --max-tokens 128`.

## Proposed rubric line (pending the run)

RUBRIC.md TT5 says: "On a pod, decode tok/s and time to first token with the sampler running stay within the run-to-run
spread of the same runs without it." Until T9 measures it, the line should read as untested rather than confirmed. After
a run, replace the bracket with the analysis:

> On a pod, decode tok/s and time to first token with the sampler running stay within the run-to-run spread of the same
> runs without it. Measured on pod-b (Strix Halo gfx1151) in T9, `lab/T9_REPORT.md`: [decode tok/s off X (sd a), on Y (sd
> b), difference D percent against a spread of S percent over 3 + 3 interleaved repeats; sampler CPU C percent of one core;
> verdict]. Confirmed / corrected to [bound]. Three repeats an arm is low confidence (telemetry ruling 7).

Citation to give: this report, `fixtures/telemetry/sampler_overhead_*.jsonl` once the run exists, and the ledger rows
`sampler-overhead.summary` (node pod-b).

## How the verdict is decided (so the run needs no interpretation)

Per metric, the difference on minus off is compared with the larger of the two arms' sd. Inside it: `within_spread`.
Outside and worse: `worse_outside_spread` (the bound is corrected). Outside and better: drift. Decode tok/s decides the
overall verdict, with time to first token; CPU percent is reported beside them. Welch's test is shown too, but with three
repeats it cannot rule out a small effect, so the rubric's within-spread rule is the verdict and the sd bound is the
claim ("no larger than the spread seen").
