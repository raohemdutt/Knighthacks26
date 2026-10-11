# T11 real run: the routing hook's closed loop on the real teacher and student

Everything in T11 was built and tested on the fakes (`lab/teacher_fake.py`, `lab/student_fake.py`, the fake power
feed). This file is what a person, or an agent with the access, runs to get the real numbers. It contacts the teacher
on pod A (account C, SJC hub) through the hub proxy and the student on the Jetson; it launches, restarts or stops
nothing, and it sends no message.

## Where it runs

On the Jetson, from the repo checkout the gateway runs from, while the gateway is up. The run needs three things the
gateway already makes:

1. The student: llama-server on the Jetson serving `atlas-base` (config/models.yaml `atlas_model`), reached at
   `http://127.0.0.1:8080`. Its chat API must accept `tools` (llama-server started with `--jinja`).
2. The power samples of both nodes in the box's event log, `data/events/<UTC day>.jsonl`: the gateway started with
   `ATLAS_TELEMETRY_SOURCE=tegrastats,pods` and `ATLAS_POD_ACCOUNTS=pod-a=C,pod-b=D`, so `telemetry.jetson.power`,
   the pod sampler's `telemetry.gpu.power` for pod-a, and `telemetry.clock.offset` for pod-a are all in that log.
3. The pod sampler (`lab/pod_power.py serve --node pod-a`) running on pod A, as T6 left it. Check it answers before
   starting; if it does not, stop here and ask, do not start it as part of T11.

## Environment (names only; values are on the Notion Credentials page and in the workspace `.env`)

- `AUP_HUB_USER_C`, `AUP_HUB_TOKEN_C`, `AUP_HUB_URL_C` (the SJC hub), `LAB_POD_ROLES=C:pod-a,D:pod-b`
- `LAB_ACTOR`, `LAB_TIER`, `LAB_TASK=T11` for the ledger rows' writer fields (optional)
- `LAB_STATE_DIR` when the ledger is not `lab/state` beside the code

## Before the run (read-only checks)

```
python -m lab.controller status --account C --probe-port 8000      # pod A up, the teacher answering on 8000
python -m lab ready                                                # the keys are found
curl -s http://127.0.0.1:8080/v1/models                            # the student answers
tail -n 3 data/events/$(date -u +%F).jsonl                         # power samples are arriving
```

No restart of pod A: if the teacher is not serving, stop and ask. Do not run `lab/pod_a_teacher.sh` for T11.

## The run

```
python -m lab routing-closed-loop --account C \
    --student-url http://127.0.0.1:8080 --student-model atlas-base \
    --power-log data/events/$(date -u +%F).jsonl \
    --window-clock jetson --n 100 --seed 11 --probes 1 \
    --settle-s 3 --wait-s 60
```

- When the run may cross midnight UTC, give both day files to `--power-log`.
- `--settle-s 3` covers the pod reader's 2 s poll (`ATLAS_POD_POWER_POLL_S`); `--wait-s 60` is the longest wait for
  one answer's joules before it is recorded as unmeasured.
- At the teacher's measured 2.83 tok/s (README, live smoke on pod C) a run of 100 requests, about half of them on the
  teacher under policy two, takes on the order of an hour. It sends one request at a time.
- The answer is one JSON line with `run_id`, the three verdict labels, `changed`, `keep_policy_two`,
  `rubric_tt6_met` and the path of `t0_result.txt`.

## The same run from a workstation

When the gateway on the Jetson does not log the pods (its restart with `ATLAS_TELEMETRY_SOURCE=tegrastats,pods`
needs sudo), the loop runs on a workstation that reaches both nodes, with three processes side by side:

```
# 1. the student, through a tunnel that is reopened when it drops
while true; do ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -i <key> -N -L 18080:127.0.0.1:8080 <user>@<jetson>; sleep 5; done
# 2. both nodes' power samples in one file (tegrastats over SSH, the pod sampler through the hub proxy)
python -m lab.t11_power_bridge --out lab/state/t11-run/power.jsonl --raw lab/state/t11-run/tegrastats.raw \
    --jetson <user>@<jetson> --ssh-key <key> --account C --minutes 240
# 3. the loop, with the outage rule and a progress file
python -m lab.t11_offbox_run --progress lab/state/t11-run/progress.jsonl -- --account C \
    --student-url http://127.0.0.1:18080 --student-model atlas-base --power-log lab/state/t11-run/power.jsonl \
    --window-clock jetson --n 100 --seed 11 --probes 1 --settle-s 3 --wait-s 60 \
    --metrics-file --row-condition "gpu=Radeon 8060S (Strix Halo), gfx1151" --row-condition hub=sjc \
    --row-condition account=C --row-condition power_period_ms=350
```

- `lab/t11_power_bridge.py` only reads. The Jetson's samples are stamped with the workstation's clock as they arrive,
  and the pod's clock offset is measured against the same clock, so `--window-clock jetson` holds here too.
- `lab/t11_offbox_run.py` runs the loop unchanged. A 5xx or a dropped connection pauses the request and asks again
  every 2 minutes, for at most 20 minutes; each pause and each answer is a line of the progress file, and
  `python -m lab.t11_offbox_run --progress <file> --summarize` reads the arms from it when a run stops early.

## After the run

1. Post the text of `lab/state/routing-closed-loop/<run id>/t0_result.txt` on T0 as the experiment result (the
   dispatcher does this; it carries its own citation: the run id, the ledger rows' seq numbers and the files).
2. Mirror the two experiments onto T0's queue with `python -m lab experiment list --print --id <E ids>`.
3. Hand the TT6 finding (the last paragraph of `t0_result.txt`) to the telemetry merge pass as a finding. Do not
   edit `lab/refine/` (plumbing ruling 5).

## What is deferred to this run (proved on the fakes only)

| Done when clause | Proved offline by | What only the real run gives |
|---|---|---|
| at least 100 lab requests from the L11 fixture, no golden question | `test_the_request_set_is_100_fixture_labeling_requests_drawn_with_a_fixed_seed_and_no_golden_question` | nothing: the same 100 requests (seed 11) |
| routed twice through T5's hook, capability and energy, interleaved | `test_every_request_is_routed_under_both_policies_with_the_arms_interleaved`, `test_policy_two_reads_the_telemetry_snapshot_and_moves_work_to_the_cheaper_node` | which node the real snapshot makes cheaper |
| joules per answer, latency and quality recorded with their spread | `test_each_request_records_joules_latency_and_the_label_agreement`, `test_each_arm_has_a_mean_and_a_spread_for_all_three`, `test_the_snapshot_is_the_joules_projections_and_every_answer_feeds_it` | the real joules (VDD_IN on the Jetson, socket_w on pod A through the clock offset), the real latencies and the real student's label agreement |
| the verdict says whether policy two changed energy or quality beyond the spread | `test_the_verdict_says_policy_two_changed_energy_and_quality_when_the_teacher_is_cheaper`, `test_the_verdict_says_no_change_when_policy_two_keeps_the_work_on_the_student` | the verdict on real numbers |
| the result is an experiment result on T0 with its citation | `test_the_t0_experiment_result_text_is_written_with_its_citation`, `test_a_ledger_row_per_arm` | posting it on T0 (Notion is out of reach of the build session) |
| TT6 reaches the rubric only as a finding | `test_rubric_test_6_is_a_finding_in_the_report_and_nothing_under_lab_refine_changes`, `test_rubric_test_6_is_not_met_when_policy_two_never_read_a_value` | the finding on real numbers, handed to the merge pass |

Not exercised offline at all: the hub proxy path to the teacher (`Teacher.from_hub`), the JSONL feed reading a log
that is being appended while the run waits (the reader is tested on a file that grows between polls,
`test_the_jsonl_feed_reads_only_new_lines`), and the clock offset between the Jetson's windows and pod A's samples
(the fakes share one clock; the projection's own tests cover the offset).

## Known effects to read the result with

- The energy policy reads the latest joules per answer per node, one value each. A short answer (a miss with no
  tool call) costs few joules and moves the next decision to that node; with the arms interleaved, the answer just
  before a B decision is usually arm A's on the same request. The report says when policy two split its requests
  for this reason (`test_a_snapshot_that_holds_only_the_latest_value_makes_policy_two_follow_the_last_answer`).
- The gateway's own requests on the Jetson in the same window share its power; joules per answer split overlapping
  windows by output tokens (T4) and mark them low confidence.
- Every joules value is low confidence today (no sensor error is measured, telemetry ruling 7), so each arm's ledger
  row is low confidence too.
