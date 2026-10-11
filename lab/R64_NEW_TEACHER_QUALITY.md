# R64: is the 4.5x faster teacher still the same teacher?

Measured 2026-10-11 between 06:02 and 06:30 UTC (2:02 to 2:30 AM ET) on pod C (account C, SJC hub, node pod-a,
Radeon 8060S, gfx1151) against the teacher that has served there since 04:33 UTC: vLLM 0.31.2.dev0+gdb9527a46.gfx1151
(tag v0.31.0 plus `lab/patches/vllm-main-gfx1151-platform.diff` from R60), `RDNAHybridW4A16LinearKernel`, the same
`RedHatAI/Qwen3.8-27B-INT4` checkpoint, served as `teacher` and `group-brain` with `--scheduling-policy priority`
(read from the process on the pod). Requests only: nothing on the pod was restarted, stopped or reconfigured, no GPU
Queue slot was taken and pod D was not touched. The "April build" below is the old teacher (vLLM 0.19.2+gfx1151 at
b1388b1 plus `lab/vllm-gfx1201.diff`). Rows are in `lab/refine/metrics.jsonl` (task R64), raw outputs under
`fixtures/lab/new_teacher_quality/`.

## 1. Greedy outputs: the new build against the April build and the A100

R60's branch holds two compare results but no record of the new build, so one was recorded
(`fixtures/lab/correctness/vllm-gfx1151-v0.31.0-2026-10-11.json`, R21 prompt set, 128 greedy tokens a prompt, thinking
off) and compared three ways with `lab/correctness.py`'s own `compare_outputs`
(`fixtures/lab/new_teacher_quality/correctness/three_way.json`). The A100 record is
`fixtures/lab/bench/a100-2026-10-10/vllm-a100-2026-10-10.json` (stock vLLM 0.31.0 on Colab, R57).

First diverging token per prompt (a dash means all 128 tokens match):

| Pair | text-kitchen | text-schedule | image-bars | image-shapes | Exact match |
|---|---|---|---|---|---|
| new build vs April build | - | 106 | 0 | 3 | 1 of 4 (0.25) |
| new build vs A100 | - | 76 | - | 67 | 2 of 4 (0.50) |
| April build vs A100 | - | 76 | 0 | 3 | 1 of 4 (0.25) |
| new build vs itself, 3 live compares minutes apart | - | - | - | - | 12 of 12 (1.00) |
| new build vs R60's run of it at 04:42 UTC | - | - | - | - | 4 of 4 |

What it says:

- The A100 disagrees with the April build at exactly the places the new build does: the two image prompts at tokens
  0 and 3. On those prompts the new build gives the A100's answer: image-bars is the same 128 tokens on both ("This is a
  vertical bar chart with five bars..."), image-shapes the same for 67 tokens, while the April build starts
  differently ("Based on the provided image, here is..."). So the 0.25 is the April build being the odd one out on
  images, which follows the vLLM version's image path, not the new W4A16 kernel.
- On text the new build diverges from the April build at token 106 of text-schedule, a 0.06 logprob near-tie and the
  same token where the April build diverged from itself in R21. The A100 diverges from both Strix builds earlier,
  at token 76, also a 0.06 near-tie.
- The new build is deterministic on a quiet server: 4 of 4 in each of three compares against its own record, and 4 of
  4 against R60's outputs from 80 minutes earlier. The April build matched itself on only 2 of 4 (R21). The record
  above is the baseline the next kernel change on this build should be held to, at 1.0 rather than 0.5.

Rows: `teacher.output_match_rate` 1.0 (self), 0.25, 0.5 and 0.25 (the three pairs), low confidence.

## 2. Label agreement on the new teacher

`python -m lab label-agreement` on the L13 labels (`fixtures/learning/tool-calling-labels`, made by the April build,
seed 0) through a copy named `tool-calling-fixture` (the verb refuses a folder whose name is not its dataset id), the
first 30 examples, second seed 1, reworded prompt, one worker, against the serving teacher (run
`agr-label-20261011T060815Z-ac17`): **0.833** overall (50 of 60 fields), verdict inconsistent against the default 0.9.
`tool_calls` agree on 30 of 30; `answer` on 20 of 30. Every answer difference keeps the same facts in other words
("at 7:41 PM in the desk zone" against "at 7:41 PM on Saturday in the desk zone", "around 2 AM" against "at 2 AM");
one leaves out the plugged-in phone charger, one adds the camera name, none contradicts the first label. The report is
`fixtures/lab/new_teacher_quality/label_agreement/`.

To tell the build from the rewording, a control arm ran the same 30 examples through the same code
(`check_agreement`) with the rewording prefix set to empty (run `agr-label-20261011T061334Z-79e0`, script beside its
report in `fixtures/lab/new_teacher_quality/label_agreement_control/`): **0.983** (59 of 60), tool calls 30 of 30,
answers 29 of 30, the one difference "around 6 PM" against "from 6 PM". So the new build reproduces the April
build's labels almost exactly; the drop to 0.833 comes from the reworded instruction, which the check adds on purpose.

There is no old-build L14 number to set this against: `lab/L14_REAL_RUN.md` on main is the run plan and holds no
result, and no `label.agreement_rate` row exists before these two. These are the first real agreement numbers.
The 30 labels took 300 s at 1 stream (10.0 s an example, 2,313 completion tokens) and about 32 kJ of socket energy.

Rows: `label.agreement_rate` 0.833 (reworded) and 0.983 (control), node pod-a.

## 3. Sync energy on pod C (no GPU)

`lab/link_speed.py`'s `measure()` (the R14 verb's code) on account C with one 1 GB seeded random file, three pushes
through the contents API (64 MiB base64 chunks, size read back) and three pulls through the proxy from the Range file
server on port 8301, each pull sha256-checked (3 of 3 passed; the pod was left as found). Each transfer call was
bracketed by a read of the running power sampler's `/status` (`lab/pod_power.py 1`, 1000 ms period, SOCKET_POWER in
1 W steps), so each window is a range of sampler sequence numbers and needs no clock alignment
(`fixtures/lab/new_teacher_quality/sync/`: samples, marks, windows, report, scripts).

| Direction | GB/s (median, range) | Window mean W | J/GB above idle (median, range) | J/GB gross |
|---|---|---|---|---|
| push, contents API | 0.0093 (0.0091 to 0.0094) | 14.1 to 14.9 | 442 (428 to 513) | 1,519 to 1,594 |
| pull, proxy Range parts | 0.046 (0.045 to 0.052) | 11.9 to 13.5 | 64 (60 to 81) | 276 to 393 |

Idle is 10.1 W, the quiet minute after the run (60 samples, GFX activity 0). The 90 s planned as the idle window
before the run had another client on the teacher (GFX activity up to 100 percent, 95 to 111 W, from 06:19:59 UTC; not
one of this task's requests), so it was not used; the transfer windows themselves stayed at 0 to 3 percent GFX
activity. A push costs about 7 times the energy of a pull per GB and takes about 5 times as long; the sensor's 1 W step
is large next to a 2 to 5 W rise, so these are low confidence. A push's seconds are its whole call; a pull's are its
requests only (wall time 19 to 33 s).

Rows: `sync.push_joules_per_gb`, `sync.pull_joules_per_gb`, `sync.push_gb_s`, `sync.pull_gb_s`, node pod-a, hub sjc,
account C, sampler `lab/pod_power.py 1`, period_ms 1000, error unknown.

## For the slide

The 4.5x faster teacher is the same teacher: it is deterministic, it writes the same tool calls as the old build on
30 of 30 examples and the same labels on 59 of 60 fields, and where its outputs differ from the old build's, they
match the A100 reference instead.
