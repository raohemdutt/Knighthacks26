# L16 retrain on tool-calling-v2: result (Sun 2026-10-11)

The rules every number here was produced under are in `lab/L16V2_RULES.md`, with five dated amendments. The pod D
numbers are the pod D seat's (lines marked `podD` in `work/L16v2-out/progress.log`, files in `work/L16v2pod-out/`).

## 1. What this fixes

The first fine-tune (run `train-20261011T032149Z-7dd098`, L16) trained on 84 rows that reword 12 questions at one
clock time, with teacher labels, no warmup, no validation, 3 epochs. It memorized them (loss 0.0095) and took
held-out tool-call agreement on `tool-calling-fixture` from 42/58 to 34/58: 0 examples improved, 8 broke, 7 of them
"in the last 60 minutes" with a wrong start time.

This work replaces the data, adds validation to the trainer, and chooses the recipe by validation only.

## 2. The scores (pod D, the packaged Q4_K_M GGUF on llama-server, tonight's flags)

Before = the base rebuilt through the package's path (`~/l4/Qwen3.5-4B-base-samepath-Q4_K_M.gguf`, sha256
84284aa2...e04f). One run per arm; the before arm of the later rows is the saved output of the first.

| adapter | test set | before | after | 0 to 1 | 1 to 0 | McNemar (examples) | sign-flip (groups) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CHOSEN: trial 12's configuration at step 32 (`atlas-tool-calling-v2-a100-f2r4`) | tool-calling-v2 held out, 228 examples in 18 groups | 95 = 0.417 | 179 = 0.785 | 90 | 6 | p = 2.5e-20 | p = 1.5e-05 |
| the same | tool-calling-fixture held out, 58 examples in 4 groups | 42 = 0.724 | 43 = 0.741 | 1 | 0 | p = 1 | p = 1 |
| grid run 1 (the fallback, `atlas-tool-calling-v2-a100-r01`) | tool-calling-v2 held out | 95 = 0.417 | 176 = 0.772 | 86 | 5 | p = 4.0e-20 | p = 1.5e-05 (17 groups up, 1 flat, 0 down) |
| the same | tool-calling-fixture held out | 42 = 0.724 | 43 = 0.741 | 1 | 0 | p = 1 | p = 1 |

The eval gate and the third full-length adapter (trial 9's configuration): pending from the pod D seat.

How to read them:

- The v2 held-out side is the clean test of the recipe: 10 objects and 8 window lengths that no training or
  validation row shows. The sign-flip test over the 18 groups is the conservative one and is what "better" rests
  on. McNemar on the 228 examples treats rewordings of one group as independent, so it is descriptive.
- For grid run 1, by tool: `last_seen` 54 to 100 of 100; `episodes_between` 41 to 76 of 128. All 5 of its losses
  and the 47 examples both arms miss are time windows (section 8).
- The old fixture's 58 are not a clean test of this recipe: the recipe was designed after tonight's failure on
  exactly those 58. They say the regression is gone (34/58 tonight, base 42/58). 15 of the 58 are "overnight"
  questions, whose start is the household's bedtime; no prompt states it, the base misses all 15, and this dataset
  deliberately never trains that shape (it is the old fixture's held-out group), so they stay missed.
- Grid run 1 was scored at 4:19 to 4:32 AM ET, before any choice existed, on the coordinator's order (rules
  amendment 3), and each finalist was scored as it arrived. The choice was made by the validation rule alone, by
  code; no held-out number went into it.

## 3. Dataset `tool-calling-v2`

Built by `lab/fixture_dataset_v2.py` on the Colab VM, with the student's tokenizer and chat template checking every
row there. It is the fixture's own method (the agent's pre-fetch code run against the fixture night), fed more.

| | |
| --- | --- |
| Examples, groups | 1,194 examples in 124 question groups (a group is one object or one window length, whatever the wording and the clock) |
| Train | 840 rows in 93 groups |
| Validation | 126 rows in 13 whole groups (8 objects, 5 window lengths) that no training row shows |
| Held out | 228 examples in 18 groups: 10 objects and 8 window lengths (25, 45, 75, 105 minutes; 7, 9, 14, 36 hours) that no other part shows |
| Targets | the generator's reference labels (the call the agent's own code makes and the answer written from its evidence); no teacher |
| What varies | 84 objects (every other one has a sighting, so `last_seen` returns something for half), 38 window lengths plus today and yesterday, 18 wordings per shape with 5 or 6 spellings of a length, and a clock per example |
| Clocks | 1,064 distinct times over the fixture night and morning; 611 as the box's clock really prints (`datetime.now(UTC)`: seconds, microseconds, +00:00), 583 in the fixture's style (whole minute, -04:00) |
| Rows | 577 to 985 tokens (mean 655), checked with the tokenizer and against the template's own rendering |
| Frozen | split sha256 `4776164b103cd9af8ffb877243cb3694cc7c75b22c7e81f78e5ff64351924377`; `rows-train.jsonl` sha256 `64e3d4222e9e007e588016a43b8914e1a3f4c9e72b1152727824d8934c135b9d` |

Regenerating it byte for byte: `python lab/fixture_dataset_v2.py --out <folder>` at this branch's commit (add
`--tokenizer <the student's folder>` to repeat the check). The build reads no clock and no random source: the first
VM's build, a local rebuild on Windows and the rebuild on the second VM gave the same seven files. The dataset
folder is not committed (4 MB of rows); the command is the record.

What is not in it, on purpose: nothing of the old fixture's held-out side (no keys, no wallet, no 60-minute or
one-hour window, no overnight wording). The build fails if one gets in. The cost: the overnight shape is not taught.

Two things the row format does not fix (both were already so for the first fixture): the training prompt lists the
three memory tools only, while the box's tool loop offers more (`TOOLS` against `self.tools` in
`contexts/agent/application/qa.py`); and the target answer is the evidence written out, not natural speech.

## 4. The trainer (`lab/train.py`)

New flags, each with a test in `tests/test_lab_train.py`: `--warmup-ratio`; `--val-rows` with `--eval-steps` (the
loss on rows that are never trained on, one line per evaluation in `validation.jsonl`); `--keep-best` (the saved
adapter is the best evaluation's checkpoint); `--early-stop N` with `--early-stop-min-rel S`; `--stop-step S` (repeat
the first S steps of a longer schedule: how pod D retrains a kept-best checkpoint without the validation passes);
and `--cuda`, the one edit that lets it run off the pod: no gfx1151 check, the pod's ROCm environment is not
exported, the data is kept, and the row says mode `cuda`, node `cuda`, with the GPU's own name in `refs.device`.

## 5. Tuning on the A100

### What the A100 can do

Measured before any sweep run (5 optimizer steps of 16 examples, the first left out):

| setting | examples/s | card memory |
| --- | --- | --- |
| the pod's own settings: batch 4 x accumulation 4, gradient checkpointing, torch fallback kernels | 2.39 | 11.2 GB |
| batch 16 x 1, checkpointing | 2.39 | 20.0 GB |
| batch 16 x 1, no checkpointing | out of memory at 40 GB (either kernel) | |
| flash-linear-attention 0.5.2, batch 16 x 1, checkpointing | 1.07 (steps of 3.6 to 4.2 s alternate with 25.9 s rebuilds per batch shape; first step 74 s) | 16.2 GB |
| 3 trainers at once on one card (the 80 GB A100) | 2.41 in total, 0.95 times one trainer (19 to 21 s a step each against 6.3 s) | 39 GB |

So one full-length run (62 steps with 8 validation passes of 23 s) is about ten minutes on the A100, six times
pod D's rate (about 0.4 examples/s there), and several trainers on a card add nothing. Not tested: the fast kernels
with every batch padded to one shape, and a run without gradient checkpointing at batch 4 x 4 on the 80 GB card.
No H100 on this account (Colab refused it). Colab granted a second A100 (the 80 GB model, high-RAM shape) and
refused a third and a fourth, so the search ran on two cards, one trainer each.

### Grid run 1

The first plan was a 12-point grid; the timing above cut it to three points, and Ilan then replaced the grid with a
Bayesian search. Its first point had finished by then: learning rate 1e-4, rank 16, alpha 32, dropout 0.05, warmup
10 percent, 62 steps = 992 example passes, 660 s. Validation loss at steps 8, 16, 24, 32, 40, 48, 56, 62: 0.0355,
0.0188, 0.0133, 0.0101, 0.0103, 0.0099, 0.0087, 0.0081 (still falling at the last step, so the kept checkpoint is
the last one). Its adapter was pulled at once as the fallback.

### The search (40 minutes, 4:13 to 4:52 AM ET)

Optuna TPE, the design copied from Ilan's earlier search (section 9), the study held in a SQLite file on the
workstation, each Colab session one worker that asks for its next point only when its last trial has finished.
A trial is 24 optimizer steps (384 example passes) with validation at steps 12 and 24; its value is the lower of
its two validation losses. `seen` is how many finished trials the sampler had when it proposed the point.

| n | seen | kind | worker | learning rate | rank | alpha | dropout | warmup | outcome | val loss, step 12 | val loss, step 24 | value | best so far |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | seed | l16v2-a100 | 0.0001 | 16 | 32 | 0.050 | 0.100 | observation (grid run 1) | n/a | 0.0133 | 0.0133 | 0.0133 |
| 2 | 1 | seed | l16v2-a100 | 0.0002 | 16 | 32 | 0.050 | 0.100 | trained | 0.0237 | 0.0159 | 0.0159 | 0.0133 |
| 3 | 2 | seed | l16v2-a100 | 5e-05 | 16 | 32 | 0.050 | 0.100 | pruned | 0.0293 | 0.0236 | 0.0293 | 0.0133 |
| 4 | 3 | init | l16v2-a100 | 6.68e-05 | 4 | 4 | 0.130 | 0.090 | pruned | 0.0644 | 0.0471 | 0.0644 | 0.0133 |
| 5 | 4 | init | l16v2-a100 | 0.000195 | 8 | 16 | 0.046 | 0.079 | pruned | 0.0260 | 0.0186 | 0.0260 | 0.0133 |
| 6 | 5 | init | l16v2-a100 | 8.03e-05 | 8 | 16 | 0.118 | 0.030 | pruned | 0.0302 | n/a | 0.0302 | 0.0133 |
| 7 | 5 | model | l16v2-a100-b | 0.000105 | 16 | 32 | 0.145 | 0.121 | pruned | 0.0258 | n/a | 0.0258 | 0.0133 |
| 8 | 6 | model | l16v2-a100 | 2.55e-05 | 16 | 32 | 0.138 | 0.006 | pruned | 0.0371 | 0.0289 | 0.0371 | 0.0133 |
| 9 | 7 | model | l16v2-a100-b | 0.000248 | 16 | 32 | 0.084 | 0.041 | trained | 0.0227 | 0.0150 | 0.0150 | 0.0133 |
| 10 | 8 | model | l16v2-a100 | 4.15e-05 | 16 | 32 | 0.061 | 0.047 | pruned | 0.0311 | n/a | 0.0311 | 0.0133 |
| 11 | 9 | model | l16v2-a100-b | 0.000156 | 16 | 32 | 0.084 | 0.140 | trained | 0.0242 | 0.0157 | 0.0157 | 0.0133 |
| 12 | 10 | model | l16v2-a100 | 0.000428 | 4 | 8 | 0.136 | 0.009 | trained | 0.0212 | 0.0142 | 0.0142 | 0.0133 |

| kind | trials run | mean val loss at step 12 (all) | completed | mean value (completed) | best value | pruned |
| --- | --- | --- | --- | --- | --- | --- |
| seed | 2 | 0.0265 | 1 | 0.0159 | 0.0159 | 1 |
| init | 3 | 0.0402 | 0 | n/a | n/a | 3 |
| model | 6 | 0.0270 | 3 | 0.0149 | 0.0142 | 3 |

Reading it:

- Trial 1 is not a 24-step trial: its value is grid run 1's loss at step 24 of a 62-step schedule, where its
  learning rate had barely decayed. The like-for-like number is in the edge round below.
- Trials 3 to 7 were pruned under the first step-12 rule (the library's median pruner, which takes its median over
  completed trials only, so after trial 2 nothing could survive); trials 3, 4 and 5 ran on to step 24 because the
  stop signal was ignored, and their step-24 losses are shown but were never used. From trial 8 the rule is the
  median of all earlier step-12 losses (rules amendment 4, a correction made after seeing seven trials).
- Did the search learn? The six model-chosen points had a mean step-12 loss of 0.0270 against 0.0402 for the three
  random startup points, and three of them completed (0.0150, 0.0157, 0.0142) where no random one did; the best
  24-step trial of the search (12) is model-chosen. Against that: the first three model-chosen points were pruned,
  two of them at learning rates the pruned seeds had already shown to be too low; the comparison is 6 points
  against 3; and the edge round's noise measurement below (0.0030 between two seeds of one configuration) is as
  large as the whole spread of the five best values. So: the model's points were better than random ones on
  average, and nothing here shows that any one of the best five configurations is better than another.

### The finalists and the choice

The three lowest values named the finalists. Trial 1 is grid run 1, already trained at full length, so it was
reused, not retrained (rules amendment 4); the other two trained at the same time, one per card, with early
stopping (stop after 2 evaluations that did not beat the best by 3 percent).

| full-length run | from | learning rate | rank | alpha | dropout | warmup | steps run | example passes (run / at best) | best val loss (step) | validation agreement | worker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r01-lr0.0001-rank16-drop0.05 | the grid's run 1 | 0.0001 | 16 | 32 | 0.050 | 0.100 | 62 | 992 / 992 | 0.0081 (62) | 110/126 | l16v2-a100 |
| final2-t12-lr0.000428-rank4-a8-drop0.136-warm0.009 (chosen) | trial 12 (model, value 0.0142) | 0.000428 | 4 | 8 | 0.136 | 0.009 | 48 | 768 / 512 | 0.0098 (32) | 113/126 | l16v2-a100 |
| final3-t09-lr0.000248-rank16-a32-drop0.084-warm0.041 | trial 9 (model, value 0.0150) | 0.000248 | 16 | 32 | 0.084 | 0.041 | 48 | 768 / 512 | 0.0107 (32) | 111/126 | l16v2-a100-b |

- Validation tool-call agreement is generated on the A100 (transformers greedy on the bf16 weights with the
  adapter attached, 126 examples in 13 groups). Base: 69/126.
- The frozen rule (the lowest validation losses, then validation agreement) chose trial 12's configuration at its
  step-32 checkpoint. The three adapters are within 3 examples of each other out of 126 and no noise was measured
  on that number, so this is the rule's pick, not a measured difference between them.
- The chosen run stopped at step 48 and kept step 32: 512 example passes, about 21 minutes on pod D.

Recipe for pod D: `--max-steps 62 --stop-step 32 --lr 0.000427627 --rank 4 --alpha 8 --dropout 0.1360 --warmup-ratio 0.0093 --batch 4 --grad-accum 4 --seed 42`

### The edge round

Pending (edge section).

## 6. Proof the data came back

Every product was pulled when its marker appeared and checked against the size and sha256 the job wrote beside it
(`work/L16v2-out/colab/downloads.txt`). The Optuna study never left the workstation
(`work/L16v2-out/bo/study.db`, with `bo.json`, `bo-edge.json` and `chosen.json` beside it).

| product (underwork/L16v2-out/colab/) | bytes | sha256 |
| --- | --- | --- |
| `dataset-tool-calling-v2.tar.gz` | 181,987 | `816847a99fc6d3c223c8e6a1bc5523ed1ee586fb24bb9ad0e6f0cedd90f6e703` |
| `failed-smoke.tar.gz` | 1,604 | `574c565b8cf48ad487d0c8e0dd9a259a5d1672401cb535553ca525ff0568c58f` |
| `fallback-adapter-r01.tar.gz` | 123,753,712 | `99538d1342349217d475a070f418263b4b6106f61d2d699f4fe0db4930db0762` |
| `run-01-r01-lr0.0001-rank16-drop0.05.tar.gz` | 14,397 | `5469fcd8820b49bb948389b8c46da9f7caa17cd9aee942ad856c18f975e80acb` |
| `run-e13-lr0.0002-rank32-a64-drop0.050-warm0.050.tar.gz` | 7,436 | `0f25014a02917027c0ce16e01527c4d65e147147d8d795e744a2f2c417c1f444` |
| `run-e14-lr0.0004-rank16-a32-drop0.050-warm0.050.tar.gz` | 7,399 | `f9b5c1a7f05a3002bb186cd78e07bb602716e6c38d6303783bdf5187555aa3f0` |
| `run-final2-t12-lr0.000428-rank4-a8-drop0.136-warm0.009.tar.gz` | 33,965,853 | `d1253d8cd9fee564e5f0319d934144e383d16cb7b92b811b559282fa94a82d01` |
| `run-final3-t09-lr0.000248-rank16-a32-drop0.084-warm0.041.tar.gz` | 123,778,973 | `d64f17569343625a50ce2a7c70ec7118f13bd9ee26da6b6163ca25c689279ec8` |
| `run-n15-trial9-seed43.tar.gz` | 7,405 | `70a4f78c2bd4d7283d267680220038086091178ca65c9670719ce23790719f24` |
| `run-n16-seedconfig-24steps.tar.gz` | 7,397 | `5de3e9681edc2de38d1566dd370498a82ccbb05b36fce4e115d2c72e1db5a6b6` |
| `run-t02-lr0.0002-rank16-a32-drop0.050-warm0.100.tar.gz` | 7,430 | `616585a35a922c9a267244f8562cd5640c1870f55c48343177e87a986e3b8f82` |
| `run-t03-lr5e-05-rank16-a32-drop0.050-warm0.100.tar.gz` | 7,408 | `bd28be7bfb4a79875bc6caac4d0f4f12a52e28682414bbb5ef434a4288a131bc` |
| `run-t04-lr6.68e-05-rank4-a4-drop0.130-warm0.090.tar.gz` | 7,422 | `5e51976edcd8a4faff49d8c71f52180028fe0d774f09d16cab7f32185d2076d8` |
| `run-t05-lr0.000195-rank8-a16-drop0.046-warm0.079.tar.gz` | 7,415 | `b27c0578d687a8c2eceac91fe9358ee732e60c86544d611e21d10e15fc3c2397` |
| `run-t06-lr8.03e-05-rank8-a16-drop0.118-warm0.030.tar.gz` | 3,533 | `31629531e590cea35eae5f600fe073d74ab8c7f4fd1d7c21a15701a4c5740846` |
| `run-t07-lr0.000105-rank16-a32-drop0.145-warm0.121.tar.gz` | 2,427 | `12432b9daf49b65f7ebd7c0298edc120ac581715c3a2b3136447fca0fded3570` |
| `run-t08-lr2.55e-05-rank16-a32-drop0.138-warm0.006.tar.gz` | 7,429 | `a529715f17745c095da32c7c835c2e71132091476de8350e78e418af1d4a2108` |
| `run-t09-lr0.000248-rank16-a32-drop0.084-warm0.041.tar.gz` | 7,430 | `f8079d69754223e7d016b75775b821ebb6000e03a58124dbc4a783b78d5cc455` |
| `run-t10-lr4.15e-05-rank16-a32-drop0.061-warm0.047.tar.gz` | 2,583 | `02b1de896c672025821a63974b896825c04ef41d95065560b732770413f86883` |
| `run-t11-lr0.000156-rank16-a32-drop0.084-warm0.140.tar.gz` | 7,453 | `b45594f30cd021be7b6e41529309894021b73d2b409f6b6f75d717b496993e70` |
| `run-t12-lr0.000428-rank4-a8-drop0.136-warm0.009.tar.gz` | 7,404 | `9e2bb114655e1cdc417e986b869d8217ee703b911d5204894ffb7346ef37128f` |
| `score-val-l16v2-a100-b.tar.gz` | 11,422 | `3b18d47f3f7a90becd1933638a2a8658fcc44c576e06ffa95c0e732a4c969a26` |
| `score-val-l16v2-a100.tar.gz` | 33,053 | `b18147d05656534756751f0241465a409bdabbaede31fbec5f35ec1c21e2f8de` |

Extracted adapters: `work/L16v2-out/colab/x-fallback-r01/adapter` (grid run 1) and
`work/L16v2-out/colab/runs/<run>/adapter` (the finalists).

Pending (sessions line).

## 7. Pod D

Pending (pod rows).

The pod D retrain with the chosen recipe (second pass): the run script is `lab/l16v2/pod_run.sh train <run id>
<recipe flags>` (a copy of `lab/l16_run.sh`'s train step with the new dataset, validation rows and flags); who runs
it and when is the coordinator's call. Pending (pod retrain).

## 8. What makes a number weaker than it looks, and what is open

- The recipe was designed after tonight's failure on the old fixture's 58; they are a continuity number, not a test.
- One run per arm on pod D (greedy decoding gave identical outputs across runs tonight, so the noise estimate was 0
  then; it was not re-measured here).
- The held-out side of v2 was read for grid run 1 before the choice existed, and for each finalist as it arrived
  (rules amendment 3). The choice did not use those numbers, but they were known when it came out.
- The choice between the three full-length adapters rests on 113, 111 and 110 correct out of 126 validation
  examples. On pod D's held-out side they are also close. Any of the three is a defensible pick.
- Two rules of the search were changed after results were seen, both on validation numbers only: the step-12
  pruning rule (amendment 4) and the search space of the edge round (amendment 5).
- A trial's value has a measured noise of about 0.003 (two seeds of one configuration), which is the size of the
  differences the search was ranking. The search found a region (learning rates of 2e-4 and up at 24 steps), not a
  winner.
- Validation and held-out examples come from the same generator as the training rows: the same 18 wordings, the
  same one night of memory, the same two clock styles. "Held out" here means unseen objects and window lengths, not
  unseen phrasing or a real household's questions.
- The time windows are where the retrained model still fails (for grid run 1: 52 of 128 held-out window examples
  missed after training, against 0 of 100 object examples). A next dataset revision should add more window lengths
  and clock times, and should decide whether to teach the overnight shape (which means giving up the old fixture's
  58 as a comparison). Nothing was tuned toward this before the choice.
- The A100-trained and pod-trained adapters differ in chip and torch build (CUDA against ROCm); the recipe itself
  is the same on both (batch 4 x 4, checkpointing, torch fallback kernels).
- Open, small: `lab/transfer.py`'s size read-back after a chunked push fails on pod D because the contents API
  answers 404 for the file unless it is asked with `type=file` (found by the pod D seat). Session l16v2-a100-b's
  CLI calls stalled several times (each cost its worker about two minutes); the driver now kills a stalled call
  and retries.

## 9. What was copied from where

| piece | copied from | what changed |
| --- | --- | --- |
| `lab/fixture_dataset_v2.py` | `lab/fixture_dataset.py` (its `reference_trace`, `instruction`, `output_schema`, label and file helpers are imported, not rewritten) | wider lists, a clock per example, reference labels as targets, the split by group, rows written at build time |
| rows | `lab/format_traces.py` (`check_trace`, `build_row`, its tool-loop results, now callable per clock as `tool_loop_results`) | nothing in the format |
| trainer flags | `lab/train.py` | section 4 |
| scoring | `lab/tool_call_agreement.py` (PR 378, merged into this branch) | the prompt takes each example's own time; the rule text is unchanged |
| Colab driver and job | `tooling/dispatcher/a100_eval_rerun.sh` and `work/colab-a100/` (setup, detached launch, status reader, the git bundle) | products with a sha256 and a marker each, pulled as they appear; a keepalive; ASCII-only reads |
| the search | Ilan's `MCA_full_pipeline/PIPELINE/training/hp_search_modernbert_rocm.py`: the TPE sampler settings with the seed offset, the median pruner, the SQLite storage (`_build_concurrent_storage`), the warm start (a finished observation plus enqueued configurations), the duplicate guard, counting only real trials, the boundary check, early stopping with patience 2 | the values (that search tuned a classifier); the study is held on the workstation and each Colab session is a worker; constant liar; the step-12 rule of amendment 4; a relative minimum improvement; the edge round |
| pod D scripts | `lab/l16_run.sh` (train), `lab/l5_run.sh` (package), the L4 run's serve, check and stop steps, `run_gate.sh` | the dataset, the flags, a version name that says where the adapter was trained |

## 10. Files

- `lab/L16V2_RULES.md`: the rules and their five amendments.
- `lab/l16v2/`: `colab_setup.sh`, `colab_worker_setup.sh`, `colab_trial.sh`, `vm_score.py`, `sweep.py` (the VM-side
  generator and scoring helpers, and the grid that ran one point), `colab_job.sh`, `colab_driver.sh`,
  `colab_keepalive.sh`, `colab_status.py` (the grid's job and driver), `colab_speed.sh`, `colab_parallel_test.sh`
  (the timings), `bo_local.py` (the search driver), `bo_edge.py` (the edge round), `report.py` (the tables here),
  `pod_run.sh`, `pod_step4.sh`, `score_pod.sh` (pod D).
- Tests: `tests/test_lab_fixture_dataset_v2.py`, `tests/test_lab_train.py`, `tests/test_lab_tool_call_agreement.py`,
  `tests/test_lab_l16v2_sweep.py`, `tests/test_lab_l16v2_bo_local.py` (needs optuna; skipped where it is missing).
