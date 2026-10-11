# L16 retrain on tool-calling-v2: result (Sun 2026-10-11)

Status of this page: the dataset, the A100 measurements, the first full-length run and its pod D scores are final.
The search table, the finalists and the choice are filled in when the search ends (section 5 says PENDING until
then). The rules every number here was produced under are in `lab/L16V2_RULES.md`, with four dated amendments.

## 1. What this fixes

The first fine-tune (run `train-20261011T032149Z-7dd098`, L16) trained on 84 rows that reword 12 questions at one
clock time, with teacher labels, no warmup, no validation, 3 epochs. It memorized them (loss 0.0095) and took
held-out tool-call agreement on `tool-calling-fixture` from 42/58 to 34/58: 0 examples improved, 8 broke, 7 of them
"in the last 60 minutes" with a wrong start time.

This work replaces the data, adds validation to the trainer, and chooses the recipe by validation only.

## 2. The scores so far (pod D, the packaged Q4_K_M GGUF on llama-server, tonight's flags)

Scored by the pod D seat, before = the base rebuilt through the package's path
(`~/l4/Qwen3.5-4B-base-samepath-Q4_K_M.gguf`, sha256 84284aa2...e04f), one run per arm.

| adapter | test set | before | after | 0 to 1 | 1 to 0 | McNemar (examples) | sign-flip (groups) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| grid run 1 (the fallback), `atlas-tool-calling-v2-a100-r01` | tool-calling-v2 held out, 228 examples in 18 groups | 95 = 0.417 | 176 = 0.772 | 86 | 5 | p = 4.0e-20 | p = 1.5e-05 (17 groups up, 1 flat, 0 down) |
| the same | tool-calling-fixture held out, 58 examples in 4 groups | 42 = 0.724 | 43 = 0.741 | 1 | 0 | p = 1 | p = 1 |

How to read them:

- The v2 held-out side is the clean test of the recipe: 10 objects and 8 window lengths that no training or
  validation row shows. The sign-flip test over the 18 groups is the conservative one, and it supports "better" at
  p = 1.5e-05. McNemar on the 228 examples treats rewordings of one group as independent, so it is descriptive.
- By tool: `last_seen` 54 to 100 of 100; `episodes_between` 41 to 76 of 128. All 5 losses and the 47 examples both
  arms miss are time windows. Section 8 lists what a next dataset revision should add because of that.
- The old fixture's 58 are not a clean test of this recipe: the recipe was designed after tonight's failure on
  exactly those 58. They say the regression is gone (34/58 tonight, 43/58 now, base 42/58). 15 of the 58 are
  "overnight" questions, whose start is the household's bedtime; no prompt states it, the base misses all 15, and
  this dataset deliberately never trains that shape (it is the old fixture's held-out group), so they stay missed.
- This reading of grid run 1 was taken at 4:19 to 4:32 AM ET, before the choice of the final adapter existed, on
  the coordinator's order (rules amendment 3). The choice below was made by the validation rule alone, by code.

The scores of the chosen adapter and the eval gate verdict: PENDING (pod D seat).

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
`--tokenizer <the student's folder>` to repeat the check). The build reads no clock and no random source: the VM's
build, a local rebuild on Windows and the rebuild on the second VM gave the same seven files (sha256 above and in
`build.json`). The dataset folder is not committed (4 MB of rows); the command is the record.

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
pod D's rate (pod D: about 0.4 examples/s), and several trainers on a card add nothing. Not tested: the fast kernels
with every batch padded to one shape, and a run without gradient checkpointing at batch 4 x 4 on the 80 GB card.
No H100 on this account (Colab refused it); a third and fourth A100 were refused as well.

### Grid run 1 (the fallback)

The first plan was a 12-point grid; the timing above cut it to three points, and Ilan then replaced the grid with a
Bayesian search. Its first point had finished by then: learning rate 1e-4, rank 16, alpha 32, dropout 0.05, warmup
10 percent, 62 steps = 992 example passes, 660 s. Validation loss at steps 8, 16, 24, 32, 40, 48, 56, 62: 0.0355,
0.0188, 0.0133, 0.0101, 0.0103, 0.0099, 0.0087, 0.0081 (still falling at the last step, so the kept checkpoint is
the last one). Its adapter was pulled at once as the fallback and is the adapter scored in section 2.

### The search

PENDING: the table of trials in order of proposal (seen, kind, parameters, losses), what the search learned, the
finalists, validation tool-call agreement, the choice and its recipe flags.

## 6. Proof the data came back

PENDING: the list of products with sizes and sha256, the study database, and the sessions stopped.

## 7. Pod D

Rows, times and the package are the pod D seat's (lines marked `podD` in `work/L16v2-out/progress.log`, files in
`work/L16v2pod-out/`). For grid run 1: GPU Queue row `3f696ef5fec581389732c09ac003aa4d`, 4:19 to 4:32 AM ET (13
minutes, extended once); package `atlas-tool-calling-v2-a100-r01`, Q4_K_M sha256
`d7e038cbdd6cdacfc14bb16700cf9966694947b19a2479702b971e742cda00fa`. An earlier row
(`3f696ef5fec581629e23dda202751b4a`, 4:16 AM) was released after 0 minutes when a transfer failed.

The pod D retrain with the chosen recipe (second pass): PENDING.

## 8. What makes a number weaker than it looks, and what is open

- The recipe was designed after tonight's failure on the old fixture's 58; they are a continuity number, not a test.
- One run per arm on pod D (greedy decoding gave identical outputs across runs tonight, so the noise estimate was 0
  then; it was not re-measured here).
- The held-out side of v2 was read once for grid run 1 before the final choice existed (rules amendment 3). The
  choice did not use it, but the person reading this page knew those numbers when the choice came out.
- The pruning rule of the search was corrected after seven trials (rules amendment 4), on validation numbers only.
- The search's trial 1 is not at the trials' fidelity: its value is step 24 of a 62-step schedule.
- Validation and held-out examples come from the same generator as the training rows: the same 18 wordings, the
  same one night of memory, the same two clock styles. "Held out" here means unseen objects and window lengths, not
  unseen phrasing or a real household's questions.
- The time windows are where the retrained model still fails (52 of 128 held-out window examples missed after
  training, against 0 of 100 object examples). A next dataset revision should add more window lengths and clock
  times, and should decide whether to teach the overnight shape (which means giving up the old fixture's 58 as a
  comparison). Nothing was tuned toward this before the choice.
- The A100-trained and pod-trained adapters differ in chip and torch build (CUDA against ROCm); the recipe itself
  is the same on both (batch 4 x 4, checkpointing, torch fallback kernels).
- Open, small: `lab/transfer.py`'s size read-back after a chunked push fails on pod D because the contents API
  answers 404 for the file unless it is asked with `type=file` (found by the pod D seat).

## 9. What was copied from where

| piece | copied from | what changed |
| --- | --- | --- |
| `lab/fixture_dataset_v2.py` | `lab/fixture_dataset.py` (its `reference_trace`, `instruction`, `output_schema`, label and file helpers are imported, not rewritten) | wider lists, a clock per example, reference labels as targets, the split by group, rows written at build time |
| rows | `lab/format_traces.py` (`check_trace`, `build_row`, its tool-loop results, now callable per clock as `tool_loop_results`) | nothing in the format |
| trainer flags | `lab/train.py` | section 4 |
| scoring | `lab/tool_call_agreement.py` (PR 378, merged into this branch) | the prompt takes each example's own time; the rule text is unchanged |
| Colab driver and job | `tooling/dispatcher/a100_eval_rerun.sh` and `work/colab-a100/` (setup, detached launch, status reader, the git bundle) | products with a sha256 and a marker each, pulled as they appear; a keepalive; ASCII-only reads |
| the search | Ilan's `MCA_full_pipeline/PIPELINE/training/hp_search_modernbert_rocm.py`: the TPE sampler settings with the seed offset, the median pruner, the SQLite storage (`_build_concurrent_storage`), the warm start, the duplicate guard, counting only real trials, the boundary check, early stopping with patience 2 | the values (that search tuned a classifier); the study is held on the workstation and each Colab session is a worker; constant liar; the step-12 rule of amendment 4; a relative minimum improvement |
| pod D scripts | `lab/l16_run.sh` (train), `lab/l5_run.sh` (package), the L4 run's serve, check and stop steps, `run_gate.sh` | the dataset, the flags, a version name that says where the adapter was trained |

## 10. Files

- `lab/L16V2_RULES.md`: the rules and their four amendments.
- `lab/l16v2/`: `colab_setup.sh`, `colab_worker_setup.sh`, `colab_trial.sh`, `vm_score.py`, `sweep.py` (the VM-side
  generator and scoring helpers, and the grid that ran one point), `colab_job.sh`, `colab_driver.sh`,
  `colab_keepalive.sh`, `colab_status.py` (the grid's job and driver), `colab_speed.sh`, `colab_parallel_test.sh`
  (the timings), `bo_local.py` (the search driver), `report.py` (the tables here), `pod_run.sh`, `pod_step4.sh`,
  `score_pod.sh` (pod D).
- Tests: `tests/test_lab_fixture_dataset_v2.py`, `tests/test_lab_train.py`, `tests/test_lab_tool_call_agreement.py`,
  `tests/test_lab_l16v2_sweep.py`, `tests/test_lab_l16v2_bo_local.py` (needs optuna; skipped where it is missing).
