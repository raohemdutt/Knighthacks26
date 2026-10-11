# L16 retrain on tool-calling-v2: result (Sun 2026-10-11)

The rules every number here was produced under are in `lab/L16V2_RULES.md`, with six dated amendments. The pod D
numbers are the pod D seat's (lines marked `podD` in `work/L16v2-out/progress.log`, files in `work/L16v2pod-out/`).

## In plain words

The box's small language model (Qwen3.5-4B) has to pick the right memory tool, with the right time window, when
someone asks it a question. Tonight's first fine-tune made it worse at that, because it memorized 84 almost
identical training examples. This work rebuilt the training data (1,194 examples, with whole groups of questions
kept back for testing), added a validation check to training, and chose the training settings by validation
questions alone: first a short Bayesian search on two rented A100 GPUs, then a longer search of complete training
runs, each repeated with a second random seed before it could win. The retrained model now on the box (trial 12's
settings, package `atlas-tool-calling-v2-a100-f2r4`) gets 179 of 228 never-seen test questions right, against 95 for
the original model, and no question group got worse (17 of 18 improved, p = 0.000015 over groups); the other two
finalists got 176 and 177, so the gain is large and clear while the differences between finalists are not. The
longer search, judged by its own rule fixed before its first result, ranked a different setting higher (rank 32 at
learning rate 2e-4: 116 of 126 validation questions on average over two seeds, against 109.5 for the box's
settings), but that margin is about the size of the seed-to-seed noise it measured, and that model has not yet been
tested on the held-out questions or packaged. So the box keeps the model it has, the better-ranked candidate is
ready for one more test, and the questions every version still gets wrong are mostly time windows such as "in the
last 45 minutes", which is what the next dataset should teach.

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
| the choice at 5:13 AM: trial 12's configuration at step 32 (`atlas-tool-calling-v2-a100-f2r4`) | tool-calling-v2 held out, 228 examples in 18 groups | 95 = 0.417 | 179 = 0.785 | 90 | 6 | p = 2.5e-20 | p = 1.5e-05 |
| the same | tool-calling-fixture held out, 58 examples in 4 groups | 42 = 0.724 | 43 = 0.741 | 1 | 0 | p = 1 | p = 1 |
| grid run 1 (the fallback, `atlas-tool-calling-v2-a100-r01`) | tool-calling-v2 held out | 95 = 0.417 | 176 = 0.772 | 86 | 5 | p = 4.0e-20 | p = 1.5e-05 (17 groups up, 1 flat, 0 down) |
| the same | tool-calling-fixture held out | 42 = 0.724 | 43 = 0.741 | 1 | 0 | p = 1 | p = 1 |
| trial 9's configuration at step 32 (`atlas-tool-calling-v2-a100-f3r16`) | tool-calling-v2 held out | 95 = 0.417 | 177 = 0.776 | 90 | 8 | p = 1.1e-18 | p = 3.1e-05 (16 groups up, 2 flat, 0 down) |
| the same | tool-calling-fixture held out | 42 = 0.724 | 42 = 0.724 | 0 | 0 | p = 1 | p = 1 |

Sources: the `podD RESULT agreement` lines of `work/L16v2-out/progress.log` (4:29 and 4:32 AM for grid run 1, 5:14
and 5:15 AM for f2r4, 5:23 and 5:25 AM for f3r16), the files under `work/L16v2pod-out/pod/<tag>/agreement-v2/` and
`agreement-old58/`, and the channel lines of 4:33, 5:16 and 5:26 AM ET. The before arm was asked once, at 4:29 AM,
and reused for the later two adapters (same prompts, checked by sha256 per example).

The eval gate (scoring rule 2) was not run for any of the three: each 15-minute pod D row was spent on packaging and
the two agreement runs. It stays open, and its verdict could only have been "reject: scored and reported" (24 scored
items, under the gate's 50).

The A100-trained f2r4 package is the one on the box (`atlas-tool-calling-v2-a100-f2r4`, Q4_K_M sha256
`db7879b12398052a77c5c707af4248991253467e6065e533835f76def0edbcde`). The extended search later changed the validation
choice (section 5); that did not change the box.

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

### The finalists and the choice at 5:13 AM

The three lowest values named the finalists. Trial 1 is grid run 1, already trained at full length, so it was
reused, not retrained (rules amendment 4); the other two trained at the same time, one per card, with early
stopping (stop after 2 evaluations that did not beat the best by 3 percent).

| full-length run | from | learning rate | rank | alpha | dropout | warmup | steps run | example passes (run / at best) | best val loss (step) | validation agreement | worker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r01-lr0.0001-rank16-drop0.05 | the grid's run 1 | 0.0001 | 16 | 32 | 0.050 | 0.100 | 62 | 992 / 992 | 0.0081 (62) | 110/126 | l16v2-a100 |
| final2-t12-lr0.000428-rank4-a8-drop0.136-warm0.009 (the choice at 5:13 AM) | trial 12 (model, value 0.0142) | 0.000428 | 4 | 8 | 0.136 | 0.009 | 48 | 768 / 512 | 0.0098 (32) | 113/126 | l16v2-a100 |
| final3-t09-lr0.000248-rank16-a32-drop0.084-warm0.041 | trial 9 (model, value 0.0150) | 0.000248 | 16 | 32 | 0.084 | 0.041 | 48 | 768 / 512 | 0.0107 (32) | 111/126 | l16v2-a100-b |

- Validation tool-call agreement is generated on the A100 (transformers greedy on the bf16 weights with the
  adapter attached, 126 examples in 13 groups). Base: 69/126.
- The frozen rule (the lowest validation losses, then validation agreement) chose trial 12's configuration at its
  step-32 checkpoint. The three adapters are within 3 examples of each other out of 126 and no noise was measured
  on that number, so this is the rule's pick, not a measured difference between them. (The extended search below
  measured it: one seed moved this configuration from 113 to 106.)
- The chosen run stopped at step 48 and kept step 32: 512 example passes, about 21 minutes on pod D.

Recipe for pod D of the 5:13 AM choice (the one on the box): `--max-steps 62 --stop-step 32 --lr 0.000427627 --rank 4 --alpha 8 --dropout 0.1360 --warmup-ratio 0.0093 --batch 4 --grad-accum 4 --seed 42`

### The edge round

Ilan asked whether the edges were explored and whether the granularity was too small; the first table said yes to
both (rank 32 never tried, rank 16 in 8 of 12 trials, the best 24-step trial on four edges at once, every learning
rate below 8e-5 a loser, and continuous warmup and dropout searched finer than a 24-step trial can see). The edge
round (rules amendment 5) widened the learning rate to 1e-3, the ranks to {2, 4, 8, 16, 32, 64}, made dropout and
warmup coarse, forced the corners the sampler had skipped, and measured the noise of a trial. Same 24-step fidelity,
20 minutes, in study `l16v2-edge` (Optuna cannot change a parameter's choices inside a study), into which 8 of the
12 first trials were copied; the 4 below the new learning-rate floor could not be.

| n | seen | kind | worker | learning rate | rank | alpha | dropout | warmup | seed | outcome | val loss, step 12 | val loss, step 24 | value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13 | 8 | edge | l16v2-a100 | 0.0002 | 32 | 64 | 0.05 | 0.05 | 42 | trained | 0.0195 | 0.0126 | 0.0126 |
| 14 | 8 | edge | l16v2-a100-b | 0.0004 | 16 | 32 | 0.05 | 0.05 | 42 | trained | 0.0182 | 0.0120 | 0.0120 |
| 15 | 9 | noise | l16v2-a100 | 0.000248 | 16 | 32 | 0.08 | 0.04 | 43 | trained | 0.0169 | 0.0120 | 0.0120 |
| 16 | 10 | noise | l16v2-a100-b | 0.0001 | 16 | 32 | 0.05 | 0.10 | 42 | trained | 0.0258 | 0.0193 | 0.0193 |
| 17 | 10 | edge | l16v2-a100 | 0.0004 | 32 | 64 | 0.05 | 0.05 | 42 | trained | 0.0195 | 0.0107 | 0.0107 |

| rank | trials (both rounds) | completed | best value | best val loss at step 12 |
| --- | --- | --- | --- | --- |
| 2 | 0 | 0 | n/a | n/a |
| 4 | 2 | 1 | 0.0142 | 0.0212 |
| 8 | 2 | 0 | n/a | 0.0260 |
| 16 | 8 | 4 | 0.0120 | 0.0182 |
| 32 | 2 | 2 | 0.0107 | 0.0195 |
| 64 | 0 | 0 | n/a | n/a |

| learning rate | trials (both rounds) | completed | best value |
| --- | --- | --- | --- |
| below 8e-5 | 4 | 0 | n/a |
| 8e-5 to 1e-4 | 1 | 0 | n/a |
| 1e-4 to 2e-4 | 3 | 1 | 0.0157 |
| 2e-4 to 4e-4 | 3 | 3 | 0.0126 |
| 4e-4 to 1e-3 | 3 | 3 | 0.0107 |

- The noise floor of a 24-step trial: trial 9's configuration gave 0.0150 with seed 42 and 0.0120 with seed 43, a
  difference of 0.0030; the seed configuration gave 0.0193 as a proper 24-step trial against the 0.0133 that
  trial 1 carried (step 24 of its 62-step run), a difference of 0.0061, which is fidelity more than noise. The
  floor used was the larger, 0.0061.
- No edge-round trial beat the best 24-step value (0.0142) by more than that floor (the best was 0.0107, rank 32 at
  4e-4), so there was no fourth finalist and the choice stood.
- What the round did show: at 24 steps the high learning rates and the larger ranks look best (0.0107 to 0.0126
  for rank 16 and 32 at 2e-4 to 4e-4), and the same table shows why that cannot be trusted: the seed recipe, which
  is the best full-length run so far (0.0081), is the worst of these as a 24-step trial (0.0193). A short trial
  rewards a recipe for being fast, not for where it ends.
- It ran short of its own plan: the budget check used the longest trial so far (414 s, slowed by the validation
  scoring that shared its card) as the estimate for the next one, so it stopped proposing at 833 of 1,200 seconds.
  Three of the five forced corners ran inside the budget; rank 4 at 8e-4 ran after it, for the table only (kind
  edge-after-budget); rank 2 at 4e-4 and every model-chosen edge point did not run at 24 steps.

### The extended search (5:23 to 6:43 AM ET)

Because of the second and third points, Ilan ordered an extended search of full-length runs until 7:00 AM ET
(rules amendment 6): every run at full length with early stopping (at most 62 steps = 992 example passes, validation
every 8 steps, the best checkpoint kept), validation tool-call agreement on the 126 validation rows for every run,
seven forced configurations first, and a challenger replacing the choice only when the mean validation agreement of
its two runs (seeds 42 and 43) is higher than the mean of the current choice's two runs. Driver `lab/l16v2/bo_ext.py`,
study `l16v2-full`, started from the three full-length runs above. Every row below is in
`lab/l16v2/record/bo-ext.json` (a copy of `work/L16v2-out/bo/bo-ext.json`) and in the `RESULT stage=ext` and
`RESULT ext_table` lines of `work/L16v2-out/progress.log`; the tables are `python lab/l16v2/report.py
work/L16v2-out/bo --ext`.

| n | kind | run | seed | learning rate | rank | alpha | dropout | warmup | steps run | best step (example passes) | best val loss | validation agreement | worker | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | earlier | final2-t12-lr0.000428-rank4-a8-drop0.136-war | 42 | 0.000428 | 4 | 8 | 0.14 | 0.01 | 48 | 32 (512) | 0.0098 | 113/126 | l16v2-a100 | 528.9 |
|  | earlier | final3-t09-lr0.000248-rank16-a32-drop0.084-w | 42 | 0.000248 | 16 | 32 | 0.08 | 0.04 | 48 | 32 (512) | 0.0107 | 111/126 | l16v2-a100-b | 534.1 |
|  | earlier | r01-lr0.0001-rank16-drop0.05 | 42 | 0.0001 | 16 | 32 | 0.05 | 0.10 | 62 | 62 (992) | 0.0081 | 110/126 | l16v2-a100 | 660.0 |
| 1 | repeat | ext01-repeat-choice-seed43 | 43 | 0.000428 | 4 | 8 | 0.14 | 0.01 | 62 | 56 (896) | 0.0073 | 106/126 | l16v2-a100 | 677.5 |
| 2 | edge | x02-lr0.0002-rank32-a64-drop0.05-warm0.05 | 42 | 0.0002 | 32 | 64 | 0.05 | 0.05 | 62 | 62 (992) | 0.0087 | 115/126 | l16v2-a100-b | 848.2 |
| 3 | repeat | x03-repeat-seed43-of-final3-t09-lr0.000248-r | 43 | 0.000248 | 16 | 32 | 0.08 | 0.04 | 62 | 48 (768) | 0.0049 | 113/126 | l16v2-a100 | 678.0 |
| 4 | repeat | x04-repeat-seed43-of-x02-lr0.0002-rank32-a64 | 43 | 0.0002 | 32 | 64 | 0.05 | 0.05 | 62 | 62 (992) | 0.0068 | 117/126 | l16v2-a100-b | 729.7 |
| 5 | repeat | x05-repeat-seed43-of-r01-lr0.0001-rank16-dro | 43 | 0.0001 | 16 | 32 | 0.05 | 0.10 | 62 | 62 (992) | 0.0092 | 110/126 | l16v2-a100 | 673.7 |
| 6 | edge | x06-lr0.0004-rank16-a32-drop0.05-warm0.05 | 42 | 0.0004 | 16 | 32 | 0.05 | 0.05 | 62 | 56 (896) | 0.0087 | 108/126 | l16v2-a100-b | 638.5 |
| 7 | edge | x07-lr0.0001-rank32-a64-drop0.05-warm0.05 | 42 | 0.0001 | 32 | 64 | 0.05 | 0.05 | 48 | 32 (512) | 0.0085 | 108/126 | l16v2-a100 | 530.8 |
| 8 | edge | x08-lr0.0001-rank64-a128-drop0.05-warm0.05 | 42 | 0.0001 | 64 | 128 | 0.05 | 0.05 | 62 | 62 (992) | 0.0086 | 116/126 | l16v2-a100-b | 788.1 |
| 9 | edge | x09-lr0.0008-rank4-a8-drop0.05-warm0.05 | 42 | 0.0008 | 4 | 8 | 0.05 | 0.05 | 62 | 62 (992) | 0.0084 | 114/126 | l16v2-a100 | 662.9 |
| 10 | edge | x10-lr0.0004-rank2-a4-drop0.05-warm0.05 | 42 | 0.0004 | 2 | 4 | 0.05 | 0.05 | lost | n/a | n/a | not scored | l16v2-a100 | n/a |
| 11 | edge | x11-lr0.0002-rank8-a16-drop0.05-warm0.05 | 42 | 0.0002 | 8 | 16 | 0.05 | 0.05 | lost | n/a | n/a | not scored | l16v2-a100-b | n/a |

| configuration run with two seeds | validation agreement, seed 42 / 43 | mean | difference | best val loss, seed 42 / 43 | difference |
| --- | --- | --- | --- | --- | --- |
| lr 0.0001, rank 16, alpha 32, dropout 0.05, warmup 0.10 | 110 / 110 | 110.0 | 0 | 0.0081 / 0.0092 | 0.0011 |
| lr 0.0002, rank 32, alpha 64, dropout 0.05, warmup 0.05 | 115 / 117 | 116.0 | 2 | 0.0087 / 0.0068 | 0.0019 |
| lr 0.000248, rank 16, alpha 32, dropout 0.08, warmup 0.04 | 111 / 113 | 112.0 | 2 | 0.0107 / 0.0049 | 0.0058 |
| lr 0.000428, rank 4, alpha 8, dropout 0.14, warmup 0.01 | 113 / 106 | 109.5 | 7 | 0.0098 / 0.0073 | 0.0026 |

Noise of one run's validation agreement, from 4 configurations run with two seeds: the two runs differ by 0 to 7 examples of 126 (mean 2.8).

| rank | full-length runs | best val loss | best validation agreement | mean validation agreement |
| --- | --- | --- | --- | --- |
| 2 | 0 | n/a | n/a | n/a |
| 4 | 3 | 0.0073 | 114/126 | 111.0 |
| 8 | 0 | n/a | n/a | n/a |
| 16 | 5 | 0.0049 | 113/126 | 110.4 |
| 32 | 3 | 0.0068 | 117/126 | 113.3 |
| 64 | 1 | 0.0086 | 116/126 | 116.0 |

| learning rate | full-length runs | best val loss | best validation agreement | mean validation agreement |
| --- | --- | --- | --- | --- |
| 8e-5 to 1e-4 | 0 | n/a | n/a | n/a |
| 1e-4 to 2e-4 | 4 | 0.0081 | 116/126 | 111.0 |
| 2e-4 to 4e-4 | 4 | 0.0049 | 117/126 | 114.0 |
| 4e-4 to 1e-3 | 4 | 0.0073 | 114/126 | 110.2 |

| kind | runs | mean best val loss | mean validation agreement |
| --- | --- | --- | --- |
| earlier (the three runs before it) | 3 | 0.0096 | 111.3 |
| edge (forced) | 5 | 0.0086 | 112.2 |
| model (the sampler's) | 0 | n/a | n/a |
| repeat (seed 43) | 4 | 0.0070 | 111.5 |

Runs that did not finish: x10-lr0.0004-rank2-a4-drop0.05-warm0.05 (lost); x11-lr0.0002-rank8-a16-drop0.05-warm0.05 (lost).

Changes of the choice, in order: lr0.000427627-rank4-a8-drop0.1360-warm0.0093 (mean 109.5/126) replaced by lr0.000248362-rank16-a32-drop0.0843-warm0.0414 (mean 112.0/126); lr0.000248362-rank16-a32-drop0.0843-warm0.0414 (mean 112.0/126) replaced by lr0.0002-rank32-a64-drop0.0500-warm0.0500 (mean 116.0/126).

How it ended. The driver stopped at 6:43 AM ET, 17 minutes before its end time, and not by its budget: from 6:34 to
6:36 AM both Colab sessions stopped answering the CLI (every status call timed out or came back empty), and after six
rounds without a status each worker counted its run as lost and left. The two lost runs were forced corners, rank 2
at 4e-4 (x10) and rank 8 at 2e-4 (x11). A read-only check of the two VMs at 6:52 AM found that both had in fact
finished training with exit 0, at 6:37:50 and 6:38:22 AM. Their validation losses at steps 8 to 48, read from
`validation.jsonl` on each VM: x10 0.0315, 0.0180, 0.0124, 0.0109, 0.0130, 0.0116 (best 0.0109 at step 32); x11
0.0314, 0.0185, 0.0120, 0.0102, 0.0138, 0.0130 (best 0.0102 at step 32). Their validation agreement was never
generated and their files were not pulled, so they took no part in the rule. They could not have changed the
outcome: one seed only makes a run a challenger, a replacement needs its second seed, and no run could start after
6:46 AM. In all, 11 runs started and 9 finished and were scored: the seed-43 repeat of the choice, 5 of the 7
forced configurations, and the seed-43 repeats of 3 challengers. No point proposed by the sampler ran: the forced
configurations and the repeats filled the time.

The rule, step by step (the `decision ext` lines of `progress.log`):

- 5:33 AM: the seed-43 repeat of the choice (trial 12's configuration) scored 106/126 against its first run's 113, so
  the number to beat became their mean, 109.5. Grid run 1 (110) and trial 9's configuration (111) were above it and
  were queued for a second seed.
- 5:41 AM: rank 32 at 2e-4 (x02) scored 115 and was queued for a second seed.
- 5:48 AM: trial 9's configuration, 111 and 113, mean 112.0, beat 109.5 and replaced the choice.
- 5:57 AM: rank 32 at 2e-4, 115 and 117, mean 116.0, beat 112.0 and replaced it again.
- 6:03 AM: grid run 1, 110 and 110, mean 110.0, did not.
- No later run scored above 116 with one seed (rank 64 at 1e-4: 116, equal, so it was not repeated; rank 4 at 8e-4:
  114; rank 16 at 4e-4 and rank 32 at 1e-4: 108), so no further second seed was run.

The outcome: the choice does not stand. By the rule as written, the extended search's final choice is learning rate
2e-4, rank 32, alpha 64, dropout 0.05, warmup 0.05, trained the full 62 steps (992 example passes; both seeds kept
their last checkpoint, so its validation loss was still falling when the budget ended), with mean validation
agreement 116.0/126 over two seeds against 109.5 for trial 12's configuration. The adapter handed on is the seed-43
run x04 (117/126, validation loss 0.0068): `work/L16v2-out/colab/runs/x04-repeat-seed43-of-x02-lr0.0002-rank32-a64-drop/adapter`,
`adapter_model.safetensors` sha256 `4938fbddfc40f55ab372c4beeeb0c83e26963fa6b90e8430350c2828a8566c8c` (recomputed after
the driver ended), pulled as `adapter-x04-repeat-seed43-of-x02-lr0.0002-rank32-a64-drop.tar.gz` (243,219,857 bytes,
sha256 `769b9db6affa4161c022507c71a8cf3c67e4f8d0c1f699969b040993879b5916`). Its recipe, from
`lab/l16v2/record/chosen.json` (rewritten by the driver at 10:42:44Z, with the 5:13 AM choice kept as
`previous_choice`): `--max-steps 62 --stop-step 62 --lr 0.0002 --rank 32 --alpha 64 --dropout 0.0500 --warmup-ratio
0.0500 --batch 4 --grad-accum 4 --seed 43`.

How much that is worth. The four configurations run with two seeds moved by 0, 2, 2 and 7 examples of 126 between
seeds; the largest is trial 12's own pair (113 and 106). The gap between the two means, 6.5 examples, is smaller than
that largest single swing. Taking the four pairs as the estimate of noise (about 2.7 examples for one run, and so
about 2.7 for the difference of two two-seed means), the gap is about 2.4 times the noise: more likely real than
not, on 13 validation groups and four pairs, but not established. Amendment 6 asks for nothing beyond the mean of two
seeds, and it was applied as written.

What it did not do. The held-out side was not read for the new choice. The A100 reading that amendment 6 places
after the final line was launched at 6:42 AM on l16v2-a100-b and all five tries got no answer; the read-only check at
6:52 AM found no `FINAL_CHOICE` file and no held-out log on that VM (the only held-out log on either VM is the reading
stopped at 5:19 AM). The new choice was not packaged and has no pod D score, and pod D now belongs to another task.
So the box keeps `atlas-tool-calling-v2-a100-f2r4` (trial 12's configuration, 179/228 held out on pod D), which this
search's rule has ranked below the rank-32 configuration on validation. Whether to package, score and deploy the
rank-32 adapter is Ilan's decision; its adapter and recipe are on the workstation.

### The finalists side by side

| | grid run 1 (r01) | trial 12's configuration (f2r4) | trial 9's configuration (f3r16) | rank 32 at 2e-4 (x02, x04) |
| --- | --- | --- | --- | --- |
| learning rate | 1e-4 | 4.28e-4 | 2.48e-4 | 2e-4 |
| rank / alpha | 16 / 32 | 4 / 8 | 16 / 32 | 32 / 64 |
| dropout / warmup | 0.05 / 0.10 | 0.136 / 0.009 | 0.084 / 0.041 | 0.05 / 0.05 |
| kept checkpoint, seed 42 (example passes) | step 62 (992) | step 32 (512) | step 32 (512) | step 62 (992) |
| best validation loss, seed 42 / 43 | 0.0081 / 0.0092 | 0.0098 / 0.0073 | 0.0107 / 0.0049 | 0.0087 / 0.0068 |
| validation agreement of 126, seed 42 / 43 | 110 / 110 | 113 / 106 | 111 / 113 | 115 / 117 |
| mean of the two seeds | 110.0 | 109.5 | 112.0 | 116.0 |
| pod D, tool-calling-v2 held out of 228 (seed-42 adapter) | 176 = 0.772 | 179 = 0.785 | 177 = 0.776 | not scored |
| pod D, old fixture of 58 | 43 | 43 | 42 | not scored |
| package | `atlas-tool-calling-v2-a100-r01` | `atlas-tool-calling-v2-a100-f2r4` (on the box) | `atlas-tool-calling-v2-a100-f3r16` | none |
| the choice | no | 5:13 to 5:48 AM | 5:48 to 5:57 AM | from 5:57 AM, final at 6:42 AM |

Base model: 69/126 on validation, 95/228 and 42/58 held out. The pod D columns are section 2's; on the 228 the three
packaged adapters are within 3 examples of each other (168 right for all three, 44 wrong for all three, time windows;
the 5:26 AM channel line).

## 6. Proof the data came back

Every product was pulled when its marker appeared and checked against the size and sha256 the job wrote beside it
(`work/L16v2-out/colab/downloads.txt`, copied to `lab/l16v2/record/downloads.txt`). After the extended search ended,
all 47 products in that list were hashed again on the workstation and every size and sha256 matched. The Optuna
study never left the workstation (`work/L16v2-out/bo/study.db`); the driver's records beside it (`bo.json`,
`bo-edge.json`, `bo-ext.json`, `chosen.json`, `edge-result.json`) are copied unchanged to `lab/l16v2/record/`, and
every table on this page can be rebuilt from them with `python lab/l16v2/report.py lab/l16v2/record --edge --ext`.

| product (under work/L16v2-out/colab/) | bytes | sha256 |
| --- | --- | --- |
| `adapter-x02-lr0.0002-rank32-a64-drop0.05-warm0.05.tar.gz` | 243,192,424 | `5b4f901ec77dfaa2f20f796d4c744247cc42a21154f3d0b04751b42ba13e3993` |
| `adapter-x03-repeat-seed43-of-final3-t09-lr0.000248-rank16.tar.gz` | 123,719,482 | `a4852bf49ed7eaaadfb342ab139f0aaa7c221152db5ee36085119a3b48e2ebd1` |
| `adapter-x04-repeat-seed43-of-x02-lr0.0002-rank32-a64-drop.tar.gz` | 243,219,857 | `769b9db6affa4161c022507c71a8cf3c67e4f8d0c1f699969b040993879b5916` |
| `dataset-tool-calling-v2.tar.gz` | 181,987 | `816847a99fc6d3c223c8e6a1bc5523ed1ee586fb24bb9ad0e6f0cedd90f6e703` |
| `failed-smoke.tar.gz` | 1,604 | `574c565b8cf48ad487d0c8e0dd9a259a5d1672401cb535553ca525ff0568c58f` |
| `fallback-adapter-r01.tar.gz` | 123,753,712 | `99538d1342349217d475a070f418263b4b6106f61d2d699f4fe0db4930db0762` |
| `run-01-r01-lr0.0001-rank16-drop0.05.tar.gz` | 14,397 | `5469fcd8820b49bb948389b8c46da9f7caa17cd9aee942ad856c18f975e80acb` |
| `run-e13-lr0.0002-rank32-a64-drop0.050-warm0.050.tar.gz` | 7,436 | `0f25014a02917027c0ce16e01527c4d65e147147d8d795e744a2f2c417c1f444` |
| `run-e14-lr0.0004-rank16-a32-drop0.050-warm0.050.tar.gz` | 7,399 | `f9b5c1a7f05a3002bb186cd78e07bb602716e6c38d6303783bdf5187555aa3f0` |
| `run-e17-lr0.0004-rank32-a64-drop0.050-warm0.050.tar.gz` | 7,435 | `f167487175304a63e48f8512ea62b47e4ae33ec83634a86ed51fd4a0e5df8390` |
| `run-e18-lr0.0008-rank4-a8-drop0.050-warm0.050.tar.gz` | 7,424 | `a0d23a2fc2e2ce05f2012fb46364a6d35f54909322692464378e13e63d1ff6b7` |
| `run-e19-lr0.0004-rank2-a4-drop0.050-warm0.050.tar.gz` | 3,584 | `50bd7a290f3b93203d91a5716605015ade2fd438dbf84c9c526bd04832e37593` |
| `run-ext01-repeat-choice-seed43.tar.gz` | 33,966,764 | `5a4a3e7955471b2104963ba5c4ba34fc12ce43281c968be2291ef89658a7f304` |
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
| `run-x02-lr0.0002-rank32-a64-drop0.05-warm0.05.tar.gz` | 11,127 | `23b85261d9e8546ffa7e6ea5a4e50f5c5459dde07082ad243f98071901da7bf6` |
| `run-x03-repeat-seed43-of-final3-t09-lr0.000248-rank16.tar.gz` | 11,065 | `9863b38cc31c45f40cd2b15c384a9d4b9810eef1ae3bdec0aae35ce73bb18598` |
| `run-x04-repeat-seed43-of-x02-lr0.0002-rank32-a64-drop.tar.gz` | 11,091 | `9ffe9913b7551b545b70f192f0ffc0f624c19806351bd857975dc1a3b6efde59` |
| `run-x05-repeat-seed43-of-r01-lr0.0001-rank16-drop0.05.tar.gz` | 11,026 | `5238a478e650d3ba8b6160d0607bcdfa974c787a4b2654f67c745e91dbe08eef` |
| `run-x06-lr0.0004-rank16-a32-drop0.05-warm0.05.tar.gz` | 11,087 | `a7a77419bb88578cd6a04938f2497b5cc442dfc4b6fc75ddf1de6409ebfd3d5d` |
| `run-x07-lr0.0001-rank32-a64-drop0.05-warm0.05.tar.gz` | 9,756 | `4a6c652af313db7a720fc796f247db6748ed1ed70997274ec562f78c921b2e8a` |
| `run-x08-lr0.0001-rank64-a128-drop0.05-warm0.05.tar.gz` | 11,107 | `711ef4934b54e62a21c53354fdb847a13114cdf44dd2c5f4c05775803defcd1f` |
| `run-x09-lr0.0008-rank4-a8-drop0.05-warm0.05.tar.gz` | 11,097 | `5c51464f963a64503566dffa9790578e11ec4900b6bbe832aa7ad5cdaa522c2c` |
| `score-ext01-repeat-choice-seed43.tar.gz` | 11,392 | `7532773e2a352bfa3df8d33eabad5c9b4cd200deacb8275d4be42dd688d444da` |
| `score-val-l16v2-a100-b.tar.gz` | 11,422 | `3b18d47f3f7a90becd1933638a2a8658fcc44c576e06ffa95c0e732a4c969a26` |
| `score-val-l16v2-a100.tar.gz` | 33,053 | `b18147d05656534756751f0241465a409bdabbaede31fbec5f35ec1c21e2f8de` |
| `score-x02-lr0.0002-rank32-a64-drop0.05-warm0.05.tar.gz` | 11,433 | `dfd5a5d3641adc2c2852df7d5a76f05d81438bc89d9a438e79429b873b6abf3c` |
| `score-x03-repeat-seed43-of-final3-t09-lr0.000248-rank16.tar.gz` | 11,483 | `4fa58e186ebc95b262f4ba97d2e944c710ff385f9a1b4de75625a8271aed5565` |
| `score-x04-repeat-seed43-of-x02-lr0.0002-rank32-a64-drop.tar.gz` | 11,493 | `12cd4e679c42ad43bdd10b65db85c5b81dbdb34ccbbb2ae6b0013bd8c0822d77` |
| `score-x05-repeat-seed43-of-r01-lr0.0001-rank16-drop0.05.tar.gz` | 11,484 | `702f98fb27c0a6a6cf60b676c2dbd823a01476e7340f94166e958020149dcf91` |
| `score-x06-lr0.0004-rank16-a32-drop0.05-warm0.05.tar.gz` | 11,434 | `58ed883852b17e9a8fa0fb70a8802300f7d18b66900ca633b74a8975133c0862` |
| `score-x07-lr0.0001-rank32-a64-drop0.05-warm0.05.tar.gz` | 11,423 | `dc6df3ba53de871ba8c1c42a7d04e1bc150b7e443b539c307a25d2bb954c4686` |
| `score-x08-lr0.0001-rank64-a128-drop0.05-warm0.05.tar.gz` | 11,445 | `751f140326b7dc631cc8739abad4b51188295606c0fb1d4b09c0e48f84fdaf3d` |
| `score-x09-lr0.0008-rank4-a8-drop0.05-warm0.05.tar.gz` | 11,413 | `f0dd7a55ed52e60bd2c450e6b3268b0a0f3f57a58da3de86af45c3a44e77ce7c` |

Extracted adapters: `work/L16v2-out/colab/x-fallback-r01/adapter` (grid run 1) and
`work/L16v2-out/colab/runs/<run>/adapter` (every finalist and every candidate of the extended search: final2,
final3, x02, x03, x04).

The Colab sessions at the end (read with `tooling/dispatcher/colab.sh sessions`, `usage`, `status` and three
read-only `exec` probes at 6:50 to 6:53 AM ET): both A100 sessions, `l16v2-a100` (Standard) and `l16v2-a100-b`
(High-RAM), were still assigned and answering again, idle, with no trainer or scorer running on either; the
account showed 634.67 compute units left at a rate of 12.07 an hour. The finished but unpulled runs x10 and x11 are
on those VMs. The driver leaves sessions up for the caller to stop; they had not been stopped when this was
written.

## 7. Pod D

Every pod D step ran inside a GPU Queue row (GPU B, exclusive). Files: `work/L16v2pod-out/pod/<tag>/` (`slot-start.json`,
`package.json`, `agreement-*`, `stop.txt`, `slot-done.json`); lines: `podD` in `work/L16v2-out/progress.log`.

| row (GPU Queue) | what | Running (ET) | minutes | outcome |
| --- | --- | --- | --- | --- |
| `3f696ef5fec581629e23dda202751b4a` | grid run 1 (r01), first try | 4:16 | 0 | the push of the inputs failed its size read-back; released with nothing done |
| `3f696ef5fec581389732c09ac003aa4d` | grid run 1 (r01) | 4:19 to 4:32 | 13.5 | packaged `atlas-tool-calling-v2-a100-r01` (Q4_K_M sha256 `d7e038cbdd6cdacfc14bb16700cf9966694947b19a2479702b971e742cda00fa`); scored, section 2 |
| `3f696ef5fec58148be53df21bb4e11f7` | finalist 2 (f2r4, the choice at 5:13 AM) | 5:06 to 5:16 | 10 | packaged `atlas-tool-calling-v2-a100-f2r4` (sha256 `db7879b1...0edbcde`); scored, section 2 |
| `3f696ef5fec5816e9ef4d44ecaa4cff4` | finalist 3 (f3r16) | 5:16 to 5:25 | 9.8 | packaged `atlas-tool-calling-v2-a100-f3r16` (sha256 `151fd328528b28074aa013cca2b66793fa2beaad5e5ae8b8a8449a83a2fad07c`); scored, section 2 |
| `3f696ef5fec58196beb4cb7484637ed3` | the retrain of f2r4's recipe on pod D's own card (podf2r4) | 5:38 to 6:03 | 25.2 | cancelled by Ilan; nothing packaged or scored (below) |

Each package was built by `python -m lab package` (merge, bf16 GGUF, Q4_K_M) at llama.cpp `23b0202`, 3,381 MB, in
about 60 s.

### The cancelled pod D retrain (it decided nothing)

What it was: the second pass of the rules ("The pod D retrain"), the 5:13 AM choice (trial 12's configuration)
trained again on pod D's AMD card (Radeon 8060S, gfx1151, ROCm torch 2.9.1) instead of the A100, with the same rows,
validation rows, seed and batch, and `--stop-step 32` so it would end at the checkpoint the A100 run kept. Flags:
`--max-steps 62 --stop-step 32 --lr 0.000427627 --rank 4 --alpha 8 --dropout 0.1360 --warmup-ratio 0.0093 --batch 4
--grad-accum 4 --seed 42 --eval-steps 8 --keep-best`, code at `5753fdc`, the power sampler on. It was to be packaged as
`atlas-tool-calling-v2-pod-f2r4` and reported beside the A100-trained f2r4; by the rules it could replace nothing.

When it stopped: the trainer started at 5:38:21 AM ET (`train.txt`). Ilan cancelled the run at 5:59 AM ET, and the
stop was carried out by the coordinator's own stop script. The trainer had logged step 24 at 6:01:02 AM; the job
ended with exit 143 (killed by a signal) 1,464.5 s after it started, about 6:02:40 AM (`train.json`). The row closed at
6:03 AM after 25.2 minutes with the servers stopped and the card's memory at 0 percent (`stop.txt`, `slot-done.json`).
The trainer deletes its outputs on every way out, so `run.json`, `ledger.jsonl` and the adapter were never written
(the pulls of the first two answered 404); nothing was packaged and nothing was scored. Pod D has since gone to
another task, and the retrain is not to be restarted.

What it left, for the record only: validation loss 0.0316, 0.0189 and 0.0134 at steps 8, 16 and 24
(`run/validation.jsonl`), against 0.0320, 0.0188 and 0.0114 for the A100 run of the same recipe
(`work/L16v2-out/colab/runs/final2-t12-lr0.000428-rank4-a8-drop0.136-warm0.009/validation.jsonl`); 46.3 s an
optimizer step at a mean 95.8 W socket power over its 24 steps (the 6:01 AM watch line; the A100 does a step in
about 6.3 s); `run/telemetry.jsonl`, `run/train.log`, and the power samples
(`run/power-samples-2026-10-11.jsonl`, 11,640,264 bytes). A partial run that ended at step 24 is not the recipe's
step-32 checkpoint, so these numbers compare nothing and decide nothing.

## 8. What makes a number weaker than it looks, and what is open

- The recipe was designed after tonight's failure on the old fixture's 58; they are a continuity number, not a test.
- One run per arm on pod D (greedy decoding gave identical outputs across runs tonight, so the noise estimate was 0
  then; it was not re-measured here).
- The held-out side of v2 was read for grid run 1 before the choice existed, and for each finalist as it arrived
  (rules amendment 3). The choice did not use those numbers, but they were known when it came out.
- The 5:13 AM choice between the three full-length adapters rested on 113, 111 and 110 correct out of 126
  validation examples, one run each. On pod D's held-out side they are also close (176 to 179 of 228). Any of the
  three is a defensible pick, and the extended search's repeats showed why: one more seed moved trial 12's
  configuration from 113 to 106.
- The extended search re-chose after pod D's held-out numbers for all three finalists were known. The base rules
  say nothing is chosen again after the held-out side is read and that no second sweep follows a held-out number;
  amendment 6, ordered by Ilan, set that aside. Its rule reads validation numbers only, and the held-out numbers
  played no part in it, but they were known when the search was ordered.
- The extended search's choice (rank 32 at 2e-4) beat trial 12's configuration by 6.5 examples of 126 on the mean
  of two seeds; the largest swing between two seeds of one configuration was 7. It has no held-out number of any
  kind (the A100 reading after the final line never ran, and it was never packaged or scored on pod D), so whether
  it is better on unseen questions is not known.
- The extended search ended 17 minutes early when both Colab sessions stopped answering; two forced corners
  (rank 2 and rank 8) trained but were never scored, and none of the sampler's own points ran. Ranks 2 and 8 and
  learning rates below 1e-4 have no full-length run with a validation agreement.
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
  and retries. From 6:34 AM both sessions stopped answering for longer than the driver's six rounds of retries,
  and it counted its last two runs as lost although they finished; a driver that re-adopts a run after an outage
  (as the first search's driver does on a restart) would have kept them.
- Open: the eval gate for any adapter; the held-out reading and a pod D package of the extended search's choice;
  stopping the two Colab sessions.

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

- `lab/L16V2_RULES.md`: the rules and their six amendments.
- `lab/l16v2/`: `colab_setup.sh`, `colab_worker_setup.sh`, `colab_trial.sh`, `vm_score.py`, `sweep.py` (the VM-side
  generator and scoring helpers, and the grid that ran one point), `colab_job.sh`, `colab_driver.sh`,
  `colab_keepalive.sh`, `colab_status.py` (the grid's job and driver), `colab_speed.sh`, `colab_parallel_test.sh`
  (the timings), `bo_local.py` (the search driver), `bo_edge.py` (the edge round), `bo_ext.py` (the extended
  search), `report.py` (the tables here), `pod_run.sh`, `pod_step4.sh`, `score_pod.sh` (pod D).
- `lab/l16v2/record/`: the driver's records, copied unchanged from `work/L16v2-out/bo/` after the extended search
  ended (`bo.json`, `bo-edge.json`, `bo-ext.json`, `chosen.json`, `edge-result.json`) and the verified download
  list (`downloads.txt`). The adapters, the dataset folder, the study database and the progress log stay on the
  workstation under `work/L16v2-out/` and `work/L16v2pod-out/`.
- Tests: `tests/test_lab_fixture_dataset_v2.py`, `tests/test_lab_train.py`, `tests/test_lab_tool_call_agreement.py`,
  `tests/test_lab_l16v2_sweep.py`, `tests/test_lab_l16v2_report.py`, `tests/test_lab_l16v2_bo_local.py` and
  `tests/test_lab_l16v2_bo_rounds.py` (the last two need optuna and are skipped where it is missing).
