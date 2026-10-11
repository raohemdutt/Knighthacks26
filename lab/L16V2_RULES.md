# L16 retrain on tool-calling-v2: the rules, fixed before the first model output

Committed before the Colab job was launched and before any model wrote anything for this work. Nothing below is
changed after a result is seen. If a rule turns out to be wrong, the result page says so and reports the number
under the rule as written.

## What tonight's first fine-tune showed, and what was decided because of it

Run `train-20261011T032149Z-7dd098` (84 rows that reword 12 questions at one clock time, teacher labels, rank 16,
learning rate 2e-4, no warmup, 3 epochs, no validation) took held-out tool-call agreement on `tool-calling-fixture`
from 42/58 to 34/58. Everything on this page (the new dataset, the grid, the validation, the choice rule) was
designed after seeing that result. So the old fixture's 58 are not a clean test of this recipe: they are reported
for continuity with tonight's number, with this sentence beside them. The clean test is the held-out side of
`tool-calling-v2`, which did not exist when the recipe was designed and is read once, after the choice.

## The data

- `tool-calling-v2` is built by `lab/fixture_dataset_v2.py` on the Colab VM, with the student's tokenizer checking
  every row. Targets are the reference labels the generator computes from the agent's own pre-fetch code. No teacher.
- The split is by whole group and is in the generator (`HELD_OUT_ITEMS`, `HELD_OUT_WINDOWS`, `VALIDATION_ITEMS`,
  `VALIDATION_WINDOWS`), so it is frozen by this commit, before any training: 93 train groups (840 rows), 13
  validation groups (126 rows, never trained on), 18 held-out groups (228 examples).
- The held-out side holds 10 objects and 8 window lengths that no train or validation example shows.
- No part of v2 holds a held-out question, object or window of `tool-calling-fixture` (no keys, no wallet, no 60
  minute or one hour window, no overnight wording). The build fails if one gets in.

## Scoring rule 1: tool-call agreement

`lab/tool_call_agreement.py` (PR 378), its `RULE` text unchanged: the prompt is the example's training text up to its
first assistant span; the output is cut at the first `<|im_end|>`; calls are read by `parse_tool_calls()`; an example
is right when its calls' names and argument objects equal the reference's, in order and in number. For
`tool-calling-v2` the system prompt of an example carries that example's own time now (its `ts`), exactly as its
training row does; for `tool-calling-fixture` it is the dataset's one instruction, as tonight.

Two engines write the text, and each number says which:

- On the A100 (validation, for the choice; and the held-out side after the choice): transformers greedy decoding of
  the bf16 base with the adapter attached, at most 512 new tokens, prompts of equal token length batched so nothing
  is padded. One run per arm (greedy on one machine; no noise estimate).
- On pod D (the reported before and after): the packaged Q4_K_M GGUF on llama-server with tonight's flags
  (`--jinja -c 4096 -ngl 99 -np 1 --reasoning-budget 0 --cache-ram 0 --top-k 1`), before = the base rebuilt through
  the package's path (`~/l4/Qwen3.5-4B-base-samepath-Q4_K_M.gguf`, sha256 84284aa2...e04f), `/completion` with
  n_predict 512 and temperature 0.2 as tonight. Before is asked twice when the 15-minute row allows it, else once;
  after is asked once. The count of runs is reported.

Reported for each test set: before and after as counts with rates, flips both ways (0 to 1 and 1 to 0), the exact
two-sided McNemar test on the paired examples, and the exact two-sided sign-flip test over question groups (the
gate's test). McNemar treats examples as independent, which rewordings of one group are not, so it is descriptive;
the sign-flip over groups is the conservative one and is the one a claim of "better" rests on.

## Scoring rule 2: the eval gate

`python -m lab eval-gate` unchanged, on `tool-calling-fixture` as tonight (its golden questions are the ones the
scorer asks), current = the base GGUF, candidate = the packaged retrained GGUF, with the gate's own decision rule
(at least 50 scored items, beat the noise, sign-flip p below 0.05). Tonight it scored 24 items, so its verdict can
only be "reject: scored and reported"; it is reported as the gate prints it.

## The choice rule (the sweep)

The grid, in running order (learning rate, rank, dropout; alpha is twice the rank): (1e-4, 16, 0.05), (2e-4, 16,
0.05), (5e-5, 16, 0.05), (1e-4, 8, 0.05), (2e-4, 8, 0.05), (5e-5, 8, 0.05), then the same six with dropout 0.1.
Fixed for every run: warmup 6 percent of the planned steps, batch 4 by accumulation 4 (the pod's pinned effective
16), seed 42, at most 62 optimizer steps (992 example passes, 1.18 epochs of the 840 rows: what pod D trains in about
45 minutes at 2.5 seconds an example), validation loss every 8 steps and after the last, the best checkpoint kept,
stop after 3 evaluations without improvement. Runs are taken in that order until the 40-minute sweep budget is
used; a run that does not fit is not run, and the page says which. Two and three epochs are outside the pod D budget
and are not in the grid.

The rule, as `lab/l16v2/sweep.py` holds it (`CHOICE_RULE`):

> Eligible: a grid run that trained and whose planned example passes (optimizer steps x 16) are at most 1,000, the
> pod D budget. Each run's score is its lowest validation loss over its evaluations (every 8 optimizer steps and
> after the last), and its adapter is the checkpoint of that evaluation. Take the 3 eligible runs with the lowest
> score (ties: fewer example passes at the best evaluation, then earlier in the grid order). For those 3, generate
> on every validation example with the kept adapter and score exact tool-call agreement by
> lab/tool_call_agreement.py's rule. Choose the highest validation agreement (ties: lower validation loss, then
> fewer example passes at the best evaluation, then earlier in the grid order). The base model's validation
> agreement is measured for reference and takes no part. The held-out side is not read until chosen.json is
> written, and nothing is chosen again after it is read.

### Amendment 1, made before any sweep run finished (Sun 2026-10-11, 3:53 AM ET)

No sweep run had produced a validation number when this was written: the first one was stopped a few steps in, and
only the smoke run (2 steps, not a grid point) and timing runs of 5 steps had run.

What was measured on the A100 (5 optimizer steps of 16 examples each, the first left out): 2.39 examples a second
with the pod's own settings (batch 4 by accumulation 4, gradient checkpointing, transformers' torch fallback for the
Gated DeltaNet and causal conv layers; 11.2 GB), 2.39 with batch 16 by 1 and checkpointing (20.0 GB), out of memory
on the 40 GB card with batch 16 and no checkpointing (with either kernel), and 1.07 with flash-linear-attention
0.5.2 installed (batch 16 by 1, checkpointing, 16.2 GB: steps of 3.6 and 4.2 s alternate with steps of 25.9 s, a
first step of 74 s, as its kernel is built again for each new batch shape; slower over the run, so it was taken out
again). causal-conv1d would not install in the time given (a source build). One validation pass over the 126 rows takes 23 s. So a run of 62 steps
with its 8 validation passes is about ten minutes, not the two or three the grid was sized for.

What changes, by Ilan's coordinator's instruction for this case:

- The grid is its first three points in the planned order: learning rate 1e-4, 2e-4, 5e-5 at rank 16 (alpha 32),
  dropout 0.05. Rank 8 and dropout 0.1 are not run. The 40-minute budget rule as written would have stopped the
  sweep after the same three runs.
- Warmup is 10 percent of the planned steps (6 of 62) instead of 6 percent.
- The A100 runs keep the pod's settings (batch 4 by accumulation 4, gradient checkpointing, torch fallback kernels),
  since nothing else was faster. Each run's row records them.

Nothing else changes: the split, the validation cadence, keep-best, early stopping after 3 evaluations without
improvement, the choice rule (its "3 eligible runs with the lowest score" are now all the runs that train), the
scoring rules, and what is never done.

### Amendment 2, made before any search trial had a result (Sun 2026-10-11, 4:16 AM ET)

Ilan rejected the fixed three-point grid of amendment 1 and asked for a Bayesian search, with runs in parallel as
long as the search still learns from them, and with compute cost no constraint. This amendment replaces the grid.

What had run when this was written: grid run 1 to its end (learning rate 1e-4, rank 16, alpha 32, dropout 0.05,
warmup 0.10; validation loss 0.0355, 0.0188, 0.0133, 0.0101, 0.0103, 0.0099, 0.0087, 0.0081 at steps 8, 16, 24, 32,
40, 48, 56, 62); grid run 2 was stopped at step 9 and run 3 never started. The first search trial had been launched
two minutes earlier and had no validation number yet. No validation tool-call agreement had been measured for any
adapter, and no held-out example had been read.

The search (`lab/l16v2/bo_local.py`; the design is copied from Ilan's earlier Optuna search,
`MCA_full_pipeline/PIPELINE/training/hp_search_modernbert_rocm.py`, its values are not):

- Space: learning rate 2e-5 to 5e-4 (log scale), rank in {4, 8, 16, 32}, alpha = rank times 1 or 2, dropout 0 to
  0.15, warmup ratio 0 to 0.15. Fixed: batch 4 by accumulation 4, gradient checkpointing, seed 42, the torch
  fallback kernels.
- Objective: validation loss on the 126 validation rows, minimized. Fidelity, fixed: a trial is 24 optimizer steps
  (384 example passes) with validation at steps 12 and 24; its value is the lower of the two, and it keeps that
  checkpoint.
- Sampler and pruner: Optuna TPE (multivariate, grouped, constant liar, seed 42 plus the trials already in the
  study) with 3 random startup trials in total across all workers after the seeded points; a median pruner (2
  startup trials, 1 warmup step) on the step-12 validation loss, and a trial it prunes is stopped there.
- Warm start: grid run 1 is added as a finished observation with its own step-24 validation loss (0.0133, from a
  62-step schedule, noted as such); the two cancelled grid points (learning rate 2e-4 and 5e-5 at rank 16, alpha 32,
  dropout 0.05, warmup 0.10) are the first two trials. A configuration handed in by hand is enqueued the same way.
- Duplicates: a draw of parameters already evaluated is skipped without training and not counted.
- Boundary check, once, after 5 finished trials: if the best trial's learning rate, dropout or warmup is within 5
  percent of an edge of its range, that range is widened once (learning rate to 1e-5 or 1e-3, dropout or warmup to
  0.25).
- Where it runs: the study is a SQLite file on the workstation and never leaves it. Each Colab A100 session is one
  worker running one trial at a time. Workers are asynchronous: a worker gets its next point only when its last
  trial has finished, so each point is proposed with every finished result in the model. Every trial records how
  many finished trials the sampler had (`seen`), its kind (seed, init for a random startup draw, model for a draw of
  the fitted sampler) and its worker. At most one worker per five trials expected in the budget.
- Budget: 40 minutes from the first trial's start (4:13 AM ET); no trial starts that would pass it.

Finalists and the choice:

- The 3 completed (not pruned) trials with the lowest value (ties: the earlier proposal) name 3 configurations.
  Each trains once at full length, at the same time on separate sessions: at most 62 optimizer steps (992 example
  passes, the pod D budget), validation every 8 steps and after the last, the best checkpoint kept, and it stops
  after 2 evaluations that did not beat the best so far by at least 3 percent of it. (Ilan's earlier search used a
  patience of 2 with an absolute minimum improvement of 0.003 on a metric near 0.7; this loss is near 0.01, so the
  minimum is relative.) This replaces "stop after 3 evaluations without improvement" for these runs.
- The choice rule is the frozen one, applied to the full-length runs (the finalists that trained and grid run 1):
  take the 3 with the lowest validation loss (ties: fewer example passes at the best evaluation); for those 3,
  generate on every validation example with the kept adapter and score exact tool-call agreement by
  `lab/tool_call_agreement.py`'s rule; choose the highest validation agreement (ties: lower validation loss, then
  fewer example passes at the best evaluation). The base model's validation agreement is measured for reference.
- The held-out side is not read until the choice is written, and nothing is chosen again after it is read.

What the result page must show about the search: the trials in order of proposal with `seen`, kind, parameters and
losses; the best-so-far curve; the mean loss of model-chosen against random startup trials. If the model-chosen
points are not better than the random ones, the page says so.

Everything else stands: the dataset and its split, the scoring rules, and what is never done.

### Amendment 3: a held-out reading before the choice, and what it may not touch (Sun 2026-10-11, 4:40 AM ET)

This page says the held-out side is not read until the choice is written. That was broken in the letter, and this
records exactly how. On the order of Ilan's coordinator (not a decision of the seat that runs the search), the pod D
seat packaged the fallback adapter, grid run 1 (learning rate 1e-4, rank 16, alpha 32, dropout 0.05, warmup 0.10,
62 steps, validation loss 0.0081), and scored it on pod D between 4:19 and 4:32 AM ET, while the search was still
running and before any choice existed: the held-out side of `tool-calling-v2` (228 examples in 18 groups) and the
first fixture's 58. Its numbers and files are in `work/L16v2pod-out/` and are reported on the result page as what
they are: a reading of one adapter taken before the choice.

What follows from it, and is checked in the code:

- The choice of the final configuration and adapter is made by the frozen validation rule alone, by
  `lab/l16v2/bo_local.py`, with no held-out number as an input. The search, the pruning, the finalists, the scoring
  of validation agreement and every tie-break read the validation rows and nothing else. Nothing in `lab/l16v2/`
  reads `work/L16v2pod-out/` or any held-out output.
- Nothing is tuned toward what that reading showed (its misses are time windows) before the choice is made: the
  search space, the fidelity, the pruning rule and the finalist rule stay as amendment 2 fixed them. What it showed
  goes on the result page as a finding for a next dataset revision.
- Because grid run 1 is one of the full-length runs the choice rule ranks, the result page says, where the chosen
  adapter is named, whether it is that adapter, whose held-out numbers were already known when the choice was made.

Two changes of mechanics, neither of which changes a rule:

- Pruning acts when it is decided. The first trials the median rule pruned (trials 3, 4 and 5) ran on to step 24,
  because the stop signal sent to the trainer was one a detached background process ignores. From 4:38 AM a pruned
  trial's trainer is stopped at once. Those three stay recorded as pruned, with their step-12 loss as their value,
  exactly as the rule says; their step-24 losses are kept in the table but were never used.
- Several trainers on one card. One session turned out to be an 80 GB A100. If the measurement on it (1, 2, 3 and
  more trainers at once) shows that trainers on one card add throughput, a session runs more than one worker, each
  still asking the one study only when it is free. A trial's validation loss does not depend on what shares its
  card; each run records how many trainers shared it. The 40-minute budget and the fidelity do not change.

### Amendment 4, made while the search was running, before its finalists were known (4:42 AM ET)

Two things, both on the coordinator's instruction, with the search at trial 9 of its budget.

A correction to the pruning rule, made after seeing seven trials, on validation numbers only. Amendment 2 handed
the step-12 decision to Optuna's median pruner with 2 startup trials. That pruner takes its median over COMPLETED
trials only, and a pruned trial never completes, so after trial 2 completed (step-12 loss 0.0237) the reference
set could not grow: trials 3 to 7 were all pruned at step 12, among them trial 5 at 0.0260 and trial 7 at 0.0258,
close to trial 2's own 0.0237, which went on to 0.0159. The search was turning into "confirm the seed". (Ilan's
earlier search, whose design this copies, used 5 startup trials before any pruning; 2 was this page's choice and was
too few.) From the next pruning decision on, the rule is: stop a trial whose step-12 validation loss is worse than
the median step-12 validation loss of ALL earlier trials that reached step 12, pruned ones included, once at least
2 have. The driver computes that median itself (each trial's step-12 loss is kept with the trial) and the library's
pruner is no longer consulted. Trials 3 to 7 stay as they were recorded. The step-12 losses when this was written:
0.0237, 0.0293, 0.0644, 0.0260, 0.0302, 0.0258 (trials 2 to 7), median 0.0277. This is a change of a rule after
results were seen; it uses validation numbers only, and the result table marks which trials were pruned under
which rule.

Also for the table: trial 1's value (0.0133) is grid run 1's loss at step 24 of a 62-step schedule, where its
learning rate had barely decayed, so it is not at the same fidelity as a 24-step trial. If a card is free when the
search budget ends, the seed configuration is run once as a proper 24-step trial and shown beside it.

To finish sooner:

- A finalist is not retrained when its configuration has already been trained at full length. That is the case
  for the seeded trial 1, which is grid run 1 itself: if it is among the three lowest values when the search
  ends, its existing adapter is that finalist, and only the other finalists train. (Amendment 2 ranked the three
  best 24-step trials and added grid run 1 as a fourth full-length run; now grid run 1 takes one of the three
  places when its value earns it. Its value is its step-24 loss on a 62-step schedule, which is not the same thing
  as a 24-step trial's; the result page says so beside the table.)
- Whether or not it is a finalist, grid run 1 stays among the full-length runs the choice rule ranks.
- TEST A measured three trainers on one card at 0.95 times one trainer's throughput, so every card stays one
  worker; the fast kernels with one batch shape and a run without gradient checkpointing were not tested.

### Amendment 5: the edge round, fixed before its first result (Sun 2026-10-11, 5:00 AM ET)

Ilan asked whether parameters on the edges were explored properly and whether the granularity was too small. The
table of the first search (12 trials, in `lab/L16V2_RESULT.md`) answers yes to both, and the coordinator ordered an
edge round. It was written while the two new finalists were training; none of its trials had started.

The evidence, all validation numbers:

- Rank 32 was never tried; rank 16 took 8 of the 12 trials (rank 8 two, rank 4 two).
- The best 24-step trial (12: learning rate 4.28e-4, rank 4, dropout 0.136, warmup 0.009, value 0.0142) sits near
  the top of the learning-rate range, at the bottom of the ranks, near the top of dropout and the bottom of warmup
  at once. Nothing was tried above 4.3e-4, or between 2.5e-4 and 4.3e-4.
- Five trials used a learning rate below 1e-4 (2.55e-5 to 8.03e-5); all five were pruned.
- The five best values (0.0133, 0.0142, 0.0150, 0.0157, 0.0159) are closer together than any measured noise,
  because none was measured: no configuration ran twice.
- A 24-step trial cannot see what the continuous ranges were asking of it: a warmup difference of 0.04 is under
  one optimizer step.

The edge round (`lab/l16v2/bo_edge.py`), same fidelity (24 optimizer steps, validation at steps 12 and 24) and
the step-12 rule of amendment 4 (the median over all earlier step-12 losses, the first search's included):

- Space: learning rate 8e-5 to 1e-3 on the log scale; rank in {2, 4, 8, 16, 32, 64}; alpha = rank times 1 or 2;
  dropout in {0, 0.05, 0.10, 0.15}; warmup in {0, 0.05, 0.10}.
- The study. Optuna does not let a categorical's choices or a parameter's type change inside a study, so the
  round continues in study `l16v2-edge` in the same database file. The first search's trials are added to it first
  as finished trials (completed or pruned, as they were), with their dropout and warmup rounded to the nearest of
  the new values. The four trials whose learning rate is below the new floor (3, 4, 8 and 10) cannot be stored in
  the new space and are left out; they all lost, and the sampler can no longer propose their region.
- Forced first, in this order, because the sampler skipped them: rank 32 at 2e-4; rank 16 at 4e-4; rank 32 at
  4e-4; rank 4 at 8e-4; rank 2 at 4e-4 (each with alpha twice the rank, dropout 0.05, warmup 0.05). Then the
  sampler chooses.
- The noise floor of a trial, measured by two repeats that are not study trials and are never pruned: trial 9's
  configuration with seed 43 (it had 42), and the seed configuration (1e-4, rank 16, alpha 32, dropout 0.05, warmup
  0.10) as a proper 24-step trial. The two differences are reported, and the floor used is the larger. (The second
  difference mixes noise with a difference of schedule: trial 1's 0.0133 is step 24 of a 62-step run. It is used
  as instructed and labeled.)
- Order of work: two forced corners, the two noise repeats, the other three corners, then the sampler's points.
- Budget: 20 minutes from the first edge trial's start; no trial starts that would pass it. A card starts taking
  edge trials the moment its finalist's adapter is on the workstation and verified. While the validation agreement
  of the finalists is still being generated on a card, an edge trial there shares it (slower steps, same losses).
- Every line carries `seen`, the finished trials the sampler had, and `kind`: edge, noise or model.

What the edge round can change:

- If an edge-round trial (edge or model) completes with a value lower than the best 24-step value of the first
  search (0.0142) by more than the noise floor, the best such configuration trains once at full length as a fourth
  finalist (the same full-length rule: at most 62 steps, validation every 8, best checkpoint kept, stop after 2
  evaluations without a 3 percent gain), and the frozen choice rule is applied again to all full-length runs: the
  3 lowest validation losses, then validation tool-call agreement.
- If none does, one RESULT line says so and the choice among the three finalists stands.
- Until that is settled the held-out side is not read on the A100: the scoring script on the VMs refuses it until
  the driver leaves a FINAL_CHOICE file. (The pod D seat scores each finalist as it arrives, on the coordinator's
  instruction; amendment 3 covers what those readings may not touch.)

Reported at the end: best value per rank and per learning-rate band, the noise floor, the edges now covered.

## The pod D retrain (second pass)

The chosen recipe is repeated on pod D with the same flags and `--stop-step <the chosen run's best step>`, so the
schedule is the same and the run ends at the checkpoint the A100 kept. It is one run, scored once, the same way.
It is compared with the A100-trained adapter side by side; neither replaces the other after the scores are seen.

## What is never done

- The held-out side of either dataset chooses nothing: no grid point, no checkpoint, no early stop, no rerun.
- No second sweep and no further tuning after a held-out number is seen. If the retrained model is not better, the
  page says that.
