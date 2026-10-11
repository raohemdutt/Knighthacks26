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

## The pod D retrain (second pass)

The chosen recipe is repeated on pod D with the same flags and `--stop-step <the chosen run's best step>`, so the
schedule is the same and the run ends at the checkpoint the A100 kept. It is one run, scored once, the same way.
It is compared with the A100-trained adapter side by side; neither replaces the other after the scores are seen.

## What is never done

- The held-out side of either dataset chooses nothing: no grid point, no checkpoint, no early stop, no rerun.
- No second sweep and no further tuning after a held-out number is seen. If the retrained model is not better, the
  page says that.
