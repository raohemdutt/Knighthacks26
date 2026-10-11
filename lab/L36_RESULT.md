# L36: the student retrained on the box's own data pooled with v2 (Sun 2026-10-11)

Decision: NO-GO. f2r4 (`atlas-tool-calling-v2-a100-f2r4`) stays the student. Training (a) did not beat f2r4 by
more than 7 questions on either held-out set. Training (b) was stopped at Ilan's order at about step 31 and was
never scored.

Every number below is traced to a file under `lab/l36/record/`.

## The data

- v2 and v3 rows share one format: the same keys (id, frame_ref, messages, tools, text, spans), the same three
  tools and the same instruction. No row was converted. The check is in `lab/l36/pool.py`, which refuses a
  mismatch.
- Pooled set `tool-calling-v3v2-pooled` (`lab/l36/pool.py`, record `lab/l36/record/pool.json`):
  - Train: v3's 721 rows, then v2's 840 (1,561 rows).
  - Validation: v3's 126 and v2's 126 (252 rows).
  - Held out: the pooled split puts both held-out sides there (v3's 240 and v2's 228), so the trainer refuses
    any of them as a row.
  - Output sha256s: rows-train `1fd5a39f...63d3`, rows-validation `358c9adf...a718`, split `de00052f...2655`.
    The trainer's run record repeats all three (`l36a-f2r4-pooled/run.json` refs), and the VM's rebuild matched
    the workstation's.
- v2's inputs came from the L16v2 run: `dataset-tool-calling-v2.tar.gz`, sha256 `816847a9...e703`, with
  rows-train sha256 `64e3d422...9d`. Both match the frozen values in `lab/L16V2_RESULT.md`.
- No held-out group of either set appears on the other set's train side. Checked by group name (frame_ref): v3
  holds out keys, backpack, cup, suitcase and 16 scenes; v2 holds out 10 objects and 8 window lengths.

## The trainings (Colab A100-SXM4-80GB, session l36-a100, `lab/l36/colab_job.py`)

Both were capped at 62 steps (batch 4 x 4), with validation every 8 steps, the best checkpoint kept, and an early
stop after 2 evaluations that did not beat the best by 3 percent.

| run | recipe | ran | steps run | kept step (example passes) | best val loss | validation agreement v3 / v2 (of 126 each) |
| --- | --- | --- | --- | --- | --- | --- |
| (a) `l36a-f2r4-pooled` | f2r4: lr 0.000427627, rank 4, alpha 8, dropout 0.136, warmup 0.0093 | 27.9 min (11:57:12 to 12:25:08 UTC) | 56 (early stop) | 40 (640) | 0.00554 | 120 / 116 |
| (b) `l36b-v3default-pooled` | v3 default: lr 1e-4, rank 16, alpha 32, dropout 0.05, warmup 0.10 | started 12:41:25 UTC, killed at about step 31 on Ilan's order | n/a | n/a | 0.01206 at step 16 (last seen) | not scored |

For reference on the same validation sets: f2r4 scores 110 / 113 and base 51 / 69.

Sources:
- (a)'s losses: `l36a-f2r4-pooled/validation.jsonl`. They were 0.01441, 0.01072, 0.00902, 0.00581 and 0.00554
  at steps 8 to 40, then 0.00574 and 0.00596 at steps 48 and 56.
- (a)'s step counts: `l36a-f2r4-pooled/run.json` metrics (trainer.best_step 40, trainer.steps 56).
- Validation agreement: `scores-table.json`, sets `v3_validation` and `v2_validation`.
- (b)'s losses (0.02870 at step 8, 0.01206 at step 16, 0.01379 at step 24) were read from the VM's log before the
  kill. They were not downloaded.

Training (b) started 16 minutes late because of a bug. The job's 35-minute check read the date column of its
times file instead of the seconds column, and the bash arithmetic error skipped (b). It is fixed in the job
script, and (b) was started by hand.

f2r4's v2 validation score here (113/126) and base's (69/126) equal the v2 record's numbers in
`lab/L16V2_RESULT.md`. That shows this scoring path reproduces the v2 search's.

## Held-out scores (A100, transformers greedy on bf16 weights with the adapter attached)

The scorer is `lab/l36/vm_score.py`. It applies `lab/tool_call_agreement.py`'s rule unchanged, with each
example's own clock through `system_for()`, and generates with `lab/l16v2/sweep.py`'s Generator, as the v2 search
scored. The rule sha256 is in `scores-table.json`.

| arm | v3 held out, right of 240 | of which last_seen (48) | of which episodes_between (192) | v2 held out, right of 228 | of which last_seen (100) | of which episodes_between (128) |
| --- | --- | --- | --- | --- | --- | --- |
| base | 109 | 45 | 64 | 136 | 93 | 43 |
| f2r4 | 179 | 48 | 131 | 174 | 100 | 74 |
| new (a) | 180 | 48 | 132 | 179 | 100 | 79 |
| new (b) | not scored (stopped) | | | not scored | | |

New (a) against f2r4, paired on the same examples (`pairs/*.f2r4-vs-a.md`):

| set | f2r4 | (a) | difference | 0 to 1 | 1 to 0 | exact McNemar p | sign-flip p (groups) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v3 held out (240, 20 groups) | 179 | 180 | +1 | 10 | 9 | 1 | 1 |
| v2 held out (228, 18 groups) | 174 | 179 | +5 | 8 | 3 | 29/128 = 0.227 | 0.125 |

Both adapters against base (`pairs/*.base-vs-*.md`):

| comparison | McNemar p |
| --- | --- |
| f2r4, v3 | 109 to 179, p = 1.0e-17 |
| f2r4, v2 | 136 to 174, p = 7.0e-08 |
| (a), v3 | 109 to 180, p = 8.4e-16 |
| (a), v2 | 136 to 179, p = 7.1e-10 |

## The rule and the decision

- The rule: GO if a new adapter beats f2r4 on both held-out sets by more than 7 questions, the between-seed
  swing measured in the v2 search. Otherwise NO-GO.
- (a) beats f2r4 by 1 (v3) and 5 (v2). Both are inside the noise, and neither McNemar p is below 0.2. NO-GO.
- (b) has no score. It was stopped before the end, so it could not change the decision in time.
- f2r4 stays, and nothing was packaged. The llama.cpp toolchain was staged on the VM (`lab/l36/colab_llama_prep.sh`:
  llama.cpp 23b0202a, llama-quantize built in 136 s, mmproj sha256 cd88edcf...f864 checked) and was not used.

## Caveats

- f2r4's 179 of 228 in `lab/L16V2_RESULT.md` was scored on pod D's Q4_K_M GGUF on llama-server. Here f2r4 scores
  174 of 228 in bf16 on the A100, and base scores 136 here against 95 there. Compare only numbers scored the same
  way. Every number in the tables above is A100 bf16.
- One run per arm. Greedy decoding, so a rerun gives the same outputs. The noise bar of 7 is between seeds of
  training, not between scoring runs.
- The base and f2r4 outputs were generated while training (a) ran on the same card. That slowed both but does not
  change greedy outputs.
- The held-out sides were read after (a)'s checkpoint was chosen by validation loss alone. Nothing here was chosen
  by a held-out number.
- (a) is better than f2r4 on both validation sets (+10 on v3, +3 on v2), but validation is not the test. The v3
  validation gain did not carry to v3's held-out side (+1).

## Files

- `lab/l36/pool.py`: the pooled dataset.
- `lab/l36/colab_job.py`: the two trainings.
- `lab/l36/vm_score.py`: generation and scoring.
- `lab/l36/colab_llama_prep.sh`: the packaging toolchain, staged and unused.
- `lab/l36/record/downloads.txt`: every product pulled, with bytes and sha256, each checked against the VM's own
  .sha256 file.
  - `run-l36a-f2r4-pooled.tar.gz`: 33,977,350 bytes, sha256 `5cfdf9a9...3dcc`. Its adapter_model.safetensors has
    sha256 `86ba0cd2...98c7`.
  - `score-l36.tar.gz`: 608,106 bytes, sha256 `f25733a8...5815`.
  - Both are kept on the workstation under `work/L36-out/colab/`, not committed.
- `lab/l36/record/`: also holds `pool.json`, `scores-table.json`, `gen-times.txt`, `pairs/` (the scorer's
  markdown per pair) and `l36a-f2r4-pooled/` (run.json, validation.jsonl, stdout, rc).
- f2r4's input was the L16v2 download `run-final2-t12-lr0.000428-rank4-a8-drop0.136-warm0.009.tar.gz`. Its
  sha256 `d1253d8c...2d01` matches `lab/l16v2/record/downloads.txt` on the v2 merge branch. Its
  adapter_model.safetensors has sha256 `80bad80b...1c13`.
