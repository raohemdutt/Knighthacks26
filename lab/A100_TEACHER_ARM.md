# A100 teacher arm: the gate's teacher scores and a second label pass, run on the Colab A100 (the other chip)

Run on 2026-10-11, 05:52 to 06:06Z, Colab A100 40 GB, stock vLLM 0.31.0 serving RedHatAI/Qwen3.8-27B-INT4 as `teacher` (greedy, thinking off, max 512 tokens, tool parser qwen3_coder), repo main 2a7d3ea for the scorer. Files under `fixtures/learning/a100-2026-10-11/`. The seat that ran it was closed on Ilan's order before it could commit; this file and the fixtures were committed by the dispatcher from the pulled tarballs.

## L4, the teacher arm of the eval gate, scored twice

| run | score | citations_ok | answer_ok | scorer_score | seconds |
| --- | --- | --- | --- | --- | --- |
| teacher.run_1 | 0.9167 | 1.000 | 0.9167 | 0.9583 | 16.0 |
| teacher.run_2 | 0.9167 | 1.000 | 0.9167 | 0.9583 | 15.9 |

Run-to-run noise 0.0 (max minus min over the items both runs scored). 24 scorable items of the 58 held-out, in 4 golden groups; 34 held-out items are unscored by construction of the committed fixture (see `scores.json`). Three items produced different answer text between the two runs with the same score (golden-01, golden-01-4, golden-02-5). The decode probe found the two runs not token-identical (`decode_probe.json`), so this is the gate's own noise rule applied to the teacher: zero on the score, not zero on the text.

What it means against the baseline: the gate had never scored a real teacher before this run (lab/L4_REAL_RUN.md). A candidate student adapter now has a measured teacher ceiling of 0.9167 on this split to be held against, and the teacher's own noise is zero at this sample size.

## L14, a second label pass on the A100 teacher

`python -m lab label-agreement`, 30 examples, second pass with seed 1 and a reworded prompt, compared field by field against the committed labels (which the old Strix Halo build produced).

| field | agree | total | rate |
| --- | --- | --- | --- |
| tool_calls | 28 | 30 | 0.933 |
| answer | 21 | 30 | 0.700 |
| overall | 49 | 60 | 0.817 |

Verdict by the harness: inconsistent against its default 0.9 line. Reading: the tool calls, which are what the student learns, agree on 28 of 30; the free-text answer field is compared as an exact string, and most of its 9 disagreements are wording ("11 PM" against "11:00 PM", "laptops and" against "laptops for"), not different facts. This pass also crosses a chip and an engine version (A100 stock v0.31.0 against the Strix Halo April build that wrote the labels), so it is an upper bound on the disagreement a same-chip rerun would show. R64 reruns it on the Strix Halo teacher.

## Conditions

node a100, hub colab, other_chip true; the equivalence citation is the R57 cross-chip correctness record (`fixtures/lab/bench/a100-2026-10-10/vllm-a100-2026-10-10.json`), and the April build matched the A100 on 1 of 4 greedy prompts exactly, so scores here are the A100's, not the pod's.
