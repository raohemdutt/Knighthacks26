# Refinement log for the kernels track

Recorded fixture for tests/test_lab_mirror.py, in the formats of lab/refine/TEMPLATES.md.

---

## Pass 3, 2026-10-10

EXAMPLE, not the record of a real pass. The quoted task text and the baseline figures are real (Notion board and lab/README.md, 2026-10-10); the pass around them is invented. Walked the kernels items on the experiment queue (E2, E3, E5) and R24 against kernel tests K1 to K8 and the two universal rules U1 and U2. Three changes, all to the plan, none measured yet. NOT CONVERGED; pass 4 repeats the walk.

Research step (sources opened):
- Notion task O4, sections "Kernel tests" and "Baseline to beat" (https://app.notion.com/p/3f596ef5fec5816aae04d8fb488302c4), opened 2026-10-10 :: the eight kernel tests and the baseline (decode 7.7 tok/s, about 26 J per token); test 1 asks for stock numbers at batch 1, 4, 8 and 16.
- lab/README.md, section "Teacher (pod A)", on main at bb351e6, opened 2026-10-10 :: the measured table, about 200 W average and 300 W peak, and the paragraph on the Quark checkpoints that vLLM b1388b1 rejects.
- Jae-Won Chung et al., "The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization", NeurIPS Datasets and Benchmarks 2025, https://arxiv.org/abs/2505.06371, opened 2026-10-10 :: the abstract reports energy measured under realistic serving conditions and optimization savings above 40 percent in some cases; it gives no batch-size figures.

Metrics before (read before changing anything):
| metric | value | unit | node | conditions | source |
| --- | --- | --- | --- | --- | --- |
| teacher.decode_tok_s | 7.7 | tok/s | pod-a | RedHatAI/Qwen3.8-27B-INT4, vllm@b1388b1, batch 1, ctx 16384, text prompt, thinking on, 160 output tokens | metrics.jsonl:3 |
| teacher.watts_avg | 200 | W | pod-a | same model and engine, ctx 16384, while generating | metrics.jsonl:7 |
| teacher.joules_per_token | 25.97 | J/token | pod-a | derived: teacher.watts_avg / teacher.decode_tok_s, marked low confidence in the file | metrics.jsonl:12 |
| teacher.decode_tok_s | none | tok/s | pod-a | batch 4, 8 and 16, same model and engine; never measured, measurement queued as E3 | none |

Walked (task or experiment :: dimensions checked :: result):
- E2 :: K2, K6, K8, U1, U2 :: BROKE K8; FIXED (3)
- E3 :: K1, K3, K6, K8, U1, U2 :: BROKE K1, K3, K6, K8; FIXED (1, 3)
- E5 :: K2, K6, K8, U1, U2 :: BROKE K2, K8; FIXED (2, 3)
- R24 :: K2, K3, K4, K5, K6, K7 :: held

Findings (by dimension):
- K1 E3 :: sweeps batch 1, 4, 16 and 64, while kernel test 1 requires stock numbers at 1, 4, 8 and 16, so batch 8 is missing and the curve cannot be compared with the baseline. FIXED (1).
- K3 E3 :: names no repeat count, so a win cannot be told from the spread. FIXED (1). RULING (a).
- K2 E5 :: one item asks two questions: whether the Quark checkpoint loads on vLLM b1388b1 (lab/README.md says it does not) and how fast the checkpoints that load run. FIXED (2).
- K8 E2, E3, E5 :: no queue item carries an expected movement or a confidence, so the queue cannot be ordered by expected energy gain times confidence. FIXED (3).
- K6 E3 :: reports a J/token curve but names no tok/s, so the speed half of the pair is missing. FIXED (1).
- K6 E2, R24 :: each pairs speed with joules per token. HELD.

Changes (3 of at most 5):
1. E3 on T0, rewritten (K1, K3, K6)
   BEFORE: "E3 batching sweep on vLLM (batch 1, 4, 16, 64: J/token curve)"
   AFTER:  "E3 (kernels) batching sweep on vLLM (batch 1, 4, 8, 16, 64; five runs per point, report the mean and the spread of tok/s and J/token; a point counts as a win only if it clears the batch-1 spread)"
   SOURCE: Notion task O4, "Kernel tests" 1, 3 and 6 (https://app.notion.com/p/3f596ef5fec5816aae04d8fb488302c4); lab/refine/PASS_RULES.md, section "The four tracks" (a queued experiment line names its track)
   WHY: the sweep skipped batch 8, which the stock baseline requires, named no repeat count and reported J/token without tok/s, so no win could be told from noise and speed was not paired with energy; five runs is a starting count, to be raised if the spread is wide.
   METRIC: teacher.joules_per_token, stored at batch 1 only today (25.97 J/token, metrics.jsonl:12); after E3 it is stored at batch 1, 4, 8, 16 and 64.
2. E5 on T0, split into E5a and E5b (K2)
   BEFORE: "E5 Teacher checkpoint shootout: AMD Quark AWQ-INT4 vs RedHat INT4 vs official FP8 (loads? tok/s, J/token, caption quality on 20 fixture frames)"
   AFTER:  "E5a (kernels) Does the AMD Quark AWQ-INT4 checkpoint load on a vLLM commit from late August 2026 or later, or with the mapper backport: yes or no, with the error text. E5b (kernels) Shootout of the checkpoints that load (RedHat INT4, official FP8, and Quark if E5a says yes): tok/s and J/token at batch 1 and 16, caption quality on 20 fixture frames."
   SOURCE: lab/README.md, "Teacher (pod A)", paragraph "Checkpoints" (the Quark checkpoints are rejected by quark.py:apply_vllm_mapper on vLLM b1388b1; loading them needs a newer vLLM commit or a backport of that one file); Notion task O4, "Kernel tests" 2 (one lever per experiment)
   WHY: E5 mixed a load question that can fail with no speed data and a speed comparison, so a failed load would have hidden the other checkpoints' numbers.
   METRIC: teacher.joules_per_token for the Quark checkpoint, none stored today; E5b stores it if E5a passes.
3. Experiment queue on T0, expected movement added to E2, E3 and E5b (K8)
   BEFORE: "(no queue item states an expected movement or a confidence)"
   AFTER:  "E2 expected: J/token differs between the two engines, direction unknown, confidence low. E3 expected: J/token changes with batch size; T0 cites the ML.ENERGY data for a 6 to 8x movement, confidence low. E5b expected: unknown until E5a answers."
   SOURCE: Notion task O4, "Kernel tests" 8 (order by expected energy gain times confidence, with the estimate and the result both on the queue); Notion task T0, "Experiment queue", E3 (the 6 to 8x note, which cites the ML.ENERGY data second-hand); Chung et al., arXiv 2505.06371, abstract (optimization recommendations save more than 40 percent in some cases; no batch-size figure, so the 6 to 8x is not verified here)
   WHY: without an estimate the queue cannot be ordered by expected energy gain times confidence; the estimates say plainly how weak they are, and the first measured result replaces each one.
   METRIC: teacher.joules_per_token, expected to change with batch size (E3); the other items have no numeric expectation yet.

Rulings raised (candidates; the merge pass numbers them into OWNER_RULINGS.md):
(a) Must a kernel win beat the run-to-run spread, or a fixed percentage?
    Now: the spread, as kernel test 3 says; E3 uses five runs per point.
    Recommend: the spread. A fixed percentage ignores how noisy a pod is on a given day, and the spread costs only the repeat runs already added.
    Source of tension: Notion task O4, "Kernel tests" 3 (a win must exceed the spread) against the 15-minute slot the GPU queue gives a benchmark, which five runs at five batch sizes may not fit.
    Answer: ____

Metrics after (only if the pass ran anything):
not run: this pass changed the plan only, so no experiment executed and no metric was re-measured.

Converged: NO - three changes made; pass 4 repeats the walk of E2, E3, E5a, E5b and R24.

Left for the next pass: re-walk every item against K1 to K8 and U1 and U2; check that T0 reads as AFTER for E3, E5a, E5b and the expected-movement lines; E3 has not run, so every batch 4, 8 and 16 reading is still none. Waits on the owner: ruling (a).

Checks run (commands and their actual output):
Re-fetched T0 after changes 1 to 3: E3, E5a, E5b and the expected-movement lines read as AFTER.

---
