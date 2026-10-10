# Owner rulings for the kernels track

Recorded fixture for tests/test_lab_mirror.py, in the format of lab/refine/TEMPLATES.md section 2.

## 4. Must a kernel win beat the run-to-run spread, or a fixed percentage? (pass 3)

Now: the spread, as kernel test 3 says; E3 uses five runs per point.
Recommend: the spread. A fixed percentage ignores how noisy a pod is on a given day, and the spread costs only the repeat runs already added.
Source of tension: Notion task O4, "Kernel tests" 3 (a win must exceed the spread) against the 15-minute slot the GPU queue gives a benchmark, which five runs at five batch sizes may not fit.
Answer: ____

## 5. Does a derived J/token count as a measurement? (merge 1)

Now: a derived J/token is reported with low_confidence true.
Recommend: keep it derived and flagged until an energy window per request exists.
Source of tension: lab/refine/kernels/RUBRIC.md K2 against lab/refine/metrics.jsonl teacher.joules_per_token.
Answer: keep it flagged as low confidence (owner, 2026-10-10)
