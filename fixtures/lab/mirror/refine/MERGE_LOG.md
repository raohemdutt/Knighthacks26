# Merge log for the four refinement tracks

Recorded fixture for tests/test_lab_mirror.py, in the merge-entry format of lab/refine/TEMPLATES.md (section 1.4).

---

## Merge 1, 2026-10-10

Reconciled the first round: telemetry pass 1 and kernels pass 3. One board change.

Walked (pass entry, task, demo beat or file :: checks :: result):
- telemetry pass 1 :: M6 :: held
- kernels pass 3 :: M6 :: held
- T3 :: M1 :: BROKE M1; FIXED (1)

Findings (by dimension):
- M1 T3 :: waits on a task of the learning track that is not in its Blocked by. FIXED (1).

Changes (1 of at most 10):
1. T3 on the board, Blocked by gains L4 (M1)
   BEFORE: "Blocked by: T1"
   AFTER:  "Blocked by: T1, L4"
   SOURCE: Notion task T3, "Done when" (the sampler reads the eval gate's run id); lab/refine/learning/RUBRIC.md, LT4
   WHY: the sampler cannot tag a run before the gate that names it exists.

Left for the next pass: re-walk T3 once L4 is Done.

---
