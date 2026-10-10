# Owner rulings for the telemetry track

Recorded fixture for tests/test_lab_mirror.py, in the format of lab/refine/TEMPLATES.md section 2.

## 1. Is the pod's idle draw subtracted before J/token is reported? (pass 1)

Now: idle is not subtracted; the whole-board figure is reported with the idle draw named beside it.
Recommend: report both, whole-board first, so a number never changes meaning between passes.
Source of tension: lab/refine/telemetry/RUBRIC.md T4 against lab/refine/research/telemetry notes on idle subtraction.
Answer: ____
