# Label agreement: tool-calling-fixture (agr-label-20261011T061334Z-79e0)

Verdict: **consistent** - overall agreement 0.983 against a threshold of 0.90 (default: lab/refine/learning/RUBRIC.md LT2 names no number, so 0.9 is used).
Sample: 30 of 30 wanted (first); second pass seed 1 (first 0), prompt NOT reworded (R64 control arm: lab/label_agreement.py check_agreement with the rewording prefix set to empty, run by hand, so no label-agreement ledger row).

| field | agree | total | rate |
| --- | --- | --- | --- |
| answer | 29 | 30 | 0.967 |
| tool_calls | 30 | 30 | 1.000 |
| overall | 59 | 60 | 0.983 |

| example | field | first | second |
| --- | --- | --- | --- |
| golden-01-2 | answer | Last night, two people worked at the table with laptops a... | Last night, two people worked at the table with laptops f... |

Proposed RUBRIC line (for the merge pass):

> LT2 label agreement: a second teacher pass (seed 1, prompt not reworded, R64 control arm) over 30 labeled examples must agree field by field on at least 0.90 (default: lab/refine/learning/RUBRIC.md LT2 names no number, so 0.9 is used); measured 0.983, verdict consistent. Source: lab/label_agreement.py run agr-label-20261011T061334Z-79e0 (ledger verb label-agreement, refs run_id agr-label-20261011T061334Z-79e0), dataset tool-calling-fixture.
