# Label agreement: tool-calling-fixture (agr-label-20261011T060815Z-ac17)

Verdict: **inconsistent** - overall agreement 0.833 against a threshold of 0.90 (default: lab/refine/learning/RUBRIC.md LT2 names no number, so 0.9 is used).
Sample: 30 of 30 wanted (first); second pass seed 1 (first 0), prompt reworded.

| field | agree | total | rate |
| --- | --- | --- | --- |
| answer | 20 | 30 | 0.667 |
| tool_calls | 30 | 30 | 1.000 |
| overall | 50 | 60 | 0.833 |

| example | field | first | second |
| --- | --- | --- | --- |
| golden-01 | answer | Overnight, the table was empty at 11 PM with a phone char... | Overnight, the table was empty with a phone charger plugg... |
| golden-01-1 | answer | Overnight, a person in a hoodie walked past the door arou... | Overnight, a person in a hoodie walked past the door arou... |
| golden-01-2 | answer | Last night, two people worked at the table with laptops a... | Last night, two people worked at the table with laptops f... |
| golden-01-3 | answer | Last night (Saturday) two people worked at the table with... | Last night (Saturday) two people worked at the table with... |
| golden-01-5 | answer | Last night, two people worked at the table with laptops a... | Last night, two people worked at the table with laptops a... |
| golden-02 | answer | Your keys were last seen at 7:41 PM in the desk zone on c... | Your keys were last seen at 7:41 PM on Saturday in the de... |
| golden-02-4 | answer | The keys were last seen at 7:41 PM in the desk zone on ca... | The keys were last seen at 7:41 PM on Saturday in the des... |
| golden-03-1 | answer | Yes, a courier dropped a cardboard box at the door at 9:0... | Yes, a courier was at the door at 9:05 AM, dropping off a... |
| v-1ad7bdb7 | answer | Your keys were last seen at 7:41 PM on Saturday in the de... | Your keys were last seen on Saturday at 7:41 PM in the de... |
| v-fb73a588 | answer | Your keys were last seen at 7:41 PM on Saturday in the de... | Your keys were last seen at 7:41 PM on Saturday in the de... |

Proposed RUBRIC line (for the merge pass):

> LT2 label agreement: a second teacher pass (seed 1, reworded prompt) over 30 labeled examples must agree field by field on at least 0.90 (default: lab/refine/learning/RUBRIC.md LT2 names no number, so 0.9 is used); measured 0.833, verdict inconsistent. Source: lab/label_agreement.py run agr-label-20261011T060815Z-ac17 (ledger verb label-agreement, refs run_id agr-label-20261011T060815Z-ac17), dataset tool-calling-fixture.
