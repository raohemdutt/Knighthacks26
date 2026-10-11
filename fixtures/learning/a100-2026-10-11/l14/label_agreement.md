# Label agreement: tool-calling-fixture (agr-a100-20261011)

Verdict: **inconsistent** - overall agreement 0.817 against a threshold of 0.90 (default: lab/refine/learning/RUBRIC.md LT2 names no number, so 0.9 is used).
Sample: 30 of 30 wanted (first); second pass seed 1 (first 0), prompt reworded.

| field | agree | total | rate |
| --- | --- | --- | --- |
| answer | 21 | 30 | 0.700 |
| tool_calls | 28 | 30 | 0.933 |
| overall | 49 | 60 | 0.817 |

| example | field | first | second |
| --- | --- | --- | --- |
| golden-01 | answer | Overnight, the table was empty at 11 PM with a phone char... | Overnight, the table was empty at 11:00 PM with a phone c... |
| golden-01-1 | answer | Overnight, a person in a hoodie walked past the door arou... | Overnight, a person in a hoodie walked past the door arou... |
| golden-01-2 | answer | Last night, two people worked at the table with laptops a... | Last night, two people worked at the table with laptops f... |
| golden-01-3 | answer | Last night (Saturday) two people worked at the table with... | Last night (Saturday) two people worked at the table with... |
| golden-01-4 | answer | While you were asleep, a person in a hoodie walked past t... | While you were asleep, a person in a hoodie walked past t... |
| golden-01-5 | answer | Last night, two people worked at the table with laptops a... | Last night, two people worked at the table with laptops a... |
| golden-02 | answer | Your keys were last seen at 7:41 PM in the desk zone on c... | Your keys were last seen at 7:41 PM on Saturday in the de... |
| golden-02-5 | answer | Your keys were last seen on Saturday at 7:41 PM in the de... | Your keys were last seen at 7:41 PM on Saturday in the de... |
| golden-03-1 | answer | Yes, a courier dropped a cardboard box at the door at 9:0... | Yes, a courier was at the door at 9:05 AM, dropping off a... |
| golden-06-2 | tool_calls | [{"name": "last_seen", "arguments": {"label": "wallet"}},... | [{"name": "last_seen", "arguments": {"label": "wallet"}}] |
| v-b0a83206 | tool_calls | [{"name": "last_seen", "arguments": {"label": "wallet"}},... | [{"name": "last_seen", "arguments": {"label": "wallet"}}] |

Proposed RUBRIC line (for the merge pass):

> LT2 label agreement: a second teacher pass (seed 1, reworded prompt) over 30 labeled examples must agree field by field on at least 0.90 (default: lab/refine/learning/RUBRIC.md LT2 names no number, so 0.9 is used); measured 0.817, verdict inconsistent. Source: lab/label_agreement.py run agr-a100-20261011 (ledger verb label-agreement, refs run_id agr-a100-20261011), dataset tool-calling-fixture.
