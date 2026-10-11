# Label agreement spot-check: `python -m lab label-agreement`

Moved here from lab/README.md (O70); the verbs list there links every page. Arguments: `python -m lab label-agreement -h`.

`lab/label_agreement.py` asks the teacher again about 30 labeled examples (the first 30, or `--sample seeded
--sample-seed N`) with another sampling seed (`--seed`, default 1; the labels were made with 0) and a reworded prompt
(`REWORD_PREFIX` in front of the instruction), compares the two labels one top-level field at a time (`tool_calls`
and `answer` for the fixture; free text and lists are one field, `label`), and writes `label_agreement.json` and
`label_agreement.md` (`--out`) with the agreement rate per field and overall, every disagreement (example id, field,
first value, second value), the examples the second pass could not label (counted as disagreeing) and a verdict:
`consistent` at or above the threshold, `inconsistent` below it, `insufficient` when fewer than 30 are labeled. The
threshold is the number LT2 in `lab/refine/learning/RUBRIC.md` names for the agreement check; it names none today,
so the default is 0.9 and the report says so (`--threshold` overrides). The report holds the rubric line the merge
pass should add, with its citation; nothing under `lab/refine/` is written. One ledger row (verb `label-agreement`,
outcome the verdict, `label.agreement_rate`, `label.agreement_sample`, `label.agreement_disagreements`). The dataset
is never written; the sample is relabeled in a temporary copy. Tests use the fake teacher (`swayed_policy` makes it
answer differently on a second seed); the run against the real teacher is in `lab/L14_REAL_RUN.md`.
