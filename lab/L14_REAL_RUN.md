# L14 real run: label agreement against the teacher

Everything in `tests/test_lab_label_agreement.py` runs on the fake teacher. The measurement that L14's Done when
asks for (30 real labels, asked again by the real teacher) has to be run where the labeled export and the teacher
are reachable. This session could not reach the teacher, a pod, the hub or Notion.

## Before

- L13's labels must exist: a dataset folder (`data/exports/<dataset_id>/`, or the fixture) with at least 30
  examples labeled by `lab/label.py`. With fewer the run ends with verdict `insufficient`.
- The teacher is served on port 8000 of a pod under the model name `teacher` (`lab/pod_a_teacher.sh`). For the
  tool-calling fixture it needs `--enable-auto-tool-choice` and the tool-call parser (see lab/README.md).
- Run from the box that holds the export (not on a pod: `lab/label.py` refuses an export there).

## Command

    python -m lab label-agreement data/exports/<dataset_id> --account C --sample first --seed 1 \
        --out reports/L14

Env var names read (values are never printed or stored): `AUP_HUB_USER_C`, `AUP_HUB_TOKEN_C`, and optionally
`AUP_HUB_URL_C` or `LAB_HUB_URL` (hub address), `LAB_HUB_USER_AGENT`. Another account letter changes the suffix.
Optional for the ledger row: `LAB_ACTOR`, `LAB_TIER`, `LAB_TASK` (= L14), `LAB_STATE_DIR`.

For a teacher reached directly instead of through the hub: `--url http://<host>:8000` and no `AUP_*` names.

## After

1. `reports/L14/label_agreement.json` and `.md` hold the rates, the disagreements and the verdict; the ledger has
   one `label-agreement` row (`label.agreement_rate`, `label.agreement_sample`, verdict as the outcome).
2. The threshold used is 0.9 unless LT2 in `lab/refine/learning/RUBRIC.md` has since named a number. The report's
   `proposed_rubric_line` is the line for the merge pass to put into LT2, citing this run's `run_id`. Tasks never write
   rubrics, so this branch does not touch `lab/refine/`.
3. Done when still open after this branch: the real numbers, and the threshold line in the rubric (merge pass).
