# Handoff: make the Learning lab show real runs

From Hemdutt's agent to Ilan1, Sat 2026-10-10. Everything described as "exists" is merged on `main` (PRs #242, #295) and
tested. What is left is on the lab's side of the fence: it needs the pods, the trainer and the lab's ledger.

## What the owner wants

The Learning lab screen (`/lab/`) should show the lab's real work to a judge: the fine-tune as it ran (loss, power),
what the model got right before and after, the gate's verdict, the swap onto the box, the benchmark numbers, and the
experiments with their verdicts. Today it shows a labelled sample run plus the real rows of `lab/refine/metrics.jsonl`.

## What exists

| Piece | Where | Reads |
| --- | --- | --- |
| Training run tab (replay) | `web/app/lab/`, `web/components/lab/LabRun.tsx`, `contexts/learning/application/lab_view.py`, `lab_api.py` | events in the box's log; a sample from `fixtures/learning/sample_run/` only when the log has no real run |
| Speed and energy tab | `web/components/lab/SpeedEnergy.tsx`, `contexts/learning/application/lab_bench.py` | `lab/refine/metrics.jsonl`, parsed directly |
| Routing tab (what-if) | `web/components/lab/Routing.tsx`, `contexts/learning/application/lab_routing.py` | the same rows; calls the real `runtime.routing.route` and the `capability` / `energy` policies |
| Experiments tab | `web/components/lab/Experiments.tsx`, `contexts/learning/application/lab_experiments.py` | the lab ledger if present (`LAB_STATE_DIR`, else the copy pushed to `<data_dir>/lab_state/ledger.jsonl`, else `lab/state/ledger.jsonl`), the seeded queue, `REFINE_LOG.md` passes |
| Import of a run, and of the ledger | `contexts/learning/lab_import.py` | a `telemetry.jsonl`, a gate `scores.json`, a package `manifest.json`, the lab's `ledger.jsonl` |
| Run folder reader | `contexts/learning/application/lab_runs.py` | `<data_dir>/lab_runs/<any>/telemetry.jsonl` (+ `run.json`), or `ATLAS_LAB_RUNS_DIR` |

Endpoints: `GET /api/learning/lab`, `GET /api/learning/lab/{run_id}`, `GET /api/learning/lab/bench`,
`POST /api/learning/lab/routing/simulate`, `GET /api/learning/lab/experiments`, `POST /api/learning/lab/import`.

## 1. Push L16's run to the box (do this first)

`lab/train.py` writes one `telemetry.train.step` per optimizer step to `<out>/telemetry.jsonl` and a `run.json` at the
end. Nothing publishes them. Push them yourself, while the run is going or after it:

```
curl -X POST --data-binary @<out>/telemetry.jsonl \
  -H "Authorization: Bearer $ATLAS_TOKEN" \
  "http://100.124.34.36:8000/api/learning/lab/import?run_id=<run_id>"
```

- Importing twice adds nothing: a step is keyed by run id and step. A file that has grown adds only its new steps, so
  you can re-post it every minute during the run and the screen plays it live (it polls every 5 s while a run trains).
- Any line that does not parse as `telemetry.train.step`, or belongs to another run, refuses the whole file (400) and
  names every bad line. The body is capped at 8 MB (413).
- The route is behind the gateway's token like other writes. With no token configured on the box it is open.
- From a checkout that has the files: `python -m contexts.learning.lab_import --telemetry <out>/telemetry.jsonl
  [--run-id <id>] [--eval scores.json] [--manifest manifest.json] [--data-dir data]`. This appends to the log on disk; a
  gateway already running reads the log only at start, so use the POST for a live box.

Then the gate and the package, when they exist:

```
curl -X POST --data-binary @<gate out>/scores.json  -H "Authorization: Bearer $ATLAS_TOKEN" "http://100.124.34.36:8000/api/learning/lab/import?kind=eval"
curl -X POST --data-binary @<pkg>/manifest.json     -H "Authorization: Bearer $ATLAS_TOKEN" "http://100.124.34.36:8000/api/learning/lab/import?kind=manifest&run_id=<run_id>"
```

`scores.json` must carry the gate's own `learning.run.evaluated` under `"event"` (it does today, `lab/eval_gate.py`).
The manifest must validate as `ModelManifest`.

**Done when:** `/lab/` on the box shows the real run with no "Sample run" banner, the loss curve and step timing from
your file, and after the eval import the gate's scores and verdict.

## 2. Fill the before/after list (the most valuable gap)

The screen's strongest section lists each held-out question with the answer before and after, regressions first. Today
it says "not in the log" for a real run, because `scores.json` has no per-question results.

The view already accepts an optional event, `learning.eval.reported` (defined in `lab_view.py`, not in `contracts/`):

```
{type: "learning.eval.reported", run_id, task, n_items, n_groups,
 correct: {current, candidate, teacher}, noise, noise_runs: [..], p_value, alpha, test, decode, split_sha256,
 low_confidence, items: [{id, group, question, expected, before: {answer, correct}, after: {answer, correct}}]}
```

`lab_import.eval_events` builds it from `scores.json` using only `n_items, n_groups, noise, p_value, split_sha256,
decode`. To fill the list, add to the gate's `scores.json` (your file, `lab/eval_gate.py`): `correct` (right-answer
counts per role), and `items` as above (the gate already scores each item per role; keep the question text and the
expected answer from the split, and each role's output and pass flag). Then extend `eval_events` to pass `correct`,
`items`, `noise_runs`, `alpha`, `test` through when present (a few lines; test in
`contexts/learning/tests/` next to the existing import tests, with a fixture written by the gate's own code).

Rules that matter: held-out items only; no household free text beyond what the export already approved; an item's
`group` is the frame group or episode the gate pairs on.

**Done when:** after `?kind=eval` the Before/after section lists real questions, flips are marked, and the summary bar
reads "n right before → n right after" from the gate's own counts.

## 3. Experiments with real verdicts

The Experiments tab reads the lab ledger through the lab's own `experiments.fold` and shows won / lost / inconclusive /
running / dropped. No ledger exists on the box or on the owner's Mac, so it shows the five seeded experiments as
queued and the refinement passes as "not converged".

Push the ledger to the box, whole, whenever it has changed (from the folder that holds it: `$LAB_STATE_DIR`, else
`lab/state/`):

```
curl -X POST --data-binary @ledger.jsonl "http://100.124.34.36:8000/api/learning/lab/import?kind=ledger"
```

No token is set on the box today. If one is set later, add `-H "Authorization: Bearer $ATLAS_TOKEN"` as for the run
import.

- The box keeps the file as it is at `<data_dir>/lab_state/ledger.jsonl` (never in `lab/state/` of its checkout) and
  the Experiments tab reads it on its next request. No environment variable, no restart per push. The tab says
  "Ledger received from the lab at 1:42 AM, 37 rows".
- The answer: `{"kind": "ledger", "changed": true, "rows": 37, "experiments": 9, "verdicts": {"a_ahead": 1,
  "b_ahead": 2, "too_early": 1}, "bytes": 41873, "received_at": "2026-10-11T05:42:10Z", "stored":
  "lab_state/ledger.jsonl"}`. The same bytes again write nothing: `"changed": false` and a note saying so.
- Every line must be a row the lab's own `ledger.validate` accepts, with an integer `seq`. One that is not refuses the
  whole body (400) and every bad line is named by number and field (`line 12: actor is missing`), never by its
  contents; the copy already on the box stays as it was. A half-written last line counts as a bad line, so push
  between appends, not during one. The body is capped at 8 MB (413).
- Each line is also checked by the repo's credential scan (the one `tests/test_no_secrets.py` runs over `lab/state`).
  A line that holds something shaped like a token, a key or a password refuses the body: `line 12: holds what looks
  like a credential (hub token)`. Take it out of the ledger, and replace the credential.
- What the box shows from the ledger is what it showed before: experiment ids, titles, levers, arms, hypotheses,
  citations, the numbers and the verdicts. A row's `note`, `source`, `conditions`, `actor` and `refs` are stored with
  the file and served nowhere.
- From a checkout on the box: `python -m contexts.learning.lab_import --ledger <file> [--data-dir data]` writes the
  same copy through the same checks; a running gateway picks it up without a restart.
- The route arrives with this code, so the box needs one pull and one restart of `atlas-gateway` to have it (ask
  Xander or Dominick, the box is theirs to restart). After that, a push needs neither.

Order of reading: `LAB_STATE_DIR` when it is set and holds a ledger, then the pushed copy, then `lab/state/` in the
box's checkout. Setting `LAB_STATE_DIR` in the box's `.env` and refreshing that folder with `rsync`/`scp` still works,
but it needs a restart to pick up the variable; the push is the way to do it.

**Done when:** `/lab/?tab=experiments` on the box shows your experiments with their verdicts and sources.

## 4. Smaller, if you have room

- **A scorer for the SPOT before/after.** The set is `python -m contexts.spot.dataset --out <dir>` (574 train in
  `spot-qa-train/`, 240 held out in `spot-qa-heldout/`, export layout, targets filled, text only;
  `lab.dataset.check(..., require_images=False)` passes). Each target is `{"answer": str, "citations": [ids]}`. A fair
  score: the set of citations equals the expected set, and every number in the expected answer appears in the output.
  Put it beside your other scorers so the gate can call it; the held-out split is by mission (anchors M022 to M027).
- **The routing test's rows (T11).** The Routing tab shows a measured result above its simulation as soon as the metrics
  file has, for the same metric and node, one row with `conditions.policy = "capability"` and one with
  `conditions.policy = "energy"`, each with at least 100 under `conditions.requests` (or `n` / `samples`). That shape is
  a guess (`closed_loop()` in `lab_routing.py`, marked TODO). Either write T11's rows in that shape, or tell Hemdutt's
  agent the real shape and the function changes in one place.
- **Tag new metrics rows with `conditions.gpu`.** Untagged pod rows older than Sat 4:25 PM ET are shown as "other chip
  (R9700)" by the same rule as `lab/slide_numbers.py`. Line 31 (pod-c, 118.3 W) is caught by it today.
- A `student.decode_tok_s` row for the Jetson and a power row for pod-b would replace two "derived" or "no row" notes.

## Things to know

- The screen never mixes sample and real: the sample is served only when the log has no real run, or with
  `?sample=1`, and carries a banner.
- A run brought in from files stays "Training" until an eval is imported: nothing else marks the trainer as finished.
- Every number on the screen opens its source row; if a figure looks wrong, the row it came from is one tap away.
- Read-only towards `lab/`: nothing in `contexts/learning/` writes there. `lab_bench.py` parses `metrics.jsonl` itself
  because the lab's reader takes a write lock and creates the file.
- Tests for all of the above: `contexts/learning/tests/test_lab_*.py`. Full run: `pytest -q` with the lab tests that
  fail on `main` deselected as usual.
- Questions: Hemdutt ↔ Dominick or Ilan ↔ Dominick on Notion, addressed to Hemdutt's agent.
