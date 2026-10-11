# Showing a run on the box: `python -m lab lab-push` (L27)

Moved here from lab/README.md (O70); the verbs list there links every page. Arguments: `python -m lab lab-push -h`.

The box's Learning lab screen (`/lab/`) shows a real run only once its files reach `POST /api/learning/lab/import`
(`contexts/learning/lab_import.py`; the handoff note is `docs/handoff/learning-lab-real-run.md`). The trainer writes
`telemetry.jsonl` and `run.json` into its `--out` folder on the pod and sends them nowhere. `lab/lab_push.py` is the
relay: it reads the files from a pod through the hub, or from a folder on this machine, and posts them. From the repo
root, with `--box` the box's address (or `ATLAS_URL` set):

```bash
# a run that is training on pod D: post its steps every 60 s until run.json appears, and keep a copy here
python -m lab lab-push --box http://<box>:8000 --account D --run-dir adapters/l16-<run_id> --follow --save ../L27-data
# a finished run in a local folder, once
python -m lab lab-push --box http://<box>:8000 --run-dir runs/<run_id>
# after the gate (run it with the trainer's --run-id): its scores and verdict
python -m lab lab-push --box http://<box>:8000 --eval lab/state/eval-gate/<run_id>/scores.json --run-id <run_id>
# after packaging: the manifest, here read from the pod
python -m lab lab-push --box http://<box>:8000 --account D --manifest data/packages/<version>/manifest.json --run-id <run_id>
# the lab's ledger, whole, for the Experiments tab (again whenever it has grown)
python -m lab lab-push --box http://<box>:8000 --ledger lab/state/ledger.jsonl
# what any of these would read and post, sending nothing to the pod or the box
python -m lab lab-push --box http://<box>:8000 --account D --run-dir adapters/l16-<run_id> --follow --dry-run
```

- **The pod is only read.** With `--account` every path is relative to the pod's Jupyter root (the home folder on SJC,
  so `~/adapters/l16-<run_id>` is `adapters/l16-<run_id>`), and each file is one `GET /files/<path>` through
  `lab.proxy.PodLink`: nothing is written, started or stopped there, so it is safe beside a run that is being measured.
  Each read carries a query parameter that is never repeated, because the hub's front end caches some GETs for hours
  (the R14 trap, `lab/link_speed.py` `FreshLink`). A round of `--follow` is two GETs: `run.json`, then the steps.
- **The steps are posted as they stand**, cut back to the last whole line (the box refuses a whole file for one line it
  cannot parse). Posting again adds nothing: the box knows a step by its run and number, so a grown file adds only its
  new steps and the screen plays the run live.
- **`--follow` ends** after the first post that follows `run.json` (that post holds the whole file), at
  `--max-minutes` (120), or at once when waiting cannot help: the box wants a token (401), has no import route (404 or
  405, a build older than the import), refuses the file (400, 413), the hub refuses the account, or the pod has
  stopped. A box that does not answer or answers 5xx, and a pod read that times out, are tried again on the next
  round. `--every` is 5 to 3600 seconds.
- **Run ids.** The steps name their run; `--run-id` overrides and makes the box refuse steps or scores of another run.
  A manifest does not name its run, so it needs `--run-id` (or steps posted in the same call). The gate's event must
  carry the trainer's run id for its scores to land on the same run: run `eval-gate --run-id <the trainer's run id>`.
- **The ledger** needs a box whose build has `kind=ledger`; an older one answers 400 and names the kinds it knows. The
  box checks every row, replaces its copy and answers with the rows, experiments and verdicts it now holds. A run id
  never rides with it.
- **The token.** A box with `ATLAS_TOKEN` set wants it on writes. The verb reads `ATLAS_TOKEN` from the environment,
  then from the `.env` files `lab/machine/ready.py` searches, sends it as `Authorization: Bearer`, and prints only
  whether one was sent.
- **Answer.** One JSON object: `outcome` is `pushed`, or names what stopped it (`no_telemetry`, `bound_reached`,
  `box_wants_token`, `route_missing`, `box_refused_file`, `too_large`, `box_unreachable`, `box_error`, `pod_error`,
  `read_error`, `missing_file`, `empty_file`, `no_run_id`, `no_box`, `no_credentials`, `unsafe_path`), with the box's
  own answer to each post. A run brought in this way reads "Training" on the screen until its eval is imported.

First run, 2026-10-10 on the Strix Halo trainer (pod D, gfx1151): L16's run `train-20261011T032149Z-7dd098`, 18 steps.
Its steps were posted by hand with these same reads and posts while it trained (five posts about a minute apart, 11 to
18 steps, the last after `run.json`), and the box listed the real run with no sample banner. This verb then read the
finished run from the pod (`--save` kept `telemetry.jsonl`, 5462 bytes, and `run.json`) and posted that copy: 200, 0 new,
18 already there. Tests: `tests/test_lab_lab_push.py`, against the lab's fake pod (`lab/proxy_fake.py`, and once
`lab/hub_fake.py`) and a small HTTP server that answers the import route with the box's own parsing.
