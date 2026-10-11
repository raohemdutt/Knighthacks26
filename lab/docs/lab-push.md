# Showing a run on the box: `python -m lab lab-push` (L27)

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

## What the box holds for L16's run (Sun 2026-10-11, 3:10 AM ET)

Everything L16's run has, pushed with this verb from `work/L27-live` in one call, then checked with the box's own API:

```bash
python -m lab lab-push --box http://<box>:8000 --run-dir <the --save copy of the pod's run folder> \
  --eval fixtures/demo/l16-gate-scores.json --manifest fixtures/demo/l16-manifest.json --ledger <ledger>
```

| File | Kept here | Box answer |
| --- | --- | --- |
| steps, 18 lines | `fixtures/demo/l16-telemetry.jsonl` | 200, 0 new, 18 already there |
| the gate's `scores.json` (pod D, 2:12 to 2:20 AM, `--run-id train-20261011T032149Z-7dd098`) | `fixtures/demo/l16-gate-scores.json` | 200, 0 new, 2 already there (first posted 2:21 AM) |
| the package manifest, both sha256s | `fixtures/demo/l16-manifest.json` | 200, 0 new, 1 already there |
| the lab's ledger, 151 rows | not committed (`lab/state` stays local) | 200 after the fix below, 5 experiments, no verdicts yet |

`GET /api/learning/lab` then answers `sample: false`, `real_runs: 1`, and one run: `train-20261011T032149Z-7dd098`,
task `tool-calling`, status `deployed`, scores current 1.0, candidate 1.0, teacher 0.9167 (24, 24 and 22 right of 24),
decision `reject`, model `atlas-tool-calling-train-20261011T032149Z-7dd098`. The run's page has the 18 losses
(0.1273 to 0.0095 over 717 s of step times), the 24 held-out questions before and after, the package's sha256 and the
swap from `atlas-base`. `GET /api/learning/lab/experiments` says "5 experiments in the lab ledger".

The teacher score on the box is the gate's own teacher (pod C, 22 of 24). The A100 teacher arm
(`fixtures/learning/a100-2026-10-11/l4/scores.json`) scored the same 22 of 24 twice; the gate's event is posted as it
is, so the box cites the gate.

Not on the box, and why:

- **The trainer's settings** (planned steps, batch 4 x 4, LoRA rank 16, samples per second). They come from `run.json`
  through `kind=run` (L29, #380), and the box's build predates it: it knows telemetry, eval and manifest only. It needs
  a pull and a gateway restart on the box, then `curl --data-binary @fixtures/demo/l16-run.json
  ".../api/learning/lab/import?kind=run&run_id=train-20261011T032149Z-7dd098"`. Until then the screen says the
  settings are not in the log.
- **Power.** No sampler ran on pod D during the run (`trainer.steps_with_power` is 0, `socket_w` null in every step),
  so there is nothing to push; the screen says so.
- **The ledger as written.** The box runs Python 3.10, whose `datetime.fromisoformat` refuses a trailing `Z`, so its
  ledger check refused every row (`line 1: ts is not what the lab's ledger accepts`). The copy pushed writes each
  `ts` as `+00:00` instead of `Z`, the same instant; nothing else changed. A lasting fix is for `lab/machine/ledger.py`
  `_parse_ts` to read `Z` itself on 3.10.
