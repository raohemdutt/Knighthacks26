# Deploy to the box: `python -m lab deploy` (L6)

Arguments: `python -m lab deploy -h`. The commands for pod D and the box are in lab/README.md, "Deploy to the box (L6)".

`lab/deploy.py` runs ON THE BOX from the repo root and takes one packaged model from the trainer pod to serving:

```
python -m lab deploy [VERSION] [--from-pod C|D] [--port 8300] [--dry-run] [--models-dir data/models]
                     [--events-dir data/events] [--part-mib 64] [--llm-url URL] [--app-url URL] [--no-app]
                     [--no-trust]
```

1. Resolves the version: the argument, or the newest complete package (by the manifest's `created`) in the listing of
   the pod's package server (`lab/package_server.py`, R10).
2. Pulls every file the manifest names through `https://<hub>/user/<account>/proxy/<port>/<version>/<file>` in Range
   parts (`lab/package_pull.py`: resumable, a sha256 check per file, manifest.json last) into `data/models/<version>/`.
   A file already there is hashed too; this is the one full sha256 pass of the deploy. After every file matched, the
   pull writes `data/models/<version>/.verified.json`: per file the name, size, sha256 and mtime (`mtime_ns`) it had
   when it was hashed, with `written_by: lab/package_pull.py`.
   The hub user, token and address are `AUP_HUB_USER_D`, `AUP_HUB_TOKEN_D` and `AUP_HUB_URL_D` (or `LAB_HUB_URL`),
   with `LAB_POD_ROLES=C:pod-a,D:pod-b`, from the environment or the .env search of `lab/hub.py` (`LAB_ENV_FILES`
   names the file directly).
3. Swaps with `jetson/atlas_swap.py`'s `swap` (R7) with `trust_verified` (the script's `--trust-verified`): checks
   every file against the manifest again, by the pull's record when the file still has the recorded size and mtime
   and by sha256 otherwise, switches `data/models/current` (same check),
   restarts `atlas-llm` (`sudo -n systemctl restart`), waits for its health and test question; `previous` keeps the
   version that was serving, so `python jetson/atlas_swap.py rollback` goes back to it.
4. Checks the model answers one request the way the gateway sends it (`runtime/llama.py`: POST
   `/v1/chat/completions`, temperature 0.2); a failed check switches back to the version that was serving and restores
   `previous`.
5. Publishes `learning.model.deployed` with version, previous_version, base, task, package_mb, pulled_seconds and
   checksums_ok: to the running app (`POST /api/learning/deployed`, `ATLAS_TOKEN` as the bearer), whose bus writes it
   to `data/events/`; straight into `data/events/YYYY-MM-DD.jsonl` when the app cannot be reached or with `--no-app`.
   A failure after the version is known is written to the event log as `learning.run.progress` with stage `failed`.

Hashing once. The verify after the pull, the swap's check, the switch (`ModelStore.activate`) and a switch back all
read `.verified.json` (`runtime/modelstore.py`: `read_verified`, `record_vouches`). A file is taken as checked only
when the record was written by the pull, names it with the manifest's sha256, and the file's size and mtime still
match; anything else (no record, another writer, a size or mtime that moved, an unreadable record) hashes the file as
before. The pull deletes the record before it touches the files and writes it only after all of them matched. On the
box, with the 3.4 GB package already there, that is one hash pass (about 40 s) plus the model server restart (67 s):
about 115 s for the whole verb, estimated until the live run, instead of the measured-basis 240 s of four passes.
`--no-trust` hashes at every step, as before. The trade: a file rewritten in place with its size and mtime put back
would be trusted; `--no-trust` is the answer if that is a worry.

`--dry-run` resolves the version, reads its manifest and file sizes and prints the plan (files, MB, free disk, the
current and previous version on the box); it writes and switches nothing.

The answer is one JSON line: `ok`, `outcome` (`deployed`, `already_serving`, `dry_run`, `error`), `kind` for an error
(`no_credentials`, `no_package`, `pull`, `swap`, `check`, `publish`), and the pull's `mb_s`, `bytes` and
`wall_seconds`, and `trust_verified` (false with `--no-trust`). The pull also writes R10's `link pull` ledger row
(`link.pull_mb_s`).

Tests: `tests/test_lab_deploy.py`, on the fixture package `fixtures/learning/package/atlas-desk-objects-fixture/`
(rebuilt by `fixtures/learning/package/make_fixture.py` with L5's manifest code), the real package server behind
`lab/package_fake.py`'s proxy, a fake box (swap, llama-server, app), and the real `atlas_swap.py` swap where the
machine can make symlinks. The hash-once cases (record present, absent, written by someone else, stale by mtime, stale
by size, a failed pull, `--no-trust`) count every sha256 the pull and the box run; `tests/test_atlas_swap.py` does
the same for the swap and its `--trust-verified` flag.

## From the button (L35)

On stage the swap runs from the Learning lab, section 5 "On the box", on the box itself; no laptop, no pod, no hub.
The card always shows the model on the box now (`data/models/current`) and the one before it (`previous`), and two
buttons:

- "Put v2 on the box": `POST /api/learning/lab/deploy {"version": "atlas-tool-calling-v2-a100-f2r4"}`. The target is
  `featured` from `GET /api/learning/lab/deploy`: that package (`ATLAS_LAB_DEPLOY_VERSION` in the gateway's
  environment changes it) when it is on disk and verified, else the newest verified version; disabled while it serves.
- "Back to base": `POST /api/learning/lab/deploy/rollback {"version": "atlas-base"}`; disabled while base serves.

Either one starts `jetson/atlas_swap.py swap <version> --trust-verified --models-dir <data>/models` as a background
subprocess of the gateway (the gateway's own Python, the checkout as PYTHONPATH, the gateway's `ATLAS_LLM_URL` as
`--url`), one at a time: a second press while one runs is a 409, a version with no `manifest.json` on disk a 404, and
nothing is ever pulled. The card polls `GET /api/learning/lab/deploy` every 2 s and lights the five stages (verify,
switch, restart, check, published) from the `STAGE <name> <start|done|failed> <unix seconds>` lines the swap prints,
with each stage's seconds; when it ends, a line says how long it took or that it rolled back. The state and the swap's
output are kept in `<data>/lab_deploy/status.json` and `swap.log`, so a gateway restart does not lose the last deploy.

Hashing from the button. There is no pull, so the swap writes `.verified.json` itself after a full hash passes (same
fields and writer marker as the pull's, `hashed_by: jetson/atlas_swap.py`). The first press for a version with no
record (f2r4 and atlas-base on the box today) hashes it once; every later press, back and forth, hashes nothing while
the files keep their size and mtime. Estimated, not yet run on the box: first press about 110 s (one 40 to 45 s hash
pass plus the 67 s restart to serving), later presses about 70 s.

Below the buttons, "Ask the model itself" sends a question to llama-server the way
`tooling/demo/live_deploy_question.py` does (`POST /api/learning/lab/deploy/ask {"question"}`: the eval's exact
prompt for a question in `live_deploy_question.json`, else that prompt with the question and the time put in) and
shows the tool call the serving model wrote, with the call the eval expects. It answers 503 while the model server
restarts.

What the box needs: the gateway's user can run `sudo -n systemctl restart atlas-llm` (the swap's restart; a gateway
unit with `NoNewPrivileges=yes` would block it), and when `ATLAS_TOKEN` is set, the browser on stage has the token in
Settings (the three POSTs are guarded like every other write). Tests:
`contexts/learning/tests/test_lab_deploy_button.py` (a fake swap script and model store: the state machine, 409, 404,
rollback, persistence, the ask route) and `tests/test_atlas_swap.py` (the record the swap writes, the stage lines).
