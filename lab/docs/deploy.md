# Deploy to the box: `python -m lab deploy` (L6)

Arguments: `python -m lab deploy -h`. The commands for pod D and the box are in lab/README.md, "Deploy to the box (L6)".

`lab/deploy.py` runs ON THE BOX from the repo root and takes one packaged model from the trainer pod to serving:

```
python -m lab deploy [VERSION] [--from-pod C|D] [--port 8300] [--dry-run] [--models-dir data/models]
                     [--events-dir data/events] [--part-mib 64] [--llm-url URL] [--app-url URL] [--no-app]
```

1. Resolves the version: the argument, or the newest complete package (by the manifest's `created`) in the listing of
   the pod's package server (`lab/package_server.py`, R10).
2. Pulls every file the manifest names through `https://<hub>/user/<account>/proxy/<port>/<version>/<file>` in Range
   parts (`lab/package_pull.py`: resumable, a sha256 check per file, manifest.json last) into `data/models/<version>/`.
   The hub user, token and address are `AUP_HUB_USER_D`, `AUP_HUB_TOKEN_D` and `AUP_HUB_URL_D` (or `LAB_HUB_URL`),
   with `LAB_POD_ROLES=C:pod-a,D:pod-b`, from the environment or the .env search of `lab/hub.py` (`LAB_ENV_FILES`
   names the file directly).
3. Swaps with `jetson/atlas_swap.py`'s `swap` (R7): checks every sha256 again, switches `data/models/current`,
   restarts `atlas-llm` (`sudo -n systemctl restart`), waits for its health and test question; `previous` keeps the
   version that was serving, so `python jetson/atlas_swap.py rollback` goes back to it.
4. Checks the model answers one request the way the gateway sends it (`runtime/llama.py`: POST
   `/v1/chat/completions`, temperature 0.2); a failed check switches back to the version that was serving and restores
   `previous`.
5. Publishes `learning.model.deployed` with version, previous_version, base, task, package_mb, pulled_seconds and
   checksums_ok: to the running app (`POST /api/learning/deployed`, `ATLAS_TOKEN` as the bearer), whose bus writes it
   to `data/events/`; straight into `data/events/YYYY-MM-DD.jsonl` when the app cannot be reached or with `--no-app`.
   A failure after the version is known is written to the event log as `learning.run.progress` with stage `failed`.

`--dry-run` resolves the version, reads its manifest and file sizes and prints the plan (files, MB, free disk, the
current and previous version on the box); it writes and switches nothing.

The answer is one JSON line: `ok`, `outcome` (`deployed`, `already_serving`, `dry_run`, `error`), `kind` for an error
(`no_credentials`, `no_package`, `pull`, `swap`, `check`, `publish`), and the pull's `mb_s`, `bytes` and
`wall_seconds`. The pull also writes R10's `link pull` ledger row (`link.pull_mb_s`).

Tests: `tests/test_lab_deploy.py`, on the fixture package `fixtures/learning/package/atlas-desk-objects-fixture/`
(rebuilt by `fixtures/learning/package/make_fixture.py` with L5's manifest code), the real package server behind
`lab/package_fake.py`'s proxy, a fake box (swap, llama-server, app), and the real `atlas_swap.py` swap where the
machine can make symlinks.
