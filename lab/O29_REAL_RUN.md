# O29: the first real run of the baseline experiment (E3 on pod B)

Everything below is for the one real run the Done when asks for. It was not run from the session that built the
runner: that session had no hub, pod, queue or Notion access, so every clause was proven against the fakes
(tests/test_lab_experiment_run.py) and the commands here are what a session with access runs, in this order, from
the repo root of a clone on `main` after this branch merged.

Why E3 is the baseline experiment: its control arm A is the stock record the kernels rubric names (vLLM b1388b1 with
`lab/vllm-gfx1201.diff` serving Qwen3.8-27B-INT4, built on pod D by R30), it is the top of T0's ranked queue, and its
run config is the one in `RUN_CONFIGS` (batch 1 against 4, 16 and 64, one shared cold spawn per repeat, 3 repeats,
a 60-minute slot). If the owner names another experiment as the baseline, give it a config with `--config FILE`.

## What has to be in place (names only, never values)

| Name | Where | What |
|---|---|---|
| `AUP_HUB_USER_D`, `AUP_HUB_TOKEN_D` | environment or the workspace `.env` | account D, pod B's account |
| `AUP_HUB_URL_D` | same | the SJC hub, `https://sjc.aupcloud.io` |
| `LAB_POD_ROLES` | same | `C:pod-a,D:pod-b`, so account D is node pod-b (the runner refuses an account with no role) |
| `NOTION_TOKEN` | same | the GPU Queue (without it the runner stops with outcome `no_token` and removes its uploads) |
| `LAB_ACTOR`, `LAB_TIER` | environment | who writes the ledger rows (optional; `unknown` and none otherwise) |

The pod must already be up (the runner never launches one) and must hold `~/vllm` and `~/models/Qwen3.8-27B-INT4`
(R30 put both there). The pod script needs `python3` with `httpx` (the openai package vLLM installs depends on it;
the script checks and exits 4 with the fix when it is missing), and `bash`, `base64`, `tar`, `timeout`, `pgrep` and
`pkill` from the image.

## The commands, in order

```
# 0. the clone and its checks
git checkout main && git pull
pip install -e ".[dev]"
python -m pytest -q tests/test_lab_experiment_run.py
python -m lab ready

# 1. the pod is up (read-only); launching it is a separate, deliberate step (lab/README.md, "Launching a pod")
python -m lab.controller status --account D

# 2. E3 is in this machine's ledger (adds only what is missing)
python -m lab experiment seed
python -m lab experiment list --id E3 --print

# 3. look before sending: the plan, the placement verdict and the script, written under lab/state/experiment_runs/
python -m lab experiment run E3 --task O29 --dry-run
python -m lab placement --check "E3 bench run: Batching vs joules per token" --gpu B --kind bench

# 4. the first call: uploads the script, asks for the 60-minute slot with E3's justification, and stops with
#    outcome "waiting" (critique_pending) because a 60-minute row waits for the referee's critique; the uploads
#    are removed again and the row stays Queued. Note the "row" in the answer.
python -m lab experiment run E3 --task O29

# 5. the referee writes its one critique on that row (a fable sitting); then read it
python -m lab referee --critique <row> --text "<one line>"
python -m lab slot status --row <row>

# 6. the run itself: answers the critique, starts the slot, runs the harness, pulls, records, releases, cleans
python -m lab experiment run E3 --task O29 --row <row> --reply "<answer to the critique>"

# 7. what it left: the verdict, the rows, the slot and the pod
python -m lab experiment list --id E3 --print
python -m lab ledger grep --verb "experiment run"
python -m lab ledger grep --verb bench.run --task T0
python -m lab slot status --row <row>
python -m lab.controller status --account D
```

Step 6 answers one JSON line. `outcome` is `recorded` when it worked; `slot_released` and `pod_cleaned` must both be
true; `readings` lists the run ids per arm and `verdict` the label, the winner and the summary. Copy that line (and
the local folder `lab/state/experiment_runs/<run id>/results/`, which holds the pulled rows, the harness's answer and
its log) onto the O29 task page as the recorded real run, with the commit the run used.

If step 4 answers `waiting` with `busy` or `not_next`, another row holds the card: run step 6 again later (same
row). If any answer says `failed`, its `failed_step` and `error` say where, the slot was still released and the pod
still cleaned; `leftovers` lists anything the cleanup could not remove.

## What the fakes could not prove (check these on the first real run)

- The SJC Jupyter server's `GET /files/<path>` with byte ranges and `DELETE /api/contents/<path>` (lab/transfer.py's
  pull and the runner's file removal), which the fake hub answers in the shapes FakePodProxy documents.
- The run-script terminal on pod D running the script from the Jupyter root (`/home/jovyan`, per R30), and the image
  having `timeout`, `pgrep`, `pkill`, `base64` and `tar`.
- That `lab/bench.py`'s default vLLM launch line starts the R30 build on pod D. If it does not, put the right line in
  a config file's `bench` list as `--serve-cmd` and pass `--config FILE`.
- That 3 cold spawns fit the 60-minute slot: R30 measured 523 s to ready from cold, so about 26 minutes of loading
  plus the measured requests. Each harness call is interrupted 4 minutes before the cap; a run that ran out of time
  shows fewer rows and, below 3 runs an arm, a `too_early` verdict. Then rerun with `--repeats 2`, or ask for a
  second slot.
- Whether batch 64 fits the KV cache on gfx1151: if not, the harness writes a skipped row and arm D has no reading;
  the verdict reads A, B and C and its note says D had none (owner ruling 5).
- Joules per token need amd-smi's SOCKET_POWER on the pod (T6 saw it answer on pod A); without power readings the
  metric is null and the run ends `failed` at `verdict`, with every measured run still a `bench.run` ledger row
  carrying its decode and TTFT numbers.
