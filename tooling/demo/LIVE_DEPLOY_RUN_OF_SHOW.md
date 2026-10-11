# The live deploy: run of show (L34)

On stage, one command on the box takes the retrained tool-calling model (v2, package
`atlas-tool-calling-v2-a100-f2r4`) from pod D's package to serving, in place of the base model. Everything that can be
done ahead was done at 5 AM ET on Sun 2026-10-11: the box is back on the base model, the v2 package is already on the
box with every sha256 checked, the v2 run's real results are in the box's Learning lab, and the command was dry-run and
timed. Expected wall-clock of the live command: **about 115 s, estimated** (one sha256 pass plus the restart; not yet
run live with the hash-once code, see "The command" below), of which the model gives no answers for the last 70 s or
so. With a box checkout that predates the hash-once change, or with `--no-trust`, it is the measured-basis 240 s.

## Starting state (set at 5:09 AM ET, checked again at 5:48 AM ET)

| What | State |
| --- | --- |
| `~/Knighthacks26/data/models/current` | `atlas-base` (Qwen3.5-4B Q4_K_M, the starting model) |
| `previous` | `atlas-tool-calling-train-20261011T032149Z-7dd098` (v1; scored 34 of 58 against base's 42, held off the stage) |
| v2 on the box, not serving | `data/models/atlas-tool-calling-v2-a100-f2r4/` (gguf 2708.8 MB, mmproj 672.4 MB, manifest.json), every sha256 matched at the pull |
| also on the box, not serving | `atlas-tool-calling-v2-a100-r01` (the first v2 package, 176 of 228; harmless, not for the stage) |
| llama-server (`atlas-llm`) | `-m .../atlas-base/Qwen3.5-4B-Q4_K_M.gguf --mmproj .../atlas-base/mmproj-F16.gguf` |
| gateway on :8000 | up; this work never restarted it (systemd stopped and started it at 5:06:26 AM ET, from outside this work, 10 s before the rollback began) |
| Learning lab | 6 real runs; the v2 run `final2-t12-lr0.000428-rank4-a8-drop0.136-warm0.009` shows score before 0.417 and after 0.785 (95 and 179 of 228 held-out), status `gated`, deploy "no learning.model.deployed event for this run's model" |

How the base was put back: `jetson/atlas_swap.py rollback` from `~/Knighthacks26`, 5:06:36 to 5:09:11 AM ET,
**154.5 s**: 88 s checking atlas-base's two files (two sha256 passes, see below), 67 s from the `atlas-llm` restart to
llama-server healthy and answering the test question, then the app was told (`learning.model.deployed` atlas-base,
previous v1, 9:09:11 UTC). One question through the gateway right after: `POST /api/ask` "Have you seen my water
bottle?" answered in 2.2 s: "Yes, your water bottle was last seen on Sunday at 5:09 AM in the desk zone."

## Ten minutes before

1. Check the box is still on base and v2 is installed (from `~/Knighthacks26`):
   ```bash
   PYTHONPATH=. ~/atlas/venv-yolo/bin/python jetson/atlas_swap.py status
   ```
   Wants `current: atlas-base`, and `atlas-tool-calling-v2-a100-f2r4` in the list.
2. Check pod D's package server answers (the deploy reads the manifest from it even with the files on the box):
   the dry run, 2.0 s this morning, wants `"ok": true, "outcome": "dry_run"`:
   ```bash
   cd ~/atlas-l6 && LAB_ENV_FILES=$HOME/Knighthacks26/.env ~/atlas/venv-yolo/bin/python -m lab deploy atlas-tool-calling-v2-a100-f2r4 --from-pod D --models-dir ~/Knighthacks26/data/models --events-dir ~/Knighthacks26/data/events --dry-run
   ```
3. Open the box's Learning lab on the v2 run (`http://<box>:8000/lab/`, run
   `final2-t12-lr0.000428-rank4-a8-drop0.136-warm0.009`). The page can take two minutes to load the first time.
4. Ask the question once on base (below) so the "before" is on screen.

## The question, before and after

The gateway's `/api/ask` is the wrong place to show the difference: it answers "have you seen my X" and "the last N
hours" in code (`contexts/agent/application/qa.py` prefetch) before the model picks a tool, and none of the 228
held-out questions reaches the gateway's tool loop. Through the gateway both models give the same answer. The
difference is in which tool the model itself picks, so the question goes straight to llama-server with the eval's
exact prompt:

```bash
cd ~/Knighthacks26 && python3 tooling/demo/live_deploy_question.py
```

Until this branch is on the box, the same two files sit in `~/l34/`: `cd ~/l34 && python3 live_deploy_question.py`.
Rehearsed on base at 5:36:56 AM ET: 6.1 s for the water bottle, 11.8 s for the door question.

Question: **"Have you seen my water bottle?"** (held-out item `o-53583b1470`, group `last_seen-water-bottle`; prompt
sha256 `ca2993ef...b192`, the same text the eval sent).

| Model | What it wrote | Right? |
| --- | --- | --- |
| atlas-base, on the box, 5:11 AM ET (3.1 s) | `search_memory(q="water bottle")` | no: searches captions instead of asking where the object was last seen |
| atlas-tool-calling-v2-a100-f2r4, pod D held-out run | `last_seen(label="water bottle")` | yes, the call the dataset expects |

The base answer was measured on the box with the box's own base GGUF; the v2 answer is from the held-out scoring on
pod D (`work/L16v2pod-out/pod/f2r4/agreement-v2/raw/held_out.after-1.jsonl`), not yet seen on the box because v2 has
never served there. Say so if asked. A second question in the same file (`--all`): "Who came to the door during the
last 14 hours?": base on the box asked `episodes_between` for the next 14 hours (07:37 to 21:37 UTC), v2 for the past
14 (17:37 to 07:37 UTC).

## The live command

On the box:

```bash
cd ~/atlas-l6 && LAB_ENV_FILES=$HOME/Knighthacks26/.env ~/atlas/venv-yolo/bin/python -m lab deploy atlas-tool-calling-v2-a100-f2r4 --from-pod D --models-dir ~/Knighthacks26/data/models --events-dir ~/Knighthacks26/data/events
```

What it does, with the expected seconds (the files are already on the box, so nothing is downloaded). The box's
checkout (`~/atlas-l6`) needs the hash-once change on main for these numbers; an older checkout hashes four times.

| Step | What happens | Expected (estimated) |
| --- | --- | --- |
| resolve | reads the manifest from pod D through the hub proxy | 2 to 3 s |
| pull | every file already there: hashes each against the manifest (the one full pass), sends nothing, writes `.verified.json` beside `manifest.json` | 40 s |
| verify | the deploy checks the installed copy against the pull's record: each file's size and mtime still match, so no hash | under 1 s |
| swap | `atlas_swap.py` swap with `--trust-verified`: checks the files by the record (no hash), switches `current` (same check, no hash), restarts `atlas-llm`, waits for health and the test question (67 s) | 67 s |
| check | one chat request through llama-server the way the gateway sends it | 3 to 10 s (waits behind a caption) |
| publish | `learning.model.deployed` (version, previous_version atlas-base, base, task, package_mb 3381.2, pulled_seconds, checksums_ok) to the app | under 1 s |
| total | | **about 115 s (110 to 120 s), estimated until the live run** |

Where the seconds come from: one sha256 pass over the package (gguf plus mmproj, 3.38 GB) measured with `sha256sum`
on the box at 5:47:43 AM ET: 40.4 s for f2r4 (32.1 s the gguf, 8.3 s the mmproj; r01, the same sizes, took 45.4 s at
5:32 AM). The code used to hash the whole package four times: `package_pull.pull_file` (a file already present is
hashed before it is kept), `deploy.deploy` (`box.verify`), `atlas_swap.swap` (`store.verify`) and
`ModelStore.activate` (verifies again), which is where the earlier 240 s estimate came from. Now only the pull hashes;
it then writes `data/models/<version>/.verified.json` (name, size, sha256 and mtime of each file it hashed), and the
verify, the swap's check and the switch trust that record for a file whose size and mtime have not changed. A file
that changed since, or no record at all, is hashed as before. The f2r4 copy on the box was pulled before the record
existed, so the live command's pull hash is what writes it; nothing has to be done ahead. The box reads from its SD
card and has about 500 MB free memory, so nothing stays cached between passes; the rollback's 88 s for two passes over
atlas-base agrees. The restart-to-serving 67 s is the rollback's, measured. The model answers nothing only from the
restart on (about 70 s); the hashing before it runs while the old model keeps serving. `--no-trust` on the command
hashes at every step again (about 240 s).

The pull line in the answer will read `"outcome": "already_present"` with no MB/s, and `pulled_seconds` in the event
will be the hash time, not a download.

The answer is one JSON line, `"ok": true, "outcome": "deployed"`, `"check": "ok"`, `"published": "app"`.

### Faster command if the segment is short

The same swap without the deploy's pull and extra verify, from `~/Knighthacks26`:

```bash
PYTHONPATH=. ~/atlas/venv-yolo/bin/python jetson/atlas_swap.py swap atlas-tool-calling-v2-a100-f2r4
```

About 150 s (two hash passes plus the restart; the rollback this morning, the same code path, took 154.5 s).
`--trust-verified` on it skips both passes only when a pull with the hash-once code has already written f2r4's
`.verified.json` (a live deploy attempt that got past its pull does); otherwise the flag changes nothing. It also
tells the app, so `learning.model.deployed` lands and the On the box card flips, but the event carries only
version and previous_version (no package_mb, pulled_seconds or checksums_ok), and it skips the gateway-shaped check.
It does not need pod D at all.

## What the audience sees

- The Learning lab, v2 run, section 5 "On the box": before the command it says "Not on the box yet" with the package
  line (gguf `atlas-tool-calling-v2-a100-f2r4-Q4_K_M.gguf`, sha256 `db7879b12398...`). When
  `learning.model.deployed` for `atlas-tool-calling-v2-a100-f2r4` lands, it shows the swap: `atlas-base` "was serving"
  on the left, the v2 model "serving since <time>" on the right. Section 4 "The gate" shows 0.417 before and 0.785
  after.
- The event itself: the deploy's last JSON line names `event_id`; on the phone's Teach screen the deployments list
  (`GET /api/learning`, `deployments`) gets a new first row, `atlas-tool-calling-v2-a100-f2r4` with previous
  `atlas-base`.
- The question again, after: `last_seen(label="water bottle")` instead of `search_memory`.

Do not point at the `model_version` in gateway answers or at `GET /health`: the gateway reads that label once when it
starts (it said v1 at 5:09 AM while base was serving) and nobody restarts the gateway during the demo.
`GET /api/learning/lab` `current_model` reads the runtime and is right.

## Rollback

From `~/Knighthacks26`:

```bash
PYTHONPATH=. ~/atlas/venv-yolo/bin/python jetson/atlas_swap.py rollback
```

After the live deploy, `previous` holds `atlas-base`: `ModelStore.activate` writes the version that was serving
(`atlas-base` now) into `previous` when it switches. So the rollback goes to base, in about 155 s (measured 154.5 s this
morning), and tells the app (`learning.model.deployed` atlas-base, previous v2). A deploy whose check fails rolls
back by itself (below), so this command is for a v2 that serves but misbehaves.

## Failure modes

| What fails | What the command does | What to do |
| --- | --- | --- |
| pod D's package server down, pod D stopped, or the hub token refused | stops at resolve, before anything on the box changes: `"ok": false, "kind": "no_package"` (or `no_credentials`); writes `learning.run.progress` stage failed only if the version was resolved | run the faster command above: it needs only the files already on the box |
| a hash mismatch (a damaged copy) | the pull deletes the bad file and downloads it again from pod D (14 to 19 min at the 3 to 4 MB/s measured this morning, too long for the stage); a mismatch at verify or swap stops before `current` changes, `"kind": "pull"` or `"swap"` | stay on base; say the box refused a model that did not match its checksum, which is the point of the check |
| llama-server does not come up on v2, or its test question fails | the swap puts `current` back to atlas-base, restores `previous`, restarts again (another 45 s hash, since atlas-base has no `.verified.json`, plus 67 s), `"kind": "swap"` | the box is on base again; carry on |
| the gateway-shaped check fails after the swap | switches back to atlas-base and writes `learning.run.progress` failed, `"kind": "check"`, `rolled_back_to` atlas-base | as above |
| the app does not take the event | the model is serving; the event goes into `data/events/` directly, `"published": "log"` with `app_note`; the running gateway does not read that file until it restarts, so the card does not flip | show the deploy's JSON line and the question instead |
| `/api/ask` slow or failing | not part of the deploy | the question goes to llama-server directly, see above |

## Measured this morning

| When (ET) | What | Result |
| --- | --- | --- |
| 5:06:36 to 5:09:11 | rollback to atlas-base | 154.5 s; `current -> atlas-base`, llama-server on the base gguf |
| 5:09:55 | `POST /api/ask` "Have you seen my water bottle?" on base | answered in 2.2 s |
| 5:11 | base's tool call on the eval prompt, on the box | `search_memory(q="water bottle")`, 3.1 s |
| 5:12:37 to 5:31:13 | pre-pull of r01 from pod D through the SJC hub proxy | 3381.2 MB in 1114.6 s wall (1091.8 s transferring), 3.10 MB/s, 52 parts of 64 MiB, 0 retries, both sha256 ok |
| 5:32:21 | `lab deploy ... r01 --dry-run` | 2.7 s, plan: pull, swap, check, publish, back to atlas-base on a failed check |
| 5:33:12 to 5:47:26 | pre-pull of f2r4, same path | 3381.2 MB in 852.9 s wall (812.0 s transferring), 4.16 MB/s, 52 parts, 0 retries, both sha256 ok; `current` stayed atlas-base |
| 5:47:43 | `sha256sum` of f2r4's two files on the box | 40.4 s, both match the manifest; `atlas_swap.py status` lists f2r4 as installed |
| 5:48:28 | `lab deploy atlas-tool-calling-v2-a100-f2r4 --dry-run` | 2.0 s, `"ok": true, "outcome": "dry_run"`, 3381.2 MB, 77.6 GB free, current atlas-base; plan: pull, swap, check, publish, back to atlas-base on a failed check |
| 5:34 | v2 run imported into the box Learning lab (`POST /api/learning/lab/import`, kinds telemetry, run, eval, manifest) | 49, 2, 3 and 2 events published, 0 already there; `real_runs` 5 to 6 |

The dry run does not look at the files on the box: its plan always lists the pull step, and it says nothing about what
is already installed. That the files are present and match was checked separately with `sha256sum` and
`atlas_swap.py status`.

### The Learning lab import

The v2 run's results were pushed to the running gateway with `POST /api/learning/lab/import?kind=<k>&run_id=final2-t12-lr0.000428-rank4-a8-drop0.136-warm0.009&task=tool-calling`:

- `telemetry`: the Colab trainer's `telemetry.jsonl` (48 steps), from `work/L16v2-out/colab/runs/<run>/`;
- `run`: the trainer's `run.json`;
- `eval`: pod D's held-out summary (`work/L16v2pod-out/pod/f2r4/agreement-v2/held_out.summary.json`, 228 items with
  both answers per item) plus the `learning.run.evaluated` event the gate would write, built from that summary with
  `lab/eval_gate.py`'s rule (228 items, delta 0.368 over noise 0.0, p 1.5e-5 under 0.05: deploy), model_version
  `atlas-tool-calling-v2-a100-f2r4`, event id `l16v2:<run>:run.evaluated`;
- `manifest`: f2r4's `manifest.json`.

Each file was first run through the box's own parser (`contexts/learning/lab_import.events_of`) on this machine.
Importing again adds nothing (every event has a fixed id).
