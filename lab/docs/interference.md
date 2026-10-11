# Interference on one chip: `python -m lab interference` (R43)

Arguments: `python -m lab interference -h`. The module docstring of `lab/interference.py` has the full account.

What a teacher-style decode load and a LoRA run cost each other on one Strix Halo (gfx1151): the placement plan
(`python -m lab placement`) says the 27B INT4 teacher and a 4B LoRA job fit one pod's 64 GB on paper once the teacher
is served at a share of 0.63; this measures whether they should share. Built and tested on the box against
`lab/bench_fake.py` and `lab/train_fake.py`; the pod run is one 30-minute GPU Queue slot on pod D, done by the
dispatcher. Until its data file is committed the plan's sharing line stays a paper estimate.

```
python -m lab interference --dry-run       # the plan, the memory check at the teacher's share, the slot minutes
python -m lab interference pod-script      # writes lab/state/interference/pod_run.sh (needs AUP_HUB_USER_D)
python tooling/dispatcher/pod_exec.py D run <clone>/lab/state/interference/pod_run.sh r43_pod_run.sh 2700 > <clone>/lab/state/interference/pod_run.out
python -m lab interference collect lab/state/interference/pod_run.out
```

In one line, in Git Bash, from a clone's root two folders below the workspace (`work/<ID>`), inside the slot:

```
.venv/Scripts/python -m lab interference pod-script && python ../../tooling/dispatcher/pod_exec.py D run lab/state/interference/pod_run.sh r43_pod_run.sh 2700 > lab/state/interference/pod_run.out && .venv/Scripts/python -m lab interference collect lab/state/interference/pod_run.out
```

`pod_exec.py` is the workspace's pod tool, not in this repo; it needs the websocket-client package, which the clone's
venv does not hold, hence the plain `python`. Its run mode uploads the script (about 270 KB: the lab files the run
imports, the bench prompt set, the trainer's requirements file and the bundled tool-calling rows, as one base64 block
with its sha256) through the Jupyter contents API and types only its `bash` line. 2700 seconds is the slot plus its one
15-minute extension. `pod-script` answers with the same commands on the script's real path.

- **Before the slot.** Stop any vLLM server on pod D: the run starts its own and stops nothing it did not start. Pod D
  must hold `~/models/Qwen3.8-27B-INT4` (R30), `~/models/Qwen3.5-4B` and the training venv `~/venvs/atlas-train` (both
  left by L16).
- **The gate and the check**, fail closed, before anything starts: rocminfo names gfx1151, the Jupyter user hashes to
  account D's (the script carries the sha256 of the user name, never the name), port 8100 is free, no vLLM server
  runs; then the teacher's checkpoint, the student's folder and the venv's Python are there and the venv holds every
  pin of `lab/requirements-train.txt`. Nothing is installed in a slot: the trainer runs from that venv with
  `--no-install` and `PYTHONNOUSERSITE=1`. Run by the image's `python3` instead, `lab/train.py` would
  `pip install --user` its pins into the user site that holds the pod's vLLM build.
- **The slot.** `--dry-run` gives 30 minutes: 29.2 with pod D's 523 s cold engine start (paste and gate 1, engine
  start 8.7, warm-up 1, decode arm 2, train arm 6.6, both arm 8.9, stop 1), about 25.5 when the engine starts from a
  warm cache. The run's own deadline is 40 minutes.
- **The three arms**, on one server started at `--gpu-memory-utilization 0.63` (at the teacher's usual 0.90 it holds
  57.6 of the pod's 58 GiB free and no LoRA job fits beside it): decode alone (`lab/bench.py`'s prompt set, 4 streams
  of 128 tokens, round after round for 120 s), the LoRA alone with the teacher idle (`lab/train.py`, batch 4,
  gradient accumulation 1, 20 measured steps), and both together (the same decode rounds for as long as the LoRA
  trains). Each arm records decode tok/s, answers per second, samples per second and steps per second (from the
  trainer's own telemetry.jsonl), watts (amd-smi through `lab/bench.py`'s LocalPower) and joules per answer or per
  step.
- **Every step the same work.** Each trainer run gets one batch's worth of rows (the first 4 of the 6 bundled), so
  every step is the same examples at one padded length, and takes 2 warm-up steps that are recorded and left out of
  its window: the train arm runs before the both arm and would otherwise pay the one-time costs the both arm finds
  ready. Samples are the examples the trainer saw, counted epoch by epoch, not steps x batch.
- **Shared against one after the other.** The both arm's window is the trainer's measured steps and its energy is
  that window's. A decode round counts by the part of its time inside the window, so the answers are the work the
  window's energy paid for. The comparison is the same answers and steps done one after the other, at the decode
  arm's joules and seconds per answer and the train arm's per step. The verdict is `share` when together took less
  time for no more energy, `serial` otherwise, `unknown` when a number is missing.
- **What stays on the pod**: `~/r43_interference/<run id>/` with the logs, the trainers' telemetry, `raw.json`,
  `result.json` and `result_bundle.b64` (the block again, should the output be cut). Each trainer run's adapter is
  deleted once its telemetry is read; a trainer still running when the run ends is stopped with the server.
- **collect** reads the marked block from the saved output, checks its sha256, and writes
  `lab/refine/research/kernels/data/interference-gfx1151-<date>.json` (schema `lab.interference/1`) and its `.md`
  table, and appends ledger rows `interference.arm` (one per arm) and `interference.summary` on node pod-b, task R43,
  low confidence (one run per arm).
- **The plan.** `python -m lab placement` reads the newest such file (or `--interference FILE`): its teacher + lora
  line then gives the measured cost with each number traced to its field, and a measured run where sharing costs time
  or energy keeps LoRA jobs off the teacher's pod even when the memory fits. The rules are in
  `lab/refine/research/plumbing/placement.md`, section "The interference run and how the plan reads it".
