# R12 real run: the ordered steps on the pod

Everything in `gpu/` was built and tested off the pod, on fixtures and fakes. This file is the real run, in order.
It runs on **pod B (SJC account D, the exclusive Strix Halo card)**, inside GPU Queue slots, and **never on pod A**
(account C, the teacher: R12 must not interrupt it). The first step posts the pre-registration; nothing is measured
before it is on T0.

Environment variable NAMES used (values live in the workspace `.env` and on the Credentials page, never here):
`AUP_HUB_USER_D`, `AUP_HUB_TOKEN_D`, `AUP_HUB_PASS_D`, `AUP_HUB_LAUNCH_D` (`form` on SJC), `AUP_HUB_GPU_D`
(`strix-halo`), `LAB_POD_ROLES` (`C:pod-a,D:pod-b`), `NOTION_TOKEN` (the slot client). On the pod, set per step below:
`TRITON_CACHE_DIR`, `VLLM_DIR`, `GFX`, `R12_STATE_DIR`, `R12_REBUILD_JSON`. `GPU_MAX_HW_QUEUES` stays unset in every
arm (stock as the teacher runs it; owner ruling 6 changes it only after E18).

Times are estimates from the R30 record on pod D: a vLLM build of 4 to 5 minutes with a warm build tree, a server
start of about 5 minutes warm and 523 s cold, decode about 2.8 tok/s at batch 1.

## 0. Before the slot (from the box, nothing on a pod)

1. Post `gpu/R12_T0_PREREG.md`'s two lines on T0 (the dispatcher). Note the E numbers they get.
2. `python -m pytest -q tests/test_gpu_r12.py` on the branch commit that will run; paste the output in the T0 note.
3. Request the slots: `python -m lab slot request --task R12 --what "R12 build, LDS and match (pod B)" --minutes 120`
   and, separately, `python -m lab slot request --task R12 --what "R12 bench, 3 arms x 5 spawns (pod B)" --minutes 330`.
   `python -m lab placement --check "R12: kernel bench" --gpu B` must not object (benchmarks go to pod-b).
4. The pod must be up (owner's call), with a clone of this repo at the run commit in `~/Knighthacks26` (as R20 needs
   for `lab/bench.py`) and `pip install --user httpx`. Every command below runs on the pod from `~/Knighthacks26`
   through `python -m lab.controller run-script --account D [--timeout S] <script> ...`, or in the pod's terminal.

## 1. Slot 1 (about 2 hours): rebuilds, LDS bytes and the token match

`slot start --row <row>` first. All servers below listen on 127.0.0.1:8100 (never 8000) and are stopped after use.

Serve command shared by every R12 server (`SERVE` below):

```
python3 -m vllm.entrypoints.openai.api_server --model ~/models/Qwen3.8-27B-INT4 --served-model-name teacher \
  --port 8100 --host 127.0.0.1 --max-model-len 16384 --limit-mm-per-prompt '{"image":1}' \
  --gpu-memory-utilization 0.90 --dtype bfloat16 --mm-encoder-attn-backend TRITON_ATTN --attention-backend TRITON_ATTN
```

The match servers add `--enforce-eager --no-enable-prefix-caching --max-num-seqs 1`.

1. **Stock rebuild** (about 5 min): `VLLM_DIR=~/vllm R12_STATE_DIR=~/.cache/r12 bash gpu/rebuild.sh stock`.
   The existing R30 build has no stamp, so this rebuilds once and records its wall-clock in
   `gpu/results/rebuild_times.json`. A non-zero exit stops slot 1: exit 3 prints the rejected hunk (record it on
   T0), exit 4 is a failed build (`~/.cache/r12/build.log`), exit 5 means the build tools are missing (run
   `bash lab/pod_a_teacher.sh build` once, then repeat).
   Then `git -C ~/vllm diff > gpu/results/stock.applied.diff`.
2. **Stock match spawns, two cold spawns** (about 12 min each). For n in 1 and 2: a fresh, empty
   `TRITON_CACHE_DIR=~/r12/triton/stock-match-$n`, start `SERVE` with the match flags in the background (log to
   `~/r12/stock-match-$n.log`), wait for `/v1/models`, then
   `python gpu/match_check.py collect --url http://127.0.0.1:8100 --pid <server pid> --build stock
   --spawn stock-$n --max-tokens 128 --out gpu/results/match/stock-$n.json`, then stop the server
   (`pkill -f vllm.entrypoints`, wait until it is gone).
   If stock does not start with `--attention-backend TRITON_ATTN`, stop here: the T0 lines record that, not a speed.
3. **Stock LDS record**: `python gpu/lds_check.py --arm stock --triton-cache ~/r12/triton/stock-match-1
   --serve-log ~/r12/stock-match-1.log --out gpu/results/lds/stock.json`. Stock is expected to pass (it serves
   today); a failure here is a finding in itself.
4. **Stages arm**: `bash gpu/rebuild.sh stages` (records its wall-clock), `git -C ~/vllm diff >
   gpu/results/stages.applied.diff`, one cold match spawn as in step 2 with `--build stages --spawn stages-1` and
   `TRITON_CACHE_DIR=~/r12/triton/stages-match-1`, then
   `python gpu/lds_check.py --arm stages --triton-cache ~/r12/triton/stages-match-1 --serve-log
   ~/r12/stages-match-1.log --out gpu/results/lds/stages.json` and
   `python gpu/match_check.py compare --stock gpu/results/match/stock-1.json --stock gpu/results/match/stock-2.json
   --changed gpu/results/match/stages-1.json --out gpu/results/match/stages.json`.
5. **Block arm**: the same as step 4 with `block` in every name.
6. **Gate.** An arm goes to slot 2 only when its `lds_check` exit is 0 (every touched kernel recorded, none over
   65,536 bytes, no OutOfResources in the log) and its `match_check compare` exit is 0 (match rate no lower than
   stock's own across the two cold spawns). An arm that fails either gate is recorded on T0 with the LDS bytes or the
   mismatches (first divergent position and logprob gap each) and gets no speed numbers.
7. `bash gpu/rebuild.sh stock` (puts the pod back to the stock build), `slot done --row <row>`.

## 2. Slot 2 (about 5.5 hours): R11 before and after, on the same prompts

`slot start --row <row>`. Five rounds; each round runs stock, then stages, then block (only the arms that passed the
gate), so drift over the slot hits every arm alike. Each arm in a round:

1. `bash gpu/rebuild.sh <arm>` (the stamp skips nothing here, since the arm changes every time; each rebuild's
   wall-clock goes to `gpu/results/rebuild_times.json`).
2. One cold spawn of R11 at batch 1 and 4 in the same spawn, on the fixed prompt set:

```
TRITON_CACHE_DIR=~/r12/triton/<arm>-bench python lab/bench.py run --engine vllm \
  --checkpoint ~/models/Qwen3.8-27B-INT4 --engine-dir ~/vllm --node pod-b \
  --patch lab/vllm-gfx1201.diff [--patch gpu/patches/unified-attention-<arm>.patch] \
  --batches 1,4 --no-64 --share-spawn --repeats 1 --max-tokens 128 --prompts fixtures/lab/bench/prompts.json \
  --out ~/r12/bench/<arm> --serve-cmd '<SERVE as a JSON list, with {port} {checkpoint} {model} {max_len}>'
```

   (stock passes only the first `--patch`; `--patch` records each file's sha256 on every row.) The serve command is
   `SERVE` above with the placeholders, and without `--enforce-eager`: speed is measured as the teacher serves.
3. After the five rounds, classify every spawn's band per arm. Each session holds one spawn, too few for R11 to
   band on its own, so `gpu/r12_bands.py` pools an arm's five sessions and runs R11's own `summarize_runs` on them
   (largest-gap split of each spawn's probe decode rate, at least 4 spawns, thresholds from the clusters):

```
python gpu/r12_bands.py --arm stock --out gpu/results/bench/stock/bands.json ~/r12/bench/stock/*/rows.jsonl
python gpu/r12_bands.py --arm stages --stock gpu/results/bench/stock/bands.json \
  --out gpu/results/bench/stages/bands.json ~/r12/bench/stages/*/rows.jsonl
python gpu/r12_bands.py --arm block --stock gpu/results/bench/stock/bands.json \
  --out gpu/results/bench/block/bands.json ~/r12/bench/block/*/rows.jsonl
```

   Each `bands.json` lists every spawn with its probe decode rate and band, the per-band statistics per batch size,
   and (for stages and block) `vs_stock`: the change in the median of each metric against stock, overall and within
   the fastest band.
4. Run `lds_check` once more per changed arm on `~/r12/triton/<arm>-bench` (the graph-mode compile of the same
   kernels) and keep it as `gpu/results/lds/<arm>-bench.json`.
5. `bash gpu/rebuild.sh stock`, `slot done --row <row>`.

## 3. Bring the results back and check them in (from the box)

`python -m lab.transfer pull-file --account D <remote> <local>` for each file under `~/Knighthacks26/gpu/results/`
and `~/r12/bench/<arm>/*/rows.jsonl` and `summary.json`, into `gpu/results/` on a branch from this one:

- `gpu/results/rebuild_times.json`: every rebuild's wall-clock (arm, patch sha256, commit, seconds).
- `gpu/results/<arm>.applied.diff`: exactly what each arm's tree carried (the diff checked into `gpu/`).
- `gpu/results/lds/<arm>.json` and `<arm>-bench.json`: LDS bytes per kernel against 65,536, failures counted.
- `gpu/results/match/*.json`: each spawn's outputs and each arm's comparison report.
- `gpu/results/bench/<arm>/rows.jsonl`, `summary.json`, `bands.json`: the R11 rows with their band per spawn.

`python -m pytest -q` before the commit (the secret scan reads the new files too). The commit message names the run
date, the pod, the slot rows and the E numbers.

## 4. Write the result on T0

Beside each pre-registered line: the measured change in `joules_per_token` at batch 1 and 4 (median over the five
spawns, per band when there are bands, with the band shares), `decode_tok_s` beside it, the LDS bytes of each
touched kernel, the match rate against stock's own, the rebuild wall-clocks, and the commit of `gpu/results/`.
**A slower result is still a result: it is recorded on T0 in the same form, with the same rows checked in, and is
never dropped or re-run until it looks better.** A result inside the pre-registered range is recorded as "no change
detectable at 5 spawns per arm", which is the expected outcome.
