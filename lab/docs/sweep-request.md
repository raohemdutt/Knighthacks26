# Request-side settings sweep: `lab/sweep_request.py` (R52)

Moved here from lab/README.md (O70); the verbs list there links every page. Arguments: `python -m lab sweep-request -h`.

The settings a request carries, measured on one server spawn of the stock teacher on pod D, so no cell restarts the
server: batch size (B concurrent requests, one each), max tokens, thinking off or on, the prompt length class and the
sampling profile. The grid is committed, `lab/refine/research/kernels/data/sweep-request-grid.json`: four slices
(`base` at 128 tokens, `tokens-32`, `thinking-on-32`, `prompt-long` at about 1,100 prompt tokens) each at batch 1, 4, 8
and 16, three repeats per cell run interleaved over the grid (forward, backward, forward). `base-b1` is the pinned
baseline cell: the operating point the inference settings sheet measured (batch 1, thinking off, temperature 0, a
short text prompt). Sampling stays greedy: the sheet marks no sampling setting as changing decode work on its pinned
server (its only sampling line is the MTP acceptance rate, and MTP is a restart setting held off); a profile under
`sampling` and a slice that sets it add the axis without a code change. The server is launched from the pinned
baseline of `searchspace-inference-gfx1151.json` (its fixed launch, the baseline value's flags and variables, its
unset list removed with `env -u`), with `lab/bench.py`'s stock vLLM line as the fallback if that launch dies before it
is ready; the result says which one served.

```
python -m lab sweep-request --dry-run [--quick]          # the grid, each cell's expected seconds, the slot minutes
python -m lab sweep-request pod-script [--quick]         # writes lab/state/sweep_request/pod_run.sh (needs AUP_HUB_USER_D)
python tooling/dispatcher/pod_exec.py D term <clone>/lab/state/sweep_request/pod_run.sh 3600 > <clone>/lab/state/sweep_request/pod_run.out
python -m lab sweep-request collect lab/state/sweep_request/pod_run.out --metrics-file
```

- **Slot.** `--dry-run` gives 55 minutes for the full grid (16 cells, 48 runs) and 30 for `--quick` (the `base` slice,
  four cells), from the grid's timing model: pod D's 523 s cold start, pod C's 2.88 tok/s at batch 1, a 5 % per-stream
  slowdown per extra stream and 400 prefill tokens a second (both guesses on the slow side), plus 10 %. The pod run
  never starts a cell its deadline (the slot minutes) cannot hold; such a cell is a skipped row with the reason.
- **The pod run.** `pod_run.sh` carries the lab files it imports, the grid, the prompt sets, the search space and the
  patch as one base64 block checked by sha256 (no tab or exclamation mark, so term mode can type it), unpacks them under
  `~/sweep_request/bundle-<sha>`, installs httpx if missing and runs `pod-run`. The gate refuses before anything
  starts unless the node is pod-b, the machine x86_64, rocminfo names gfx1151, the Jupyter user hashes to account D's
  (the script holds the hash, not the name), port 8100 is free and no vLLM server runs. Then idle watts, one spawn,
  idle watts loaded, one discarded 16-token warm-up round per batch size and prompt class, every cell through
  `lab/bench.py`'s `Session.measure` (the same TTFT, decode rate, amd-smi watts, joules per token and conditions as the
  harness), and the stop, however the run ends. It prints three headline lines and the result block, and keeps both in
  `~/sweep_request/last_result.b64` (`pod_exec.py D tail sweep_request/last_result.b64 3000` fetches it if the terminal
  dropped).
- **collect** writes `lab/refine/research/kernels/data/sweep-request-gfx1151-<date>.json` and `.md` (a `-quick` suffix
  for the quick grid): per cell the median of its repeats with the lowest and highest, decode tok/s per stream and in
  total, TTFT, watts and joules per token, the change against `base-b1`; the best cell by aggregate decode tok/s and its
  change over the baseline with the range the two cells' lowest and highest repeats allow; the lowest joules per token;
  what stays open. It appends one `sweep_request.cell` ledger row per cell on node pod-b and, with `--metrics-file`, the
  baseline and best cells' headline medians and the derived change to `lab/refine/metrics.jsonl`, all low_confidence
  (three repeats on one spawn). The raw result is kept under `lab/state/sweep_request/<run>/`.
- **Tested on this machine** (`tests/test_lab_sweep_request.py`) against `lab/bench_fake.py` spawned as a real server:
  the grid parser, the cell-to-harness mapping, one spawn for the whole grid, the fallback launch, the deadline, stuck
  streams, the summary arithmetic on chosen numbers, the rows, the result block, the gate and the pod script (its
  bundle runs the dry run with the standard library alone). `fixtures/lab/sweep_request/recorded_full.out` is the
  committed grid run on the fake, as the pod prints it (`make_fixture.py` records it again; the test checks the grid's
  sha256). Not yet seen on a pod: the pinned launch's flags on vLLM b1388b1, the real rates and the warm start time.
