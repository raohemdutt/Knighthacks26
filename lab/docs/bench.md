# Benchmark harness: `lab/bench.py` (R11)

Moved here from lab/README.md (O70); the verbs list there links every page. Arguments: `python -m lab bench -h`.

Decode tokens per second and joules per generated token, the measuring stick of the kernels track (rubric K1, K3, K6).
A run is one cold server spawn; repeats are spawns. On the pod the server runs on (R20 runs it on pod B; it needs the
repo clone and `pip install --user httpx`):

```
python lab/bench.py run --engine vllm --checkpoint ~/models/Qwen3.8-27B-INT4 --engine-dir ~/vllm     --patch lab/vllm-gfx1201.diff --node pod-b [--batches 1,4,8,16] [--repeats 5] [--metrics-file]
python lab/bench.py run --engine llamacpp --checkpoint ~/models/<student>.gguf --node pod-b      # same harness
python -m lab bench attach --account C --batches 1 --requests-per-stream 4 --warmup 0 --no-probe  # from the box
```

- **Requests.** `fixtures/lab/bench/prompts.json`: two text prompts and two drawn images (`make_images.py` writes
  them; never camera frames). Greedy (temperature 0, top_p 1, seed 0), a fixed `--max-tokens` (128) with
  `ignore_eos`, thinking off, and a per-request nonce at the front whenever prefix caching may be on (both engines'
  default). Batch size B is B concurrent streams; the defaults are 1, 4, 8 and 16, plus 64 only when the KV-cache
  capacity in the serve log holds 64 requests of the workload (owner ruling 5), else a `skipped` row with the reason.
- **One run.** Idle watts with no server, spawn (`pod_a_teacher.sh`'s vLLM serve line, llama-server's, or
  `--serve-cmd` with `{port}` `{log}` `{batch}` `{max_len}` `{model}` `{checkpoint}`), wait for `/v1/models`, read the
  serve log and the process, loaded-idle watts, warm-up (one request per prompt, then one round at B; discarded and
  listed under `warmup`), one probe request alone (single-stream rate, `mem_busy_percent` while it decodes), the
  measured requests, stop. It refuses a port something already listens on and defaults to 8100: spawning on pod C
  would collide with the teacher serving there (port 8000 and most of the memory), so spawn-per-run measuring is
  pod B's (R20).
- **Numbers.** From the client's own token stream: TTFT, per-request decode rate (first to last token), the median
  per stream and the total over the decode window; watts from amd-smi through `lab/pod_power.py`'s reader and parser
  (or, attached, the T6 sampler on port 8790 through the proxy), integrated over the decode window: average, peak,
  energy, joules per token, with the idle readings beside them and THROTTLE_STATUS.
- **Conditions on every row** (rubric K1 b): engine commit and patch sha256, checkpoint, quantization, max model
  length, batch size, node, ROCm, torch, Triton, the environment read back from `/proc/<pid>/environ` (secrets left
  out, `GPU_MAX_HW_QUEUES` always present), the linear kernel classes, attention backends and KV-cache capacity from
  the serve log, the host (hub, account, hostname, kernel release, amdgpu driver, firmware, ASIC serial and the chip,
  from `amd-smi static`/`firmware` under /opt/rocm/bin), the KFD queue count, time since spawn. What cannot be read
  is null with its reason under `unavailable`. The harness reads the chip and never checks it.
- **Bands.** Spawns are banded by their own probe decode rates (largest-gap split, 10 % of the median, at least 4
  spawns); the rate and `mem_busy_percent` thresholds between bands come from those clusters. Each configuration
  reports per-spawn values with median, min, max, mean and sd, overall and per band with each band's share.
- **Rows.** `lab/state/bench/<session>/rows.jsonl` (one per run) and `summary.json`; a `bench.run` ledger row per run
  and the `bench` row of the whole call (`@recorded`); with `--metrics-file`, each configuration's medians in
  `lab/refine/metrics.jsonl`, `low_confidence` under 5 spawns.

Tested on the box against `lab/bench_fake.py` (a fake OpenAI-compatible server spawned as a real process, writing a
vLLM- or llama-server-style log, a fake `/proc` and `/sys`, and a fake pod whose amd-smi prints the Strix Halo output
recorded on pod C). Not yet seen on a pod, so R20 should check them first: the exact serve-log lines of vLLM b1388b1
for the kernel classes and KV cache (`fixtures/lab/bench/vllm_serve_synthetic.log` is made from the source's log
statements), the output of `amd-smi static --driver`/`--asic` and `amd-smi firmware` on the SJC image
(`amdsmi_*_synthetic.txt` follow amd-smi's layout, not a recording), and whether the pod exposes
`/sys/class/kfd/kfd/proc/<pid>/queues` and `mem_busy_percent`.

Live smoke, 2026-10-10, attached to the teacher on pod C through the proxy (4 requests at batch 1, 64 tokens each,
no spawn, no change on the pod): decode 2.83 tok/s per request (2.67 over the window), TTFT 0.52 s text and 3.63 s
for the 1280x720 image, 112 W average and 124 W peak from the T6 sampler, 42 J per generated token with nothing
subtracted. No row went to the metrics file: a single attached run is not a stock record.
