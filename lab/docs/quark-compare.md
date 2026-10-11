# Quark INT4 checkpoint against the teacher's: `python -m lab quark-compare` (R23)

`amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16` loads at b1388b1 only with the loader hunks at the end of
`lab/vllm-gfx1201.diff` (Python only: the mapper takes the export's `algo_config`, a list of dicts, and the
`QuarkW4A16Int4` scheme of vLLM PR #48606 is backported). Its key names are the ones b1388b1 reads;
`lab/quark_keys.py` gives every key's disposition and is the pre-flight. The runner compares it with RedHat INT4 on
pod B, same build and settings sheet (the pinned baseline of `data/searchspace-inference-gfx1151.json`):

```
python -m lab quark-compare preflight                    # checkpoint sizes, config keys, time plan; changes nothing
python -m lab quark-compare slot [--row ID --reply TEXT] # the 60-minute pod B row (critique, then reply)
python -m lab quark-compare run --batches 1,4            # on pod B in the slot: hunks on, bench, correctness, hunks off
python -m lab quark-compare record --result F --control-rows R20_ROWS [--reference F] --row ID
```

`record` reads tok/s and J/token B against A per batch size with `lab/machine/verdict.py`, the exact-match rate
against A's greedy outputs, writes a `quark-compare` ledger row, T0's E5 result (arms A and B) and
`R23_T0_result.txt`, and releases the slot; `all` does the four in one process. Steps for the real run:
`lab/R23_REAL_RUN.md`.
### Baseline matrix on pod D (R20, partial)

The stock build on pod D (SJC account D, Strix Halo, gfx1151; node `pod-b` in the ledger): vLLM 0.19.2+gfx1151 at
b1388b1 plus `lab/vllm-gfx1201.diff` (sha256 0b5ac113...), `Qwen3.8-27B-INT4` (compressed-tensors, W4A16 through
`ConchLinearKernel`), max model length 16384, the `pod_a_teacher.sh` serve flags (`--mm-encoder-attn-backend
TRITON_ATTN`, `--gpu-memory-utilization 0.90`, bfloat16), port 8100, `GPU_MAX_HW_QUEUES` unset, thinking off, greedy,
128 tokens per request, the bench-v1 prompt set, 4 requests per stream. Every cell is a cold spawn with its warm-up
discarded. 2026-10-10 evening ET, run with `python3 -m lab bench run ... --node pod-b --hub sjc --account D` from
`~/Knighthacks26` on the pod. Raw per-spawn rows: `fixtures/lab/bench/pod-b-2026-10-10/`. Medians also went to
`lab/refine/metrics.jsonl` as `teacher.*` on node `pod-b` (low_confidence: under the 5 spawns `verdict()` needs).

| Batch | Spawns | tok/s per stream (median, min, max) | tok/s total (median) | TTFT s (median) | J/token gross (median, min, max) | Watts avg |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 (J: 2) | 2.73 (2.73, 2.73) | 2.67 | 0.96 | 36.4 (36.0, 36.8) | 97 |
| 4 | 2 | 2.58 (2.58, 2.58) | 9.48 | 5.15 | 10.3 (10.1, 10.5) | 98 |
| 8 | 2 | 2.37 (2.36, 2.38) | 16.5 | 8.49 | 5.81 (5.80, 5.81) | 96 |
| 16 | 2 | 2.06 (2.01, 2.12) | 27.4 | 11.9 | 3.50 (3.49, 3.50) | 96 |
| 64 | 2 | 0.88 (0.86, 0.90) | 49.0 | 24.2 | 1.96 (1.93, 2.00) | 96 |

Partial: 3 spawns per batch size were asked for (15 spawns); this table holds 11 that measured: three at batch 1 (J/token on two of them) and two each at batch 4, 8, 16 and 64. The third spawn of batch 4, 8, 16 and 64 was cut by order (pod D went to other jobs); a medians-of-three matrix needs those four. Median of two is the
mean of the two, so min and max bracket it. The batch-1 spawn at 03:58 UTC decoded 2.73 tok/s but its power samples
did not cover the decode window (J/token null). Idle (no server) read 15 W, server loaded 16 W; the sampler reads a
flat 96 to 97 W under load on this chip, so J/token is nearly watts / total tok/s. All spawns fall into one band (no
decode-rate gap; `mem_busy_percent` is not exposed on the SJC image). The KV cache holds 304192 tokens (55.6 requests
of 16384), and 64 streams of ~250 tokens fit. Three things went wrong: the first spawn died in the engine core during
torch.compile (exit -2, no traceback), the third repeat was cut by SIGTERM, and a stray 64 arm was added to a batch-1
run (stopped). The R30 teacher build serves on pod D from a cold start in about 5 minutes with a warm compile cache.

What the harness could not read on pod D: ROCm version (`/opt/rocm/.info/version`), the amdgpu driver and ASIC serial
(amd-smi prints N/A), the KFD queue count, `mem_busy_percent`. Attention backend per the serve log: `ROCM_ATTN`.
