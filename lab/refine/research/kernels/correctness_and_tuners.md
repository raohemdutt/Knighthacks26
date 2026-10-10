# Checking changed kernels, and how kernel tuners choose configs

Research note for the kernels track (Task Board O8), searched on 2026-10-10. Sources are indexed in
`SOURCES.md`; labels, the Implication and Experiment lines, and the `KE` and `V` experiment ids
follow the conventions stated at the top of `w4a16_decode_gfx1201.md`, whose section 7 ranks the
experiments and defines V1 and V2.

What this note covers: how to tell a correct-but-different kernel from a wrong one (kernels rubric
test 4 and task R21), why exact greedy match is fragile on our platform, how vLLM's and PyTorch's
tuners pick a config by time alone, and what is known about tuning for energy instead.

## 1. Checking a changed kernel

### 1.1 Per-operation tests come first

- **[practitioner]** vLLM's tests for the RDNA hybrid W4A16 kernel compare FP16 and BF16 outputs
  against an FP32 reference, add dispatch-boundary tests at the M and K x M limits, and assert that a
  row-padded weight gives bit-identical output (`rtol=0, atol=0`) through both the HIP decode
  kernel and the Triton prefill kernel. Source: [S16] vllm-project/vllm PR #52619, test plan; [S18]
  vllm-project/vllm PR #56301, accuracy section.

  Implication: before any end-to-end comparison, a changed kernel should pass shape-by-shape tests
  against an FP32 dequantize-and-matmul reference at the exact shapes we serve, with a stated
  tolerance; layout-only changes (padding, repacking) should be bit-identical.

  Experiment: V2, per-shape tests for every GEMM shape we serve (K = 5120, 6144 and 17408 at M = 1,
  2, 4, 8 and 16, group 128) with relative L2 error recorded; expected about 1e-3 for a correct
  kernel; minimum 1 run per kernel arm; confidence high (0.8).

- **[measured]** vLLM issue #60608 (open, 2026-10-08): the ExLlama GPTQ kernels and the RDNA3 W4A16
  kernels step K in blocks of 32 and switch quantization groups only at block starts, so with group
  size 16 they silently return wrong results: relative L2 error against an FP32 reference of 0.55 to
  0.71 at group 16, against about 0.0005 to 0.04 at groups 32 and 64, reproduced on CUDA sm_120 and
  ROCm gfx1100, with nothing in vLLM rejecting the configuration. Source: [S36]
  vllm-project/vllm issue #60608, body and error table.

  Implication: a kernel can be fast, load without error and still be wrong for a shape class that
  end-to-end greedy checks on a few prompts may not exercise; our group size is 128, which these
  kernels handle, but any kernel arm (KE2, KE4, KE5, KE6) must pass V2 at our group size and shapes
  before its speed counts.

  Experiment: V2, as above; expected a failing arm to show relative L2 error above 0.1 at some shape;
  minimum 1 run per arm; confidence high.

- **[practitioner]** The vLLM wrapper around Conch at b1388b1 carries a fix for a past silent bug:
  "The conch kernel computes stride_mul = block_k / group_size; passing -1 produces a negative
  stride that reads out-of-bounds scale values for all K-blocks after the first", so channel-wise
  group size -1 is mapped to K before the call. Source: [S07] vLLM
  `vllm/model_executor/kernels/linear/mixed_precision/conch.py` at b1388b1, `apply_weights`.

  Implication: retuning Conch's BLOCK_K (KE5) interacts with the group-size arithmetic
  (`stride_mul = block_k / group_size`), so every BLOCK_K in the sweep must be tested against the
  FP32 reference at group 128, and BLOCK_K values that are not multiples or divisors of 128 need
  particular care.

  Experiment: KE5's sweep runs V2 for each BLOCK_K before timing it; expected BLOCK_K of 64, 128 and
  256 to pass; minimum 1 run per config; confidence medium.

### 1.2 Why exact greedy match is fragile

- **[measured]** Thinking Machines Lab sent the same prompt 1,000 times at temperature 0 to
  Qwen3-235B-A22B-Instruct-2507 and got 80 distinct completions, identical for 102 tokens and
  diverging at token 103, because the serving kernels are not batch-invariant: split-K matmul,
  split-KV attention and tile choices change the reduction order with batch size, and the batch
  size depends on server load. With batch-invariant kernels all 1,000 completions matched; the
  deterministic run took 42 to 55 s against 26 s for the default. Source: [S55] Horace He and
  Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference", 2025-09-10.

  Implication: on a correct kernel, exact match against a recorded greedy output holds only if the
  batch composition is the same; a kernel that changes the reduction order (split-K, a different
  K tile) will produce some first-divergence positions without being wrong. R21's exact-match rate
  needs a stated concurrency and a way to judge divergences.

  Experiment: V1, the noise floor: run R21's 50 prompts on the unchanged server on two spawns at
  concurrency 1 and once at concurrency 8, and record exact-match rate and first-divergence token
  per prompt. Expected 100% at concurrency 1 across spawns and below 100% at concurrency 8; minimum
  2 spawns; confidence medium (0.6).

- **[practitioner]** vLLM's batch-invariant mode (`VLLM_BATCH_INVARIANT=1`) lists "NVIDIA GPUs with
  compute capability 8.0 or higher" and "Intel XPUs with Triton support" as supported hardware and
  does not mention ROCm. Source: [S56] vLLM documentation, "Batch Invariance", hardware
  requirements.

  Implication: we cannot switch on deterministic kernels on the pods, so R21 should run at
  concurrency 1 with a fixed `max-num-seqs` and judge a divergence by its logit margin: a
  divergence where the top two tokens were within a small logprob gap is rounding, one with a wide
  gap is a bug.

  Experiment: R21 records the top-2 logprob gap at each first divergence (vLLM can return
  logprobs); expected rounding divergences to show gaps under about 0.1 nats; minimum the 50
  prompts once per arm; confidence low to medium (0.4), because the threshold itself needs V1 data.

- **[measured]** Upstream kernel PRs for this card class report accuracy in ways R21 can copy: #56301
  ran GSM8K (200 questions, 5-shot, greedy) three times per arm on gfx1151 and got 88.00% on every
  repetition of both arms, with output token counts of 23,209 against 23,209 to 23,214; #54965
  reported byte-identical text and token ids before and after; #40977 reported GSM8K 0.835 before
  and 0.855 after on 200 samples. Source: [S18] PR #56301, "Accuracy"; [S17] PR #54965, "Test
  Result"; [S13] PR #40977, accuracy line.

  Implication: when exact match fails, a task metric with repeats is the arbiter; a single 200-item
  GSM8K run moves by about 2 points between equivalent arms (0.835 against 0.855), so one run per arm
  cannot show a difference smaller than that.

  Experiment: every GEMM arm (KE2, KE4, KE5, KE6) runs a 200-question task check three times per
  arm; expected identical or within 2 points for a correct kernel; minimum 3 repetitions per arm;
  confidence medium.

- **[practitioner]** PyTorch TunableOp discards a candidate GEMM implementation "if the given
  solution doesn't achieve the same accuracy as the default implementation". Source: [S51] PyTorch
  2.9 documentation, "TunableOp", tuning description.

  Implication: even a pure-speed tuner checks accuracy per candidate; a sweep we build (KE5, KE15)
  should reject configs that fail V2 automatically, not only report them.

  Experiment: KE5's sweep script rejects configs that fail V2 before timing; expected a few configs
  rejected at unusual BLOCK_K values; minimum 1 run per config; confidence medium.

## 2. How the tuners choose: time only

- **[practitioner]** vLLM's fused-MoE tuner at b1388b1 times each candidate with CUDA events (20
  iterations while tuning) and keeps it only `if kernel_time < best_time`; candidates that raise
  `triton.runtime.autotuner.OutOfResources` are skipped silently. Source: [S49] vLLM
  `benchmarks/kernels/benchmark_moe.py` at b1388b1, `tune` and `benchmark_config`.

  Implication: the shipped configs optimize time, not joules, and the tuner's silent skip of LDS
  overflows is fine for tuning but means the kernels rubric's "resource failures count as failures"
  has to be enforced in our own sweep logs, not inherited.

  Experiment: KE15 (energy-aware selection) reuses a tuner like this with joules per call as the
  objective; see section 3.

- **[practitioner]** vLLM's benchmark for the RDNA hybrid W4A16 kernel times each GEMM with
  `triton.testing.do_bench` (median of quantiles) and zeroes a 256 MB buffer between repetitions so
  the weights come from DRAM. Source: [S50] vLLM main
  `benchmarks/kernels/benchmark_rdna_hybrid_w4a16_gemm.py`, `benchmark` docstring.

  Implication: this is the time objective done carefully; our KE5 sweep should copy its cache flush
  and add a power read around the same calls.

  Experiment: KE5, as specified in `triton_gfx12.md` section 4.

- **[practitioner]** PyTorch TunableOp currently implements only a tunable GEMM for ROCm, choosing
  between rocBLAS and hipBLASLt solutions; each candidate runs for 100 iterations or as many as fit
  in 30 ms, whichever is smaller, the fastest average wins, results are stored in a CSV keyed by
  library versions, and a rotating-buffer option limits cache-flattered timings. Source: [S51]
  PyTorch 2.9 documentation, "TunableOp".

  Implication: TunableOp tunes torch GEMMs (our bf16 prefill projections and `lm_head` above four
  tokens), not Triton or HIP W4A16 kernels, so it cannot touch the decode bottleneck; at most it
  trims TTFT.

  Experiment: KE16, TunableOp on for prefill; expected TTFT 0 to -10% and no J/token change;
  minimum 5 spawns per arm; confidence low (0.3).

## 3. Tuning for energy instead of time

- **[measured]** Kernel Tuner's energy study tuned a GEMM on four NVIDIA GPUs and found that "the
  fastest configuration returned by race-to-idle is not the most energy efficient for any of the
  GPUs" and that "auto-tuning the GEMM kernel for energy efficiency does not lead to the same
  optimal configuration as tuning for time"; the trade-off is GPU-specific (on the RTX A4000 a
  28.4% speed reduction bought 5.8% more energy efficiency, on the A100 a 27.5% speed reduction
  bought 50.9%). Because NVML refreshes power at about 10 Hz, Kernel Tuner runs each configuration
  repeatedly for about one second per energy measurement. Source: [S52] Schoonhoven, Veenboer,
  van Werkhoven, Batenburg, "Going green: optimizing GPUs for energy efficiency through
  model-steered auto-tuning", PMBS workshop at SC22, arXiv:2211.07260, sections III.B and V.A.

  Implication: a time-only tuner can leave energy on the table, but most of the large gains in that
  study came with clock-frequency changes, which our pods cannot make; at fixed clocks the
  reachable gain is from tile and occupancy choices only, and every energy measurement needs a
  window long enough for the pod's sampler (task O5 covers the sampler's rate).

  Experiment: KE15, re-run the KE5 shortlist with joules per call (pod watts integrated over a
  window of at least 1 s per config, 5 repeats) as the objective; expected 0 to 5% J/token at fixed
  clocks; confidence low (0.25). This is the post-hackathon energy-aware tuning thread in the spec.

- **[opinion]** At fixed clocks in memory-bound decode, time is a reasonable proxy for energy,
  because board power varies less across configs than run time does; it stops being a good proxy
  when a faster config also raises power by pulling more bandwidth, which is exactly the regime a
  better decode kernel moves us into. Source: our reading of [S52] and [S53] (decode power far below
  the board limit on H200).

  Implication: report joules per token next to tok/s for every kernel arm (kernels rubric test 6)
  and expect the J/token gain of KE2 to be smaller than its speedup.

  Experiment: KE2 reports watts per arm; expected load power to rise from about 200 W toward 230
  to 280 W as decode speeds up; minimum 5 spawns per arm; confidence medium (0.5).
