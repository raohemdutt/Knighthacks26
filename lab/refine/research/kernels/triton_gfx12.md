# Triton and HIP kernels on gfx12: LDS, wave32, WMMA and the tuning knobs

Research note for the kernels track (Task Board O8), searched on 2026-10-10. Sources are indexed in
`SOURCES.md`; labels, the Implication and Experiment lines, and the `KE` experiment ids follow the
conventions stated at the top of `w4a16_decode_gfx1201.md`, whose section 7 ranks the experiments.

What this note covers: the hardware limits a kernel on the R9700 has to respect, the matrix
instructions RDNA4 offers, the Triton knobs that have measurably mattered on gfx12, and how to
benchmark a kernel on a card with a 64 MB last-level cache. AMD's own RDNA4 ISA reference
(document 70651) sits behind a portal that returned HTTP 401 to our fetch, so the instruction facts
below come from AMD's GPUOpen articles and from the LLVM AMDGPU backend instead.

## 1. Hardware limits

- **[practitioner]** AMD's ROCm hardware table lists the Radeon AI PRO R9700 as RDNA4, LLVM target
  gfx1201, 32 GiB, 64 compute units, wavefront size 32 or 64, LDS 128 KiB, Infinity Cache 64 MiB,
  L2 8 MiB, L0 vector cache 32 KiB, VGPR file 768 KiB, SGPR file 32 KiB, GFXIP 12.0. AMD's product
  page adds a boost clock of up to 2920 MHz, 191 TFLOPs of FP16 matrix throughput (383 with
  structured sparsity), a 256-bit bus at 640 GB/s and 300 W total board power. Source: [S37] ROCm
  documentation, "GPU hardware specifications", RDNA4 table, row "Radeon AI PRO R9700"; [S60] AMD
  product page "AMD Radeon AI PRO R9700", specifications.

  Implication: the decode roofline in the main note (about 41 tok/s at 15.5 GB per step) uses
  640 GB/s; the 64 MiB Infinity Cache is large enough to hold any single GEMM's int4 weights in
  this model, which distorts microbenchmarks (section 4).

  Experiment: KE0 records the gfx and memory clocks during decode next to kernel times; the T0
  note records a gfx clock range of 500 to 2350 MHz on the pod ([L2]), below the 2920 MHz boost,
  which would matter only for compute-bound phases; minimum 2 spawns; confidence high.

- **[practitioner]** In the LLVM AMDGPU backend, the maximum LDS a single workgroup can allocate is
  "For GFX10-GFX12 in WGP mode ... limited to 64k even though the WGP has 128k in total"; on our card
  Triton reports the limit as "Hardware limit: 65536" when a kernel overflows it. Source: [S43]
  LLVM `llvm/lib/Target/AMDGPU/AMDGPUSubtarget.h`, comments on `getLocalMemorySize` and
  `getAddressableLocalMemorySize`; [S29] vLLM issue #48723, error text.

  Implication: the per-kernel LDS check in the kernels rubric (test 5) is 65,536 bytes per
  workgroup, and two workgroups at up to 64 KB each can share one WGP's 128 KB.

  Experiment: V2's kernel tests also record each compiled kernel's shared-memory size from the
  Triton kernel metadata; expected every shipped config under 65,536 bytes; minimum 1 run per
  config; confidence high.

- **[measured]** Our own vLLM build needed `num_stages=1` in both launches of the Triton unified
  attention kernel because default pipelining asked for 66,560 bytes of LDS on gfx1201 and failed
  with `OutOfResources`. Source: [L3] `lab/vllm-gfx1201.diff`, hunk in
  `vllm/v1/attention/ops/triton_unified_attention.py` and its comment, merged in PR #37.

  Implication: this is the first recorded instance of test 5 on our stack; it costs software
  pipelining in attention, which matters at long context (KE8), not for short captions.

  Experiment: KE8 measures whether a backend that fits the LDS budget without dropping pipelining
  (for example `ROCM_SEGMENTED_ATTN`) beats the `num_stages=1` kernel at 4k to 16k context;
  expected +20% to +80% at long context; minimum 5 spawns per arm; confidence medium (0.45).

- **[practitioner]** vLLM's RDNA hybrid W4A16 kernel states the same budget in code ("AMD RDNA has
  128 KiB total LDS per CU, but 64 KiB per workgroup") and sizes its skinny decode path from it:
  the activations of a decode GEMM are cached in LDS up to 32,768 fp16 elements, and after #52619 a
  "medium" variant accepts up to 1.2 times that by keeping a prefix in LDS and reading the rest from
  global memory. Source: [S14] vLLM main `rdna_hybrid_w4a16.py`, constants `LDS_CAPACITY_ELEMENTS`
  and `MEDIUM_SKINNY_LIMIT_ELEMENTS`; [S16] vLLM PR #52619, purpose section.

  Implication: on our shapes one decode token fits easily (K up to 17,408 elements), and the
  K x M limit is why batch sizes of 3 to 5 fall back to Triton for `down_proj`.

  Experiment: KE2 reports tok/s separately at concurrency 1, 2, 4 and 8 to show where the fallback
  starts; expected a step in per-token cost between 2 and 3 sequences; minimum 5 spawns; confidence
  medium.

## 2. The WMMA instruction set

- **[practitioner]** GFX12 adds wave32 WMMA at 16x16x16 for f16 and bf16 (accumulating in f32 or in
  the input type), for iu8, for iu4 (also 16x16x32), and four fp8 and bf8 combinations accumulating
  in f32, plus sparse SWMMAC variants at 16x16x32 (16x16x64 for iu4). There is no fp4, fp6 or
  block-scaled (MX) matrix instruction in the set. Source: [S42] LLVM PR #77795, "[AMDGPU] Add GFX12
  WMMA and SWMMAC instructions", merged 2024-01-24, intrinsic and builtin definitions.

  Implication: on gfx1201, MXFP4 and int4 weights must be unpacked in registers to bf16, fp8 or iu4
  before a matrix instruction; there is no hardware path that consumes MX scales, which is why
  upstream vLLM emulates MXFP4 here (precision note, "MXFP4").

  Experiment: none directly; it bounds KE13 (MXFP4 is a software-dequant kernel on this card).

- **[practitioner]** AMD's GPUOpen article gives theoretical matrix throughput per clock per compute
  unit for the RX 9070 XT (RDNA4) of 1024 FP16, 1024 BF16 and 2048 INT8 operations, against 512, 512
  and 512 on the RX 7900 XTX (RDNA3), and says the RDNA4 WMMA register layout changed so RDNA3 WMMA
  code is not backward compatible. Source: [S40] Harada and Yoshimura, "Using the Matrix Cores of
  AMD RDNA 4 architecture GPUs", AMD GPUOpen, 2025-07-11, throughput table and section
  "Transitioning from AMD RDNA 3 WMMA to RDNA 4 WMMA".

  Implication: kernels written for RDNA3 matrix cores (such as the gfx11-only W4A16 kernel in vLLM
  #41394) cannot be reused on gfx1201 without rework; for compute-bound shapes INT8 matrix math is
  twice the FP16 rate, which is the case for W8A8-int8 prefill (precision note).

  Experiment: none directly; see "INT8 and INT4 matrix paths" in `precision_kv_batching.md`.

- **[practitioner]** On RDNA4, global and shared memory loads go up to 128 bits per lane; FP16 WMMA
  operands fill that width, while FP8 and INT8 operands use only 64-bit loads, so AMD's guide fuses
  two WMMA instructions into a double-K step to use 128-bit loads, "effectively doubling memory
  throughput for these narrower data types". Source: [S41] Hui Zhang, "WMMA guide for AMD RDNA 4
  architecture GPUs - part 2", AMD GPUOpen, 2026-06-02.

  Implication: a 4-bit decode kernel should fetch packed weights in 128-bit vectors (32 int4 values
  per lane per load); a kernel that loads 32 bits per lane leaves three quarters of the load width
  unused, which is one thing to look for in the KE5 and KE6 assembly.

  Experiment: KE5 shortlists Conch configs whose compiled code issues 128-bit global loads for the
  weight tile (checked in the dumped ISA); expected better bandwidth for those configs; minimum 1
  microbenchmark per config; confidence low.

## 3. Triton tuning knobs that matter on gfx12

- **[practitioner]** The ROCm "Optimizing Triton kernels" page covers auto-tunable configurations,
  MLIR and ISA analysis and occupancy, and points to "the AMD Instinct MI300 Series / MI350 Series
  workload optimization guide" for parameter guidance; it says nothing about RDNA. Source: [S38]
  ROCm documentation, "Optimizing Triton kernels".

  Implication: there is no vendor-written tuning recipe for gfx12 Triton kernels; CDNA knobs such as
  `matrix_instr_nonkdim` do not transfer, and our configs have to be tuned on the R9700 itself with
  the LDS check and cache-flushed timing below.

  Experiment: KE5 (Conch retune) and KE11 (GDN configs) are the two places this applies first.

- **[measured]** vLLM PR #60400 (open) tunes the Gated DeltaNet packed-decode kernel per arch: on the
  R9700 the new `(BV, num_warps, waves_per_eu)` rows cut kernel time 7.6% to 10.0% (geometric mean
  over batch 1 to 512, bf16 or fp32 state, TP 1 to 4) against the default `BV=32, num_warps=1`; the
  author notes that "on gfx1201, low-warp configs without the hint are erratic across batch sizes
  (e.g. BV=8/2 warps is -15% at B=1 but +125% at B=2); with `waves_per_eu=2` they are consistent".
  Source: [S30] vllm-project/vllm PR #60400, claims table and "Why waves_per_eu".

  Implication: `waves_per_eu` is a live knob on gfx12, and a config that wins at batch 1 can lose
  badly at batch 2, so every tuned table must be checked at each batch size we serve.

  Experiment: KE11 (backport the GDN config table, Python only). Expected the GDN kernel 7.6% to
  10% faster and the decode step 1% to 2% faster; minimum 5 spawns per arm; confidence medium-high
  (0.6) for the kernel, low for the step.

- **[measured]** vLLM PR #58225 narrowed the KV tile (`BLOCK_N` 128 to 32) of the Triton prefill
  attention kernel on RDNA3 and RDNA4: kernel medians were 2.25x to 2.30x faster on gfx1201 and
  2.56x to 2.67x on gfx1100, and end-to-end embedding throughput rose 1.08x to 1.12x on gfx1201,
  with nDCG@10 moving by -0.0005. Source: [S31] vllm-project/vllm PR #58225, test results.

  Implication: default tiles chosen for CDNA are often too wide for RDNA's 64 KB per workgroup and
  wave32; this kernel serves encoder and non-causal attention, so it affects embeddings, not our
  decode, but the pattern argues for narrow tiles in any Triton kernel we tune.

  Experiment: KE5 includes BLOCK_N 32 in the Conch sweep; expected the 5120-wide outputs to gain
  most because their grid doubles; minimum 5 spawns for the shortlisted config; confidence low
  (0.4).

## 4. Benchmarking a kernel on a card with a 64 MB cache

- **[practitioner]** Two recent vLLM kernel PRs for this card class time kernels so that data comes
  from DRAM: #60400 times each config inside a captured HIP graph that rotates over at least 1 GB of
  state slots ("vs its 64 MB Infinity Cache"), alternating default and new configs over 5 rounds and
  reporting the median; the RDNA hybrid W4A16 benchmark uses `triton.testing.do_bench` because it
  zeroes a 256 MB buffer between repetitions, noting that "with a cudagraph replay the weight stays
  resident in the 32 MB MALL, which flatters wide-BLOCK_N tiles and hides row-stride effects".
  Source: [S30] PR #60400, performance methodology; [S50] vLLM main
  `benchmarks/kernels/benchmark_rdna_hybrid_w4a16_gemm.py`, docstring of `benchmark`.

  Implication: our largest int4 GEMMs (`gate_proj`, `up_proj` and `down_proj`, 17408 x 5120 int4
  values) are 44.6 MB each and fit in the R9700's 64 MB Infinity Cache, so a microbenchmark that
  repeats the same GEMM measures cache bandwidth, not DRAM bandwidth, and can rank configs wrongly
  for decode.

  Experiment: KE5's shortlist microbenchmark rotates at least 1 GB of distinct weight copies (or
  flushes a 256 MB buffer) between calls and reports the median over 5 alternating rounds;
  expected effective bandwidth well under 640 GB/s, unlike a cache-resident loop; minimum 5 rounds
  per config; confidence high (0.8) that the flush changes the ranking for at least one config.

## 5. Conch's resource use on this card

- **[opinion]** Conch's fixed tile (BLOCK_M 128, BLOCK_N 64, BLOCK_K 64, 2 stages) would need about
  2 x (128 x 64 + 64 x 64) x 2 bytes = 48 KB of LDS if both operands were staged in bf16, which fits
  under 64 KB but allows only two such workgroups per WGP; a decode tile with BLOCK_M 16 would cut
  the activation staging eightfold and leave room for more workgroups in flight. This is an
  estimate, because the real allocation depends on how Triton's AMD backend lays out the dequantized
  operand. Source: arithmetic on [S10] and [S43].

  Implication: the compiled kernel's actual shared-memory size is the number that matters for test
  5 and for occupancy; it should be read from the Triton metadata, not assumed.

  Experiment: KE5's first step dumps the shared-memory size and VGPR count of the stock Conch
  kernel and of each candidate; expected stock usage between 24 KB and 48 KB; minimum 1 compile per
  config; confidence medium.
