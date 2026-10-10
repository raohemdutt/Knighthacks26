# Precision formats, KV cache, recurrent state and batching on the R9700

Research note for the kernels track (Task Board O8), searched on 2026-10-10. Sources are indexed in
`SOURCES.md`; labels, the Implication and Experiment lines, and the `KE` experiment ids follow the
conventions stated at the top of `w4a16_decode_gfx1201.md`, whose section 7 ranks the experiments.

What this note covers: FP8 matrix math on gfx1201 (stated as a claim that task R25 is asked to
confirm or correct), MXFP4, the INT8 and INT4 matrix paths, KV-cache quantization, the precision
of the Gated DeltaNet recurrent state, and what batching does to joules per token.

## 1. FP8 on gfx1201

- **[practitioner]** FP8 is present at every layer of our stack on gfx1201: the matrix unit has four
  fp8 and bf8 WMMA variants accumulating in f32 [S42]; hipBLASLt supports the OCP `hipblaslt_f8` and
  `hipblaslt_bf8` types on gfx12 [S39]; vLLM at b1388b1 returns true from `supports_fp8()` for any
  arch containing gfx94, gfx95 or gfx12 [S58]; and the dense skinny FP8 GEMM `wvSplitKQ` has a gfx12
  path built on `__builtin_amdgcn_dot4_f32_fp8_fp8`, where the hardware reads FP8 as OCP e4m3fn on
  gfx12 and as fnuz on gfx9 [S20]. Source: [S42] LLVM PR #77795; [S39] hipBLASLt "Data type
  support"; [S58] vLLM `vllm/platforms/rocm.py` at b1388b1, `supports_fp8`; [S20] vLLM PR #34709,
  commit 2 description.

  Implication: the claim for R25 to confirm or correct is this: `Qwen/Qwen3.8-27B-FP8` loads on our
  b1388b1 build and runs its linear layers through a native FP8 kernel rather than an fp32 upcast,
  with block-FP8 shapes going to Triton GEMMs that have no tuned R9700 configs (so vLLM warns per
  shape, and the community measured +33% single-stream from adding such configs,
  `rdna4_vllm_ports.md` 1.2). FP8 checkpoints are OCP e4m3fn; an fnuz checkpoint made for MI300
  would be the wrong format for this card.

  Experiment: KE14 (load test first, then the R20 matrix). Expected a native FP8 kernel class in
  the serve log and per-shape missing-config warnings; minimum 1 spawn for the load test, 3 spawns
  for speed; confidence medium (0.6) for the claim, low (0.2) for a useful speedup given VRAM.

- **[opinion]** VRAM decides KE14 more than speed does: the FP8 checkpoint is 28.8 GB ([L1]
  `lab/README.md`, "Checkpoints") on a 32 GiB card, so at `--gpu-memory-utilization 0.90` (about
  30.9 GB) roughly 2 GB remain for KV cache, the float32 GDN state (about 151 MB per sequence slot,
  section 5), activations and CUDA graphs. That fits a few short sequences, not the 16 concurrent
  labeling sequences of KE1, and not the 16k context the teacher serves today. Source: arithmetic
  on [L1], [S25] and [S37].

  Implication: even if FP8 decode is 2x faster than our Conch INT4 path at one stream, it loses the
  batching lever, which is worth more in J/token for labeling; FP8 is a single-stream option only.

  Experiment: KE14 records the KV-cache token capacity vLLM reports at startup with the FP8
  checkpoint next to the INT4 one (about 160k tokens for INT4, from the roughly 10x concurrency at
  16k measured in [L1]); expected a small fraction of that for FP8, possibly too little to start at
  a 16k maximum length; minimum 1 spawn each; confidence medium.

## 2. MXFP4

- **[practitioner]** Upstream vLLM returns true from `supports_mx()` only on gfx95x and gfx1250, so on
  gfx1201 a Quark MXFP4 or MXFP6 layer logs that "Simulated weight dequantization and activation QDQ
  (quantize and dequantize) will be used, with the linear layers computed in high precision".
  Source: [S23] vLLM main `vllm/platforms/rocm.py` `supports_mx` and
  `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py`, constructor warnings.

  Implication: the AMD Quark MXFP4 checkpoint of our teacher would run slower than the INT4 one on
  stock vLLM on this card, because emulation dequantizes weights before a high-precision GEMM.

  Experiment: KE13 only through the Capicua fork's RDNA MXFP4 kernel; expected stock-vLLM MXFP4 to be
  slower than INT4 (not worth a slot); confidence medium (0.6).

- **[practitioner]** A vLLM PR adding an MXFP4 inline-dequant Triton kernel for RDNA4 described
  gfx12 as hardware "which lacks `tl.dot_scaled` support required by the existing OAI Triton MXFP4
  path" and was closed without merging. Source: [S34] vllm-project/vllm PR #34632 (2026-02-16,
  closed).

  Implication: native-speed MXFP4 on gfx1201 exists only in community forks, so choosing MXFP4 means
  carrying a fork; together with the instruction set ([S42], no MX matrix instruction) it confirms
  MXFP4 is a software-dequant format on this card.

  Experiment: none beyond KE13.

- **[opinion]** MXFP4 does not shrink our bytes: it stores 4-bit values with an 8-bit E8M0 scale
  per 32 values (4.25 bits per weight), while our INT4 checkpoint stores a bf16 scale per 128 values
  (4.125 bits per weight). Source: arithmetic on [S03] (group 32, E8M0) and [S25] (group 128).

  Implication: any MXFP4 speed or energy advantage on gfx1201 would come from a better kernel, not
  from fewer bytes, so KE2 (a better INT4 kernel) dominates KE13 unless the fork's kernel is
  markedly better.

  Experiment: none; the comparison is KE2 against KE13 on the same R20 matrix.

## 3. INT8 and INT4 matrix paths

- **[practitioner]** RDNA4's INT8 matrix rate is 2048 operations per clock per compute unit, twice
  its FP16 rate [S40]; GFX12 also has iu4 WMMA [S42]; and vLLM at b1388b1 offers two ROCm INT8
  W8A8 kernels, `AiterInt8ScaledMMLinearKernel` and `TritonInt8ScaledMMLinearKernel` [S06]. Source:
  [S40] GPUOpen RDNA4 matrix cores article, throughput table; [S42] LLVM PR #77795; [S06] vLLM
  `kernels/linear/__init__.py` at b1388b1, `_POSSIBLE_INT8_KERNELS`.

  Implication: a W8A8-int8 or W4A8 path would double matrix throughput for prefill and large
  batches, but single-stream decode is bandwidth-bound, so it would not move W1 J/token; activation
  quantization also adds accuracy risk for a model whose outputs become training labels.

  Experiment: not queued now. If KE0 shows prefill above a third of a labeling job's time, test a
  W8A8-int8 prefill path for W2; expected TTFT -20% to -40% at concurrency 16; minimum 5 spawns per
  arm plus the R21 gate; confidence low (0.2).

## 4. KV cache quantization

- **[practitioner]** vLLM's FP8 KV cache supports `fp8_e4m3` on CUDA and ROCm; with
  `kv_cache_dtype="fp8"` and no calibration "All quantization scales are set to 1.0", calibration
  with llm-compressor is the recommended path, and `--kv-cache-dtype-skip-layers` keeps sensitive
  layers in the native dtype. Source: [S57] vLLM documentation, "Quantized KV Cache", sections
  "Scale Calibration Approaches", "Additional kv_cache_dtype Options" and "Skipping Specific Layers".

  Implication: FP8 KV is a supported switch on our platform; whether it uses good scales depends on
  the checkpoint.

  Experiment: KE9 (FP8 KV). See the next finding.

- **[practitioner]** Our checkpoint ships a bf16 `self_attn.k_scale` and `self_attn.v_scale` for each
  of the 16 full-attention layers. Source: [S25] `RedHatAI/Qwen3.8-27B-INT4`
  `model.safetensors.index.json` and the `model.safetensors` header.

  Implication: FP8 KV can use calibrated per-layer scales instead of 1.0, provided vLLM loads them;
  the serve log or a debug print has to show they loaded, or KE9 measures the uncalibrated case.

  Experiment: KE9 with `--kv-cache-dtype fp8`. Expected -10% to -20% bytes per step at concurrency
  16 and 8k context, twice the KV token capacity, about no change at concurrency 1; minimum 5 spawns
  per arm plus the R21 gate and a task-accuracy check; confidence low (0.3) for the J/token gain.

- **[opinion]** One token of context costs 64 KB of bf16 KV (16 layers x K and V x 4 KV heads x 256
  x 2 bytes) and the 48 Gated DeltaNet layers keep fixed-size state instead. At concurrency 1 with a
  short prompt that is under 1% of a 15.5 GB decode step; at concurrency 16 with 8k tokens each it
  is about 8.6 GB per step, more than half of the weight bytes. Source: arithmetic on [S25].

  Implication: KV precision matters for J/token only in the batched long-context regime (W2), and
  there mostly through capacity (more sequences fit), so KE9 belongs after KE1, not before.

  Experiment: KE9 runs at concurrency 16 with 8k-token prompts, the regime where it can matter.

- **[measured]** An open vLLM PR enabling FP8 KV in the gfx12 custom paged-attention kernel doubled
  KV capacity from about 132K to 264K tokens for Qwen3-Coder-30B AWQ on an R9700, and earlier
  validation on ROCm 7.2.3 reported mean TPOT 32.81% lower over 10 Qwen3-30B-A3B workloads; the
  kernel keeps "head size 128 and no ALiBi" and limits GQA ratios. Source: [S33] vllm-project/vllm
  PR #34741 (open), summary, changes and validation.

  Implication: that custom kernel does not cover our head dimension of 256, so on our model FP8 KV
  goes through a Triton attention path, with the e4m3 conversion cost and the probability-flush
  accuracy risk described in `rdna4_vllm_ports.md` 1.2.

  Experiment: KE9 records the attention backend chosen with FP8 KV; expected a Triton backend, not
  the custom paged kernel; minimum 1 spawn; confidence high (0.8).

## 5. Gated DeltaNet state precision

- **[practitioner]** The checkpoint's text config sets `mamba_ssm_dtype: float32`, and vLLM at b1388b1
  copies that value into `mamba_ssm_cache_dtype` when the flag is left at "auto"; an explicit
  `--mamba-ssm-cache-dtype` overrides it with a warning. Source: [S25] `config.json`, `text_config`;
  [S59] vLLM `vllm/model_executor/models/config.py` at b1388b1,
  `Qwen3_5ForConditionalGenerationConfig.verify_and_update_config`.

  Implication: our teacher keeps float32 recurrent state: 48 layers x 48 value heads x 128 x 128 x 4
  bytes, about 151 MB per sequence, read and written every decode step, and the same amount of VRAM
  per concurrent sequence.

  Experiment: KE10 (bf16 state). See below.

- **[measured]** vLLM PR #60400 describes the Gated DeltaNet packed-decode kernel as "bound by
  reading/writing the recurrent state (per sequence and layer: 1 MB in bf16, 2 MB in fp32 at TP=1)"
  for Qwen3.5 shapes with 32 value heads; its new launch configs cut that kernel's time 7.6% to 10%
  on the R9700. Source: [S30] vllm-project/vllm PR #60400, "Why per batch size and per state dtype"
  and the claims table.

  Implication: with our 48 value heads the state is 1.5 MB (bf16) or 3 MB (fp32) per sequence and
  layer; at concurrency 16 in float32 that is about 4.8 GB of state traffic per decode step, a
  third of the weight bytes, so state precision and the GDN kernel config both matter for W2.

  Experiment: KE10, `--mamba-ssm-cache-dtype bfloat16`. Expected state traffic 4.8 to 2.4 GB per
  step at concurrency 16, -8% to -12% J/token at W2 and about -1% at W1, plus room for more
  sequences; minimum 5 spawns per arm plus the R21 gate on long outputs (recurrent error
  accumulates with length); confidence low to medium (0.4).

- **[measured]** In the Capicua port, `--mamba-ssm-cache-dtype bfloat16` raised request capacity on a
  hybrid GDN model with speculative decoding: aggregate throughput at 32 users went from 228 to 319
  tok/s on a page-poor configuration. Source: [S03] `RDNA4-PORT.md`, section "Release rc9", serving
  findings.

  Implication: on this card the bf16 state is already in production use for the same model family,
  which lowers the accuracy risk of KE10 without removing the need to check it on our prompts.

  Experiment: KE10, as above.

- **[measured]** On an H200 with roughly 4B-parameter models, Gated DeltaNet decode is "two-thirds
  elementwise operations", and lowering the SM clock from 1980 to 780 MHz saved 49 W (30%) "at zero
  throughput loss". Source: [S53] Ma, Afzal, Eitzinger, Wellein, "The Illusion of Power Capping in
  LLM Decode: A Phase-Aware Energy Characterisation Across Attention Architectures",
  arXiv:2605.11999, 2026, section 5.1.

  Implication: the GDN part of our decode step is memory- and latency-bound rather than
  compute-bound; on the pods, without clock control, its levers are state precision (KE10) and
  launch configuration (KE11), not frequency.

  Experiment: KE11 (GDN launch configs). Expected the GDN kernel 7.6% to 10% faster; minimum 5
  spawns per arm; confidence medium (0.6) for the kernel time.

## 6. Batching and joules per token

- **[measured]** On an H200, decode of roughly 4B-parameter models drew only 137 to 300 W on a 700 W
  GPU, so power caps never triggered; increasing batch size from 1 to 32 "reduces energy-per-token
  by over 20x by amortising the cost of loading weights"; static clock locking recovered up to 32%
  of decode energy at less than 1% throughput loss. Source: [S53] arXiv:2605.11999, abstract and
  sections 1 and 4.2.

  Implication: our teacher draws about 200 W of a 300 W board while decoding, the same memory-bound
  regime, so batching is the dominant J/token lever and clock locking would be the next one, but
  the pods give us no root to lock clocks ([L1]); that lever belongs to the Jetson (task R27).

  Experiment: KE1 (batching sweep). See the last finding of this section.

- **[measured]** With Hugging Face TGI serving Llama 8B in bfloat16 on H100, server-side batching cut
  mean energy per request from 1.2e-1 Wh to 9.6e-3 Wh (12.5x) against a naive baseline, and fixed
  500 ms request spacing cut it to 1.1e-3 Wh, about 100x below the baseline. Source: [S54]
  Delavande, Pierrard, Luccioni, arXiv:2601.22362, 2026, section 5.2.

  Implication: how requests reach the server (arrival shaping) changes J/token as much as the
  kernel does; the labeling job should submit work in batches sized to the server's sequence
  slots, not one request at a time.

  Experiment: KE1 also compares a paced client (fixed spacing) with a burst client at the same
  total work; expected the burst client to reach the higher concurrency and lower J/token; minimum
  3 spawns; confidence medium.

- **[measured]** Three studies from groups that measure inference energy at scale agree that the
  serving configuration, not the model, sets much of the energy bill: automated optimization of
  configuration saved "sometimes more than 40%" of energy "without changing what is being computed
  by the model" (ML.ENERGY Benchmark, NeurIPS Datasets and Benchmarks 2025); across 46 models and
  1,858 configurations on H100 and B200, differences in GPU utilization alone produced 3x to 5x
  energy differences (the ML.ENERGY group's 2026 follow-up); and optimizations sized to the workload cut
  total energy by up to 73% from unoptimized baselines (Fernandez et al., 2025). Source: [S62]
  Chung et al., "The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and
  Optimization", NeurIPS D&B 2025, arXiv:2505.06371, abstract; [S64] Chung, Wu, Ma, Chowdhury,
  "Where Do the Joules Go? Diagnosing Inference Energy Consumption", arXiv:2601.22076, 2026,
  abstract; [S63] Fernandez, Na, Tiwari, Bisk, Luccioni, Strubell, "Energy Considerations of Large
  Language Model Inference and Efficiency Optimizations", arXiv:2504.17674, 2025, abstract.

  Implication: utilization is the common thread: a kernel that streams weights at 19% of peak and
  a server that decodes one sequence at a time are both low-utilization states, so KE1 and KE2
  attack the two largest factors these studies name.

  Experiment: KE1 and KE2; the R20 matrix should log GPU utilization (busy percent) next to watts
  so the utilization story can be shown on the slide; minimum 3 spawns; confidence medium.

- **[opinion]** With Conch's 128-row tile, rows 2 to 16 of a decode step ride along in the same
  GEMM tiles at almost no cost, so the step grows only by the extra GDN state (about 0.30 GB per
  sequence), KV reads and sampling: we estimate a step at concurrency 16 takes 1.15x to 1.4x a
  single-sequence step, giving 85 to 110 tok/s in aggregate; at 210 to 260 W that is about 2 to 3
  J/token against 26 today. Memory allows it: 16 sequences need about 2.4 GB of float32 GDN state
  plus their KV, inside the roughly 10x concurrency at 16k tokens that [L1] measured. Source:
  arithmetic on [S10], [S25], [L1]; batching behavior of good kernels in [S44].

  Implication: for the labeling and tool-trace jobs (W2), batching beats every kernel change on
  J/token, and the slide's J/token claim should say which concurrency it was measured at.

  Experiment: KE1, concurrency 1, 4, 8, 16 (and 32 if the KV cache allows) with the R11 harness
  (tasks R26 and T0 E3). Expected aggregate 60 to 110 tok/s and 2 to 4 J/token at concurrency 16,
  per-request TPOT 1.1x to 1.5x of single-stream; minimum 3 spawns with 10 requests per level;
  confidence 0.75.
