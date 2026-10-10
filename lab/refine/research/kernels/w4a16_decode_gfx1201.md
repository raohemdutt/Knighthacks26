# W4A16 decode on gfx1201: why it is slow and what can fix it (main note)

Research note for the kernels track, written for Task Board task O8 and searched on 2026-10-10.
`SOURCES.md` in this folder indexes every source; `[Sxx]` ids point into it, and `[L1]` to `[L3]`
are our own measurements and files. The other notes in this set are `rdna4_vllm_ports.md`,
`triton_gfx12.md`, `precision_kv_batching.md` and `correctness_and_tuners.md`. The ranked
experiment queue for the whole set is section 7 of this note.

Labels: **[measured]** is a number someone measured, with the setup stated; **[practitioner]** is
documented behavior or a documented recommendation from the people who build or run the thing
(vendor docs, a project's own code or pull request read at a pinned commit); **[anecdote]** is a
single report without enough setup to reproduce it; **[opinion]** is argument or arithmetic
without new field evidence, ours included. Under each finding, "Implication" says what it means
for the kernels metric set (tokens per second at each batch size, time to first token, joules per
token, VRAM, output match rate against the baseline) and "Experiment" names a candidate
experiment with its expected movement, minimum runs and confidence. Experiment ids `KE0` to `KE16`
and `V1` to `V2` are local to this note set; a kernels pass that queues one on T0 gives it the next
`E<n> (kernels)` number and carries the citation with it.

## Summary

The teacher decodes at 7.7 tok/s on one R9700 because every 4-bit linear layer runs through the
Conch Triton kernel, which launches one fixed tile for every shape and moves about a fifth of the
card's 640 GB/s. The same model at 4 bits decodes at 29 to 31 tok/s on the same card in llama.cpp,
and upstream vLLM has since added an RDNA W4A16 kernel (a HIP skinny GEMM for decode) that cut
time per output token 3.2x to 4.8x on a sibling RDNA part; our vLLM commit predates it.
Independent of the kernel, batching the labeling jobs is the largest joules-per-token lever,
because with this kernel a step costs about the same for one sequence or sixteen. Speculative
decoding with the checkpoint's own MTP layer is the third lever for single-stream answers.

## 1. Baseline and byte budget

### 1.1 What we measured

- **[measured]** On pod A (one R9700, ROCm 7.13, torch 2.9.1+rocm7.13.0, vLLM commit b1388b1 plus
  `lab/vllm-gfx1201.diff`), `RedHatAI/Qwen3.8-27B-INT4` with bf16 activations and a 16k context
  decoded at 7.8 tok/s (text, thinking off, 76 tokens), 7.7 tok/s (thinking on, 160 tokens) and
  6.6 tok/s (1280x720 caption, 28 tokens), with TTFT 0.55 s, 0.28 s and 1.09 s, about 200 W average
  and 300 W peak while generating; 200 W / 7.7 tok/s is about 26 J per generated token. Source:
  [L1] `lab/README.md`, section "Teacher (pod A)", merged in PR #37 (merge commit bb351e6).

  Implication: these are the "old" values for single-stream tok/s, TTFT and J/token, but they come
  from one server spawn with no stated repeats, so their spread is unknown; on this card vLLM decode
  can differ by 1.3x to 2.2x between spawns of the same command (`rdna4_vllm_ports.md`, "Field
  reports that change how we measure").

  Experiment: KE7, cold-restart the unchanged server at least 6 times and record decode per spawn.
  Expected: either one band near 7.7 tok/s (most likely) or a second band 1.3x to 2.2x faster;
  minimum 6 spawns; confidence that a faster band exists: low (0.25).

- **[measured]** Pod idle power is 38 to 40 W (amd-smi SOCKET_POWER and rocm-smi), with the gfx
  clock between 500 and 2350 MHz and the memory clock between 96 and 1258 MHz. Source: [L2] T0
  task page, Findings, Sat 5:55 AM.

  Implication: about 5 of the 26 J/token are the idle floor (39 W / 7.7 tok/s), so any speedup
  removes that share in proportion even if load watts rise; joules per token falls by less than
  the speedup only to the extent load power rises with it, which is why every speed result has to
  carry its watts.

  Experiment: KE2 (section 7), measured with the pod sampler; expected 9 to 12 J/token from
  20 to 30 tok/s at 230 to 280 W; minimum 5 spawns per arm; confidence medium (0.6).

### 1.2 What one decoded token has to read

- **[practitioner]** The checkpoint is compressed-tensors `pack-quantized`: int4, symmetric,
  group size 128, static activation order, on every Linear except the vision tower, `lm_head`, the
  MTP layer and the Gated DeltaNet `in_proj_a` and `in_proj_b` (kept bf16). The text model has 64
  layers (48 `linear_attention` Gated DeltaNet layers and 16 `full_attention` layers, one in four),
  hidden size 5120, MLP width 17408, 24 query heads and 4 KV heads of dimension 256, vocabulary
  248,320 with untied embeddings, one MTP layer, and `mamba_ssm_dtype: float32` for the recurrent
  state. Source: [S25] `RedHatAI/Qwen3.8-27B-INT4` `config.json`, `model.safetensors.index.json`
  and the `model.safetensors` header (tensor shapes), Hugging Face, read 2026-10-10.

  Implication: the per-token byte budget below follows from these shapes, and every kernel the
  decode step needs is fixed: W4A16 GEMMs at K = 5120, 6144 and 17408, a bf16 `lm_head` GEMM, the
  GDN recurrent update, and attention on 16 layers only.

  Experiment: KE0, profile one decode step by kernel; expected movement none, it decides which
  kernel row of section 7 runs first; minimum 2 spawns, 20 decode steps each at concurrency 1 and
  16; confidence high (0.8) that it attributes at least 90% of step time.

- **[opinion]** From those shapes, one decode step at concurrency 1 reads 12.60 GB of layer weights
  (int4 values plus bf16 group scales), 2.54 GB of bf16 `lm_head`, about 0.30 GB of float32 GDN
  state (48 layers x 48 heads x 128 x 128 x 4 bytes, read and written) and 64 KB of bf16 KV cache
  per token of context: about 15.5 GB per step at a short context. At 640 GB/s that caps decode
  near 41 tok/s, and at the 84% of peak that llama.cpp reaches on this card about 35 tok/s. The
  measured 7.7 tok/s moves about 119 GB/s, 19% of peak. Source: arithmetic on [S25]; bandwidth from
  [S60] (AMD R9700 specifications: 640 GB/s, 300 W); the 84% from [S26].

  Implication: the 4-bit GEMM path, not the hardware, sets single-stream J/token today; a kernel at
  70% to 85% of bandwidth would make decode 3x to 4x faster at similar watts. `lm_head` alone is 17%
  of the bytes, which matters once the 4-bit layers are fast.

  Experiment: KE2 then KE12 (section 7). KE12 expected -6% to -8% J/token after KE2 from an FP8 or
  int8 `lm_head`; minimum 5 spawns per arm plus the correctness gate; confidence low (0.3).

## 2. Why the 4-bit path is slow at our commit

### 2.1 Kernel selection

- **[practitioner]** At b1388b1 vLLM lists the ROCm weight-only kernels as `[ConchLinearKernel,
  ExllamaLinearKernel]`, in priority order, while CUDA gets `CutlassW4A8`, `Machete`, `AllSpark`,
  `Marlin`, then `Conch` and `Exllama`; `choose_mp_linear_kernel` returns the first kernel whose
  `can_implement` passes, and `VLLM_DISABLED_KERNELS` removes kernels by class name. Source: [S06]
  vLLM `vllm/model_executor/kernels/linear/__init__.py` at b1388b1, `_POSSIBLE_KERNELS` and
  `choose_mp_linear_kernel`.

  Implication: the kernel is chosen by list order, not by measurement, so on ROCm the first kernel
  that accepts the layer wins whatever its speed.

  Experiment: KE4 (disable Conch, see 3.3). Expected 1.5x to 3x decode if ExLlama's fp16 kernel runs
  well on wave32; minimum 5 spawns per arm plus correctness and a task-accuracy check; confidence
  low (0.3).

- **[practitioner]** `ExllamaLinearKernel.can_implement` returns false unless activations are
  float16 ("Exllama only supports float16 activations"); it also needs a positive group size that
  divides K and supports only `uint4b8` and `uint8b128`. Our server passes `--dtype bfloat16`, so
  ExLlama is skipped and Conch serves every layer. Source: [S08] vLLM
  `mixed_precision/exllama.py` at b1388b1, `can_implement`; [L3] `lab/pod_a_teacher.sh`, `serve()`.

  Implication: switching kernels on this commit needs `--dtype float16`, which changes numerics for
  a bf16-trained model (see 3.3) and must pass the correctness gate.

  Experiment: KE4, as above.

### 2.2 The Conch kernel

- **[practitioner]** vLLM at b1388b1 pins `conch-triton-kernels==1.2.1`, and Conch 1.2.1 launches
  every mixed-precision GEMM with one tile from a hard-coded function: `BLOCK_M 128, BLOCK_N 64,
  BLOCK_K 64, GROUP_M 16, num_warps 8, num_stages 2`, with no autotuning, no split-K, no
  per-device table, and cache-eviction hints only on NVIDIA. The grid is
  `cdiv(M, 128) * cdiv(N, 64)` programs and each program loops over K in steps of 64. Source:
  [S11] vLLM `requirements/rocm.txt` at b1388b1, line 19; [S10] Conch v1.2.1
  `conch/kernels/quantization/gemm.py`, `_get_tuning_parameters` and `mixed_precision_gemm_launcher`.

  Implication: the tile is not tunable from vLLM's tree; a config sweep (task R24) has to patch the
  `conch-triton-kernels` package itself or monkeypatch `_get_tuning_parameters` at server start, so
  the patch cannot live only in `lab/vllm-gfx1201.diff`.

  Experiment: KE5 (Conch tile retune). Expected 1.2x to 1.8x decode, -15% to -40% J/token; minimum
  5 spawns per arm, one lever per run; confidence low to medium (0.4).

- **[opinion]** At one decoded token the Conch grid is N/64 workgroups: 80 for the 5120-wide outputs
  (`down_proj`, `out_proj`, `o_proj`) and 272 for the 17408-wide `gate_proj` and `up_proj`, on a
  64-CU card; each `down_proj` workgroup walks K = 17408 in 272 sequential steps of 2 KB of int4
  weights, and 127 of the 128 activation rows in every tile are masked padding. That leaves too
  few memory requests in flight to stream weights at full bandwidth, which fits the 19% of peak
  in 1.2. Source: arithmetic on [S10] and [S25].

  Implication: the cure is more parallelism over K (split-K, or a skinny GEMV that splits K across
  a wave), not a different N tile; a pure tile sweep can at most double the 80-workgroup grid with
  BLOCK_N 32.

  Experiment: KE6 (split-K on the Triton path) if KE2 is blocked. Expected 1.5x to 2.5x decode,
  -35% to -60% J/token; minimum 5 spawns per arm plus per-shape tests against an fp32 reference;
  confidence low (0.3).

- **[measured]** Conch's own benchmark table, on an NVIDIA H100: the Int4 x FP16 mixed-precision
  GEMM takes 1.656 ms in Conch's Triton kernel against 0.500 ms for the CUDA baseline, a ratio of
  0.30; the README lists A10, H100 and MI300X (ROCm 6.2.4) as supported hardware and no RDNA part.
  Source: [S09] stackav-oss/conch README, performance table (median of 10,000 iterations).

  Implication: Conch is a portable fallback that is already 3.3x slower than the tuned kernel on the
  hardware it was written for, so its 19% of peak on gfx1201 is expected rather than a gfx1201
  defect.

  Experiment: KE2 (replace it). See section 7.

- **[measured]** The AMD Quark INT4 W4A16 checkpoint of this same model, served by current vLLM on
  an MI350P (gfx950), runs the generic `triton_w4a16_gemm_kernel` at about 261 us per GEMM, a TPOT
  of 82 ms with memory utilization near 5%, against a TPOT of 9.5 ms for the MXFP4 checkpoint on the
  same card. Source: [S24] vLLM issue #60581 (2026-10-08), body.

  Implication: generic Triton W4A16 kernels are slow for this model even with 8 TB/s of HBM, so the
  fix is a shape-aware decode kernel, not more bandwidth; moving our pod to current vLLM without the
  RDNA kernel in the selection list would not help.

  Experiment: KE2, which selects the RDNA kernel and records the chosen class in the serve log.

### 2.3 Why the CUDA kernels do not apply

- **[practitioner]** Machete targets NVIDIA Hopper and later: it is built on CUTLASS 3.5.1 and uses
  `wgmma` instructions, the Tensor Memory Accelerator and warp specialization, and pre-shuffles
  weights offline so one 128-bit load feeds the tensor-core register layout. Source: [S45] Lucas
  Wilkinson, "Introducing Machete, a mixed-input GEMM kernel optimized for NVIDIA Hopper GPUs",
  Red Hat Developer, 2024-10-14.

  Implication: nothing in Machete ports to gfx1201, but the offline pre-shuffle idea does: the RDNA
  hybrid kernel stores weights once in an ExLlama-style shuffled layout for the same reason (3.1).

  Experiment: KE2.

- **[measured]** Marlin supports batch sizes up to 16 to 32 "with close to maximum" 4x speedup over
  FP16 for 4-bit weights, decreasing gradually up to 64 to 128, and gave up to 2.8x end-to-end in
  vLLM; its techniques are asynchronous memory access, task scheduling and pipelining. Source:
  [S44] Frantar et al., "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large
  Language Models", PPoPP 2025, doi:10.1145/3710848.3710871 (arXiv:2408.11743), abstract.

  Implication: a good 4-bit kernel keeps the cost of a decode step nearly flat up to 16 to 32
  sequences, which is what makes batching nearly free in J/token; our metric set must record
  tok/s at batch 1, 4, 8 and 16 for every kernel arm, not only batch 1.

  Experiment: KE1 (batching sweep). See section 7.

## 3. What already exists upstream

### 3.1 The RDNA W4A16 kernels in current vLLM

- **[practitioner]** Current vLLM main (c41b2639e2, 2026-10-10) orders the ROCm weight-only kernels
  `RDNA3W4A16`, `RDNAHybridW4A16`, `TritonW4A16`, `Conch`, `Exllama`. Our b1388b1 sits on a release
  branch that diverged before `TritonW4A16LinearKernel` landed (#37352, merged 2026-04-10), so the
  pod has none of the first three. Source: [S06] the same file on main; [S12] vLLM PR #37352;
  GitHub compare of #37352's merge commit d468322dc1 with b1388b1 (diverged, 32 ahead and 336
  behind).

  Implication: the fastest known W4A16 path for gfx12 is one rebuild away; the AMD-track story can
  cite upstream AMD-contributed kernels and our measurement of them on the R9700.

  Experiment: KE2.

- **[practitioner]** `RDNAHybridW4A16LinearKernel` (#40977, merged 2026-07-14) targets gfx11 and
  gfx12: it routes to the HIP skinny GEMM `wvSplitK_int4_g` when M <= 5 and K x M <= 1.2 x 32,768
  (the 64 KB per-workgroup LDS budget in fp16 elements, after #52619), and to a Triton fused-dequant
  GEMM otherwise, with tile heuristics tuned on gfx1201 (code comment: "Radeon AI PRO R9700") and
  gfx1151. It stores weights once in an ExLlama-shuffled `[N, K//2]` int8 buffer, accepts
  compressed-tensors (`output_dim` 0) and AWQ layouts, symmetric `uint4b8` and asymmetric `uint4`,
  group sizes 32, 64 and 128, and fp16 or bf16 activations. The int4 skinny kernel compiles for
  `__GFX11__` and `__GFX12__` and dequantizes with marlin-style magic numbers into `v_dot2` fp16 or
  bf16 dot products. Source: [S13] vLLM PR #40977, description; [S14] vLLM main
  `rdna_hybrid_w4a16.py`, module docstring and `can_implement`; [S15] vLLM main
  `csrc/rocm/skinny_gemms_int4.cu`.

  Implication: our checkpoint meets every condition (compressed-tensors, symmetric, group 128,
  bf16). At one token every one of our shapes takes the HIP path (K x M <= 39,321 for K = 5120,
  6144 and 17408); at three to five sequences `down_proj` (K = 17408) falls back to Triton, so
  tok/s at batch 4 is worth measuring separately.

  Experiment: KE2, either a vLLM commit after 2026-10-07 (to include #52619) built against the pod's
  torch, or a backport onto b1388b1 of `skinny_gemms_int4.cu` with its bindings, the Python kernel,
  its registry entry and the `on_gfx12x` and `on_gfx1151` helpers (b1388b1 has only `on_gfx1x`).
  Expected single-stream decode 7.7 to 20-30 tok/s, TTFT -10% to -40%, J/token 26 to 9-12; minimum
  5 spawns per arm, interleaved, plus the R21 correctness gate; confidence medium (0.6).

- **[measured]** With `--max-num-seqs 1` on gfx1151 (Strix Halo), the hybrid kernel cut TPOT 3.2x to
  4.8x against the previous path on five W4A16 models (for example `RedHatAI/Qwen3-8B-quantized.w4a16`
  at 128 in and 128 out: 116.99 ms to 24.55 ms) and TTFT 1.1x to 2.1x, while GSM8K on
  `RedHatAI/Qwen3-4B-quantized.w4a16` went from 0.835 to 0.855 on 200 samples. Source: [S13] vLLM
  PR #40977, benchmark tables.

  Implication: this is the closest measured evidence for KE2; it is on a sibling RDNA part with
  LPDDR5X, not on gfx1201 with GDDR6, so the size of the gain on our card is unmeasured.

  Experiment: KE2, as above.

- **[measured]** Follow-ups on the same kernel: the medium-skinny dispatch (#52619, merged
  2026-10-07) raised aggregate decode at batch 3 on Qwen3-8B GPTQ from 41.70 to 68.16 tok/s
  (+63%, gfx1151); keeping asymmetric zero points packed (#54965) cut TPOT 26.4% on a group-32
  31B AWQ model and 1.5% on Qwen3-8B; stride padding (#56301, gated to gfx1151) cut TPOT 2% to 3%.
  Source: [S16] vLLM PR #52619; [S17] vLLM PR #54965; [S18] vLLM PR #56301, test results.

  Implication: take a commit at or after 2026-10-07 for KE2, and expect asymmetric AWQ checkpoints
  (task R23) to depend more on zero-point layout than symmetric ones.

  Experiment: KE2, as above.

### 3.2 Neighbors of the kernel that are already in our commit

- **[measured]** The dense `wvSplitK` skinny GEMM was enabled for gfx11 and gfx12 decode in #34709
  (merged 2026-03-20) with "~15% decode token/s improvement on AMD Radeon AI PRO R9700" on
  Qwen3-Coder-30B-A3B AWQ; it adds a wave32 DPP reduction and `v_dot2_f32_f16`, plus an FP8
  `wvSplitKQ` path using `__builtin_amdgcn_dot4_f32_fp8_fp8`. b1388b1 contains it (GitHub compare:
  294 ahead, 0 behind). Source: [S20] vLLM PR #34709.

  Implication: our bf16 `lm_head` already runs a skinny HIP kernel at one to four tokens, so the
  slow part is the 4-bit layers only, and the same skinny-kernel family is proven on this card.

  Experiment: KE0 will show `lm_head` time next to the Conch GEMMs; expected `lm_head` near 5% of
  the step today; minimum 2 spawns; confidence medium (0.6).

### 3.3 ExLlama and the RDNA3 kernel

- **[measured]** On two RX 7900 XTX (gfx1100, TP2) serving Qwen3.6-27B GPTQ W4A16 group 32 at
  `max-num-seqs 8`, aggregate throughput was 82.4 tok/s with the Triton W4A16 kernel in bf16, 255.0
  with ExLlama (fp16 only) and 205.3 (bf16) or 270.2 (fp16) with the native RDNA3 kernel. The PR
  explains that bf16-trained Qwen models cast to fp16 can overflow and produce inf or NaN and
  garbled tool calls. The RDNA3 kernel is gated to gfx11; "CDNA / MI3xx / RDNA4 fall through".
  Source: [S19] vLLM PR #41394, motivation and summary table.

  Implication: on RDNA, ExLlama's fp16 kernel was about 3x the Triton W4A16 kernel, which makes
  KE4 a cheap probe, but fp16 activations are a correctness risk for this model family, so KE4 is
  judged on the correctness gate and task accuracy as much as on speed.

  Experiment: KE4: `VLLM_DISABLED_KERNELS=ConchLinearKernel` with `--dtype float16` on the unchanged
  build. Expected decode 1.5x to 3x, J/token -30% to -60%; minimum 5 spawns per arm plus the R21
  gate and a 200-question GSM8K-style check; confidence low (0.3).

### 3.4 Quark checkpoints and vendor libraries

- **[practitioner]** Native Quark W4A16 INT4 and UINT4 loading (#48606, merged 2026-09-18) builds
  an `MPLinearLayerConfig` and calls `choose_mp_linear_kernel`, the same selection the
  compressed-tensors path uses. Source: [S21] vLLM PR #48606 and main
  `quantization/quark/schemes/quark_w4a16_int4.py`.

  Implication: AMD's `amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16` runs the same kernel as the RedHat INT4
  checkpoint on any given commit (Conch at b1388b1 once the loader is backported), so task R23 and
  T0 E5 should expect tok/s within noise and look for differences in accuracy and in zero-point
  bytes, not in kernel path.

  Experiment: R23 as written, with this expectation recorded: tok/s difference within the spawn
  band, accuracy difference measured on the R21 prompts; minimum 5 spawns per arm; confidence
  medium (0.6) that tok/s is within the band.

- **[measured]** The Quark AWQ model card reports INT4 weight-only, group 128, GSM8K 5-shot in
  thinking mode at 91.21% and 90.67% (flexible and strict) against 93.33% and 93.33% for bf16
  (97.7% recovery), 100.9% recovery in non-thinking mode, and Wikitext perplexity 8.825 against
  8.436; it names vLLM PRs #48606 and #46110 as required. Source: [S22] Hugging Face model card
  `amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16`.

  Implication: a 4-bit teacher loses about 2 points of GSM8K in thinking mode, which is the noise
  scale any kernel-level accuracy check has to beat.

  Experiment: V1 (section 7): measure the match-rate and accuracy noise floor of the unchanged
  server before judging any kernel arm.

- **[practitioner]** hipBLASLt 1.4.1 supports the OCP `hipblaslt_f8` and `hipblaslt_bf8` types only
  on gfx950 and gfx12 (the fnuz types only on gfx942), supports int8, and lists no int4 type ("Data
  types not listed in the table below are not supported"). Source: [S39] hipBLASLt documentation,
  "Data type support", sections "hipDataType enumeration" and "Supported data types overview".

  Implication: there is no vendor-library W4A16 path on gfx1201; 4-bit decode has to come from a
  custom HIP or Triton kernel, and hipBLASLt helps only for FP8 W8A8 or for prefill after a
  dequantize to bf16 (the Capicua port does exactly that above 512 rows, `rdna4_vllm_ports.md`).

  Experiment: KE14 (FP8 W8A8 checkpoint, the vendor-library path). See section 7.

### 3.5 Speculative decoding with the checkpoint's own MTP layer

- **[practitioner]** vLLM at b1388b1 already has `vllm/model_executor/models/qwen3_5_mtp.py` and
  documents MTP speculative decoding as `--speculative_config '{"method":"mtp",
  "num_speculative_tokens":1}'`, recommending a small depth; the RedHat checkpoint keeps its MTP
  layer (`mtp.layers.0.*`) unquantized in bf16. Source: [S61] vLLM tree at b1388b1 and
  `docs/features/speculative_decoding/mtp.md` at b1388b1; [S25] `model.safetensors.index.json`.

  Implication: KE3 needs no new weights and no new code, only a server flag; with Conch's 128-row
  tile the verify rows of a speculative step cost almost nothing extra in the 4-bit GEMMs, while
  each draft token reads the bf16 MTP layer (about 0.75 GB) and the bf16 `lm_head` (2.54 GB) once.

  Experiment: KE3, `num_speculative_tokens` 1, 2 and 3 at concurrency 1, measured under greedy and
  under our production sampling. Expected 1.4x to 2.6x tok/s on the current build and -30% to -60%
  J/token; minimum 5 spawns per arm with acceptance rate logged; confidence medium-low (0.45).

## 4. What the ceiling looks like on this card

- **[measured]** llama.cpp on one R9700 with Unsloth `Qwen3.8-27B` `UD-Q4_K_XL` (16.34 GiB):
  `llama-bench` tg128 of 30.50 tok/s on Vulkan, 31.26 with `rm_kq=1`, and 29.07 with a
  gfx1201-targeted HIP build (Lemonade b1322); the poster computes 30.5 tok/s as about 84% of
  640 GB/s, and reports 55.3 to 57.5 tok/s with MTP (3 draft tokens). A generic, not
  gfx1201-targeted, HIP build ran at about 5 tok/s ("wrong kernels"). Source: [S26] llama.cpp
  discussion #21043, comment by misterkerns, 2026-09-03.

  Implication: about 30 tok/s single-stream is a measured ceiling for a 4-bit 27B on this card, so
  the vLLM gap is about 4x; E2 on T0 (vLLM against llama.cpp) should expect that ratio until KE2
  lands, and the "wrong kernels" 5 tok/s shows how much a generic build costs.

  Experiment: KE2 and T0 E2. Expected after KE2: vLLM within 0.7x to 1.0x of llama.cpp's tok/s at
  batch 1; minimum 5 spawns per arm; confidence medium (0.5).

- **[measured]** In the same thread, other single-R9700 runs agree: `Qwen3.5-27B` Q4_K_M at 29.07
  (AMDVLK, stock) and 32.25 tok/s (RADV, stock), and `Qwen3.6-27B` UD-Q4_K_XL with q4_0 KV at
  28.2 tok/s baseline and 47.4 tok/s with MTP-3 (tg128, three repetitions). Source: [S26]
  discussion #21043, original post by JohnTDI-cpu (2026-03-26) and comments by zedbytes (2026-03-28
  and 2026-05-18).

  Implication: MTP adds 1.6x to 1.9x on top of a bandwidth-efficient 4-bit kernel on this card,
  which bounds what KE3 can add after KE2.

  Experiment: KE3 (MTP). See section 7.

- **[measured]** On an RX 9070 XT (the same gfx1201 silicon) at concurrency 1 with `Qwen3.5-9B`,
  vLLM with FP8 weights and an `fp8_e4m3` KV cache averaged 48 tok/s against 62 tok/s for
  llama-server on Vulkan with a Q6_K GGUF (10 measured runs after 2 warmups, temperature 0, 512
  tokens). Source: [S27] ivangotoy/llama-server-rdna4-vulkan, README.

  Implication: vLLM's FP8 path on gfx1201 reaches roughly three quarters of llama.cpp's speed for
  a bandwidth-bound decode, so the 4-bit path is the outlier, not vLLM on ROCm as a whole.

  Experiment: KE14 (FP8 W8A8 checkpoint). See section 7.

## 5. Kernel lessons that transfer to gfx1201

- **[measured]** A Triton W4A16 fused dequant-GEMM with SplitK work decomposition improved skinny
  GEMMs (m < n = k, as in Llama-style inference) by 65% on average on A100 and 124% on H100, with a
  peak of 295%. Source: [S48] Hoque, Wright, Yang, Srivatsa, Ganti, "Accelerating a Triton Fused
  Kernel for W4A16 Quantized Inference with SplitK work decomposition", arXiv:2402.00025, 2024,
  abstract.

  Implication: splitting K is the measured remedy for the few-workgroup problem in 2.2, and the
  HIP `wvSplitK` family in 3.1 is that idea for RDNA.

  Experiment: KE6 if KE2 is blocked; KE2 otherwise.

- **[measured]** GPTQ quantizes to 3 or 4 bits with negligible accuracy loss and reports end-to-end
  inference speedups over FP16 of about 3.25x on A100 and 4.5x on A6000. Source: [S47] Frantar,
  Ashkboos, Hoefler, Alistarh, "GPTQ: Accurate Post-Training Quantization for Generative
  Pre-trained Transformers", ICLR 2023, arXiv:2210.17323, abstract.

  Implication: 4-bit weights pay off only with a kernel that streams them near peak; our 7.7 tok/s
  is the case where they do not.

  Experiment: KE2.

- **[measured]** AWQ's TinyChat runtime gave more than 3x over the Hugging Face FP16 implementation
  on desktop and mobile GPUs for 4-bit weight-only models. Source: [S46] Lin et al., "AWQ:
  Activation-aware Weight Quantization for LLM Compression and Acceleration", MLSys 2024 (Best
  Paper), arXiv:2306.00978, abstract.

  Implication: same lesson from a second group; and AWQ checkpoints are usually asymmetric, which
  brings zero points into the byte budget (3.1, #54965).

  Experiment: R23 records the zero-point layout of the Quark checkpoint next to its tok/s.

- **[measured]** On H100 with Hugging Face TGI and transformers, bitsandbytes int4 used about as
  much decode energy as float32 and int8 used 2 to 3 times more, because the dequantization kernels
  are "small, memory-bound, and irregular"; the authors conclude that precision matters "only in
  compute-bound regimes". Source: [S54] Delavande, Pierrard, Luccioni, "Understanding Efficiency:
  Quantization, Batching, and Serving Strategies in LLM Energy Use", arXiv:2601.22362, 2026,
  sections 3.2 and 9.

  Implication: quantization by itself does not lower J/token; the dequant-and-multiply kernel
  decides it, so every quantization experiment on our card pairs tok/s with joules and names the
  kernel class that ran.

  Experiment: KE2, with the kernel class recorded per run.

## 6. What this means for the open kernel tasks

- **[opinion]** For R24 (the Conch config sweep, P0): because Conch's tile is a hard-coded function
  in the `conch-triton-kernels` package, the sweep needs a package patch or a startup monkeypatch,
  and because Conch has no split-K a tile sweep can at most double the grid for the 5120-wide
  outputs; expect KE5's modest gain, and treat KE2 as the larger lever if a rebuild is allowed.
  Source: [S10], [S11], [S14], [S48].

  Implication: R24's result should be compared with KE2 on the same prompts before the slide
  claims a kernel win.

  Experiment: KE5 and KE2 on the same R20 baseline matrix; expected KE2 at least 2x KE5's gain;
  minimum 5 spawns per arm; confidence medium (0.5).

- **[opinion]** For R12 (first HIP kernel experiment, whose first named candidate is the unified
  attention kernel's stage and block settings): at short caption prompts the 4-bit GEMMs dominate
  the step (2.2), while attention dominates long multi-turn contexts, where a new Triton backend
  measured +56% to +181% on gfx1201 (`rdna4_vllm_ports.md`, "Attention at long context"). Source:
  [S24], [S13], [S32].

  Implication: let the KE0 profile at the workload's real context length choose between the
  attention candidate and the GEMM.

  Experiment: KE0 before R12; KE8 if attention exceeds a third of the step at the target context.

## 7. Ranked experiment queue

Each row is a candidate for T0, scored by expected joules-per-token gain times confidence.
"Gain" is the expected fractional cut in J/token against the 26 J/token baseline on the workload
named in the row: WL1 is single-stream decode (interactive answers and captions at concurrency 1),
WL2 is batched labeling at concurrency 16 with 2k to 4k tokens of context (WL1 and WL2 name
workloads; they are unrelated to the W1 to W6 work items in `lab/refine/kernels/RUBRIC.md`). "Confidence" is our
probability that the experiment delivers at least half of that gain, judged from the labels of
its evidence (measured on gfx1201 above measured on a sibling RDNA part above practitioner above
opinion). Gains do not add: KE2, KE4, KE5, KE6 and KE13 are alternatives for the same GEMMs, and
KE1 and KE3 multiply with whichever of them wins.

Protocol for every row, from the field reports in `rdna4_vllm_ports.md` and the checks in
`correctness_and_tuners.md`: each arm is at least 5 cold server spawns, interleaved with the other
arm, each with 2 warmup and 10 measured requests on a fixed prompt set at temperature 0 and a
fixed `max_tokens` with `ignore_eos`; report the per-spawn means and the band across spawns; time
decode from the client stream, not from the server's inter-token-latency histogram; integrate pod
watts over the decode window for J/token; record the kernel class and attention backend from the
serve log; a win must clear the union of both arms' spawn bands and pass the R21 gate at
concurrency 1. Rows with very large expected effects (KE1) may use 3 spawns per arm.

| Rank | Id | Lever | Workload | Expected movement | Gain | Confidence | Score | Minimum runs | Board home |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | KE1 | Batch the labeling jobs: concurrency 1, 4, 8, 16 (32 if KV fits) | WL2 | aggregate 7.7 to 60-110 tok/s at c=16; 26 to 2-4 J/token; per-request TPOT up 1.1x to 1.5x | 0.85 | 0.75 | 0.64 | 3 spawns, 10 requests per level | R26, T0 E3 |
| 2 | KE2 | RDNAHybridW4A16 kernel (newer vLLM, or backport onto b1388b1) | WL1 | 7.7 to 20-30 tok/s; TTFT -10% to -40%; 26 to 9-12 J/token | 0.60 | 0.60 | 0.36 | 5 spawns per arm plus R21 | new task |
| 3 | KE3 | MTP speculative decoding, 1 to 3 draft tokens, from the checkpoint's own MTP layer | WL1 | 1.4x to 2.6x tok/s on Conch (verify rows ride free in its 128-row tile), 1.6x to 1.9x after KE2; -30% to -60% J/token | 0.45 | 0.45 | 0.20 | 5 spawns per arm, greedy and production sampling | new task |
| 4 | KE4 | `VLLM_DISABLED_KERNELS=ConchLinearKernel` with `--dtype float16` (ExLlama path) | WL1 | 1.5x to 3x tok/s; -30% to -60% J/token; accuracy risk from fp16 | 0.50 | 0.30 | 0.15 | 5 spawns per arm plus R21 and a GSM8K-style check | new task |
| 5 | KE6 | Split-K Triton W4A16 (only if KE2 is blocked) | WL1 | 1.5x to 2.5x tok/s; -35% to -60% J/token | 0.45 | 0.30 | 0.14 | 5 spawns per arm plus per-shape fp32-reference tests | R12 candidate |
| 6 | KE5 | Conch tile retune (patch `_get_tuning_parameters`: BLOCK_M 16 for decode, BLOCK_N 32-128, BLOCK_K 64-256, 2-8 warps, 1-2 stages, LDS checked) | WL1 | 1.2x to 1.8x tok/s; -15% to -40% J/token | 0.25 | 0.40 | 0.10 | 5 spawns per config shortlisted from a cache-flushed microbenchmark | R24 |
| 7 | KE14 | FP8 W8A8 checkpoint `Qwen/Qwen3.8-27B-FP8` (28.8 GB) | WL1 only | 14-17 tok/s at batch 1; -40% to -50% J/token at WL1; almost no room for KV, so WL2 gets worse | 0.40 | 0.20 | 0.08 | 3 spawns after a load test | R25 |
| 8 | KE13 | Quark MXFP4 checkpoint through the Capicua RDNA4 fork (upstream only emulates MXFP4 on gfx1201) | WL1 | similar to KE2 if the fork runs on our stack; it expects torch 2.10 | 0.55 | 0.15 | 0.08 | 3 spawns per arm | R23-adjacent |
| 9 | KE7 | Spawn-lottery check and CUDA-graph mode (default `FULL_AND_PIECEWISE` vs `PIECEWISE`) | WL1 | either no second band, or a band 1.3x to 2.2x faster at the same watts | 0.30 | 0.25 | 0.075 | 6 spawns per mode | R20 protocol |
| 10 | KE8 | Attention at long context: `ROCM_SEGMENTED_ATTN` (newer vLLM) or unified-attention stage and block tuning | WL1 and WL2 at 4k-16k | +20% to +80% tok/s at long context, about 0 at short captions | 0.15 | 0.45 | 0.068 | 5 spawns per arm per context length | R12 |
| 11 | KE10 | GDN state in bf16 (`--mamba-ssm-cache-dtype bfloat16`; the checkpoint asks for float32) | WL2 | state traffic 4.8 to 2.4 GB per step at c=16; -8% to -12% J/token at WL2, about -1% at WL1 | 0.08 | 0.40 | 0.032 | 5 spawns per arm plus R21 on long outputs | new task |
| 12 | KE9 | FP8 KV cache with the checkpoint's calibrated `k_scale` and `v_scale` | WL2 | -10% to -20% bytes per step at c=16 and 8k context, 2x KV capacity; about 0 at WL1 | 0.10 | 0.30 | 0.030 | 5 spawns per arm plus R21 and a task-accuracy check | R25 |
| 13 | KE12 | FP8 or int8 `lm_head` (2.54 GB of bf16 per token) | WL1 after KE2 | -6% to -8% J/token after KE2 | 0.07 | 0.30 | 0.021 | 5 spawns per arm plus top-1 logit agreement | new task |
| 14 | KE11 | GDN packed-decode launch configs for gfx1201 (#60400, Python only) | WL1 and WL2 | GDN kernel -7.6% to -10%; about -1% to -2% of the step | 0.015 | 0.60 | 0.009 | 5 spawns per arm | new task |
| 15 | KE15 | Energy-aware config selection: pick tiles by joules per call instead of time | WL1 | 0 to -5% J/token at fixed clocks | 0.03 | 0.25 | 0.008 | 1 s windows per config, 5 repeats | post-hackathon thread |
| 16 | KE16 | PyTorch TunableOp for the bf16 prefill GEMMs | WL1 TTFT | TTFT 0 to -10%, J/token about 0 | 0.01 | 0.30 | 0.003 | 5 spawns per arm | none |

Run order is not the rank order. KE0 (the kernel profile) and KE7 run first, inside the R20
baseline matrix, because every other row needs a trustworthy baseline and a known split of step
time. KE1 needs no code and runs next. Among the GEMM alternatives, KE4 is the cheapest probe (no
build), KE2 is the expected winner, and KE5 and KE6 matter only if a rebuild is not allowed.

Two validation experiments carry no J/token gain but set the thresholds the rows above are judged
against (details in `correctness_and_tuners.md`):

V1 measures the noise floor of the R21 match rate and of task accuracy: the unchanged server on two
spawns at concurrency 1 and once at concurrency 8. Expected 100% exact match at concurrency 1
across spawns and below 100% at concurrency 8; minimum 2 spawns; confidence medium.

V2 runs per-shape kernel tests against an fp32 reference for every GEMM shape we serve (K = 5120,
6144 and 17408 at M = 1, 2, 4, 8 and 16, group 128) and records relative L2 error per shape.
Expected error near 1e-3 for a correct kernel against about 0.5 for the failure mode in vLLM issue
#60608; minimum 1 run per kernel arm; confidence high.
