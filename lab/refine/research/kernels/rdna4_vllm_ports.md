# Community RDNA4 vLLM ports and field reports from R9700 owners

Research note for the kernels track (Task Board O8), searched on 2026-10-10. Sources are indexed in
`SOURCES.md`; labels, the Implication and Experiment lines, and the `KE` experiment ids follow the
conventions stated at the top of `w4a16_decode_gfx1201.md`, whose section 7 ranks the experiments.

What this note covers: the two public vLLM ports for the R9700 named on the task page and what
they changed, the earlier FP8 WMMA patch from the vLLM forum, and reports from people running vLLM
on gfx1201 that change how we have to measure. None of the ports targets our exact case
(single card, W4A16 int4, no speculative decoding), so the value is in their kernels and their
lessons, not their headline numbers.

## 1. The Capicua25x port (ROCm/vllm RFC 1229)

### 1.1 What it is and what it reports

- **[measured]** RFC #1229 on ROCm/vllm (opened 2026-08-20 by the port's author) offers AMD a vLLM
  0.26.1 port run in production since June on two Radeon AI PRO R9700 (TP2) serving Qwen3.8-27B
  ("dense hybrid GDN/attention") with MTP-3 speculative decoding, a 262,144-token window and 32
  sequence slots. Short prompts: config B (vendor FP8 weights, fp8 KV, bf16 SSM state) 66 tok/s for
  one user and 1,014 tok/s aggregate at 32 users; config C (Quark MXFP4, bf16 KV) 62 and 713. No AMD
  maintainer had replied when read. The fork has since moved to a vLLM 0.28.0 branch (current
  release rc12, 2026-08-28), carried as rebased branches with a "native MXFP4 linear kernel +
  spec-decode verify fix". Source: [S01] ROCm/vllm issue #1229, body and benchmark tables; [S02]
  Capicua25x/vllm-rocm-rdna4, README.

  Implication: these numbers are two cards plus speculative decoding, so they do not compare with
  our single-card 7.7 tok/s; what transfers is the list of kernel changes below and the fact that
  this exact model runs at production quality on gfx1201.

  Experiment: none from the headline; the kernels below map to KE3, KE9, KE13 and KE14.

### 1.2 The kernels it changed

- **[practitioner]** The port adds `RdnaMxfp4LinearKernel` (A16 dense linear: a Triton kernel
  dequantizes 4-bit tiles to the activation dtype, then WMMA; "Weights never materialize in high
  precision") and `RdnaMxfp4Fp8LinearKernel` (MXFP4 weights times e4m3 activations on RDNA4 FP8
  WMMA: a fused Triton GEMM for decode at M <= 128, a weight-only kernel for mid M, and above 512
  rows an integer dequantize to bf16 followed by a hipBLASLt bf16 GEMM). Source: [S03]
  `RDNA4-PORT.md` on branch `rdna4-port-0.26.1`, section "What's in the port".

  Implication: the community converged on the same split vLLM upstream uses for int4 on RDNA
  (a dedicated decode kernel plus a different prefill path), which supports KE2's design; their
  MXFP4 kernel is the only route to native-speed MXFP4 on gfx1201 today.

  Experiment: KE13 (Quark MXFP4 through this fork). Expected similar single-stream gain to KE2 if
  it runs on our stack; minimum 3 spawns per arm; confidence low (0.15) because the fork expects
  torch 2.10 and the pod image ships 2.9.1.

- **[measured]** Triton 3.6 open-codes the f32 to e4m3 conversion on gfx1201 at about 33 VALU
  instructions per element even though the chip has `v_cvt_pk_fp8_f32`; the port's LLVM-IR extern
  library cut the fp8-Q attention kernel by 33% (inner loop 1,568 to 814 instructions per
  iteration) with bit-identical outputs. Source: [S01] issue #1229; [S03] `RDNA4-PORT.md`, section
  "Release rc9 (2026-08-19)".

  Implication: any FP8 path through Triton on our pod (FP8 KV, FP8 activations) pays a conversion
  tax the byte count does not show, so R25's FP8 measurements need the kernel time, not only the
  bytes.

  Experiment: KE9 (FP8 KV). Expected -10% to -20% bytes per step at concurrency 16 and 8k context,
  less in time because of the conversion tax; minimum 5 spawns per arm; confidence low (0.3).

- **[measured]** With an fp8 KV cache, the software e4m3 downcast of attention probabilities
  flushes values at or below 2^-10 of the row maximum; scaling by 2^8 before the downcast
  (`VLLM_RDNA_P_SCALE`, default 256) moved GSM8K strict on the fp8-KV config from 0.70 to 0.88.
  Source: [S03] `RDNA4-PORT.md`, section "Release rc9 (2026-08-19)".

  Implication: FP8 KV on gfx1201 can lose accuracy silently in the attention kernel, so the output
  match rate and a task-accuracy check are part of any FP8-KV arm, not an afterthought.

  Experiment: KE9 with the R21 gate and a 200-question task check per arm; expected match-rate drop
  measurable at concurrency 1 if the flush happens; minimum 2 spawns; confidence medium (0.5) that
  stock vLLM at our commit shows a measurable accuracy drop with fp8 KV.

- **[measured]** Tuned per-shape w8a8 block-FP8 GEMM configs for `AMD_Radeon_R9700`, which vLLM
  did not ship, measured +33% single-stream and about 1.9x aggregate at 32 users over the default
  heuristics; in-tree configs regenerated with vLLM's tuner moved the FP8 arm from 754 to 1,014
  tok/s at short-prompt c32 (+34%). Source: [S01] issue #1229; [S03] `RDNA4-PORT.md`, sections
  "Release rc9" and "Throughput".

  Implication: on gfx1201 the FP8 path is configuration-starved: without device configs it runs on
  generic tiles, so the FP8 W8A8 arm of R25 measures the untuned path unless configs are added.

  Experiment: KE14 records whether vLLM logs a missing-config warning per GEMM shape; expected
  warnings on stock vLLM; minimum 1 spawn; confidence high (0.8).

- **[measured]** The port relaxes upstream's 3D split-KV gate in unified attention so small-query
  speculative-verify shapes use it (`MAX_QLEN_3D=8`), reported as about 2x long-context decode, and
  runs MTP-3 by default. With MTP-3 on the MXFP4 arm, greedy decoding through `/v1/completions`
  read about 61 tok/s single-stream (mean acceptance 3.10) while the sampled chat path read about
  51 tok/s on the same weights. Source: [S01] issue #1229; [S03] `RDNA4-PORT.md`, section "What's in
  the port" and the rc6 effect paragraph.

  Implication: MTP gains depend on the decoding mode, so KE3 must be measured under our production
  sampling settings as well as greedy, and long-context verify may need the gate fix to pay off.

  Experiment: KE3 (MTP), measured in both modes. Expected 1.4x to 2.6x tok/s at concurrency 1 on
  our Conch build; minimum 5 spawns per arm; confidence medium-low (0.45).

- **[practitioner]** The port's MXFP4 recipe quantizes only the MLP and MoE-expert projections to 4
  bits (group 32, E8M0 scales, data-free); attention, norms, embeddings, `lm_head`, routers,
  `conv1d`, vision and MTP stay bf16. Source: [S03] `RDNA4-PORT.md`, section "Quantization recipe
  (Quark MXFP4)".

  Implication: their MXFP4 model reads more bf16 bytes per token than our int4 checkpoint (which
  also quantizes attention and GDN projections), so per-card bandwidth math from their numbers
  would understate what a good int4 kernel can do for us.

  Experiment: none; context for KE13.

- **[practitioner]** For TP2 over PCIe the port sets `NCCL_PROTO=Simple` (the LL protocol measured
  2.8x slower, 205 against 73 us per op) and says custom all-reduce on gfx12 over PCIe "initializes
  and returns garbage on this pair". Source: [S03] `RDNA4-PORT.md`, section "Release rc9".

  Implication: none for us today, because our two R9700s are on separate pods and cannot run
  tensor parallel; it explains why every community number above is TP2 and not reproducible on
  one card.

  Experiment: none.

## 2. The drwolfen radiance port

- **[anecdote]** The radiance image (vLLM 0.28, ROCm 7.14, PyTorch 2.12.1, Triton 3.7.1, AITER
  0.1.20, two R9700) ships 25 hand-written RDNA4 HIP kernels (`libr4d`), including a 16x16x16 FP8
  WMMA kernel described as "running at 325 TFLOP/s on gfx1201 (vs 160 TFLOP/s for f16 WMMA and 43
  TFLOP/s for stock Triton)" and a W4A8 FP8-WMMA GEMM enabled by `RADIANCE_MXFP4_W4A8`; the README
  publishes no method for those figures. Source: [S04] drwolfen/radiance-vllm-r9700, README,
  sections on libr4d and environment variables.

  Implication: if stock Triton reaches roughly a quarter of what hand-written WMMA code does on
  gfx1201, compute-bound shapes (prefill, large batches) have large headroom; single-stream decode
  is bandwidth-bound, so this matters for TTFT and for WL2, not for KE2's WL1 gain.

  Experiment: KE0 includes prefill; if prefill GEMMs exceed a third of a labeling job's time, a
  W8A8 or W4A8 FP8 prefill path becomes a candidate (precision note, "INT8 and INT4 matrix
  paths"); confidence low.

- **[measured]** On `Ornith-1.5-35B-A3B-FP8` (a 256-expert MoE) with FP8 KV over TP2, with both cards
  undervolted by -70 mV and capped at 235 W, total throughput went from 74.8 tok/s at one stream
  (TPOT 12.0 ms) to 550.1 tok/s at 16 streams (TPOT 23.8 ms). Source: [S04] radiance README,
  benchmark "Config A" table.

  Implication: on this card class, 16 streams gave 7.4x the throughput of one stream while
  per-stream latency doubled; under a fixed power cap that is roughly a 7x cut in J/token, which
  supports KE1.

  Experiment: KE1 (batching sweep). Expected 26 to 2-4 J/token at concurrency 16 on our build;
  minimum 3 spawns, 10 requests per level; confidence 0.75.

- **[anecdote]** Owners tune R9700 power directly: the radiance setup runs -70 mV with a 235 W
  cap per card, and a llama.cpp owner ran a 210 W cap, a -500 MHz clock offset and -88 mV. Source:
  [S04] radiance README, "Hardware Power & Voltage Tuning"; [S26] llama.cpp discussion #21043,
  comment by zedbytes, 2026-03-27.

  Implication: these levers need root and writable sysfs; our pods have neither ([L1]
  `lab/README.md`: "No root, no Docker, no SSH"), so on the pods our J/token levers are kernel
  efficiency, batching and decoding strategy only, and clock or power experiments belong on the
  Jetson (task R27).

  Experiment: none on the pods.

## 3. The FP8 WMMA patch from the vLLM forum

- **[measured]** In November 2025 an R9700 owner (vLLM 0.11.1rc6, ROCm 7.0.0) found that FP8
  weights were upcast to FP32 (`torch._scaled_mm` with `out_dtype=torch.float32`), routed gfx1201
  to AITER's Triton block-scale FP8 GEMM with 16 hand-made config files, and measured decode going
  from about 160 to about 200 tok/s on Qwen3-0.6B (+25%) and from about 52 to about 85 tok/s on
  Qwen3-30B-2507 (+63%); raising the memory clock from 1258 to 1375 MHz added about 5%. FP8 ops
  needed M >= 16, so single-token generation needed padding. Source: [S05] vLLM forum thread
  "Native FP8 WMMA Support for AMD RDNA4 (RX 9070 XT / R9700) in vLLM", original post, 2025-11-13.

  Implication: FP8 on gfx1201 needed patches in late 2025; our commit's `supports_fp8()` already
  includes gfx12 ([S58]), so R25 should first establish whether FP8 GEMMs run natively unpatched
  and which kernel class they use.

  Experiment: KE14's load test records the FP8 linear kernel class chosen at startup; expected a
  native FP8 path, not an fp32 upcast; minimum 1 spawn; confidence medium (0.6).

## 4. Field reports that change how we measure

- **[measured]** ROCm issue #6347: on a single R9700, the same vLLM model and command land in one of
  two decode bands per process spawn, with almost no spread inside a spawn: Qwen3-14B-FP8 at about
  26.5 or 33.6 tok/s (standard deviation about 0.02 within a spawn; 2 of 7 spawns fast with the
  AITER attention backend, 1 of 8 with `ROCM_ATTN`), and gemma-4-26B-A4B at 42 or 72 tok/s. A second
  owner (two R9700, TP2, Qwen3.8-27B-FP8) saw 30.1 or 13.8 tok/s, with the penalty confined to FULL
  CUDA-graph replay (about +37 ms per replay); vLLM forces GDN models to `FULL_AND_PIECEWISE`
  ("CUDAGraphMode.FULL is not supported with GDNAttentionBackend"). Slow spawns drew the same
  power (about 274 W) as fast ones, and some `FULL_AND_PIECEWISE` processes later dropped from about
  74 to about 45 tok/s after idling. Source: [S28] ROCm/ROCm issue #6347, body and comments by
  taisunyoung and yanghoeg, 2026-06-11 to 2026-08-28.

  Implication: repeats inside one server process understate the real spread by an order of
  magnitude, so the kernels rubric's "a win must exceed the spread" needs repeats across cold
  spawns; and because the slow band burns the same watts, J/token moves with the band.

  Experiment: KE7, at least 6 cold spawns of the unchanged server, then the same with
  `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'`. Expected one band (most likely) or two;
  minimum 6 spawns per mode; confidence low (0.25) that our W4A16 build shows two bands.

- **[practitioner]** In the same thread, throughput computed from vLLM's
  `vllm:inter_token_latency_seconds` sum and count over a window read 11.0 tok/s on a pod whose
  controlled single-stream measurement was 29.4 tok/s, because prefill steps were mixed into the
  window; keeping only windows with mean tokens per iteration below 2 fixed it. Source: [S28]
  comment by yanghoeg, 2026-08-27.

  Implication: the R11 harness should time decode per request from the client's token stream
  (first token to last token), not from the server's latency histogram.

  Experiment: V1 also cross-checks client-side decode tok/s against the server metric on the same
  requests; expected agreement within 2% only for pure-decode windows; minimum 1 spawn; confidence
  high (0.8).

- **[practitioner]** vLLM issue #48723: the AITER update in vLLM v0.25.0 generated a
  unified-attention kernel (`kernel_unified_attention_3d`) that exceeded gfx1201's 65,536-byte LDS
  limit (`triton.runtime.errors.OutOfResources`), on a setup where v0.24.0 started fine. Source:
  [S29] vllm-project/vllm issue #48723 (2026-07-15), body.

  Implication: LDS overflows arrive with upgrades as well as with our own changes, so the kernels
  rubric's per-kernel LDS check (test 5) has to run at server start after every vLLM bump,
  including KE2's.

  Experiment: KE2 records server start success and the attention backend chosen; expected a clean
  start with our `num_stages=1` patch carried forward or made unnecessary; minimum 1 spawn;
  confidence medium (0.6).

- **[practitioner]** vLLM issue #50436: the ROCm tuned-config lookup keys off a device name that can
  differ from the shipped file name (`AMD_Radeon_8060S` against `Radeon_8060S_Graphics`), so
  `get_moe_configs()` silently returns None and the default config runs. Source: [S35]
  vllm-project/vllm issue #50436 (2026-07-30) and its last comment.

  Implication: any tuned config file we add for the R9700 (FP8 block configs, MoE configs) has to
  be confirmed as loaded in the serve log, or the "tuned" arm silently equals the baseline arm.

  Experiment: any config-file row (KE14 with configs, KE5 if done by file) logs the config path
  that loaded; expected silent misses without that check; minimum 1 spawn; confidence medium.

## 5. Attention at long context

- **[measured]** The opt-in `ROCM_SEGMENTED_ATTN` backend (vLLM #59132, merged 2026-10-06; RDNA3,
  RDNA3.5 and RDNA4; FP8 KV gated to gfx12) was compared with `ROCM_ATTN` on two R9700 (TP2) serving
  `Qwen/Qwen3.8-27B-FP8` with FP8 KV in an 8-turn agentic workload with 2,048 prompt tokens per
  turn: per-request output went from 4.19 to 6.53 tok/s at one stream (+56%) and from 7.84 to 22.03
  tok/s at eight streams (+181%), with mean TTFT 18% higher at one stream in a cold run that
  logged Triton compilation. On gfx1151 with a W4A16 AutoRound build of the same model, one stream
  went from 5.49 to 10.79 tok/s. Source: [S32] vllm-project/vllm PR #59132, "gfx1201 serving
  comparison" and "gfx1151 serving comparison".

  Implication: at multi-turn contexts the attention backend, not the 4-bit GEMM, can dominate the
  step on this model; our prompt sets must record context length, and the KE0 profile must run
  at the context lengths of real jobs (captions are short, tool-call traces are long).

  Experiment: KE8 (attention at long context). Expected +20% to +80% tok/s at 4k to 16k context
  and about 0 at short captions; minimum 5 spawns per arm per context length; confidence medium
  (0.45).
