# Which runtime settings can change vLLM decode on the gfx1151 teacher, and which ones should Ax search

Every environment variable and engine setting that can change decode tokens per second or joules per
token for the teacher server (vLLM 0.19.2 at b1388b1 plus `lab/vllm-gfx1201.diff`, built for gfx1151,
serving `RedHatAI/Qwen3.8-27B-INT4` as model `teacher`) on the SJC Strix Halo pods (Radeon 8060S,
gfx1151, RDNA 3.5, 64 GB shared, ROCm 7.13, torch 2.9.1+rocm7.13.0, Triton 3.5.1, Python 3.12). For each
setting it gives the mechanism from its source, the value in use today, and a pinned baseline; it ends
with a shortlist of at most ten settings with predicted effects, which is the space the Ax run (R38)
searches, and the search-space file `data/searchspace-inference-gfx1151.json`. It does not cover kernel
rewrites (the main note, `w4a16_decode_gfx1201.md`) or training settings, and no number here was
measured for this note: nothing ran on a pod or a GPU.

## Summary

How to read the value columns. "Script" means `lab/pod_a_teacher.sh` sets it on the serve line or in
the environment. "Code default" means the script does not set it and the value is the default in the
source named in the finding below, at the pinned ref. "Image default, unread" means the pod image
(`ghcr.io/amdresearch/auplc-base:latest`) may set the variable in its environment and nobody has read
the serve process's environment yet, so the value in use today is not known from git; the code default
is what applies if the image leaves it unset. The pinned baseline is what R38 sets explicitly on every
trial, so no arm inherits an unread image value. Searched settings are in the Shortlist section.

| Setting | Layer | Value today | Pinned baseline | Searched |
| --- | --- | --- | --- | --- |
| `--dtype` with `VLLM_DISABLED_KERNELS` (the W4A16 kernel path) | vLLM | script: `bfloat16`, nothing disabled, so `ConchLinearKernel` | `bfloat16`, `VLLM_DISABLED_KERNELS` empty | yes (`linear_kernel`) |
| `--speculative-config` (MTP) | vLLM | code default: off | off | yes (`num_speculative_tokens`) |
| `GPU_MAX_HW_QUEUES` | HIP runtime | image default, unread (code default 4) | `4` | yes |
| `cudagraph_mode` / `--enforce-eager` | vLLM | code default: `FULL_AND_PIECEWISE` (hybrid model rule), not eager | `FULL_AND_PIECEWISE` | yes |
| `--mamba-ssm-cache-dtype` | vLLM | code default `auto`, resolved to `float32` from the checkpoint | `float32` | yes |
| `VLLM_ROCM_USE_SKINNY_GEMM` | vLLM | image default, unread (code default true) | `1` | yes |
| `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` | vLLM | image default, unread (code default 1) | `1` | yes |
| `--attention-backend` | vLLM | code default: `ROCM_ATTN` by priority | `ROCM_ATTN` | yes |
| `--enable-prefix-caching` | vLLM | code default: on | on | yes |
| `AMDGCN_USE_BUFFER_OPS` | Triton | image default, unread (code default off) | `0` | yes |
| `--max-num-seqs` | vLLM | code default: 256 | 256 | no |
| `--max-num-batched-tokens` | vLLM | code default: 2048 | 2048 | no |
| `--block-size` | vLLM | code default 16, raised by the hybrid rule (derived 784, unread) | unset (let the rule pick) | no |
| `--kv-cache-dtype` | vLLM | code default `auto` (bf16) | `auto` | no |
| `--gpu-memory-utilization` | vLLM | script: 0.90 | 0.90 | no |
| `--max-model-len` | vLLM | script: 16384 | 16384 | no |
| `--mm-encoder-attn-backend` | vLLM | script: `TRITON_ATTN` on gfx115x | `TRITON_ATTN` | no |
| `--compilation-config` capture sizes | vLLM | code default: up to min(2 x max_num_seqs, 512) | default | no |
| `VLLM_ROCM_USE_AITER` and its family | vLLM | image default, unread (code default off; AITER is MI300-only here) | `0` | no |
| `HSA_ENABLE_SDMA` | ROCr | image default, unread (code default: runtime decides) | unset | no |
| `HSA_ENABLE_INTERRUPT` | ROCr | image default, unread (code default on) | unset | no |
| `HSA_XNACK`, `HSA_NO_SCRATCH_RECLAIM`, `HSA_SCRATCH_SINGLE_LIMIT`, `HSA_MAX_QUEUES` | ROCr | image default, unread | unset | no |
| `HIP_FORCE_DEV_KERNARG` | HIP runtime | image default, unread (code default true) | unset | no |
| `AMD_SERIALIZE_KERNEL`, `HIP_LAUNCH_BLOCKING` | HIP runtime | image default, unread (code default 0) | `0` | no |
| `GPU_SINGLE_ALLOC_PERCENT`, `DEBUG_HIP_FORCE_GRAPH_QUEUES` | HIP runtime | image default, unread (code defaults 100 and 4) | unset | no |
| `TORCH_BLAS_PREFER_HIPBLASLT` / `preferred_blas_library` | torch | image default, unread (code: hipBLAS on gfx1151) | unset | no |
| `PYTORCH_TUNABLEOP_ENABLED` and family | torch | image default, unread (code default 0) | `0` | no |
| `PYTORCH_HIP_ALLOC_CONF` / `PYTORCH_ALLOC_CONF` | torch | image default, unread (code default: no options) | unset | no |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | torch | script: unset (R29 found it wrong for the vision encoder) | unset | no |
| `TRITON_CACHE_DIR`, `TRITON_HOME`, `TRITON_CACHE_AUTOTUNING`, `TRITON_ALWAYS_COMPILE` | Triton | image default, unread (code default `~/.triton/cache`, off, off) | per-arm cache dir (see KS10) | no |
| `TRITON_HIP_USE_BLOCK_PINGPONG`, `TRITON_HIP_USE_ASYNC_COPY`, prefetch knobs | Triton | image default, unread (code: off on gfx1151) | unset | no |
| `power_dpm_force_performance_level`, `pp_power_profile_mode`, `power1_cap` | amdgpu sysfs | readable, not settable without root; values unread | read and recorded, never set | no |

Baseline numbers every prediction is relative to: decode 2.88 tok/s at batch 1 with 120 W average
SOCKET_POWER while decoding and 8 W idle with no server on pod C, which is about 41.7 J per generated
token gross (120 W / 2.88 tok/s, derived) and about 38.9 J net of idle; source `lab/README.md`, section
"Teacher on SJC today (pod C, Strix Halo)".

## What runs today: the serve line and the build

- **[practitioner]** The teacher is started by `lab/pod_a_teacher.sh` `serve()` with `--model <checkpoint> --served-model-name teacher --port 8000 --host 127.0.0.1 --max-model-len 16384 --limit-mm-per-prompt '{"image":1}' --gpu-memory-utilization 0.90 --dtype bfloat16`, plus `--mm-encoder-attn-backend TRITON_ATTN` when the gfx target matches `gfx115*`; it exports no runtime environment variable for the server (the build step sets only build-time variables such as `PYTORCH_ROCM_ARCH`, `MAX_JOBS` and `SETUPTOOLS_SCM_PRETEND_VERSION=0.19.2+<gfx>`), and it builds vLLM at commit `b1388b1fb` with `lab/vllm-gfx1201.diff` applied. Source: `lab/pod_a_teacher.sh::serve` and `lab/pod_a_teacher.sh::build` at origin/main 722d845 (R29, 498e161, PR #194; R30, 0f51e01, PR #220).
  - Implication: every setting not on that line is a code default or an image default; the values in the Summary table's "script" rows are the only ones git can vouch for. The runtime environment of the serve process has never been read back, so every "image default, unread" row stays unknown until KS0 runs.
  - Candidate experiment: KS0, read the environment of the running vLLM process on pod C read-only (`/proc/<pid>/environ` with secrets dropped, as R11's harness already does for its own spawns), plus the serve log lines that name the attention backend, the linear kernel class, the resolved block size and the KV capacity. Expected movement: none in decode tok/s or joules per token; it fills the "value today" column. Minimum sample size: 1 read per pod. Confidence: high that the read is possible without root (the process belongs to the pod user).

- **[measured]** On pod C the same build served 2.88 tok/s decode at batch 1 (text, 31 in, 77 out, 16k context, temperature 0, thinking off) at 120 W average and 129 W peak SOCKET_POWER, 8 W idle with no server and 9.2 W with the teacher loaded, and the engine was ready about 5 min after launch with a warm torch.compile cache. Source: `lab/README.md`, section "Teacher on SJC today (pod C, Strix Halo)", and `lab/refine/metrics.jsonl` rows on node `pod-c`.
  - Implication: these are the baseline numbers for every prediction below: about 41.7 J per token gross. SOCKET_POWER is the whole APU package (CPU included), so a setting that keeps the host CPU busy moves joules per token even when decode speed does not.
  - Candidate experiment: KS-base, the pinned baseline of `data/searchspace-inference-gfx1151.json` run as its own arm in R38, interleaved with every other arm. Expected movement: none by construction; its spread across cold spawns sets the noise floor each other arm must beat. Minimum sample size: 6 cold spawns. Confidence: high.

## The W4A16 kernel path: dtype and VLLM_DISABLED_KERNELS

- **[practitioner]** At b1388b1 the ROCm weight-only kernel list is `[ConchLinearKernel, ExllamaLinearKernel]` in priority order, the first kernel whose `can_implement` passes wins, and `VLLM_DISABLED_KERNELS` (a comma-separated list of class names, default empty) removes kernels from the choice; `ExllamaLinearKernel.can_implement` refuses anything but float16 activations, so with `--dtype bfloat16` Conch serves every 4-bit layer. Source: [S06] vLLM `vllm/model_executor/kernels/linear/__init__.py` at b1388b1, `_POSSIBLE_KERNELS` and `choose_mp_linear_kernel`; [S08] `mixed_precision/exllama.py` at b1388b1, `can_implement`; [S81] vLLM `vllm/envs.py` at b1388b1, `VLLM_DISABLED_KERNELS`.
  - Implication: the kernel path is one categorical setting with two legal values on this build: `conch_bf16` (today) and `exllama_fp16` (`--dtype float16` with `VLLM_DISABLED_KERNELS=ConchLinearKernel`). Decode is weight-read bound (about 15.5 GB a step at a short context, from `w4a16_decode_gfx1201.md` section 1.2), so the kernel that streams int4 weights closer to the memory bandwidth decides tok/s almost alone.
  - Candidate experiment: KS1, `linear_kernel` = `exllama_fp16` against `conch_bf16`, each arm a cold spawn at batch 1 and batch 4, judged on the R21 correctness gate as well as speed, because bf16-trained Qwen models cast to fp16 can overflow [S19]. Expected movement: decode +50% to +200%, joules per token -30% to -65% (power may rise up to 20% as bandwidth use rises); the size waits on the R33 chip sheet (measured bandwidth and dequant rate on gfx1151). Minimum sample size: 5 cold spawns per arm. Confidence: low (0.3).

## Speculative decoding with the MTP layer

- **[practitioner]** vLLM at b1388b1 has `qwen3_5_mtp.py` and documents MTP speculative decoding as `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`, recommending a small depth; the RedHat checkpoint keeps its MTP layer in bf16. Source: [S61] vLLM `docs/features/speculative_decoding/mtp.md` at b1388b1; [S25] `RedHatAI/Qwen3.8-27B-INT4` `model.safetensors.index.json`.
  - Implication: `num_speculative_tokens` in {0, 1, 2, 3} is a single integer setting with no new weights. Each accepted draft token saves one full weight read of the 4-bit layers, which is the dominant cost on this chip; each draft costs one read of the bf16 MTP layer and the bf16 `lm_head`.
  - Candidate experiment: KS2, `num_speculative_tokens` 0, 1, 2, 3 at batch 1 with the acceptance rate logged from the serve metrics, greedy and under production sampling. Expected movement: decode +30% to +120% at depth 1 or 2 and less at 3, joules per token -20% to -50%; the size waits on the R32 note (whether the gfx1151 memory system rewards the extra bf16 reads) and on the acceptance rate. Minimum sample size: 5 cold spawns per arm. Confidence: low (0.4).

- **[practitioner]** In the Gated DeltaNet layer at b1388b1 the packed recurrent decode fast path runs only when `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` is on, the batch has no prefills, and `spec_sequence_masks` is None, that is, when no speculative tokens are in flight. Source: [S89] vLLM `vllm/model_executor/layers/mamba/gdn_linear_attn.py` at b1388b1, `_forward_core` and `_forward_core_decode_non_spec`.
  - Implication: KS2 and KS7 interact: with MTP on, the GDN layers leave the packed path, so a speculative arm pays the general GDN path on top of the draft cost. Ax must see both parameters together rather than tune them one at a time.
  - Experiment: KS2 and KS7, run as a joint factor in R38 (no separate run).

## Hardware queues: GPU_MAX_HW_QUEUES

- **[practitioner]** The HIP runtime's `GPU_MAX_HW_QUEUES` is a release flag with default 4, "The maximum number of HW queues allocated per device"; the same flags file defines `DEBUG_HIP_FORCE_GRAPH_QUEUES` (default 4, "the number of streams for the graph parallel execution"). Source: [S82] ROCm `rocm-systems` `projects/clr/rocclr/utils/flags.hpp` at 71a246ac.
  - Implication: the value in use today is 4 unless the image sets it (unread). It is an integer with useful values 1, 2 and 4.
  - Candidate experiment: KS3, `GPU_MAX_HW_QUEUES` 1, 2 and 4, each arm alternating cold spawns at batch 1, with per-spawn decode, watts and the KFD queue count recorded, so a bimodal arm is visible as bands rather than an average. Expected movement: if gfx1151 shows the per-spawn decode bands reported on gfx1201, mean decode +0% to +70% and joules per token down by the same factor at queue 1; if it shows one band, within 1%; whether the hardware-queue mechanism exists the same way on this APU waits on the R32 note. Minimum sample size: 8 cold spawns per arm. Confidence: low (0.25).

- **[measured]** On one R9700 (gfx1201), `GPU_MAX_HW_QUEUES=1` moved a dense Qwen3-14B-FP8 server from 2 of 12 fast spawns to 12 of 12 (slow band 25.74 to 26.52 tok/s, fast 33.65 to 33.68), and its cost was within run-to-run noise at concurrency 1 to 16. Source: [S65] ROCm/legacy-rocm-build issue #6347, comments of 2026-09-13, 2026-09-15 and 2026-09-20.
  - Implication: this is the only measured evidence for KS3 and it is on the other chip; it justifies searching the knob, not the size of the prediction. Our gfx1151 baseline of 2.88 tok/s is one spawn, so it is not yet known which band, if any, it sits in.
  - Experiment: KS3, as specified in the previous finding.

## CUDA graphs and eager mode

- **[practitioner]** At b1388b1 `cudagraph_mode` accepts NONE, PIECEWISE, FULL, FULL_DECODE_ONLY and FULL_AND_PIECEWISE, documented as the V1 default and "the most performant mode for most models"; capture sizes default to `[1, 2, 4] + range(8, 256, 8) + range(256, max + 1, 16)` with `max_cudagraph_capture_size = min(max_num_seqs * 2, 512)`; and for hybrid models `MambaModelConfig.verify_and_update_config` enables FULL_AND_PIECEWISE by default "required to get good performance for mamba layers in V1". `--enforce-eager` is the switch to no graphs. Source: [S78] vLLM `vllm/config/compilation.py` at b1388b1, `CompilationConfig.cudagraph_mode` and `max_cudagraph_capture_size`; [S59] `vllm/model_executor/models/config.py` at b1388b1, `MambaModelConfig`; [S76] `vllm/engine/arg_utils.py` at b1388b1, `--enforce-eager`.
  - Implication: today the teacher replays full graphs for decode batches (the 5 min start includes the capture). One decode step is about 350 ms at 2.88 tok/s, so launch overhead that graphs remove is a few percent of the step at most; graphs matter here mostly as the place the gfx1201 band penalty lived (next finding).
  - Candidate experiment: KS4, `cudagraph_mode` in {FULL_AND_PIECEWISE, PIECEWISE, NONE} at batch 1 and batch 4. Expected movement: NONE (eager) decode -1% to -6% and joules per token +1% to +6%; PIECEWISE within -3% to +3% when one band is seen, higher if the full-graph replay penalty seen on gfx1201 appears on gfx1151. Minimum sample size: 6 cold spawns per arm. Confidence: medium (0.5).

- **[measured]** On two R9700 (TP2) serving Qwen3.8-27B-FP8, the slow decode band (13.8 against 30.1 tok/s) was confined to FULL CUDA-graph replay, about +37 ms per replay. Source: [S28] ROCm/ROCm issue #6347, comment by yanghoeg.
  - Implication: on the other chip the graph mode and the hardware-queue limit act on the same penalty, which is why KS3 and KS4 are searched together and why each arm needs several cold spawns.
  - Experiment: KS4, as specified in the previous finding.

## Gated DeltaNet state precision

- **[practitioner]** vLLM at b1388b1 accepts `--mamba-ssm-cache-dtype` values `auto`, `float32` and `float16` only (`MambaDType`), and for Qwen3.5-family models copies the checkpoint's `mamba_ssm_dtype` (float32 for this checkpoint) into it when left at `auto`; an explicit value overrides it with a warning. Source: [S77] vLLM `vllm/config/cache.py` at b1388b1, `MambaDType` and `mamba_ssm_cache_dtype`; [S59] `Qwen3_5ForConditionalGenerationConfig.verify_and_update_config`; [S25] `config.json`.
  - Implication: the searchable values are `float32` (today) and `float16`; bf16 is not a legal value on this build, so the bf16 arm of KE10 in `precision_kv_batching.md` cannot run here as written. The state is about 0.30 GB of the roughly 15.5 GB read per decode step, so the speed ceiling of halving it is about 1%.
  - Candidate experiment: KS5, `mamba_ssm_cache_dtype` `float16` against `float32` at batch 1 and batch 8, judged on the R21 gate over a long generation (recurrent error accumulates with length). Expected movement: decode +0.5% to +2%, joules per token -0.5% to -2%, and about half the per-sequence state memory. Minimum sample size: 5 cold spawns per arm. Confidence: medium (0.6) on the size, low on correctness.

## Unquantized GEMMs: the skinny kernels, hipBLAS and hipBLASLt, TunableOp

- **[practitioner]** At b1388b1 every unquantized bf16 or fp16 linear on ROCm goes through `rocm_unquantized_gemm_impl`: with `VLLM_ROCM_USE_SKINNY_GEMM` true (the default) on gfx9 or gfx1x and `k % 8 == 0`, it calls the `wvSplitK` kernel for 1 to 4 tokens when the output has more than 8 rows, the `LLMM1` kernel for 1 token when `k <= 8192` and there is no bias, and `torch.nn.functional.linear` otherwise. Source: [S90] vLLM `vllm/model_executor/layers/utils.py` at b1388b1, `rocm_unquantized_gemm_impl`; [S81] `vllm/envs.py`, `VLLM_ROCM_USE_SKINNY_GEMM`.
  - Implication: at batch 1 to 4 the bf16 `lm_head` (2.54 GB, about 16% of a step's bytes) and the bf16 GDN `in_proj_a` and `in_proj_b` run vLLM's own skinny kernels, not a BLAS library; above batch 4 they fall to torch and from there to hipBLAS. The flag is a boolean worth one guard arm.
  - Candidate experiment: KS6, `VLLM_ROCM_USE_SKINNY_GEMM=0` against `1` at batch 1 and batch 4. Expected movement: decode -10% to +5% with the flag off and joules per token the inverse; a gain with it off would mean the skinny kernels are mistuned for gfx1151's CU count. Minimum sample size: 5 cold spawns per arm. Confidence: low (0.35).

- **[practitioner]** In torch 2.9.1 the default BLAS backend on ROCm is hipBLAS (`Cublas`) unless every visible GPU is one of gfx90a, gfx942, gfx1200, gfx1201 or gfx950, and a request for hipBLASLt (`TORCH_BLAS_PREFER_HIPBLASLT=1`, `TORCH_BLAS_PREFER_CUBLASLT=1`, or `torch.backends.cuda.preferred_blas_library`) is overridden back to hipBLAS with the warning "Attempting to use hipBLASLt on an unsupported architecture!" on any GPU outside gfx90a, gfx942, gfx1100, gfx1101, gfx1200, gfx1201, gfx908 and gfx950. gfx1151 is on neither list. Source: [S84] PyTorch v2.9.1 `aten/src/ATen/Context.cpp`, `Context::blasPreferredBackend`, and `aten/src/ATen/Context.h`, `blas_preferred_backend`.
  - Implication: on an upstream torch 2.9.1 the hipBLAS-against-hipBLASLt choice is not a live setting on gfx1151: it always resolves to hipBLAS. The pod's torch is AMD's `2.9.1+rocm7.13.0` build, which may carry a different arch list (unread), so the knob is excluded from the search and KS0 checks the warning in the serve log instead.
  - Experiment: none beyond KS0 (grep the serve log for the hipBLASLt override warning with `TORCH_BLAS_PREFER_HIPBLASLT=1` set once). Expected movement: none. Minimum sample size: 1 spawn. Confidence: high (0.8) that it resolves to hipBLAS.

- **[practitioner]** PyTorch TunableOp is off by default (`PYTORCH_TUNABLEOP_ENABLED`, default 0); when on it tunes torch GEMMs at first use (`PYTORCH_TUNABLEOP_TUNING` default 1) across rocBLAS and hipBLASLt solutions (`PYTORCH_TUNABLEOP_ROCBLAS_ENABLED` and `PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED`, default 1), bounded by `PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS` (default 30) and `PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS` (default 100), and writes `tunableop_results.csv` (`PYTORCH_TUNABLEOP_FILENAME`). Source: [S85] PyTorch v2.9.1 `aten/src/ATen/cuda/tunable/README.md`, environment variable table.
  - Implication: at batch 1 to 4 the only torch GEMMs left in a decode step are the ones the skinny path above does not take, and the 4-bit layers are Triton kernels TunableOp never sees, so TunableOp cannot move batch-1 decode by more than noise. It is excluded from the search and pinned off; it belongs to a prefill or batch-8-and-up study.
  - Experiment: none beyond KE16 in `correctness_and_tuners.md` (prefill). Expected movement: none on decode at batch 1. Minimum sample size: 0. Confidence: medium (0.6).

## Attention backend and block size

- **[practitioner]** At b1388b1 the ROCm attention priority for a non-MLA model is `ROCM_ATTN`, then `ROCM_AITER_FA` when AITER MHA is enabled, then `ROCM_AITER_UNIFIED_ATTN` when AITER is found and supported, then `TRITON_ATTN`; `--attention-backend` overrides the choice after validating it. `is_aiter_found_and_supported()` returns true only on MI3xx (gfx942 or gfx950), and `VLLM_ROCM_USE_AITER` defaults to off. Source: [S58] vLLM `vllm/platforms/rocm.py` at b1388b1, `_get_backend_priorities` and `RocmPlatform.get_attn_backend_cls`; [S80] `vllm/_aiter_ops.py` at b1388b1, `is_aiter_found_and_supported`; [S81] `vllm/envs.py`, `VLLM_ROCM_USE_AITER`.
  - Implication: on gfx1151 the legal attention backends are `ROCM_ATTN` (today, by priority) and `TRITON_ATTN`; the AITER family is out of reach on this chip and pinned off. Attention runs on 16 of the 64 layers, so at a short context it is a small share of a step.
  - Candidate experiment: KS8, `--attention-backend` `TRITON_ATTN` against `ROCM_ATTN`, at batch 1 with a 300-token and an 8k-token context. Expected movement: within -3% to +3% decode at the short context and up to -10% to +10% at 8k, joules per token the inverse. Minimum sample size: 5 cold spawns per arm. Confidence: medium (0.5) that the short-context difference is under 3%.

- **[practitioner]** `ROCM_ATTN` at b1388b1 accepts any block size that is a multiple of 16, but its native HIP paged-attention kernel handles only block sizes 16 and 32; other sizes ("qwen3_5 with 784 and 1056", in the code's own comment) are routed to its Triton path. Source: [S79] vLLM `vllm/v1/attention/backends/rocm_attn.py` at b1388b1, `get_supported_kernel_block_sizes` and `do_kv_cache_update`.
  - Implication: for this hybrid model `ROCM_ATTN` never runs its native HIP kernel, so KS8 compares two Triton implementations, not HIP against Triton; that is why the predicted difference is small.
  - Experiment: KS8, as specified in the previous finding.

- **[practitioner]** For hybrid attention plus recurrent models vLLM at b1388b1 raises the attention block size, "even if the user has explicitly set it", until one attention page is at least as large as one recurrent-state page: without mamba prefix caching (or in `align` mode) to `16 * ceil(mamba_page / (16 * attn_bytes_per_token))`, and in `all` mode to a multiple of the chunk size; the default block size is 16 before that rule. Source: [S59] vLLM `vllm/model_executor/models/config.py` at b1388b1, `HybridAttentionMambaModelConfig.verify_and_update_config`; [S77] `vllm/config/cache.py`, `DEFAULT_BLOCK_SIZE`.
  - Implication: block size is not a free setting here. From the checkpoint shapes (4 KV heads of 256 in bf16 is 4096 bytes per token per layer; a recurrent page of 48 x 128 x 128 x 4 bytes of state plus the conv state, about 3.2 MB) the rule gives 784 tokens in the `none` and `align` paths (derived, matching the code comment), and `--kv-cache-dtype fp8` would roughly double it. It is pinned unset and its resolved value is read from the serve log line "Setting attention block size to N tokens" in KS0.
  - Experiment: none beyond KS0. Expected movement: none. Minimum sample size: 1 spawn. Confidence: high (0.8) that the log names 784 or a chunk multiple near it.

- **[practitioner]** `--enable-prefix-caching` defaults to on at b1388b1, and for a hybrid model it switches the recurrent cache to `mamba_cache_mode` `all` (or `align` when the model does not support `all`), which caches recurrent state at block boundaries and sets the mamba block size to the attention block size; with prefix caching off the mode is `none` and the mamba block size is `max_model_len`. Source: [S77] vLLM `vllm/config/cache.py` at b1388b1, `enable_prefix_caching` and `mamba_cache_mode`; [S59] `MambaModelConfig.verify_and_update_config`.
  - Implication: prefix caching changes the recurrent-state bookkeeping of every decode step, not only prefill reuse; it is a boolean worth one arm. The R11 harness already puts a per-request nonce at the front whenever prefix caching may be on, so the measurement stays fair in both arms.
  - Candidate experiment: KS9, `--no-enable-prefix-caching` against the default at batch 1 and batch 4. Expected movement: decode +0% to +5% with it off and joules per token the inverse; TTFT worse for repeated prompts (not a metric of this search). Minimum sample size: 5 cold spawns per arm. Confidence: low (0.35).

## KV cache dtype, batch caps and memory share

- **[practitioner]** At b1388b1 `--kv-cache-dtype` accepts `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2` and others, and both `ROCM_ATTN` and `TRITON_ATTN` list the fp8 forms as supported; but `RocmPlatform.supports_fp8()` is true only for gfx94, gfx95 and gfx12, not for gfx11. Source: [S77] vLLM `vllm/config/cache.py` at b1388b1, `CacheDType`; [S79] `rocm_attn.py` and `triton_attn.py` at b1388b1, `supported_kv_cache_dtypes`; [S58] `vllm/platforms/rocm.py` at b1388b1, `supports_fp8`.
  - Implication: an fp8 KV cache on gfx1151 would run through software conversion on a chip vLLM itself does not mark as fp8-capable, and at batch 1 with a short context the KV read is under 1% of a step's bytes, so it cannot pay at the batch-1 operating point. It stays pinned `auto` and out of the search; it belongs to a long-context or high-concurrency study after a correctness check.
  - Experiment: none beyond KE9 in `precision_kv_batching.md`. Expected movement: none at batch 1. Minimum sample size: 0. Confidence: medium (0.6).

- **[practitioner]** When `--max-num-seqs` and `--max-num-batched-tokens` are not set, vLLM at b1388b1 picks them by device memory: on a device with less than 70 GiB the OpenAI server gets `max_num_seqs` 256 and `max_num_batched_tokens` 2048, and the larger pair (1024 and 8192) only above 70 GiB. Source: [S76] vLLM `vllm/engine/arg_utils.py` at b1388b1, `EngineArgs.get_batch_defaults`.
  - Implication: with 64 GB shared on this pod the defaults are 256 and 2048 whatever torch reports as device memory (below 70 GiB either way). Neither caps a batch-1 or batch-16 decode; concurrency is the harness's workload variable, not a server setting, so both are pinned at those values explicitly and kept out of the search. `max_num_seqs` also sets the graph capture range (KS4), which is one more reason to pin it rather than search it.
  - Experiment: none beyond E11 in the kernels queue (the batching sweep). Expected movement: none at fixed concurrency. Minimum sample size: 0. Confidence: high (0.8).

- **[practitioner]** `--gpu-memory-utilization` is the share of device memory vLLM may use (default 0.9 at b1388b1), and the script passes 0.90 explicitly. Source: [S77] vLLM `vllm/config/cache.py` at b1388b1, `gpu_memory_utilization`; `lab/pod_a_teacher.sh` (`GPU_UTIL` default 0.90).
  - Implication: it sizes the KV cache and leaves the rest of the 64 GB to the CPU side of the APU; it does not change the decode step. Pinned at 0.90.
  - Experiment: none beyond KS0 (the KV capacity line). Expected movement: none. Minimum sample size: 0. Confidence: high (0.8).

## ROCr and HIP runtime environment

- **[practitioner]** The ROCr runtime reads `HSA_ENABLE_SDMA` as three-state (`0` disables SDMA copy engines, `1` enables, unset leaves the runtime's default), `HSA_ENABLE_INTERRUPT` as on unless `0` (off makes waits poll instead of sleeping on interrupts), `HSA_XNACK` as `0`, `1` or unchanged, `HSA_NO_SCRATCH_RECLAIM=1` to keep scratch memory allocated, `HSA_MAX_QUEUES` and `HSA_SCRATCH_SINGLE_LIMIT` as integers. Source: [S83] ROCm `rocm-systems` `projects/rocr-runtime/runtime/hsa-runtime/core/util/flag.h` at 71a246ac, `Flag::Refresh`.
  - Implication: SDMA only moves copies, and a graph-replayed decode step copies a few token ids, so `HSA_ENABLE_SDMA` cannot move decode tok/s by more than noise; `HSA_ENABLE_INTERRUPT=0` trades host CPU cycles for wake-up latency, and on an APU whose SOCKET_POWER includes the CPU that trade can raise joules per token; the others are memory-fault and scratch policies with no decode path. All stay unset (their defaults), all are recorded by KS0, and none is searched.
  - Experiment: none beyond KS0. Expected movement: none. Minimum sample size: 0. Confidence: medium (0.6).

- **[practitioner]** The HIP runtime defines `HIP_FORCE_DEV_KERNARG` (default true, "Force device mem for kernel args"), `AMD_SERIALIZE_KERNEL` and `HIP_LAUNCH_BLOCKING` (default 0; nonzero waits for each kernel to finish around its enqueue), `GPU_SINGLE_ALLOC_PERCENT` (default 100, the largest single allocation as a share of total) and `ROC_AQL_QUEUE_SIZE` (default 16384 packets). Source: [S82] ROCm `rocm-systems` `projects/clr/rocclr/utils/flags.hpp` at 71a246ac.
  - Implication: the serialize flags only slow decode (they are debug switches) and are pinned 0; the kernel-argument and queue-size flags act on launch cost, which graph replay already hides; the allocation share could only matter if one allocation exceeded the whole device. None is searched.
  - Experiment: none beyond KS0. Expected movement: none. Minimum sample size: 0. Confidence: medium (0.6).

## Allocator

- **[practitioner]** In torch 2.9.1 the caching allocator reads its options from `PYTORCH_ALLOC_CONF`, with `PYTORCH_CUDA_ALLOC_CONF` and `PYTORCH_HIP_ALLOC_CONF` accepted as deprecated aliases (a warning names `PYTORCH_ALLOC_CONF` as the replacement); the options include `max_split_size_mb`, `garbage_collection_threshold`, `roundup_power2_divisions` and `expandable_segments`, all off or zero by default. Source: [S86] PyTorch v2.9.1 `c10/core/AllocatorConfig.cpp`, `AcceleratorAllocatorConfig`.
  - Implication: vLLM allocates the KV cache and the graph pools at start-up and a replayed decode step allocates nothing new, so allocator options can change start-up memory headroom and fragmentation, not the decode step. Pinned unset and recorded by KS0; if the image sets `PYTORCH_HIP_ALLOC_CONF`, the serve log shows the deprecation warning.
  - Experiment: none beyond KS0. Expected movement: none. Minimum sample size: 0. Confidence: medium (0.6).

## Triton compiler flags

- **[practitioner]** Triton 3.5.1 keeps compiled kernels under `TRITON_CACHE_DIR` (default `$TRITON_HOME/.triton/cache`, `TRITON_HOME` defaulting to `~`), caches autotuning results only with `TRITON_CACHE_AUTOTUNING`, recompiles regardless of the cache with `TRITON_ALWAYS_COMPILE`, and has AMD backend switches `AMDGCN_USE_BUFFER_OPS` (default off; when on, tensor arguments within 2 GB are lowered to buffer loads and stores), `TRITON_HIP_USE_BLOCK_PINGPONG` (default on only for gfx942, and for gfx950 with async copy), `TRITON_HIP_USE_ASYNC_COPY`, `TRITON_HIP_GLOBAL_PREFETCH` and `TRITON_HIP_LOCAL_PREFETCH`. Source: [S87] Triton v3.5.1 `python/triton/knobs.py`, `cache_knobs`, `autotuning_knobs` and `amd_knobs`, and `third_party/amd/backend/compiler.py`, `is_pingpong_schedule_enabled`, `get_arg_specialization` and `make_ttgir`.
  - Implication: the cache flags change start-up and compile time, not steady decode, and the ping-pong and async-copy paths are gated to CDNA3 and CDNA4 by default; of the AMD switches only buffer ops changes the code generated for every Triton kernel on gfx1151, which is where the decode time goes (Conch, the GDN kernels and attention). That makes `AMDGCN_USE_BUFFER_OPS` the one Triton flag worth a search dimension.
  - Candidate experiment: KS10, `AMDGCN_USE_BUFFER_OPS=1` against `0`, each arm with its own empty `TRITON_CACHE_DIR` so the kernels really recompile, at batch 1. Expected movement: decode -5% to +10%, joules per token the inverse; whether the gfx11 lowering supports buffer ops cleanly waits on the R32 note. Minimum sample size: 5 cold spawns per arm. Confidence: low (0.3).

## Power and clock state a non-root pod can read

- **[practitioner]** The amdgpu driver exposes `power_dpm_force_performance_level` (auto, low, high, manual, profile_standard, profile_min_sclk, profile_min_mclk, profile_peak), `pp_dpm_sclk` and `pp_dpm_mclk`, and `pp_power_profile_mode` as read-write attributes created with mode `S_IRUGO | S_IWUSR` (world-readable, owner-writable), `gpu_metrics` as read-only, and in hwmon `power1_average` and `power1_input` (on APUs these include the CPU) as read-only and `power1_cap` as read-write for the owner. Source: [S88] Linux v6.17 `drivers/gpu/drm/amd/pm/amdgpu_pm.c` (DOC blocks and `amdgpu_device_attrs`, hwmon attributes) and `drivers/gpu/drm/amd/pm/inc/amdgpu_pm.h` (`AMDGPU_DEVICE_ATTR_RW`).
  - Implication: the pod user (no root, R2 and R29 hour-0 facts in `lab/README.md`) can read the performance level, the clock tables, the power profile and the power cap but cannot set them, so power and clock state is a recorded condition on every trial, never a search dimension; the energy side of the search moves only through decode speed and through how hard each setting drives the chip at the driver's own policy.
  - Experiment: none beyond KS0, which also reads `power_dpm_force_performance_level`, `pp_power_profile_mode` and `power1_cap` once per spawn. Expected movement: none. Minimum sample size: 1 read per spawn. Confidence: high (0.8).

- **[measured]** On this chip `amd-smi metric --power` returns SOCKET_POWER (the whole APU package) and GFX_ACTIVITY, with CLOCK and ENERGY reading N/A, so the lab sampler fills clocks from `rocm-smi --showclocks` and integrates energy from power. Source: `lab/README.md`, sections "GPU power sampler: `lab/pod_power.py`" and "Teacher on SJC today (pod C, Strix Halo)"; `lab/pod_power.py`.
  - Implication: joules per token in R38 is integrated SOCKET_POWER over the decode window with the idle reading beside it, as R11 defines; a setting that keeps CPU threads spinning shows up in it.
  - Experiment: none beyond KS-base. Expected movement: none. Minimum sample size: 0. Confidence: high (0.8).

## Shortlist

The ten settings R38 searches, in the order of expected joules-per-token gain times confidence. Every
other row of the Summary table is pinned at its baseline. Predictions are relative to the KS-base
arm (2.88 tok/s, about 41.7 J/token at batch 1); "waits on" marks a size that the R33 chip sheet or the
R32 hardware note (both in flight) will move.

| # | Setting | Values (baseline first) | Decode tok/s | J/token | Confidence | Waits on |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `linear_kernel` | `conch_bf16`, `exllama_fp16` | +50% to +200% | -30% to -65% | low (0.3) | R33 (bandwidth, dequant rate) |
| 2 | `num_speculative_tokens` | 0, 1, 2, 3 | +30% to +120% | -20% to -50% | low (0.4) | R32 (memory system), acceptance rate |
| 3 | `gpu_max_hw_queues` | 4, 2, 1 | 0% to +70% (only if bimodal) | 0% to -40% | low (0.25) | R32 (queues on this APU) |
| 4 | `cudagraph_mode` | `FULL_AND_PIECEWISE`, `PIECEWISE`, `NONE` | -6% to +3% | -3% to +6% | medium (0.5) | none |
| 5 | `mamba_ssm_cache_dtype` | `float32`, `float16` | +0.5% to +2% | -0.5% to -2% | medium (0.6) | none |
| 6 | `vllm_rocm_use_skinny_gemm` | true, false | -10% to +5% | -5% to +10% | low (0.35) | R33 (lm_head GEMM rate) |
| 7 | `fla_packed_recurrent_decode` | true, false | -10% to 0% with it off | 0% to +10% | low (0.35) | none |
| 8 | `attention_backend` | `ROCM_ATTN`, `TRITON_ATTN` | -3% to +3% (short context) | -3% to +3% | medium (0.5) | none |
| 9 | `enable_prefix_caching` | true, false | 0% to +5% with it off | 0% to -5% | low (0.35) | none |
| 10 | `amdgcn_use_buffer_ops` | false, true | -5% to +10% | -10% to +5% | low (0.3) | R32 (gfx11 buffer instructions) |

The flags each value maps to are in `data/searchspace-inference-gfx1151.json` under `mapping`; R38 reads
them from there. Two pairs interact and must be searched jointly, not one at a time: 2 with 7 (MTP
turns the packed GDN decode path off) and 3 with 4 (on gfx1201 the band penalty lived in full-graph
replay).

## Candidate experiments, ranked

1. KS1 `linear_kernel`, with the R21 correctness gate as a hard constraint.
2. KS2 `num_speculative_tokens`, jointly with KS7.
3. KS3 `gpu_max_hw_queues`, jointly with KS4, 8 cold spawns per arm.
4. KS6 `vllm_rocm_use_skinny_gemm` (a guard arm; a gain with it off is a kernel bug to file).
5. KS10 `amdgcn_use_buffer_ops`, each arm with a fresh Triton cache.
6. KS9 `enable_prefix_caching`.
7. KS4 `cudagraph_mode`.
8. KS8 `attention_backend`.
9. KS5 `mamba_ssm_cache_dtype`.
10. KS0 is not ranked: it is the read-back that fills the value-today column and runs first, once per pod, before any trial.

## Open questions

- What the pod image sets in the serve process's environment (every "image default, unread" row): KS0, one read-only read on pod C or D.
- Whether AMD's `torch 2.9.1+rocm7.13.0` build carries the upstream hipBLASLt arch lists or adds gfx1151; the serve log answers it with one spawn.
- Whether `qwen3_5_mtp` speculative decoding runs on this ROCm build at all; KS2's first spawn answers it.
- Which `mamba_cache_mode` (`all` or `align`) the teacher resolves to, and so its exact attention block size; the serve log of KS0.
- How large each prediction's band really is on gfx1151; the R33 chip sheet and the R32 note (both in flight) will move the sizes in rows 1, 2, 3, 6 and 10.

## Search log

- 2026-10-10, budget 3 h, used about 2 h. Starting points: the committed launch script and recipes (`lab/pod_a_teacher.sh` at 722d845, R29 498e161, R30 0f51e01, `lab/README.md`, `lab/vllm-gfx1201.diff`) and the five kernels notes in this folder. Read at pinned refs through GitHub raw files: vLLM b1388b1 (`platforms/rocm.py`, `engine/arg_utils.py`, `config/cache.py`, `config/compilation.py`, `models/config.py`, `envs.py`, `_aiter_ops.py`, `layers/utils.py`, `attention/backends/rocm_attn.py` and `triton_attn.py`, `layers/mamba/gdn_linear_attn.py`); ROCm `rocm-systems` develop at 71a246ac (`clr` flags, `rocr-runtime` flags); PyTorch v2.9.1 (`Context.cpp`, `Context.h`, `AllocatorConfig.cpp`, the TunableOp README); Triton v3.5.1 (`knobs.py`, the AMD backend compiler); Linux v6.17 (`amdgpu_pm.c`, `amdgpu_pm.h`). The rendered documentation sites (rocm.docs.amd.com, docs.vllm.ai, docs.pytorch.org, docs.kernel.org) were refused by this session's egress proxy, so every mechanism is cited from the source or the in-repo docs file those sites are built from. Web search for "ROCm environment variables HSA_ENABLE_SDMA GPU_MAX_HW_QUEUES HIP_FORCE_DEV_KERNARG" returned only rendered rocm.docs.amd.com pages, which are not cited because they could not be opened. Stopped when the last three files opened (the GDN layer, the unquantized GEMM dispatcher, the amdgpu sysfs attributes) settled the remaining rows and no setting was left without a mechanism. No pod, GPU or Jetson was touched.
