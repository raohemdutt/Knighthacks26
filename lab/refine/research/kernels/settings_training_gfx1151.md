# Training settings for the student LoRA on gfx1151: every knob, its mechanism, the pinned baseline and a shortlist

Research note for the kernels track, written for Task Board task R36 on 2026-10-10. It lists every environment
variable and runtime setting that can change one LoRA training step of the student (Qwen3.5-4B, bf16, PEFT LoRA,
TRL `SFTTrainer`) on the trainer pod: AMD Strix Halo, Radeon 8060S, gfx1151 (RDNA 3.5), 64 GB of memory shared
between CPU and GPU, ROCm 7.13, torch 2.9.1+rocm7.13.0, triton 3.5.1, Python 3.12, pod D on the SJC hub (account
D, node pod-b). The stock image has no flash-linear-attention and no cmake. Each setting gets its mechanism from
its source, the value in use today and a pinned baseline; section 13 is the settings table, section 14 the
shortlist with predicted effects, and `data/searchspace-training-gfx1151.json` is the Ax search space for R39.

Nothing here was run on a pod or a GPU. Every number about our own step comes from `lab/smoke_lora.py` [L4] or
from the R9700 smoke run [L5], which is another chip and is never a gfx1151 baseline. The step profile (R34) and
the chip sheet (R33) have not run yet, so every predicted size that depends on where a step's time goes is marked
"waits on R34". The real trainer (L3) is in progress in another session and has no pushed branch or open PR on
2026-10-10, so "value today" means `lab/smoke_lora.py`, which the README names as the template for L3.

`SOURCES.md` in this folder indexes every source; `[Sxx]` and `[Lx]` ids point into it. Labels follow
`lab/refine/RESEARCH_FORMAT.md`. The metric set for this note: training samples per second
(`trainer.samples_per_s`, the smoke script's `samples_per_s`) and peak memory (the smoke script's `peak_mem_gb`,
which is `torch.cuda.max_memory_allocated`, allocated bytes and not reserved bytes). Experiment ids `TS1` to
`TS16` are local to this note.

## 1. The step being tuned and the node it runs on

- **[practitioner]** Our training step today is `lab/smoke_lora.py`: `AutoModelForCausalLM` in bf16 on device 0 with `use_cache=False`, LoRA r 16, alpha 32, dropout 0 on `all-linear`, and a TRL `SFTConfig` with per-device batch 4, accumulation 1, lr 2e-4, `bf16=True`, `max_length` 256, `dataloader_num_workers=2`, `dataloader_persistent_workers=False`, `torch_compile=False`; it does not set `gradient_checkpointing`, `packing`, `optim` or `attn_implementation`, and it reports samples per second and `max_memory_allocated`. `lab/pod_b_trainer.sh` installs PEFT, TRL, transformers and accelerate with `pip --user` and never reinstalls torch, torchvision or triton. Source: [L4] `lab/smoke_lora.py::main`, `lab/pod_b_trainer.sh`
  - Implication: every setting this note does not name is at its library default, so the defaults found below are the values in use today; the baseline for R39 is this file plus the pinned environment in section 13. Its `peak_mem_gb` is allocated memory, so allocator settings that only change reserved memory do not move it.
  - Candidate experiment: TS1, the pinned-baseline run: the smoke step with the section 13 environment exported explicitly and a printout of the effective values (`torch.backends.cuda.preferred_blas_library()`, `trainer.args.optim`, `trainer.args.gradient_checkpointing`, `model.config._attn_implementation`, which Gated DeltaNet implementation was bound, `torch.cuda.mem_get_info()`). Expected movement: none against an unpinned run beyond noise; it fixes the reference samples per second and peak memory every other experiment is compared with. Minimum sample size: 5 runs of 50 steps after 5 warm-up steps. Confidence: high.

- **[measured]** On one R9700 (gfx1201, a 32 GB discrete card, another chip) the same smoke step ran at 0.51 samples/s with an 8.7 GB peak, loss 2.23 to 0.16 over 20 steps. Source: [L5] `lab/refine/research/learning/lora_recipe.md` section 4, from PR #37
  - Implication: none for the gfx1151 baseline: the owner ruled the R9700 numbers another chip. It does say the step fits in under 10 GB at batch 4 and 256 tokens, which bounds how much batch, sequence length or checkpointing-off headroom a 64 GB shared pod has.
  - Experiment: TS1, as specified in section 1 (the gfx1151 numbers replace these as the reference).

- **[practitioner]** AMD's Strix Halo optimization page recommends keeping the BIOS VRAM reservation small (for example 0.5 GB), says that by default the GTT limit is "approximately 50% of total system RAM", and that the limit is raised through the TTM `pages_limit` (in pages) or the `amd-ttm --set <GB>` tool, which needs Linux 6.18.4 or newer. Source: [S91] ROCm 7.2.0 docs, `strixhalo.rst`, "Memory settings" and "Configuring shared memory limits on linux"
  - Implication: the memory a training step can allocate on the pod is set by the host's carve-out and GTT limit, which we do not control (no root on a pod); peak memory is judged against what `torch.cuda.mem_get_info()` reports on the pod, not against 64 GB.
  - Experiment: TS1, as specified in section 1 (it records `mem_get_info` total and free).

## 2. HIP and HSA runtime variables

- **[practitioner]** HIP reads `GPU_MAX_HW_QUEUES` as "The maximum number of hardware queues allocated per device", default 4, reusing queues when an application creates more streams; `AMD_SERIALIZE_KERNEL` (default 0) serializes kernel enqueue for debugging; `HIP_VISIBLE_DEVICES` limits the devices HIP sees. Source: [S67] ROCm 7.13.0 environment variables, HIP debug variables; [S90] HIP `debug_hip_env.rst`
  - Implication: a training step uses one compute stream plus copy streams, so `GPU_MAX_HW_QUEUES` should not change samples per second; `AMD_SERIALIZE_KERNEL` must stay unset (0), since serializing every launch only slows a step. On gfx1201 a lower queue count removed a per-kernel wait in serving [S66], which is another chip and another workload.
  - Candidate experiment: TS2, the runtime-variable no-op check: baseline against `GPU_MAX_HW_QUEUES=1` in one pinned session. Expected movement: samples per second within noise (under 2 percent), peak memory unchanged. Minimum sample size: 3 runs per arm of 50 steps. Confidence: medium.

- **[practitioner]** The HIP runtime defines `HIP_FORCE_DEV_KERNARG` with default true ("Force device mem for kernel args."). Source: [S90] `clr/rocclr/utils/flags.hpp`
  - Implication: already on by default; nothing to tune, pinned as unset.
  - Experiment: none beyond TS1.

- **[practitioner]** The ROCr runtime documents `HSA_XNACK` ("Enables XNACK", legal values 0 or 1), `HSA_ENABLE_SDMA` (default 1, DMA engines for copies) and `HSA_ENABLE_SCRATCH_ASYNC_RECLAIM` (default 1, reclaim scratch when a device allocation fails). Source: [S89] ROCr `env_variables.rst`
  - Implication: on an APU the host-to-device copies of a text batch are small, so `HSA_ENABLE_SDMA` should not move samples per second; `HSA_XNACK=1` enables recoverable page faults, which unified-memory code can use but PyTorch's caching allocator does not need. Both stay unset in the baseline.
  - Experiment: TS2, as specified in section 2 (add an `HSA_ENABLE_SDMA=0` arm only if TS1 shows copies above 1 percent of a step).

- **[practitioner]** HIP documents `GPU_MAX_HEAP_SIZE` (default 100, the GPU heap as a percent of board memory) and `GPU_SINGLE_ALLOC_PERCENT` (default 100). Source: [S90] HIP `memory_management_hip_env.rst`
  - Implication: at their defaults neither caps a step that peaks near 10 GB; they matter only if a larger batch or checkpointing off hits an allocation failure before `mem_get_info` says the pod is full.
  - Experiment: none beyond TS1.

## 3. The PyTorch allocator on ROCm

- **[practitioner]** PyTorch 2.9.1 reads its allocator options from `PYTORCH_ALLOC_CONF` first, then the deprecated `PYTORCH_CUDA_ALLOC_CONF` and `PYTORCH_HIP_ALLOC_CONF` (each deprecated name warns "use PYTORCH_ALLOC_CONF instead"); the options are `max_split_size_mb` (default unlimited), `garbage_collection_threshold` (default 1.0, off), `roundup_power2_divisions` and `expandable_segments` (experimental, default False). Source: [S80] `c10/core/AllocatorConfig.cpp`; [S79] `notes/cuda.rst`, "Memory management"
  - Implication: these settings change how much memory is reserved and how fragmented it gets, not how much is allocated, so they move `torch.cuda.memory_reserved` and the risk of an out-of-memory error, not our `peak_mem_gb`. Use `PYTORCH_ALLOC_CONF` on the pod.
  - Candidate experiment: TS3, the allocator check: baseline against `PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512` at the largest batch that fits, recording peak reserved as well as peak allocated. Expected movement: peak allocated and samples per second within noise; peak reserved down by 0 to 15 percent. Minimum sample size: 3 runs per arm of 50 steps. Confidence: medium.

- **[practitioner]** In PyTorch 2.9.1 the real `ExpandableSegment` is compiled only under `!defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)`; on ROCm the stub asserts "expandable segment not supported", and the config parser warns "expandable_segments not supported on this platform" and keeps it off. Source: [S80] `c10/cuda/CUDACachingAllocator.cpp` at v2.9.1
  - Implication: `expandable_segments:True` is a no-op on our torch, so it stays out of the search space; R39 must not spend trials on it.
  - Experiment: none beyond TS1 (TS1 logs any allocator warning at start-up).

- **[practitioner]** The HIP semantics page says `PYTORCH_NO_HIP_MEMORY_CACHING=1` disables caching, that hipBLAS workspaces are set with `HIPBLAS_WORKSPACE_CONFIG=:[SIZE]:[COUNT]` (default 32 MiB, 128 MiB on MI300 and newer; `CUBLAS_WORKSPACE_CONFIG` also accepted), and that setting the hipFFT plan cache size "is not supported". Source: [S78] `notes/hip.rst` at v2.9.1
  - Implication: disabling caching would slow every step and is a debugging switch only; the BLAS workspace default is fine for a 4B model's GEMMs. Both stay unset.
  - Experiment: none beyond TS1.

## 4. Matmuls: rocBLAS, hipBLASLt and TunableOp

- **[practitioner]** In PyTorch 2.9.1 `Context::blasPreferredBackend` picks hipBLASLt by default only when every device is gfx90a, gfx942, gfx1200, gfx1201 (ROCm 6.4 or newer) or gfx950, and allows a requested hipBLASLt only on gfx90a, gfx942, gfx1100, gfx1101, gfx1200, gfx1201, gfx908 or gfx950; on any other arch it warns "Attempting to use hipBLASLt on an unsupported architecture! Overriding blas backend to hipblas". gfx1151 is in neither list. `TORCH_BLAS_PREFER_HIPBLASLT=1` and `torch.backends.cuda.preferred_blas_library("hipblaslt")` are the two ways to request it. Source: [S76] `aten/src/ATen/Context.cpp::Context::blasPreferredBackend` at v2.9.1; [S85] `torch/backends/cuda/__init__.py::preferred_blas_library`
  - Implication: on upstream 2.9.1 every LoRA matmul on gfx1151 runs through hipBLAS (rocBLAS) whatever we set; the hipBLASLt knob is a no-op unless the pod's `2.9.1+rocm7.13.0` build patched the list. TS1's printout of `preferred_blas_library()` after a request settles which.
  - Candidate experiment: TS4, the BLAS backend check: `TORCH_BLAS_PREFER_HIPBLASLT=1` against unset, confirming the effective backend each time. Expected movement: none if the warning appears (upstream lists); if the pod build accepts hipBLASLt, samples per second within plus or minus 10 percent (waits on R34 for the GEMM share of a step), peak memory up by the hipBLASLt workspace only. Minimum sample size: 3 runs per arm of 50 steps. Confidence: high that it is a no-op on upstream lists, low on the size otherwise.

- **[practitioner]** PyTorch 2.10.0 adds gfx1150 and gfx1151 to the hipBLASLt supported list under ROCm 7.0 and later, and main prefers hipBLASLt on gfx1151 from ROCm 7.13. Source: [S77] `aten/src/ATen/Context.cpp` at v2.10.0 and main `aten/src/ATen/cuda/detail/CUDAHooks.cpp`
  - Implication: upstream treats hipBLASLt on gfx1151 as supported only from 2.10 and as the better default only on main; on our 2.9.1 the decision is the pod build's. A torch upgrade is outside this task (the image's torch is never reinstalled).
  - Experiment: TS4, as specified in section 4.

- **[practitioner]** TunableOp is off by default (`PYTORCH_TUNABLEOP_ENABLED`, "Default is 0"); when enabled with `PYTORCH_TUNABLEOP_TUNING=1` (default 1) it times every candidate GEMM solution from rocBLAS and hipBLASLt (`PYTORCH_TUNABLEOP_ROCBLAS_ENABLED` and `PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED`, both default 1) for each new shape, up to `PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS` (default 30) and `PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS` (default 100), and writes the winners to `PYTORCH_TUNABLEOP_FILENAME` (default `tunableop_results.csv`, device ordinal inserted) for reuse; "Currently only a TunableGemm for ROCm is implemented". Source: [S81] `aten/src/ATen/cuda/tunable/README.md` at v2.9.1
  - Implication: TunableOp is the one GEMM lever that does not depend on the arch lists above, because it times the libraries' solutions directly; a step with fixed shapes (fixed `max_length`, or packing) tunes once, while variable-length padded batches create a new shape per length and pay tuning time repeatedly. Tune in a first run, then train with `PYTORCH_TUNABLEOP_TUNING=0` reading the file.
  - Candidate experiment: TS5, TunableOp: one tuning run of 30 steps writing the results file, then baseline against `PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0` reading it, at a fixed sequence length. Expected movement: samples per second up 0 to 15 percent (waits on R34 for the GEMM share), peak memory unchanged. Minimum sample size: 3 runs per arm of 50 steps. Confidence: low.

## 5. Attention: SDPA backends and AOTriton

- **[practitioner]** transformers uses SDPA by default ("By default, if available, SDPA will be used", otherwise `eager`); Qwen3.5's eight gated-attention layers have `head_dim` 256. Source: [S94] `modeling_utils.py`, `from_pretrained` docstring; [S103] `configuration_qwen3_5.py` (`head_dim: int = 256`)
  - Implication: the attention path today is SDPA, and which SDPA kernel runs decides the cost of those 8 of 32 layers.
  - Experiment: TS6, as specified in section 5.

- **[practitioner]** PyTorch 2.9.1 builds SDPA's flash and memory-efficient backends on ROCm from AOTriton 0.11b, limits gfx11 to `head_dim` 256 with that version, and refuses both backends on an arch AOTriton marks experimental unless `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` ("Flash Efficient attention on Current AMD GPU is still experimental."); AOTriton 0.11b's `isArchExperimentallySupported` returns true for gfx950, gfx1151 and gfx1201. With both refused, SDPA runs its math backend. `TORCH_ROCM_FA_PREFER_CK` selects the CK backend instead of AOTriton. Source: [S82] `sdp_utils.cpp` and `cmake/External/aotriton.cmake` at v2.9.1; [S83] ROCm/aotriton `v2src/util.cc` at 0.11b; [S85] `preferred_rocm_fa_library`
  - Implication: today the 8 attention layers run SDPA's math path on gfx1151, which materializes the full attention matrix in each layer; `head_dim` 256 is inside the gfx11 limit, so the flag is the only thing between the step and a fused kernel. On 256-token sequences the matrix is small, so the gain is mostly launch count and memory traffic, and it grows with `max_length` and packing.
  - Candidate experiment: TS6, AOTriton attention: `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` against unset, with `torch.nn.attention.sdpa_kernel` restricted to flash then to efficient in two extra arms to confirm which kernel ran, and the loss curve compared step by step with the baseline over 50 steps. Expected movement: samples per second up 2 to 10 percent at 256 tokens and more at 1024 or with packing (waits on R34 for the attention share), peak memory down a few percent. Minimum sample size: 3 runs per arm of 50 steps. Confidence: medium.

- **[anecdote]** A ROCm TheRock issue on flash attention with gfx1151 says that at runtime "you also need to set the env var TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 to enable aotriton", and its closing comment says AOTriton is built for every arch in the nightlies except gfx1152, gfx1153 and gfx103X. Source: [S84] ROCm/TheRock#1364
  - Implication: gives no speed or accuracy figure, but says a TheRock-built torch (which a `+rocm7.13.0` wheel may be) ships the gfx1151 AOTriton kernels behind the same flag.
  - Experiment: TS6, as specified in section 5.

- **[practitioner]** `torch.backends.cuda.enable_flash_sdp`, `enable_mem_efficient_sdp` and `enable_math_sdp` are beta flags, and `torch.backends.cuda.sdp_kernel()` is deprecated in favor of `torch.nn.attention.sdpa_kernel()`. Source: [S85] `torch/backends/cuda/__init__.py` at v2.9.1
  - Implication: R39 sets the backend with the environment flag above and confirms it with `sdpa_kernel` in a probe, rather than toggling beta global flags inside the trainer.
  - Experiment: TS6, as specified in section 5.

## 6. The recurrent layers: Gated DeltaNet and causal conv1d

- **[practitioner]** In transformers, Qwen3.5's `chunk_gated_delta_rule` and `fused_recurrent_gated_delta_rule` are bound to the `fla` package and `causal_conv1d_fn` and `causal_conv1d_update` to the `causal_conv1d` package through `use_kernel_func_from_hub_with_fallback`, whose order is hub kernels (if requested), then the original package, then the torch path; on the torch path it logs once that the function "is falling back to its reference PyTorch implementation", with a comment that for `chunk_gated_delta_rule` "the gap is more than an order of magnitude on an H100". The torch `torch_chunk_gated_delta_rule` casts query, key, value, beta and decay to float32. Source: [S104] `hub_kernels.py::use_kernel_func_from_hub_with_fallback`; [S103] `modeling_qwen3_5.py::torch_chunk_gated_delta_rule`
  - Implication: the stock image has no `flash-linear-attention` and no `causal-conv1d`, so today all 24 Gated DeltaNet layers run the float32 torch reference in forward and backward. This is the largest single setting on the list; how much of a step it costs is exactly what R34 measures.
  - Candidate experiment: TS7, the Gated DeltaNet kernel: install `flash-linear-attention` into the user site without touching torch or triton (`--no-deps`, then check `torch.__version__` and `triton.__version__` are unchanged), confirm the fallback warning is gone and `fla` is bound, then baseline against fla. Expected movement: samples per second up 1.5x to 3x if the Gated DeltaNet layers are more than half of the step (the learning note's LR-1 predicts they are; waits on R34), peak memory down 10 to 30 percent (no float32 copies of q, k, v); loss within noise of the baseline over 50 steps. Minimum sample size: 3 runs per arm of 50 steps, after 10 warm-up steps for Triton compilation. Confidence: medium.

- **[practitioner]** The flash-linear-attention README says all implementations are "platform-agnostic and verified on NVIDIA, AMD, and Intel hardware", installs on ROCm with `pip install flash-linear-attention[rocm]` after a ROCm torch from the PyTorch rocm7.2 index, and says a bare `pip install flash-linear-attention` does not install torch or triton; `chunk_gated_delta_rule` takes `g` in log space and a `use_qk_l2norm_in_kernel` flag. Source: [S101] fla `README.md` and `fla/ops/gated_delta_rule/chunk.py` at f3bec72
  - Implication: the `[rocm]` extra would pull a torch from another index and must not be used on the pod; the bare package on top of the image's torch 2.9.1 and triton 3.5.1 is the install that keeps the stack pinned. Whether fla's minimum triton is at or below 3.5.1 is checked before the install.
  - Experiment: TS7, as specified in section 6.

- **[anecdote]** A flash-linear-attention issue reports that "the Qwen3.5 gated-delta fast path has pathological first-run latency when Triton starts from a cold cache" on a Radeon 890M (torch 2.11.0+rocm7.1, triton 3.6.0), and the proposed PR was "not a standalone fix for the pathological cold-cache case". Source: [S102] fla-org/flash-linear-attention#804
  - Implication: the first steps after an fla install, or after a new sequence length, are dominated by Triton compilation on an RDNA 3.5 part; samples per second must exclude warm-up, and the Triton cache directory must persist between runs (the pod's home directory does).
  - Experiment: TS7, as specified in section 6 (the 10 warm-up steps; also record the first-step time).

- **[practitioner]** transformers main maps the whole `Qwen3_5GatedDeltaNet` layer, for ROCm devices of capability 11.5 ("AMD Strix Halo (gfx1151, capability 11.5)"), to the hub repository `Atlas-Inference/gdn` at a pinned revision with `trust_remote_code=True`; the mapping is used only when hub kernels are requested (`use_kernels=True`), and the `chunk_gated_delta_rule` and `causal_conv1d` hub mappings list CUDA and XPU only. Source: [S104] `hub_kernels.py`, `_build_kernel_mapping`
  - Implication: a second route to a fast Gated DeltaNet on exactly our arch exists, but it runs third-party code from a non-allow-listed publisher, may not exist in the pod's transformers version, and is not documented as training-capable. It stays out of the search space until someone reads that repository's code and confirms a backward pass.
  - Candidate experiment: TS8, the hub-layer probe: in a scratch environment, load the student with `use_kernels=True`, run one forward and backward pass on a fixed batch and compare the LoRA gradients with the torch path. Expected movement: none to the training numbers until it passes; if it passes, samples per second similar to TS7. Minimum sample size: 1 run on 1 fixed batch. Confidence: low.

- **[practitioner]** The torch fallback `torch_chunk_gated_delta_rule` accepts `**kwargs` but has no `cu_seqlens` parameter, while the layer passes `cu_seqlens=kwargs.pop("cu_seq_lens_q", None)` to it; the fallback wrapper filters keyword arguments to the implementation's own parameters. Source: [S103] `modeling_qwen3_5.py::Qwen3_5GatedDeltaNet.forward`; [S104] `use_kernel_func_from_hub_with_fallback`
  - Implication: our inference from the code: on the torch path, sequence boundaries are dropped, so a packed or padding-free batch carries the recurrent state from one example into the next. Packing is only correct once fla (TS7) is bound and receives `cu_seqlens`; the search space encodes that dependency.
  - Experiment: TS9, as specified in section 9.

- **[practitioner]** causal-conv1d's maintainer says it "already works on ROCm/HIP with `--no-build-isolation`. Tested on gfx1201, all tests pass." Source: [S105] Dao-AILab/causal-conv1d#99
  - Implication: the package must be compiled for gfx1151 on the pod (hipcc is present; the build needs no cmake), and no gfx1151 test is reported. Without it, the depthwise conv runs through PyTorch's convolution (section 7).
  - Candidate experiment: TS10, the conv kernel: build `causal-conv1d` with `--no-build-isolation` for gfx1151, run its tests, then TS7's fla arm with and without it. Expected movement: samples per second up 0 to 5 percent (the conv is a small share; waits on R34), peak memory unchanged. Minimum sample size: 3 runs per arm of 50 steps. Confidence: low.

- **[anecdote]** A PEFT issue reports a Qwen3.8-27B LoRA run where "Nothing in the logs said which delta-rule implementation was in use". Source: [S106] huggingface/peft#3718
  - Implication: every run R39 records must log which implementation was bound, read from the bound function, not inferred from the absence of a warning.
  - Experiment: TS1, as specified in section 1 (the printout includes the bound implementation).

## 7. MIOpen find and cache modes

- **[practitioner]** PyTorch 2.9.1 sends a float, half or bf16 convolution on a ROCm device to MIOpen when `torch.backends.cudnn.enabled` is set; 1-D input is viewed as 2-D, and a depthwise convolution (channels equal to groups) selects `MiopenDepthwise`. Source: [S86] `aten/src/ATen/native/Convolution.cpp` at v2.9.1
  - Implication: without the `causal-conv1d` package, Qwen3.5's short causal convolution in the 24 Gated DeltaNet layers is the only part of our step that reaches MIOpen; with it, MIOpen is not used at all and the settings below do nothing.
  - Experiment: TS11, as specified in section 7.

- **[practitioner]** MIOpen's find modes are NORMAL (1), FAST (2), HYBRID (3), DYNAMIC_HYBRID (5), TRUST_VERIFY (6) and TRUST_VERIFY_FULL (7) set by `MIOPEN_FIND_MODE`; the page says "The default find mode is ``DYNAMIC_HYBRID``" and also describes HYBRID as the behaviour with `MIOPEN_FIND_MODE` unset, so the docs disagree with themselves; DYNAMIC_HYBRID "offers faster start-up times than hybrid find, but GPU performance might decrease". The user database defaults to `$HOME/.config/miopen` (`MIOPEN_USER_DB_PATH`), the kernel cache to `$HOME/.cache/miopen` (`MIOPEN_CUSTOM_CACHE_DIR`), and `MIOPEN_FIND_ENFORCE` takes NONE, DB_UPDATE, SEARCH, SEARCH_DB_UPDATE or DB_CLEAN. Source: [S87] MIOpen `find-and-immediate.rst`, "Find modes"; [S88] MIOpen `tuningdb.rst`, `perfdb.rst`, `cache.rst`
  - Implication: a first run on a fresh pod pays find and kernel compilation for the conv shapes; both caches sit in the persistent home directory, so later runs reuse them. Pin `MIOPEN_FIND_MODE=NORMAL` in tuning runs so the effective mode does not depend on which reading of the docs the library follows, and exclude warm-up from samples per second.
  - Candidate experiment: TS11, MIOpen find mode on the torch conv path: `MIOPEN_FIND_MODE=NORMAL` against `FAST`, each with a warm cache. Expected movement: samples per second within plus or minus 3 percent (the conv is a small share; waits on R34), first-step time lower with FAST, peak memory unchanged. Minimum sample size: 3 runs per arm of 50 steps. Confidence: low.

## 8. Precision: bf16, TF32 and reduced-precision reductions

- **[practitioner]** TRL's `SFTConfig` documents `bf16` as defaulting to True when `fp16` is not set; transformers' Trainer delegates autocast to accelerate ("We rely on accelerate for autocast, hence we do nothing here") with `mixed_precision="bf16"`; and the smoke script also loads the weights in bf16. Source: [S95] `trl/trainer/sft_config.py`; [S93] `training_args.py` (`bf16`)
  - Implication: the step is bf16 weights plus bf16 autocast; fp32 is not a candidate (more memory, slower GEMMs) and fp16 risks overflow without a loss scaler. bf16 is pinned, not searched.
  - Experiment: none beyond TS1.

- **[practitioner]** The HIP semantics page says "TF32 is not supported on ROCm"; in PyTorch 2.9.1 `allowTF32CuBLAS` returns false on ROCm unless `HIPBLASLT_ALLOW_TF32=1`, and setting `allow_tf32` without it logs that it "is not supported on ROCm by default". Source: [S78] `notes/hip.rst`; [S76] `Context::allowTF32CuBLAS`
  - Implication: TF32 settings (`torch.backends.cuda.matmul.allow_tf32`, `set_float32_matmul_precision`, the trainer's `tf32`) do nothing to a bf16 step on gfx1151; RDNA 3.5 has no TF32 path through rocBLAS anyway. They stay unset.
  - Experiment: none beyond TS1.

- **[practitioner]** PyTorch's CUDA notes say `allow_bf16_reduced_precision_reduction` "is set to `True` by default for BF16" GEMMs. Source: [S79] `notes/cuda.rst`, "Reduced Precision Reduction in BF16 GEMMs"
  - Implication: already at the fast setting by default; turning it off could only slow GEMMs, and whether rocBLAS honours it on gfx1151 is not documented. Pinned at the default.
  - Experiment: none beyond TS1.

## 9. Batch, accumulation, checkpointing and packing

- **[practitioner]** TRL's `SFTConfig` overrides transformers and defaults `gradient_checkpointing` to True; transformers enables it with `use_reentrant=False` when no kwargs are given and calls `enable_input_require_grads` for PEFT, and documents checkpointing as allowing "larger models or batch sizes at the cost of ~20% slower training". Source: [S95] `sft_config.py` (the NOTE on changed defaults); [S94] `modeling_utils.py::gradient_checkpointing_enable`; [S93] `training_args.py` (`gradient_checkpointing`)
  - Implication: the smoke step checkpoints every layer today although the R9700 run peaked under 10 GB; on a 64 GB shared pod, turning it off trades memory we have for speed we want.
  - Candidate experiment: TS12, gradient checkpointing off: `gradient_checkpointing=False` against the baseline at batch 4 and at the largest batch that fits. Expected movement: samples per second up 15 to 30 percent, peak memory up 1.5x to 3x of activations (from under 10 GB to roughly 15 to 25 GB at batch 4 and 256 tokens; waits on R34 for the activation share). Minimum sample size: 3 runs per arm of 50 steps. Confidence: medium.

- **[measured]** In torchtune's single-device table (Llama 3.2 3B, bf16, A100, batch 2, sequence 4096), activation checkpointing cut peak memory from 42.9 to 24.9 GiB and throughput from 9,174 to 7,210 tokens/s. Source: `lab/refine/research/learning/lora_recipe.md` section 4, finding citing the torchtune README
  - Implication: on another model and chip, checkpointing cost about a fifth of throughput and saved about two fifths of memory; TS12's predicted sizes take that order.
  - Experiment: TS12, as specified in section 9.

- **[practitioner]** transformers defines the effective batch as `per_device_train_batch_size x num_devices x gradient_accumulation_steps` (accumulation default 1), and its single-GPU guide says larger batches train faster and to "avoid too many gradient accumulation steps because it can really slow down training". Source: [S93] `training_args.py` (`gradient_accumulation_steps`); [S97] `perf_train_gpu_one.md` at v4.57.1, "Batch size" and "Gradient accumulation"
  - Implication: for a fixed effective batch, a larger per-device batch with fewer accumulation steps raises samples per second until the GPU is saturated, at the cost of activation memory. The search space holds the effective batch fixed and searches the split, so R39 changes speed, not the optimization recipe.
  - Candidate experiment: TS13, the batch split: per-device batch 1, 2, 4, 8 and 16 with accumulation set to keep the effective batch at the value L3 uses. Expected movement: samples per second up 20 to 60 percent from batch 4 to 16 if a step at batch 4 is launch-bound (waits on R34), peak memory roughly linear in per-device batch for activations. Minimum sample size: 3 runs per arm of 50 optimizer steps. Confidence: medium.

- **[practitioner]** TRL documents `packing` (default False) with `packing_strategy` "bfd" (default), "bfd_split" or "wrapped", says that with "bfd" padding-free is enabled whatever `padding_free` says, warns that attention implementations other than its flash-attention variants (flash_attention_2 and 3 and the kernels-community flash-attn repositories; SDPA is not one) "may lead to cross-contamination between samples", and its memory guide says packing is for "setups that use **FlashAttention** (or its variants)". `max_length` defaults to 1024 and sets the packed length. Source: [S95] `sft_config.py` and `sft_trainer.py`; [S96] `reducing_memory_usage.md`, "Packing" and "Padding-free"
  - Implication: packing removes the padding a batch of short chat examples carries, but on our stack it is incorrect twice: SDPA is not a flash-attention variant, and the torch Gated DeltaNet path drops sequence boundaries (section 6). It is gated on fla (TS7) plus a varlen attention path on gfx1151 that we have not found; until then the search space keeps it off.
  - Candidate experiment: TS9, packing: once TS7 passes and a flash-attention variant loads on gfx1151, `packing=True` (bfd) against the baseline at the same `max_length`, checking per-example loss against an unpacked run on the same 48 examples. Expected movement: samples per second up 1.5x to 3x on short examples (most of a padded batch is padding), peak memory flat at a fixed `max_length`. Minimum sample size: 3 runs per arm of 50 steps. Confidence: low.

- **[practitioner]** transformers' `DataCollatorWithFlattening` "concatenates the entire mini batch into single long sequence of shape [1, total_tokens]" and returns `position_ids`, with the tip "Make sure your attention computation is able to handle it!" Source: [S109] `data_collator.py::DataCollatorWithFlattening`
  - Implication: the same correctness gate as packing applies to padding-free collation; it is not a separate search dimension.
  - Experiment: TS9, as specified in section 9.

- **[practitioner]** transformers main offers `train_sampling_strategy="group_by_length"`, which "Uses `LengthGroupedSampler` to group samples of roughly the same length together (to minimize padding"; older releases expose the same sampler as the boolean `group_by_length`. Source: [S93] `training_args.py` (`train_sampling_strategy`)
  - Implication: length grouping cuts padding without packing's correctness problems, so it is the safe padding lever while packing is gated. It changes the order of examples, so it is a recipe change for L3 as well as a speed change.
  - Candidate experiment: TS14, length grouping: `group_by_length` against random sampling on a real L3 dataset with varied lengths. Expected movement: samples per second up 10 to 40 percent depending on the length spread, peak memory unchanged or lower; held-out score checked by L4, not here. Minimum sample size: 3 runs per arm of one epoch. Confidence: medium.

## 10. Data loading

- **[practitioner]** PyTorch's DataLoader loads in the main process when `num_workers` is 0 (default), `prefetch_factor` defaults to 2 when workers exist, `persistent_workers` defaults to False, and "Host to GPU copies are much faster when they originate from pinned (page-locked) memory"; transformers' `dataloader_pin_memory` defaults to True and `dataloader_num_workers` to 0. Source: [S92] `torch/utils/data/dataloader.py` and `docs/source/data.md`, "Memory Pinning"; [S93] `training_args.py`
  - Implication: our batches are a few kilobytes of token ids, so on an APU neither workers nor pinning should move samples per second; the smoke script's 2 workers with a forkserver start add process start-up per epoch. Pinned at the smoke values (2 workers, pin on, not persistent), outside the search space.
  - Candidate experiment: TS15, data loading: workers 0 against 2 with pin memory on. Expected movement: samples per second within 2 percent, peak memory unchanged. Minimum sample size: 3 runs per arm of 50 steps. Confidence: high.

- **[anecdote]** A Triton issue (open) reports "HIP: invalid device ordinal on gfx1151 (AMD Strix Halo iGPU) in subprocess" with Triton 3.6.0, ROCm 7.12.0 and PyTorch 2.10.0+rocm7.12.0. Source: [S100] triton-lang/triton#10036
  - Implication: worker or compile subprocesses that touch the GPU can fail on gfx1151; DataLoader workers do not touch the GPU, but torch.compile's compile workers and fla's Triton compilation do, so a failure there in TS7 or TS16 has a known lead.
  - Experiment: TS15, as specified in section 10.

## 11. Optimizer

- **[practitioner]** transformers defaults `optim` to `adamw_torch_fused` from torch 2.8 ("defaults to `"adamw_torch"` (for torch>=2.8 `"adamw_torch_fused"`)"); PyTorch's fused optimizers support float32, float16 and bfloat16 parameters and are enabled for device types including "cuda", which a ROCm build reports; foreach uses about the size of the parameters in extra peak memory. Source: [S93] `training_args.py` (`optim`); [S98] `torch/optim/optimizer.py` and `torch/utils/_foreach_utils.py::_get_fused_kernels_supported_devices`
  - Implication: the step already uses the fused AdamW on torch 2.9.1 if the pod's transformers has the same default (TS1 prints it); LoRA's trainable parameters are a small fraction of 4B, so the optimizer step is a small share and the fused-against-foreach choice should not show. Pinned, not searched.
  - Experiment: TS1, as specified in section 1 (it prints `trainer.args.optim`).

## 12. torch.compile

- **[practitioner]** transformers' `torch_compile` defaults to False ("Can provide 20-50% speedup with no code changes"), with `torch_compile_backend` "inductor" and `torch_compile_mode` "default", "reduce-overhead" or "max-autotune"; PEFT's guide lists LoRA and Trainer training as working and warns "If you don't see an error, it doesn't necessarily mean that `torch.compile` worked correctly." Source: [S93] `training_args.py` (`torch_compile`); [S99] PEFT `torch_compile.md`
  - Implication: compile is off today. Its blockers on our stack: inductor generates Triton kernels for gfx1151 (Triton compilation in subprocesses has an open gfx1151 bug, [S100]); the torch Gated DeltaNet path has Python-level chunk loops that may graph-break; variable sequence lengths recompile; and gradient checkpointing under compile always preserves RNG state [S108]. It comes after TS7, so it does not compile a reference implementation that is about to be replaced.
  - Candidate experiment: TS16, torch.compile: `torch_compile=True` (inductor, default mode) at a fixed `max_length` after TS7, counting graph breaks with `TORCH_LOGS=graph_breaks` and recording compile time separately. Expected movement: samples per second up 0 to 25 percent after warm-up (waits on R34 for the share of small elementwise kernels), peak memory down 0 to 10 percent, first steps several minutes slower. Minimum sample size: 3 runs per arm of 50 steps after 20 warm-up steps. Confidence: low.

## 13. Settings table: value today and pinned baseline

"Today" is `lab/smoke_lora.py` with the image's defaults [L4]. "Pinned baseline" is what TS1 exports explicitly so every trial starts from the same state; it equals today's value except where a default is ambiguous or deprecated. "In search space" refers to `data/searchspace-training-gfx1151.json`.

| Setting | Mechanism (section) | Value today | Pinned baseline | In search space |
| --- | --- | --- | --- | --- |
| Gated DeltaNet kernel | fla `chunk_gated_delta_rule` or torch float32 reference (6) | torch reference (no fla in the image) | torch | yes, `gdn_kernel` |
| causal conv1d kernel | `causal_conv1d` package or `nn.Conv1d` through MIOpen (6, 7) | `nn.Conv1d` through MIOpen | torch | yes, `causal_conv1d_kernel` |
| Hub GDN layer | `use_kernels=True`, `Atlas-Inference/gdn` for capability 11.5 (6) | off | off | no (TS8 probe first) |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | gates SDPA flash and efficient on gfx1151 (5) | unset, SDPA math | unset | yes, `aotriton_experimental` |
| `attn_implementation` | SDPA by default, `eager` otherwise (5) | `sdpa` | `sdpa` | no |
| `TORCH_ROCM_FA_PREFER_CK` | CK instead of AOTriton for SDPA (5) | unset | unset | no |
| `TORCH_BLAS_PREFER_HIPBLASLT` / `preferred_blas_library` | hipBLASLt request, overridden on archs not listed (4) | unset, hipBLAS | unset | yes, `blas_backend` |
| `PYTORCH_TUNABLEOP_ENABLED`, `_TUNING`, `_FILENAME` | GEMM solution search across rocBLAS and hipBLASLt (4) | 0 | 0 | yes, `tunableop` |
| `HIPBLAS_WORKSPACE_CONFIG` | hipBLAS workspace, 32 MiB default (3) | unset | unset | no |
| `HIPBLASLT_ALLOW_TF32`, `allow_tf32`, `tf32` | TF32 not supported on ROCm by default (8) | unset | unset | no |
| `allow_bf16_reduced_precision_reduction` | default True for bf16 GEMMs (8) | True | True | no |
| `bf16` and model dtype | bf16 weights and accelerate bf16 autocast (8) | True, bf16 | True, bf16 | no |
| `gradient_checkpointing` | TRL default True, `use_reentrant=False` (9) | True | True | yes, `gradient_checkpointing` |
| `per_device_train_batch_size` | micro-batch per step (9) | 4 | 4 | yes, `per_device_train_batch_size` |
| `gradient_accumulation_steps` | effective batch = micro-batch x accumulation (9) | 1 | effective batch / micro-batch | derived, `gradient_accumulation_steps` |
| `max_length` | truncation and packed length (9) | 256 | 256 | no (a data choice for L3) |
| `packing`, `packing_strategy`, `padding_free` | bfd packing, padding-free collation (9) | off, "bfd", off | off | yes, `packing` (only with fla) |
| `train_sampling_strategy` / `group_by_length` | `LengthGroupedSampler` (9) | random | random | yes, `group_by_length` |
| `dataloader_num_workers` | worker processes (10) | 2 | 2 | no |
| `dataloader_pin_memory` | pinned host buffers (10) | True | True | no |
| `dataloader_persistent_workers` | keep workers between epochs (10) | False | False | no |
| `dataloader_prefetch_factor` | batches prefetched per worker (10) | 2 (torch default) | 2 | no |
| `optim` | fused AdamW from torch 2.8 (11) | `adamw_torch_fused` (TS1 confirms) | `adamw_torch_fused` | no |
| `torch_compile`, backend, mode | inductor Triton kernels (12) | False | False | yes, `torch_compile` |
| `PYTORCH_ALLOC_CONF` | allocator options; `expandable_segments` a no-op on ROCm 2.9.1 (3) | unset | unset | no (TS3 measures reserved memory) |
| `PYTORCH_NO_HIP_MEMORY_CACHING` | disables the caching allocator (3) | unset | unset | no |
| `MIOPEN_FIND_MODE` | find mode for the torch conv path (7) | unset (docs disagree: DYNAMIC_HYBRID or HYBRID) | `NORMAL` | no |
| `MIOPEN_USER_DB_PATH`, `MIOPEN_CUSTOM_CACHE_DIR` | find database and kernel cache location (7) | `~/.config/miopen`, `~/.cache/miopen` | same (persistent home) | no |
| `MIOPEN_FIND_ENFORCE` | find database update policy (7) | unset (NONE) | unset | no |
| `GPU_MAX_HW_QUEUES` | hardware queues per device (2) | unset (4) | unset | no |
| `AMD_SERIALIZE_KERNEL` | serialize launches, debugging (2) | unset (0) | unset | no |
| `HIP_FORCE_DEV_KERNARG` | kernel arguments in device memory (2) | unset (true) | unset | no |
| `HSA_XNACK`, `HSA_ENABLE_SDMA` | page-fault retry, DMA copy engines (2) | unset | unset | no |
| `GPU_MAX_HEAP_SIZE`, `GPU_SINGLE_ALLOC_PERCENT` | heap and single-allocation caps (2) | unset (100, 100) | unset | no |
| GTT limit, `pages_limit`, BIOS carve-out | host-side shared memory limits (1) | host-set, no root | recorded by TS1 | no |

## 14. Candidate experiments, ranked

At most ten settings, ranked by predicted samples-per-second gain times confidence. Sizes marked R34 depend on where a step's time goes and are finalised when the R34 profile lands; until then they are ranges from the sources above. Each row's full experiment is in the section named.

| Rank | Setting (experiment) | Predicted samples per second | Predicted peak memory | Waits on R34 | Confidence |
| --- | --- | --- | --- | --- | --- |
| 1 | `gdn_kernel` fla against torch (TS7, section 6) | up 1.5x to 3x | down 10 to 30 percent | yes (Gated DeltaNet share) | medium |
| 2 | `gradient_checkpointing` off (TS12, section 9) | up 15 to 30 percent | up, to roughly 15 to 25 GB at batch 4 | yes (activation share) | medium |
| 3 | `per_device_train_batch_size` 1 to 16, effective batch fixed (TS13, section 9) | up 20 to 60 percent from 4 to 16 | roughly linear in micro-batch | yes (launch-bound or not) | medium |
| 4 | `packing` bfd, only with fla and a varlen attention path (TS9, section 9) | up 1.5x to 3x on short examples | flat at fixed `max_length` | no (padding share is a data property) | low |
| 5 | `group_by_length` (TS14, section 9) | up 10 to 40 percent | unchanged or lower | no | medium |
| 6 | `aotriton_experimental` SDPA flash or efficient (TS6, section 5) | up 2 to 10 percent at 256 tokens, more when longer | down a few percent | yes (attention share) | medium |
| 7 | `torch_compile` inductor after fla (TS16, section 12) | up 0 to 25 percent after warm-up | down 0 to 10 percent | yes (elementwise share) | low |
| 8 | `tunableop` tuned file, tuning off in training (TS5, section 4) | up 0 to 15 percent | unchanged | yes (GEMM share) | low |
| 9 | `causal_conv1d_kernel` package (TS10, section 6) | up 0 to 5 percent | unchanged | yes (conv share) | low |
| 10 | `blas_backend` hipBLASLt (TS4, section 4) | none if overridden (upstream 2.9.1); plus or minus 10 percent otherwise | up by a workspace | yes (GEMM share) | high that it is a no-op on upstream lists |

TS1 (pinned baseline) runs before all of them. TS2, TS3, TS8, TS11 and TS15 are checks that keep settings out of the search, not candidates.

## Open questions

- Does the pod's `torch 2.9.1+rocm7.13.0` build carry upstream 2.9.1's hipBLASLt arch lists (gfx1151 refused) or a patched list? TS1 answers it in one line.
- Which flash-linear-attention release is the newest whose Triton floor is at or below 3.5.1, and do its Gated DeltaNet backward kernels pass their tests on gfx1151?
- Is there a varlen attention path on gfx1151 that TRL accepts as a flash-attention variant (a ROCm flash-attn build, or a kernels-community repository with a ROCm entry)? Without one, packing stays off.
- Does the `Atlas-Inference/gdn` hub layer have a backward pass, and does the pod's transformers version carry its mapping?
- How much memory does `torch.cuda.mem_get_info()` report on pod D, given the GTT limit default of about half the system RAM?
- What effective batch will L3 use? The search space holds it fixed at 16 (the batch 4 by accumulation 4 of Qwen's own fine-tuning example, cited in the learning note) until L3 says otherwise.

## Search log

- 2026-10-10, R36, budget about 3 hours (kernels), used about 25 minutes, about 6:35 to 7:00 PM ET. Started from the task's Done when list, `lab/smoke_lora.py`, `lab/pod_b_trainer.sh` and the learning note `lora_recipe.md` sections 4 and 6. docs.pytorch.org, rocm.docs.amd.com and huggingface.co did not resolve from this session, so every doc was read from its source file on GitHub at a pinned ref (PyTorch v2.9.1, v2.10.0 and main; transformers, TRL, fla and Ax main at the commits named in `SOURCES.md`; MIOpen and ROCr docs at the develop commits named there; the ROCm 7.2.0 Strix Halo page). Queries: the ROCm environment variables reference, MIOpen find modes, hipBLASLt arch lists and TunableOp, the allocator config and expandable segments on ROCm, AOTriton gfx1151 status, SDPA ROCm gating, DataLoader and TrainingArguments defaults, TRL packing and padding-free, fused optimizers, torch.compile on gfx1151, fla on AMD, causal-conv1d on ROCm, the transformers Qwen3.5 kernel fallback, and the Ax parameter configs. Stopped when the last three lookups (a pytorch issue search for compile on gfx1151, a second fla issue search for ROCm, and the AMD Triton install page) returned nothing beyond the sources already listed; no pod was used.
