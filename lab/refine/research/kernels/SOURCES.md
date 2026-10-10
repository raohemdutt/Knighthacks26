# Kernels research sources

Index of every source cited by the kernels research notes in this folder, written for Task Board
task O8. Every source below was opened on 2026-10-10, and every number taken from it was checked
against the source text (page, file at a pinned commit, or thread) before it was cited. Ids are
stable: a later research pass appends new ids after the last one and never renumbers.

Notes in this folder:

| Note | Covers |
| --- | --- |
| `w4a16_decode_gfx1201.md` (main) | Why W4A16 decode is slow on gfx1201 in vLLM, what exists to fix it, the byte budget, and the ranked experiment queue (section 7) |
| `rdna4_vllm_ports.md` (ports) | The Capicua25x and radiance ports, the forum FP8 patch, and field reports that change how we measure |
| `triton_gfx12.md` (triton) | LDS and cache limits, the GFX12 WMMA set, Triton knobs that mattered on gfx12, cache-aware benchmarking |
| `precision_kv_batching.md` (precision) | FP8, MXFP4, INT8 and INT4 paths, KV-cache quantization, Gated DeltaNet state precision, batching and J/token |
| `correctness_and_tuners.md` (correctness) | Checking changed kernels, batch invariance, time-only tuners, energy-aware tuning |
| `strix_halo_hardware.md` (strix) | The gfx1151 chip in the SJC pods from AMD, LLVM, Linux and library sources: CUs, wavefronts, registers and occupancy, LDS, caches, memory split, WMMA, power files, ROCm settings, and the questions for R33 |
| `settings_inference_gfx1151.md` (settings) | Every runtime setting that can change vLLM decode on the gfx1151 teacher, its mechanism, value today, pinned baseline, and the ten-setting Ax shortlist |

Short names in the "Cited in" column refer to the notes above. "Code" sources are files read at the
commit named in the link; "b1388b1" is the vLLM commit our pod runs
(b1388b1fbf5aaef47937fabe98931211684666a6, 2026-04-18) and "main" is vLLM main at
c41b2639e29c3bc01add1d34bef3032a6d9d8aca (2026-10-10).

## Our own measurements and files

| Id | Kind | Source | Link or location | Opened | Cited in |
| --- | --- | --- | --- | --- | --- |
| L1 | our measurement | `lab/README.md`, sections "Teacher (pod A)" and "Checkpoints": decode, TTFT, watts, VRAM and concurrency for `RedHatAI/Qwen3.8-27B-INT4` on pod A; merged in PR #37 | this repo, `lab/README.md` (merge commit bb351e6) | 2026-10-10 | main, ports, precision |
| L2 | our measurement | T0 task page, Findings line of Sat 5:55 AM: pod idle power 38 to 40 W, gfx clock 500 to 2350 MHz, memory clock 96 to 1258 MHz | https://app.notion.com/p/3f596ef5fec581819588ec7a702a4bb1 | 2026-10-10 | main, triton |
| L3 | our files | `lab/pod_a_teacher.sh` (serve flags, `--dtype bfloat16`) and `lab/vllm-gfx1201.diff` (unified attention `num_stages=1`, 66,560 bytes of LDS requested by default), merged in PR #37 | this repo, `lab/` (merge commit bb351e6) | 2026-10-10 | main, triton |

## Community ports, forums and field reports

| Id | Kind | Source | Link or location | Opened | Cited in |
| --- | --- | --- | --- | --- | --- |
| S01 | issue | Capicua25x, "[RFC] RDNA4 (gfx1201) serving: a production-proven port exists - offering it to AMD for adoption", ROCm/vllm issue #1229, 2026-08-20 | https://github.com/ROCm/vllm/issues/1229 | 2026-10-10 | ports |
| S02 | repository | Capicua25x, `vllm-rocm-rdna4` README (vLLM 0.28.0 branch, release rc12 of 2026-08-28) | https://github.com/Capicua25x/vllm-rocm-rdna4 | 2026-10-10 | ports |
| S03 | repository doc | Capicua25x, `RDNA4-PORT.md` on branch `rdna4-port-0.26.1` (sections "What's in the port", "Quantization recipe (Quark MXFP4)", "Release rc9 (2026-08-19)", "Throughput", "Measurement update 2026-08-24") | https://github.com/Capicua25x/vllm-rocm-rdna4/blob/rdna4-port-0.26.1/RDNA4-PORT.md | 2026-10-10 | ports, precision |
| S04 | repository | drwolfen, `radiance-vllm-r9700` README (libr4d kernels, environment variables, benchmark tables, power tuning) | https://github.com/drwolfen/radiance-vllm-r9700 | 2026-10-10 | ports |
| S05 | forum | claviger, "Native FP8 WMMA Support for AMD RDNA4 (RX 9070 XT / R9700) in vLLM", vLLM forum, 2025-11-13, with replies to 2026-02-12 | https://discuss.vllm.ai/t/native-fp8-wmma-support-for-amd-rdna4-rx-9070-xt-r9700-in-vllm/1900 | 2026-10-10 | ports |
| S26 | discussion | JohnTDI-cpu and others, "RDNA4 Llama Experiments - Squeezing Every Token/s from the R9700", ggml-org/llama.cpp discussion #21043, 2026-03-26; comments by zedbytes (2026-03-27, 2026-03-28, 2026-05-18) and misterkerns (2026-09-03) | https://github.com/ggml-org/llama.cpp/discussions/21043 | 2026-10-10 | main, ports |
| S27 | repository | ivangotoy, `llama-server-rdna4-vulkan` README (RX 9070 XT, Qwen3.5-9B, llama-server Vulkan against vLLM ROCm) | https://github.com/ivangotoy/llama-server-rdna4-vulkan | 2026-10-10 | main |
| S28 | issue | taisunyoung, "R9700 (gfx1201) - vLLM decode throughput is bimodal (~33 vs ~26 tok/s), fixed at process spawn and not recoverable without restart", ROCm/ROCm issue #6347, 2026-06-11, with comments by taisunyoung and yanghoeg to 2026-08-28 | https://github.com/ROCm/ROCm/issues/6347 | 2026-10-10 | ports |

## vLLM pull requests and issues

| Id | Kind | Source | Link or location | Opened | Cited in |
| --- | --- | --- | --- | --- | --- |
| S12 | PR | jatseng-ai, "[Kernel][Hardware][AMD] Add TritonW4A16LinearKernel for ROCm", vLLM #37352, merged 2026-04-10 | https://github.com/vllm-project/vllm/pull/37352 | 2026-10-10 | main |
| S13 | PR | mgehre-amd, "[ROCm][Kernel] Add HybridW4A16LinearKernel: Triton prefill + HIP skinny decode", vLLM #40977, merged 2026-07-14 | https://github.com/vllm-project/vllm/pull/40977 | 2026-10-10 | main, correctness |
| S16 | PR | tangzzycc, "[ROCm][Perf] Enable medium-skinny dispatch in RDNAHybridW4A16LinearKernel", vLLM #52619, merged 2026-10-07 | https://github.com/vllm-project/vllm/pull/52619 | 2026-10-10 | main, triton, correctness |
| S17 | PR | mgehre-amd, "[ROCm][Perf] W4A16: keep skinny GEMM zero-points packed 4-bit", vLLM #54965, merged 2026-09-14 | https://github.com/vllm-project/vllm/pull/54965 | 2026-10-10 | main, correctness |
| S18 | PR | mgehre-amd, "[ROCm][Perf] W4A16: pad gfx11 weight and activation strides", vLLM #56301, merged 2026-10-06 | https://github.com/vllm-project/vllm/pull/56301 | 2026-10-10 | main, correctness |
| S19 | PR | JartX, "[Kernel][ROCm] Native W4A16 kernel for AMD RDNA3 (gfx1100) - fp16 + bf16", vLLM #41394, merged 2026-05-29 | https://github.com/vllm-project/vllm/pull/41394 | 2026-10-10 | main |
| S20 | PR | laudney, "[ROCm] Enable wvSplitK skinny GEMM kernel for RDNA4/gfx1x decode", vLLM #34709, merged 2026-03-20 | https://github.com/vllm-project/vllm/pull/34709 | 2026-10-10 | main, precision |
| S21 | PR and code | limitmhw, "[Quantization] Support native Quark W4A16 INT4/UINT4 exports in vLLM", vLLM #48606, merged 2026-09-18; and `vllm/model_executor/layers/quantization/quark/schemes/quark_w4a16_int4.py` at main | https://github.com/vllm-project/vllm/pull/48606 | 2026-10-10 | main |
| S24 | issue | fyun89, "[Usage]: ROCm gfx950: MXFP4 activation quant not fused into GEMM; Quark INT4 W4A16 uses a very slow Triton kernel", vLLM #60581, 2026-10-08 | https://github.com/vllm-project/vllm/issues/60581 | 2026-10-10 | main |
| S29 | issue | dantolinp, "[Bug]: [ROCm][gfx1201] AITER update in v0.25.0 generates unified-attention kernel exceeding LDS limit", vLLM #48723, 2026-07-15 | https://github.com/vllm-project/vllm/issues/48723 | 2026-10-10 | ports, triton |
| S30 | PR | nemanjaudovic, "[ROCm][Perf] GDN packed decode: tuned launch configs for gfx1100 and gfx1201", vLLM #60400, open (2026-10-07) | https://github.com/vllm-project/vllm/pull/60400 | 2026-10-10 | triton, precision |
| S31 | PR | lijipeng787, "[Perf][ROCm][Attention] Narrow the Triton prefill-attention KV tile on RDNA3/RDNA4", vLLM #58225, merged 2026-09-23 | https://github.com/vllm-project/vllm/pull/58225 | 2026-10-10 | triton |
| S32 | PR | big-yellow-duck, "[ROCM][Perf] Add ROCM_SEGMENTED_ATTN attention-backend works across RDNA3/3.5/4", vLLM #59132, merged 2026-10-06 | https://github.com/vllm-project/vllm/pull/59132 | 2026-10-10 | main, ports |
| S33 | PR | laudney, "[ROCm] Enable FP8 KV-cache and relax constraints for RDNA4 custom paged attention", vLLM #34741, open (2026-02-17) | https://github.com/vllm-project/vllm/pull/34741 | 2026-10-10 | precision |
| S34 | PR | laudney, "[ROCm] Add MXFP4 inline dequant Triton kernel for RDNA4/gfx12", vLLM #34632, closed without merge (2026-02-16) | https://github.com/vllm-project/vllm/pull/34632 | 2026-10-10 | precision |
| S35 | issue | Lafunamor, "[Bug]: ROCm tuned-config lookup keys off an unstable device name - shipped gfx1151 configs are unreachable", vLLM #50436, 2026-07-30 | https://github.com/vllm-project/vllm/issues/50436 | 2026-10-10 | ports |
| S36 | issue | JWhiteCat, "[Bug]: Exllama GPTQ and RDNA3 W4A16 kernels silently produce wrong output when `group_size % 32 != 0`", vLLM #60608, 2026-10-08 | https://github.com/vllm-project/vllm/issues/60608 | 2026-10-10 | correctness |

## vLLM, Conch and LLVM source code

| Id | Kind | Source | Link or location | Opened | Cited in |
| --- | --- | --- | --- | --- | --- |
| S06 | code | vLLM `vllm/model_executor/kernels/linear/__init__.py` (`_POSSIBLE_KERNELS`, `_POSSIBLE_INT8_KERNELS`, `choose_mp_linear_kernel`) at b1388b1 and at main | https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/kernels/linear/__init__.py and https://github.com/vllm-project/vllm/blob/c41b2639e29c3bc01add1d34bef3032a6d9d8aca/vllm/model_executor/kernels/linear/__init__.py | 2026-10-10 | main, precision |
| S07 | code | vLLM `vllm/model_executor/kernels/linear/mixed_precision/conch.py` at b1388b1 (`ConchLinearKernel`) | https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/kernels/linear/mixed_precision/conch.py | 2026-10-10 | correctness |
| S08 | code | vLLM `vllm/model_executor/kernels/linear/mixed_precision/exllama.py` at b1388b1 (`ExllamaLinearKernel.can_implement`) | https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/kernels/linear/mixed_precision/exllama.py | 2026-10-10 | main |
| S09 | repository | Stack AV Co., `conch` README ("A 'standard library' of Triton kernels", supported hardware, performance table) | https://github.com/stackav-oss/conch | 2026-10-10 | main |
| S10 | code | Conch v1.2.1 `conch/kernels/quantization/gemm.py` (`_get_tuning_parameters`, `mixed_precision_gemm_launcher`) | https://github.com/stackav-oss/conch/blob/v1.2.1/conch/kernels/quantization/gemm.py | 2026-10-10 | main, triton, precision |
| S11 | code | vLLM `requirements/rocm.txt` at b1388b1 (`conch-triton-kernels==1.2.1`) | https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/requirements/rocm.txt | 2026-10-10 | main |
| S14 | code | vLLM `vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py` at main | https://github.com/vllm-project/vllm/blob/c41b2639e29c3bc01add1d34bef3032a6d9d8aca/vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py | 2026-10-10 | main, triton |
| S15 | code | vLLM `csrc/rocm/skinny_gemms_int4.cu` at main | https://github.com/vllm-project/vllm/blob/c41b2639e29c3bc01add1d34bef3032a6d9d8aca/csrc/rocm/skinny_gemms_int4.cu | 2026-10-10 | main |
| S23 | code | vLLM `vllm/platforms/rocm.py` (`supports_mx`, `supports_fp8`) and `vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py` at main | https://github.com/vllm-project/vllm/blob/c41b2639e29c3bc01add1d34bef3032a6d9d8aca/vllm/platforms/rocm.py and https://github.com/vllm-project/vllm/blob/c41b2639e29c3bc01add1d34bef3032a6d9d8aca/vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py | 2026-10-10 | precision |
| S42 | PR (compiler) | Mirko Brkusanin, "[AMDGPU] Add GFX12 WMMA and SWMMAC instructions", llvm/llvm-project #77795, merged 2024-01-24 | https://github.com/llvm/llvm-project/pull/77795 | 2026-10-10 | triton, precision |
| S43 | code | LLVM `llvm/lib/Target/AMDGPU/AMDGPUSubtarget.h` (`getLocalMemorySize`, `getAddressableLocalMemorySize` comments), main as of 2026-10-10 | https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUSubtarget.h | 2026-10-10 | triton |
| S49 | code | vLLM `benchmarks/kernels/benchmark_moe.py` at b1388b1 (`tune`, `benchmark_config`) | https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/benchmarks/kernels/benchmark_moe.py | 2026-10-10 | correctness |
| S50 | code | vLLM `benchmarks/kernels/benchmark_rdna_hybrid_w4a16_gemm.py` at main (`benchmark` docstring) | https://github.com/vllm-project/vllm/blob/c41b2639e29c3bc01add1d34bef3032a6d9d8aca/benchmarks/kernels/benchmark_rdna_hybrid_w4a16_gemm.py | 2026-10-10 | triton, correctness |
| S58 | code | vLLM `vllm/platforms/rocm.py` at b1388b1 (`supports_fp8`, `on_gfx1x`, attention backend priorities) | https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/platforms/rocm.py | 2026-10-10 | ports, precision |
| S59 | code | vLLM `vllm/model_executor/models/config.py` (`Qwen3_5ForConditionalGenerationConfig`) and `vllm/model_executor/layers/mamba/mamba_utils.py` at b1388b1 | https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/models/config.py | 2026-10-10 | precision |
| S61 | code and doc | vLLM `vllm/model_executor/models/qwen3_5_mtp.py` and `docs/features/speculative_decoding/mtp.md` at b1388b1 | https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/docs/features/speculative_decoding/mtp.md | 2026-10-10 | main |

## Model files and model cards

| Id | Kind | Source | Link or location | Opened | Cited in |
| --- | --- | --- | --- | --- | --- |
| S22 | model card | AMD, `amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16` (scheme, accuracy against bf16, serve command, required vLLM PRs) | https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16 | 2026-10-10 | main |
| S25 | model files | Red Hat AI, `RedHatAI/Qwen3.8-27B-INT4`: `config.json` (`quantization_config`, `text_config`), `model.safetensors.index.json`, and the `model.safetensors` header read by HTTP range request (tensor shapes and dtypes); cross-checked against `Qwen/Qwen3.8-27B` `config.json` | https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4 and https://huggingface.co/Qwen/Qwen3.8-27B | 2026-10-10 | main, precision |

## Vendor documentation

| Id | Kind | Source | Link or location | Opened | Cited in |
| --- | --- | --- | --- | --- | --- |
| S37 | vendor doc | AMD ROCm documentation, "GPU hardware specifications", RDNA4 table, row "Radeon AI PRO R9700" | https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html | 2026-10-10 | triton, precision |
| S38 | vendor doc | AMD ROCm documentation, "Optimizing Triton kernels" | https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html | 2026-10-10 | triton |
| S39 | vendor doc | hipBLASLt 1.4.1 documentation, "Data type support" (sections "hipDataType enumeration", "Supported data types overview") | https://rocmdocs.amd.com/projects/hipBLASLt/en/latest/reference/data-type-support.html | 2026-10-10 | main, precision |
| S40 | vendor article | Takahiro Harada and Atsushi Yoshimura, "Using the Matrix Cores of AMD RDNA 4 architecture GPUs", AMD GPUOpen, 2025-07-11 | https://gpuopen.com/learn/using_matrix_core_amd_rdna4/ | 2026-10-10 | triton, precision |
| S41 | vendor article | Hui Zhang, "WMMA guide for AMD RDNA 4 architecture GPUs - part 2", AMD GPUOpen, 2026-06-02 | https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-2/ | 2026-10-10 | triton |
| S51 | framework doc | PyTorch 2.9 documentation, "TunableOp" (`torch.cuda.tunable`) | https://docs.pytorch.org/docs/2.9/cuda.tunable.html | 2026-10-10 | correctness |
| S56 | framework doc | vLLM documentation, "Batch Invariance" (latest) | https://docs.vllm.ai/en/latest/features/batch_invariance/ | 2026-10-10 | correctness |
| S57 | framework doc | vLLM documentation, "Quantized KV Cache" (latest) | https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/ | 2026-10-10 | precision |
| S60 | vendor spec | AMD, "AMD Radeon AI PRO R9700" product page, specifications (64 CUs, boost up to 2920 MHz, 191 TFLOPs FP16 matrix, 64 MB Infinity Cache, 256-bit, 640 GB/s, 300 W TBP) | https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html | 2026-10-10 | main, triton |

## Papers and lab write-ups

| Id | Kind | Source | Link or location | Opened | Cited in |
| --- | --- | --- | --- | --- | --- |
| S44 | paper | Elias Frantar, Roberto L. Castro, Jiale Chen, Torsten Hoefler, Dan Alistarh, "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models", PPoPP 2025, pp. 239-251 | doi:10.1145/3710848.3710871 (https://doi.org/10.1145/3710848.3710871); arXiv:2408.11743 | 2026-10-10 | main, precision |
| S45 | engineering blog | Lucas Wilkinson, "Introducing Machete, a mixed-input GEMM kernel optimized for NVIDIA Hopper GPUs", Red Hat Developer, 2024-10-14 | https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel | 2026-10-10 | main |
| S46 | paper | Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han, "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration", MLSys 2024 (Best Paper Award) | arXiv:2306.00978 (https://arxiv.org/abs/2306.00978) | 2026-10-10 | main |
| S47 | paper | Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", ICLR 2023 | arXiv:2210.17323 (https://arxiv.org/abs/2210.17323) | 2026-10-10 | main |
| S48 | paper | Adnan Hoque, Less Wright, Chih-Chieh Yang, Mudhakar Srivatsa, Raghu Ganti, "Accelerating a Triton Fused Kernel for W4A16 Quantized Inference with SplitK work decomposition", 2024 | arXiv:2402.00025 (https://arxiv.org/abs/2402.00025) | 2026-10-10 | main |
| S52 | paper | Richard Schoonhoven, Bram Veenboer, Ben van Werkhoven, K. Joost Batenburg, "Going green: optimizing GPUs for energy efficiency through model-steered auto-tuning", PMBS 2022 workshop at SC22 | arXiv:2211.07260 (https://arxiv.org/abs/2211.07260) | 2026-10-10 | correctness |
| S53 | paper | Bole Ma, Ayesha Afzal, Jan Eitzinger, Gerhard Wellein, "The Illusion of Power Capping in LLM Decode: A Phase-Aware Energy Characterisation Across Attention Architectures", 2026 | arXiv:2605.11999 (https://arxiv.org/abs/2605.11999) | 2026-10-10 | precision, correctness |
| S54 | paper | Julien Delavande, Regis Pierrard, Sasha Luccioni, "Understanding Efficiency: Quantization, Batching, and Serving Strategies in LLM Energy Use", 2026 | arXiv:2601.22362 (https://arxiv.org/abs/2601.22362) | 2026-10-10 | main, precision |
| S55 | lab write-up | Horace He and Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference", 2025-09-10 | https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ | 2026-10-10 | correctness |
| S62 | paper | Jae-Won Chung, Jeff J. Ma, Ruofan Wu, Jiachen Liu, Oh Jun Kweon, Yuxuan Xia, Zhiyu Wu, Mosharaf Chowdhury, "The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization", NeurIPS Datasets and Benchmarks 2025 (Spotlight) | arXiv:2505.06371 (https://arxiv.org/abs/2505.06371) | 2026-10-10 | precision |
| S63 | paper | Jared Fernandez, Clara Na, Vashisth Tiwari, Yonatan Bisk, Sasha Luccioni, Emma Strubell, "Energy Considerations of Large Language Model Inference and Efficiency Optimizations", 2025 | arXiv:2504.17674 (https://arxiv.org/abs/2504.17674) | 2026-10-10 | precision |
| S64 | paper | Jae-Won Chung, Ruofan Wu, Jeff J. Ma, Mosharaf Chowdhury, "Where Do the Joules Go? Diagnosing Inference Energy Consumption", 2026 | arXiv:2601.22076 (https://arxiv.org/abs/2601.22076) | 2026-10-10 | precision |

## Searched but not cited

The AMD RDNA4 Instruction Set Architecture reference guide (document 70651) moved to
https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture, which returned HTTP 401 to an
unauthenticated fetch; the instruction and LDS facts it would have supplied are cited from S40, S41,
S42 and S43 instead. The search was bounded at about three hours and stopped when new results
repeated sources already listed.

## Added by kernels refinement pass 1 (2026-10-10)

| Id | Kind | Source | Link or location | Opened | Cited in |
| --- | --- | --- | --- | --- | --- |
| S65 | issue | The same issue as S28 at its new home, read through the GitHub API to its last comment (2026-09-25): the opening post's idle sweep, Tanguille's report of `Qwen3.8-27B` W4A16 on one R9700 (2026-09-13), taisunyoung's `GPU_MAX_HW_QUEUES=1` results (2026-09-15) and cost measurement (2026-09-20) | https://github.com/ROCm/legacy-rocm-build/issues/6347 | 2026-10-10 | ports, main |
| S66 | issue | gazelleboy, "[gfx1201] Pending `hipStreamWaitEvent` on other streams adds ~17 us to every small kernel (not on gfx1030; gone with GPU_MAX_HW_QUEUES<=2)", ROCm/legacy-rocm-build issue #6685, 2026-09-25 | https://github.com/ROCm/legacy-rocm-build/issues/6685 | 2026-10-10 | ports |
| S67 | vendor doc | AMD ROCm 7.13.0 documentation, "ROCm environment variables", HIP environment variables, debug variables (`GPU_MAX_HW_QUEUES`) | https://rocm.docs.amd.com/en/7.13.0-preview/reference/environment-variables/index.html | 2026-10-10 | ports |
| S68 | paper | Prabhu Vellaisamy, Vanessa Lam, Shawn Blanton, John Paul Shen, "Characterization of Request and Token Energy Costs for LLM Inference Workloads on GPU Platforms", IISWC 2026 (accepted) | arXiv:2608.28044 (https://arxiv.org/abs/2608.28044) | 2026-10-10 | main |
| S69 | forum | u/lummr1, Reddit post on vLLM against llama.cpp for a dense 27B on one R9700 (vLLM 0.19.1, ROCm 7.13.0), read from a snapshot of 2026-07-17 | https://reddit.sentinel-team.org/posts/1utuh4k/snapshots/2026-07-17T18%3A53%3A30.39359Z | 2026-10-10 | main |
| S70 | repository doc | Capicua25x, `RDNA4-PORT.md` at commit 19f71e90764c, section "Reproducibility - same config, different outputs, and why" and the benchmark-harness note (the same document as S03 at another ref) | https://github.com/Capicua25x/vllm-rocm-rdna4/blob/19f71e90764c/RDNA4-PORT.md | 2026-10-10 | kernels pass 1 log entry |

## Added by kernels refinement pass 2 (2026-10-10)

| Id | Kind | Source | Link or location | Opened | Cited in |
| --- | --- | --- | --- | --- | --- |
| S71 | issue | The same issue as S66, read through the GitHub API to its last comment (2026-09-30): taisunyoung's third host (2026-09-26: half the cost, eager and graph alike, and the image that set `GPU_MAX_HW_QUEUES=1`), gazelleboy's reply (2026-09-26), lohiaj's fourth host with no cost (2026-09-30) and taisunyoung's run with the HIP runtime swapped on one host (2026-09-30) | https://github.com/ROCm/legacy-rocm-build/issues/6685 | 2026-10-10 | ports |
| S72 | issue | taisunyoung and others, "[Bug]: RowWiseTorchFP8ScaledMMLinearKernel is selected on RDNA4 (gfx1201) from v0.28 and costs 5-24% decode", vLLM #57838, read to its comment of 2026-10-10: `VLLM_DISABLED_KERNELS` honored for the scaled-mm choice, `--attention-backend` holding with AITER on, and image corruption under AITER unified attention | https://github.com/vllm-project/vllm/issues/57838 | 2026-10-10 | correctness |
| S73 | issue | taisunyoung, "gemma-4-12B on gfx1201: ROCM_AITER_UNIFIED_ATTN is 4.65x faster at prefill than TRITON_ATTN (A/B on rocm/vllm:latest + aiter main)", ROCm/aiter #5683, 2026-09-19, with the image check and AMD's reply of 2026-09-20 | https://github.com/ROCm/aiter/issues/5683 | 2026-10-10 | correctness |
| S74 | vendor doc | AMD SMI documentation, "AMD SMI CLI tool usage" (AMD SMI 27.1.0): `static --driver`, `static --asic` and `firmware` | https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-cli-tool.html | 2026-10-10 | ports |
| S75 | other | Ilan's Team Comms page (Ilan and Dominick), FYI of Sat 2:47 PM ET: the GPU pods moved to AMD's SJC hub under accounts C and D, quota is off there, and no pod had been launched there yet | https://app.notion.com/p/3f596ef5fec58163bbefd0e7ea5acea7 | 2026-10-10 | ports |

## Added by R32, Strix Halo hardware note (2026-10-10)

The session's network policy refused amd.com, rocm.docs.amd.com, docs.amd.com, llvm.org,
docs.kernel.org and gpuopen.com, so each documentation page below was read from its published
source on GitHub at the commit named: `ROCm/ROCm` builds rocm.docs.amd.com, `llvm/docs` builds
llvm.org/docs, and the kernel-doc comments in the Linux tree build docs.kernel.org/gpu/amdgpu.
"63fd839" is ROCm/ROCm develop at 63fd8390bf645ae3c0c715537025137127f892e3 (2026-09-01); "1a54676"
is llvm/llvm-project main at 1a54676b01047a16c355770b20e2f97d0945c764 (2026-10-10); "3857c2f" is
torvalds/linux at 3857c2fe5449541e24afc5efdb0f81a8a8f9a3a0 (the commit the telemetry note reads).

| Id | Kind | Source | Link or location | Opened | Cited in |
| --- | --- | --- | --- | --- | --- |
| L4 | our measurement | `lab/README.md`, section "Teacher on SJC today (pod C, Strix Halo)": gfx1151 build notes, the AOTriton finding, teacher TTFT, decode and watts on pod C, idle, and the T6 amd-smi sampler paragraph; merged in PR #194 (commit 498e161) | this repo, `lab/README.md` (commit 498e161) | 2026-10-10 | strix |
| L5 | our measurement | R30 task page, Hemdutt's agent's hour-0 facts table of Sat 5:40 PM (pod D beside R29's pod C: GPU, host RAM, kernel, HIP, torch, idle SOCKET_POWER reads) | https://app.notion.com/p/3f596ef5fec58118b653e4f3ae2d3024 | 2026-10-10 | strix |
| S74 | vendor doc (source) | AMD ROCm documentation source, "AMD GPU specifications", Ryzen APU table, row "AMD Ryzen AI Max+ PRO 395" (gfx1151) and its neighbors, `docs/reference/gpu-specs.rst` lines 793 to 900 at 63fd839 | https://github.com/ROCm/ROCm/blob/63fd8390bf645ae3c0c715537025137127f892e3/docs/reference/gpu-specs.rst | 2026-10-10 | strix |
| S75 | vendor doc (source) | AMD ROCm documentation source, "AMD RDNA3.5 system optimization" (memory settings, GTT and TTM, required Linux kernel, per-release support table), `docs/reference/system-optimization/rdna3-5.rst` at 63fd839 | https://github.com/ROCm/ROCm/blob/63fd8390bf645ae3c0c715537025137127f892e3/docs/reference/system-optimization/rdna3-5.rst | 2026-10-10 | strix |
| S76 | vendor doc (source) | AMD ROCm documentation source, "Data types and precision support", section "Matrix core support", `docs/reference/precision-support.rst` lines 589 to 797 at 63fd839 | https://github.com/ROCm/ROCm/blob/63fd8390bf645ae3c0c715537025137127f892e3/docs/reference/precision-support.rst | 2026-10-10 | strix |
| S77 | vendor doc (source) | AMD ROCm Core SDK release notes (development, ROCm 10.0.0), "Improved roofline support on gfx1150 (Strix Point), gfx1151 (Strix Halo and Gorgon Halo), and gfx1152", `docs/about/release-notes.md` line 178 at 63fd839 | https://github.com/ROCm/ROCm/blob/63fd8390bf645ae3c0c715537025137127f892e3/docs/about/release-notes.md | 2026-10-10 | strix |
| S78 | vendor doc | AMD ROCm release notes `RELEASE.md` at tags rocm-7.0.2 (hipBLAS 3.0.2 and rocBLAS 5.0.2 lines enabling gfx1151) and rocm-7.1.0 (line 811) | https://github.com/ROCm/ROCm/blob/rocm-7.0.2/RELEASE.md and https://github.com/ROCm/ROCm/blob/rocm-7.1.0/RELEASE.md | 2026-10-10 | strix |
| S79 | compiler doc (source) | LLVM, "User Guide for AMDGPU Backend": processor table (gfx1151 row), target features `cumode` and `wavefrontsize64`, "Memory Model GFX10-GFX11", function attributes, `llvm/docs/AMDGPUUsage.rst` at 1a54676 | https://github.com/llvm/llvm-project/blob/1a54676b01047a16c355770b20e2f97d0945c764/llvm/docs/AMDGPUUsage.rst | 2026-10-10 | strix |
| S80 | code (compiler) | LLVM `llvm/lib/Target/AMDGPU/AMDGPU.td` at 1a54676: `FeatureISAVersion11_5_Common`, `FeatureISAVersion11_5_1` (`Feature1536VGPRs`), GFX11 features (wave sizes, LDS, `FeatureLDSBankCount32`, `FeatureMaxWavesPerEU16`), WMMA and dot-instruction feature definitions | https://github.com/llvm/llvm-project/blob/1a54676b01047a16c355770b20e2f97d0945c764/llvm/lib/Target/AMDGPU/AMDGPU.td | 2026-10-10 | strix |
| S81 | code (compiler) | LLVM `llvm/lib/TargetParser/AMDGPUTargetParser.cpp` (total VGPRs, addressable SGPRs, LDS size) and `llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp` (VGPR granule, occupancy, SGPRs and occupancy) at 1a54676 | https://github.com/llvm/llvm-project/blob/1a54676b01047a16c355770b20e2f97d0945c764/llvm/lib/TargetParser/AMDGPUTargetParser.cpp and https://github.com/llvm/llvm-project/blob/1a54676b01047a16c355770b20e2f97d0945c764/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp | 2026-10-10 | strix |
| S82 | code (compiler) | LLVM `llvm/lib/Target/AMDGPU/VOP3PInstructions.td` at 1a54676 (WMMA instructions under `HasWMMA256bInsts`, fp8 and SWMMAC predicates, dot-instruction encodings) | https://github.com/llvm/llvm-project/blob/1a54676b01047a16c355770b20e2f97d0945c764/llvm/lib/Target/AMDGPU/VOP3PInstructions.td | 2026-10-10 | strix |
| S83 | code (compiler) | LLVM `llvm/lib/Target/AMDGPU/GCNSubtarget.cpp` (wave32 default for GFX10 and later) and clang `clang/lib/Basic/Targets/AMDGPU.cpp` (wavefront size) at 1a54676 | https://github.com/llvm/llvm-project/blob/1a54676b01047a16c355770b20e2f97d0945c764/llvm/lib/Target/AMDGPU/GCNSubtarget.cpp and https://github.com/llvm/llvm-project/blob/1a54676b01047a16c355770b20e2f97d0945c764/clang/lib/Basic/Targets/AMDGPU.cpp | 2026-10-10 | strix |
| S84 | vendor doc (driver source) | Linux `drivers/gpu/drm/amd/pm/amdgpu_pm.c` (hwmon power and clock documentation, attribute permissions, device attributes, power labels) and `drivers/gpu/drm/amd/pm/inc/amdgpu_pm.h` at 3857c2f; the source of docs.kernel.org "drm/amdgpu thermal and power" | https://github.com/torvalds/linux/blob/3857c2fe5449541e24afc5efdb0f81a8a8f9a3a0/drivers/gpu/drm/amd/pm/amdgpu_pm.c | 2026-10-10 | strix |
| S85 | vendor doc (driver source) | Linux `drivers/gpu/drm/ttm/ttm_tt.c` (`pages_limit`), `drivers/gpu/drm/ttm/ttm_device.c` (50% default), `drivers/gpu/drm/amd/amdgpu/amdgpu_drv.c` (`gttsize`, deprecated) and `amdgpu_gtt_mgr.c` (`mem_info_gtt_total`) at 3857c2f | https://github.com/torvalds/linux/blob/3857c2fe5449541e24afc5efdb0f81a8a8f9a3a0/drivers/gpu/drm/ttm/ttm_tt.c and https://github.com/torvalds/linux/blob/3857c2fe5449541e24afc5efdb0f81a8a8f9a3a0/drivers/gpu/drm/amd/amdgpu/amdgpu_drv.c | 2026-10-10 | strix |
| S86 | vendor doc (source) | Linux `Documentation/gpu/amdgpu/apu-asic-info-table.csv` line 18 ("Ryzen AI Max 300 series, Strix Halo", GC 11.5.1, MP1 14.0.1) at 3857c2f | https://github.com/torvalds/linux/blob/3857c2fe5449541e24afc5efdb0f81a8a8f9a3a0/Documentation/gpu/amdgpu/apu-asic-info-table.csv | 2026-10-10 | strix |
| S87 | vendor doc (source) | AMD HIP environment-variable references (`projects/hip/docs/reference/env_variables/debug_hip_env.rst`, `memory_management_hip_env.rst`, `gpu_isolation_hip_env.rst`) and ROCr `projects/rocr-runtime/runtime/docs/data/env_variables.rst` in ROCm/rocm-systems at 71a246ac15a1c010d0c80de568c05eb206687f46 | https://github.com/ROCm/rocm-systems/tree/71a246ac15a1c010d0c80de568c05eb206687f46/projects/hip/docs/reference/env_variables | 2026-10-10 | strix |
| S88 | vendor doc (source) | AMD SMI documentation in ROCm/rocm-systems at 71a246a: `projects/amdsmi/docs/how-to/amdsmi-cli-tool.md` (`--gtt`, `--mem-carveout`, prerequisites, lines 1740 to 1752) and `projects/amdsmi/docs/reference/amdsmi-py-api.md` line 3962 (`apu_metrics.*` on gfx1151) | https://github.com/ROCm/rocm-systems/blob/71a246ac15a1c010d0c80de568c05eb206687f46/projects/amdsmi/docs/how-to/amdsmi-cli-tool.md | 2026-10-10 | strix |
| S89 | vendor doc and code | ROCm/rocm-libraries at 1bc5695a91428822970389ef006c277c31683721: rocBLAS `projects/rocblas/docs/reference/env-variables.rst` (`ROCBLAS_USE_HIPBLASLT`) and `projects/rocblas/CMakeLists.txt` line 90 (gfx1151 in the ROCm 7.0 target list); hipBLASLt `Tensile/Logic/asm_full/gfx1151` directory (also at ROCm/hipBLASLt tags rocm-7.0.2 and rocm-7.2.0) | https://github.com/ROCm/rocm-libraries/tree/1bc5695a91428822970389ef006c277c31683721/projects/rocblas | 2026-10-10 | strix |
| S90 | code | PyTorch at 012dcaf4836ab311701c841f6ff7d1354aba52ca: `aten/src/ATen/cuda/detail/CUDAHooks.cpp` (hipBLASLt supported and preferred architectures, lines 555 to 572) and `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp` (gfx11 head-dimension cap, experimental flag) | https://github.com/pytorch/pytorch/blob/012dcaf4836ab311701c841f6ff7d1354aba52ca/aten/src/ATen/cuda/detail/CUDAHooks.cpp | 2026-10-10 | strix |
| S91 | code | ROCm/aotriton at b5e8cfb909e554c62081f56055b11d6ba3fd6bfa: `v3src/hip-dependent/util.cc` lines 97 to 103 (experimental targets) and `python/gpu_targets.py` (gfx1151 mapped to gfx1100 tuning) | https://github.com/ROCm/aotriton/blob/b5e8cfb909e554c62081f56055b11d6ba3fd6bfa/v3src/hip-dependent/util.cc | 2026-10-10 | strix |
| S92 | code | triton-lang/triton at 3078daa76bbd55c1c3509f7ee8e899b1ed6a127e: `third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp` (GFX11 is the RDNA3 family) and `third_party/amd/backend/compiler.py` line 174 (gfx115 in-thread transpose) | https://github.com/triton-lang/triton/blob/3078daa76bbd55c1c3509f7ee8e899b1ed6a127e/third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp | 2026-10-10 | strix |
| S93 | vendor doc | ROCm/TheRock at 74744339975d50c76a942aad3e090c4d95116af6: `SUPPORTED_GPUS.md` line 63 (gfx1151 build passing, sanity tested, release ready) and `RELEASES.md` (device-gfx1151 wheels) | https://github.com/ROCm/TheRock/blob/74744339975d50c76a942aad3e090c4d95116af6/SUPPORTED_GPUS.md | 2026-10-10 | strix |

Searched for R32 but not opened (refused by the session's network policy, HTTP 403 at the proxy or
DNS failure): AMD's Ryzen AI Max+ 395 product page
(https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html),
not opened; the RDNA 3.5 ISA reference guide on docs.amd.com, not opened; GPUOpen's WMMA on RDNA 3
article and RDNA performance guide, not opened; the rendered rocm.docs.amd.com, llvm.org and
docs.kernel.org pages, not opened (their sources are S74 to S88); and the techpowerup.com,
chipsandcheese.com, notebookcheck.net and phoronix.com pages on the Radeon 8060S, not opened.
## Added by R35, the gfx1151 inference settings note (2026-10-10)

Note `settings_inference_gfx1151.md` (settings) also cites S06, S08, S19, S25, S28, S58, S59, S61 and S65
above. Every file below was read as a raw file at the commit in its link; rocm.docs.amd.com,
docs.vllm.ai, docs.pytorch.org and docs.kernel.org were refused by the session's egress proxy and are
not cited. "v2.9.1" is the PyTorch tag (d38164a545b4), "v3.5.1" the Triton tag (0add68262ab0), "v6.17"
the Linux tag (e5f0a698b34e), and 71a246ac is ROCm rocm-systems develop on 2026-10-10.

| Id | Kind | Source | Link or location | Opened | Cited in |
| --- | --- | --- | --- | --- | --- |
| S76 | code | vLLM `vllm/engine/arg_utils.py` at b1388b1 (`EngineArgs.get_batch_defaults`, `--enforce-eager`) | https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/engine/arg_utils.py | 2026-10-10 | settings |
| S77 | code | vLLM `vllm/config/cache.py` at b1388b1 (`CacheDType`, `MambaDType`, `DEFAULT_BLOCK_SIZE`, `gpu_memory_utilization`, `enable_prefix_caching`, `mamba_cache_mode`) | https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/config/cache.py | 2026-10-10 | settings |
| S78 | code | vLLM `vllm/config/compilation.py` at b1388b1 (`CompilationConfig.cudagraph_mode`, `max_cudagraph_capture_size`) | https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/config/compilation.py | 2026-10-10 | settings |
| S79 | code | vLLM `vllm/v1/attention/backends/rocm_attn.py` and `triton_attn.py` at b1388b1 (`supported_kv_cache_dtypes`, `get_supported_kernel_block_sizes`, `do_kv_cache_update`) | https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/attention/backends/rocm_attn.py and https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/attention/backends/triton_attn.py | 2026-10-10 | settings |
| S80 | code | vLLM `vllm/_aiter_ops.py` at b1388b1 (`is_aiter_found_and_supported`) | https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/_aiter_ops.py | 2026-10-10 | settings |
| S81 | code | vLLM `vllm/envs.py` at b1388b1 (`VLLM_DISABLED_KERNELS`, `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE`, `VLLM_ROCM_USE_AITER`, `VLLM_ROCM_USE_SKINNY_GEMM`) | https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/envs.py | 2026-10-10 | settings |
| S82 | code | ROCm `rocm-systems` `projects/clr/rocclr/utils/flags.hpp` at 71a246ac (HIP runtime release flags: `GPU_MAX_HW_QUEUES`, `HIP_FORCE_DEV_KERNARG`, `AMD_SERIALIZE_KERNEL`, `HIP_LAUNCH_BLOCKING`, `GPU_SINGLE_ALLOC_PERCENT`, `ROC_AQL_QUEUE_SIZE`, `DEBUG_HIP_FORCE_GRAPH_QUEUES`) | https://github.com/ROCm/rocm-systems/blob/71a246ac15a1c010d0c80de568c05eb206687f46/projects/clr/rocclr/utils/flags.hpp | 2026-10-10 | settings |
| S83 | code | ROCm `rocm-systems` `projects/rocr-runtime/runtime/hsa-runtime/core/util/flag.h` at 71a246ac (`Flag::Refresh`: `HSA_ENABLE_SDMA`, `HSA_ENABLE_INTERRUPT`, `HSA_XNACK`, `HSA_NO_SCRATCH_RECLAIM`, `HSA_MAX_QUEUES`, `HSA_SCRATCH_SINGLE_LIMIT`) | https://github.com/ROCm/rocm-systems/blob/71a246ac15a1c010d0c80de568c05eb206687f46/projects/rocr-runtime/runtime/hsa-runtime/core/util/flag.h | 2026-10-10 | settings |
| S84 | code | PyTorch v2.9.1 `aten/src/ATen/Context.cpp` (`Context::blasPreferredBackend`) and `aten/src/ATen/Context.h` (`blas_preferred_backend`, `TORCH_BLAS_PREFER_HIPBLASLT`) | https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/aten/src/ATen/Context.cpp and https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/aten/src/ATen/Context.h | 2026-10-10 | settings |
| S85 | project doc | PyTorch v2.9.1 `aten/src/ATen/cuda/tunable/README.md` (TunableOp environment variable table) | https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/aten/src/ATen/cuda/tunable/README.md | 2026-10-10 | settings |
| S86 | code | PyTorch v2.9.1 `c10/core/AllocatorConfig.cpp` (`AcceleratorAllocatorConfig`, `PYTORCH_ALLOC_CONF` and its deprecated aliases) | https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/c10/core/AllocatorConfig.cpp | 2026-10-10 | settings |
| S87 | code | Triton v3.5.1 `python/triton/knobs.py` (`cache_knobs`, `autotuning_knobs`, `amd_knobs`) and `third_party/amd/backend/compiler.py` (`is_pingpong_schedule_enabled`, `get_arg_specialization`, `make_ttgir`) | https://github.com/triton-lang/triton/blob/0add68262ab0a2e33b84524346cb27cbb2787356/python/triton/knobs.py and https://github.com/triton-lang/triton/blob/0add68262ab0a2e33b84524346cb27cbb2787356/third_party/amd/backend/compiler.py | 2026-10-10 | settings |
| S88 | code | Linux v6.17 `drivers/gpu/drm/amd/pm/amdgpu_pm.c` (DOC blocks for `power_dpm_force_performance_level`, `pp_dpm_sclk`, `pp_power_profile_mode`, `gpu_metrics`, hwmon) and `drivers/gpu/drm/amd/pm/inc/amdgpu_pm.h` (`AMDGPU_DEVICE_ATTR_RW`) | https://github.com/torvalds/linux/blob/e5f0a698b34ed76002dc5cff3804a61c80233a7a/drivers/gpu/drm/amd/pm/amdgpu_pm.c and https://github.com/torvalds/linux/blob/e5f0a698b34ed76002dc5cff3804a61c80233a7a/drivers/gpu/drm/amd/pm/inc/amdgpu_pm.h | 2026-10-10 | settings |
| S89 | code | vLLM `vllm/model_executor/layers/mamba/gdn_linear_attn.py` at b1388b1 (`_forward_core`, `_forward_core_decode_non_spec`) | https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/mamba/gdn_linear_attn.py | 2026-10-10 | settings |
| S90 | code | vLLM `vllm/model_executor/layers/utils.py` at b1388b1 (`rocm_unquantized_gemm_impl`) | https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/utils.py | 2026-10-10 | settings |
## Added for the gfx1151 training settings sheet (R36, 2026-10-10)

Cited by `settings_training_gfx1151.md` (short name "settings"). docs.pytorch.org, rocm.docs.amd.com and
huggingface.co did not resolve from the session that wrote the note, so the rendered pages were read from the
source files they are generated from, at the ref named in each link. "main" refs were read on 2026-10-10 at
transformers 536ecc007387a50e77603bb5d92100e9b07514cc, TRL 72cf37df714d3183e8e96ec391d474b26eaa30b6,
flash-linear-attention f3bec720f7847d4f176872b34f002cc580e2cc6e, Ax f4e58e05f9e888cc70da28ef02df9e755b38f985,
ROCm/rocm-libraries develop 1bc5695a91428822970389ef006c277c31683721 and ROCm/rocm-systems develop
71a246ac15a1c010d0c80de568c05eb206687f46; PyTorch v2.9.1 is d38164a545b4a4e4e0cf73ce67173f70574890b6.

| Id | Kind | Source | Link or location | Opened | Cited in |
| --- | --- | --- | --- | --- | --- |
| L6 | our files | `lab/smoke_lora.py` (the training step in use today: bf16 model, PEFT LoRA r 16 on all-linear, TRL `SFTConfig` with batch 4, accumulation 1, `max_length` 256, two DataLoader workers, `torch_compile=False`) and `lab/pod_b_trainer.sh` (torch, torchvision and triton come from the pod image and are never reinstalled) | this repo, `lab/smoke_lora.py`, `lab/pod_b_trainer.sh` (main 722d845) | 2026-10-10 | settings |
| L7 | our measurement | The smoke LoRA on one R9700 (gfx1201, another chip): 0.51 samples/s, 8.7 GB peak, loss 2.23 to 0.16, as recorded in the learning note | `lab/refine/research/learning/lora_recipe.md` section 4, from PR #37 | 2026-10-10 | settings |
| S94 | code | PyTorch v2.9.1 `aten/src/ATen/Context.cpp`, `Context::blasPreferredBackend` (hipBLASLt preferred and supported arch lists) and `Context::allowTF32CuBLAS` (`HIPBLASLT_ALLOW_TF32`) | https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/Context.cpp | 2026-10-10 | settings |
| S95 | code | PyTorch v2.10.0 `aten/src/ATen/Context.cpp` (hipBLASLt supported list adds gfx1150 and gfx1151 under ROCm 7.0) and main `aten/src/ATen/cuda/detail/CUDAHooks.cpp` (gfx1151 preferred from ROCm 7.13) | https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/Context.cpp | 2026-10-10 | settings |
| S96 | project doc | PyTorch v2.9.1 `docs/source/notes/hip.rst`, "HIP (ROCm) semantics": TF32 on ROCm, hipBLAS workspaces, hipFFT plan cache, `PYTORCH_NO_HIP_MEMORY_CACHING` | https://github.com/pytorch/pytorch/blob/v2.9.1/docs/source/notes/hip.rst | 2026-10-10 | settings |
| S97 | project doc | PyTorch v2.9.1 `docs/source/notes/cuda.rst`, "Memory management" (allocator options) and "Reduced Precision Reduction in BF16 GEMMs" | https://github.com/pytorch/pytorch/blob/v2.9.1/docs/source/notes/cuda.rst | 2026-10-10 | settings |
| S98 | code | PyTorch v2.9.1 `c10/core/AllocatorConfig.cpp` (read order `PYTORCH_ALLOC_CONF`, then the deprecated CUDA and HIP names) and `c10/cuda/CUDACachingAllocator.cpp` (`ExpandableSegment` compiled only when `!defined(USE_ROCM)`, otherwise "expandable segment not supported") | https://github.com/pytorch/pytorch/blob/v2.9.1/c10/cuda/CUDACachingAllocator.cpp | 2026-10-10 | settings |
| S99 | project doc | PyTorch v2.9.1 `aten/src/ATen/cuda/tunable/README.md` (TunableOp environment variables and `torch.cuda.tunable` API) | https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/cuda/tunable/README.md | 2026-10-10 | settings |
| S100 | code | PyTorch v2.9.1 `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp` (`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` gate, head_dim 256 limit on gfx11 with AOTriton 0.11) and `cmake/External/aotriton.cmake` (AOTriton 0.11b) | https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp | 2026-10-10 | settings |
| S101 | code | ROCm/aotriton 0.11b `v2src/util.cc::isArchExperimentallySupported` (gfx950, gfx1151, gfx1201) | https://github.com/ROCm/aotriton/blob/0.11b/v2src/util.cc | 2026-10-10 | settings |
| S102 | issue | "[Issue]: PyTorch Flash Attention with gfx1151", ROCm/TheRock #1364, opened 2025-08-30, closed 2026-01-08 | https://github.com/ROCm/TheRock/issues/1364 | 2026-10-10 | settings |
| S103 | code | PyTorch v2.9.1 `torch/backends/cuda/__init__.py` (`preferred_blas_library`, `preferred_rocm_fa_library`, `enable_*_sdp`) | https://github.com/pytorch/pytorch/blob/v2.9.1/torch/backends/cuda/__init__.py | 2026-10-10 | settings |
| S104 | code | PyTorch v2.9.1 `aten/src/ATen/native/Convolution.cpp` (`use_miopen`, 1-D input viewed as 2-D, depthwise to `MiopenDepthwise`) | https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/Convolution.cpp | 2026-10-10 | settings |
| S105 | project doc | MIOpen `docs/how-to/find-and-immediate.rst`, "Find modes" | https://github.com/ROCm/rocm-libraries/blob/1bc5695a91428822970389ef006c277c31683721/projects/miopen/docs/how-to/find-and-immediate.rst | 2026-10-10 | settings |
| S106 | project doc | MIOpen `docs/conceptual/tuningdb.rst`, `perfdb.rst` and `cache.rst` (user database, `MIOPEN_FIND_ENFORCE`, kernel cache) | https://github.com/ROCm/rocm-libraries/blob/1bc5695a91428822970389ef006c277c31683721/projects/miopen/docs/conceptual/tuningdb.rst | 2026-10-10 | settings |
| S107 | project doc | ROCr runtime `runtime/docs/data/env_variables.rst` (`HSA_XNACK`, `HSA_ENABLE_SDMA`, scratch reclaim) | https://github.com/ROCm/rocm-systems/blob/71a246ac15a1c010d0c80de568c05eb206687f46/projects/rocr-runtime/runtime/docs/data/env_variables.rst | 2026-10-10 | settings |
| S108 | project doc | HIP `docs/reference/env_variables/debug_hip_env.rst` (`AMD_SERIALIZE_KERNEL`) and `memory_management_hip_env.rst` (`GPU_MAX_HEAP_SIZE`, `GPU_SINGLE_ALLOC_PERCENT`), with `clr/rocclr/utils/flags.hpp` (`HIP_FORCE_DEV_KERNARG` default true) | https://github.com/ROCm/rocm-systems/blob/71a246ac15a1c010d0c80de568c05eb206687f46/projects/hip/docs/reference/env_variables/memory_management_hip_env.rst | 2026-10-10 | settings |
| S109 | vendor doc | ROCm 7.2.0 `docs/how-to/system-optimization/strixhalo.rst`, "Memory settings" and "Configuring shared memory limits on linux" | https://github.com/ROCm/ROCm/blob/docs/7.2.0/docs/how-to/system-optimization/strixhalo.rst | 2026-10-10 | settings |
| S110 | code | PyTorch main `torch/utils/data/dataloader.py` (DataLoader docstring) and `docs/source/data.md`, "Memory Pinning" | https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataloader.py | 2026-10-10 | settings |
| S111 | code | transformers `src/transformers/training_args.py` (dataloader, gradient checkpointing, accumulation, `train_sampling_strategy`, bf16, `optim` default, `torch_compile`) | https://github.com/huggingface/transformers/blob/536ecc007387a50e77603bb5d92100e9b07514cc/src/transformers/training_args.py | 2026-10-10 | settings |
| S112 | code | transformers `src/transformers/modeling_utils.py::gradient_checkpointing_enable` (`use_reentrant` False by default) and the `from_pretrained` `attn_implementation` docstring | https://github.com/huggingface/transformers/blob/536ecc007387a50e77603bb5d92100e9b07514cc/src/transformers/modeling_utils.py | 2026-10-10 | settings |
| S113 | code | TRL `trl/trainer/sft_config.py` (defaults: `gradient_checkpointing` True, `max_length` 1024, `packing`, `packing_strategy` "bfd", `padding_free`) and `trl/trainer/sft_trainer.py` (flash-attention variants, cross-contamination warning) | https://github.com/huggingface/trl/blob/72cf37df714d3183e8e96ec391d474b26eaa30b6/trl/trainer/sft_config.py | 2026-10-10 | settings |
| S114 | project doc | TRL `docs/source/reducing_memory_usage.md`, "Packing" and "Padding-free" | https://github.com/huggingface/trl/blob/72cf37df714d3183e8e96ec391d474b26eaa30b6/docs/source/reducing_memory_usage.md | 2026-10-10 | settings |
| S115 | project doc | transformers v4.57.1 `docs/source/en/perf_train_gpu_one.md`, "Methods and tools for efficient training on a single GPU" (on main it redirects to `optimization_overview`) | https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/perf_train_gpu_one.md | 2026-10-10 | settings |
| S116 | code | PyTorch main `torch/optim/optimizer.py` (`_foreach_doc`, `_fused_doc`) and `torch/utils/_foreach_utils.py::_get_fused_kernels_supported_devices` | https://github.com/pytorch/pytorch/blob/main/torch/optim/optimizer.py | 2026-10-10 | settings |
| S117 | project doc | PEFT `docs/source/developer_guides/torch_compile.md` | https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/torch_compile.md | 2026-10-10 | settings |
| S118 | issue | "HIP: invalid device ordinal on gfx1151 (AMD Strix Halo iGPU) in subprocess", triton-lang/triton #10036 (open) | https://github.com/triton-lang/triton/issues/10036 | 2026-10-10 | settings |
| S119 | code | fla-org flash-linear-attention `README.md` (ROCm install, platform note) and `fla/ops/gated_delta_rule/chunk.py` (`chunk_gated_delta_rule`) | https://github.com/fla-org/flash-linear-attention/blob/f3bec720f7847d4f176872b34f002cc580e2cc6e/README.md | 2026-10-10 | settings |
| S120 | issue | "Qwen3.5 gated-delta fast path has pathological first-run latency" on a Radeon 890M, fla-org/flash-linear-attention #804 (closed 2026-04-17) | https://github.com/fla-org/flash-linear-attention/issues/804 | 2026-10-10 | settings |
| S121 | code | transformers `src/transformers/models/qwen3_5/modeling_qwen3_5.py` (`torch_chunk_gated_delta_rule`, `causal_conv1d_fn`, `Qwen3_5GatedDeltaNet.forward`) and `configuration_qwen3_5.py` (`head_dim` 256, linear-attention head sizes) | https://github.com/huggingface/transformers/blob/536ecc007387a50e77603bb5d92100e9b07514cc/src/transformers/models/qwen3_5/modeling_qwen3_5.py | 2026-10-10 | settings |
| S122 | code | transformers `src/transformers/integrations/hub_kernels.py` (`use_kernel_func_from_hub_with_fallback`, the `Qwen3_5GatedDeltaNet` ROCm gfx1151 layer mapping) | https://github.com/huggingface/transformers/blob/536ecc007387a50e77603bb5d92100e9b07514cc/src/transformers/integrations/hub_kernels.py | 2026-10-10 | settings |
| S123 | issue | "ROCm support", Dao-AILab/causal-conv1d #99 (closed; maintainer: works on ROCm with `--no-build-isolation`, tested on gfx1201) | https://github.com/Dao-AILab/causal-conv1d/issues/99 | 2026-10-10 | settings |
| S124 | issue | huggingface/peft #3718 (open; title paraphrased here): a Qwen3.8-27B LoRA run where "Nothing in the logs said which delta-rule implementation was in use" | https://github.com/huggingface/peft/issues/3718 | 2026-10-10 | settings |
| S125 | code | Ax `ax/api/configs.py` (`RangeParameterConfig`, `ChoiceParameterConfig`, `DerivedParameterConfig`) | https://github.com/facebook/Ax/blob/f4e58e05f9e888cc70da28ef02df9e755b38f985/ax/api/configs.py | 2026-10-10 | settings |
| S126 | code | PyTorch main `torch/utils/checkpoint.py` (`use_reentrant` guidance) | https://github.com/pytorch/pytorch/blob/main/torch/utils/checkpoint.py | 2026-10-10 | settings |
| S127 | code | transformers `src/transformers/data/data_collator.py::DataCollatorWithFlattening` | https://github.com/huggingface/transformers/blob/536ecc007387a50e77603bb5d92100e9b07514cc/src/transformers/data/data_collator.py | 2026-10-10 | settings |

Opened and set aside for this note: pytorch/pytorch #173367 ("Segfault on AMD Strix Halo (Radeon 8060S) with PyTorch ROCm 7.1 / Fedora 43"), title only from a search result, not opened; AMD's Triton install page for Radeon (rocm.docs.amd.com/projects/radeon), search result only, not opened; the Framework community thread on flash attention for Strix Halo, not opened (the host did not resolve).
