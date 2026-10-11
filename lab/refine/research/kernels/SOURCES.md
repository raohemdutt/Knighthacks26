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
| `graphics_pipeline_for_ml.md` (graphics) | Graphics-pipeline techniques (Brook, texture units, block-compressed weights, Vulkan engines) for inference and fine-tuning on the Strix Halo, with the G1 to G3 proposals |
| `energy_model_gfx1151.md` (energy) | The joules-per-token model for the gfx1151 teacher and student from the chip sheet, its knees, its check against R26's measured frontier, the measurement protocol and the optimizer's objective and constraints |

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
| S13 | PR | mgehre-amd, "[ROCm][Kernel] Add HybridW4A16LinearKernel: Triton prefill + HIP skinny decode", vLLM #40977, merged 2026-07-14 | https://github.com/vllm-project/vllm/pull/40977 | 2026-10-10 | main, correctness, graphics |
| S16 | PR | tangzzycc, "[ROCm][Perf] Enable medium-skinny dispatch in RDNAHybridW4A16LinearKernel", vLLM #52619, merged 2026-10-07 | https://github.com/vllm-project/vllm/pull/52619 | 2026-10-10 | main, triton, correctness |
| S17 | PR | mgehre-amd, "[ROCm][Perf] W4A16: keep skinny GEMM zero-points packed 4-bit", vLLM #54965, merged 2026-09-14 | https://github.com/vllm-project/vllm/pull/54965 | 2026-10-10 | main, correctness |
| S18 | PR | mgehre-amd, "[ROCm][Perf] W4A16: pad gfx11 weight and activation strides", vLLM #56301, merged 2026-10-06 | https://github.com/vllm-project/vllm/pull/56301 | 2026-10-10 | main, correctness, graphics |
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


## Added by R48, graphics pipeline note (2026-10-10)

Every source below was opened on 2026-10-10 for `graphics_pipeline_for_ml.md` (short name
"graphics"). Sources published before 2024 are marked background in the Kind column. S13 and S18
above were re-opened for this note. "23b0202" is llama.cpp master at
23b0202a189c44a54625aadcb37a946dd1d6278d (2026-10-10), "bffd919" is Mesa main at
bffd91947efc1932da99aca69fc540d6d4201b0c (2026-10-10), and "71a246a" is ROCm/rocm-systems at
71a246ac15a1c010d0c80de568c05eb206687f46 (2026-10-10).

| Id | Kind | Source | Link or location | Opened | Cited in |
| --- | --- | --- | --- | --- | --- |
| L8 | our measurement | `lab/refine/metrics.jsonl` lines 24, 29 and 34: R29 on pod C (Radeon 8060S, gfx1151), vLLM b1388b1 with `lab/vllm-gfx1201.diff` built for gfx1151, `RedHatAI/Qwen3.8-27B-INT4`, batch 1: decode 2.88 tok/s, 120.3 W average over decode, engine start 297 s | this repo, `lab/refine/metrics.jsonl` | 2026-10-10 | graphics |
| L9 | our page | T0 task page, queue lines E19 (RDNA hybrid W4A16 kernel against Conch, Sat 3:18 PM ET) and E20 (vLLM against llama.cpp, a reference, Sat 3:28 PM ET), and the merge 2 line of Sat 4:39 PM ET (next free number E22) | https://app.notion.com/p/3f596ef5fec581819588ec7a702a4bb1 | 2026-10-10 | graphics |
| S128 | paper (background, 2004) | Ian Buck, Tim Foley, Daniel Horn, Jeremy Sugerman, Kayvon Fatahalian, Mike Houston, Pat Hanrahan, "Brook for GPUs: Stream Computing on Graphics Hardware", SIGGRAPH 2004, ACM Transactions on Graphics 23(3), pp. 777-786 | https://graphics.stanford.edu/papers/brookgpu/brookgpu.pdf (doi:10.1145/1015706.1015800) | 2026-10-10 | graphics |
| S129 | paper (background, 2005) | John D. Owens, David Luebke, Naga Govindaraju, Mark Harris, Jens Krueger, Aaron E. Lefohn, Timothy J. Purcell, "A Survey of General-Purpose Computation on Graphics Hardware", Eurographics 2005 State of the Art Reports, pp. 21-51 (journal version: Computer Graphics Forum 26(1), 2007) | https://research.nvidia.com/sites/default/files/pubs/2005-08_A-Survey-of/ASurveyofGeneralPurposeComputationonGraphicsHardware.pdf | 2026-10-10 | graphics |
| S130 | encyclopedia (background) | Wikipedia, "Close to Metal" (ATI's CTM, the AMD Stream SDK of December 2007, Brook+ as its open-source language, the later switch to OpenCL); opened, not cited (secondary) | https://en.wikipedia.org/wiki/Close_to_Metal | 2026-10-10 | not cited |
| S131 | vendor doc | AMD, HIP 7.16.0 documentation, "Texture fetching" (texture objects, filter modes, address modes, texture cache) | https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/device_memory/texture_fetching.html | 2026-10-10 | graphics |
| S132 | vendor doc | AMD, ROCm Compute Profiler 3.6.0 documentation (ROCm 7.13.0 preview), "Workgroup processor (WGP)" for RDNA 3.5 (TCP as the L0 vector cache, VMEM counting global, buffer and texture operations) | https://rocm.docs.amd.com/projects/rocprofiler-compute/en/7.13.0-preview/conceptual/rdna/wgp.html | 2026-10-10 | graphics |
| S133 | vendor doc | Microsoft Learn, "Block Compression (Direct3D 10)" (BC1 to BC5 layouts, BC4_UNORM and BC4_SNORM decode rules), page updated 2025-04-15 | https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression | 2026-10-10 | graphics |
| S134 | blog (background, 2021) | The ryg blog (fgiesen), "GPU BCn decoding", 2021-10-04 (vendor decoder differences, BC4 and BC5 decode precision) | https://fgiesen.wordpress.com/2021/10/04/gpu-bcn-decoding/ | 2026-10-10 | graphics |
| S135 | paper | Apple, "Apple Intelligence Foundation Language Models: Tech Report 2025", arXiv v3 2025-08-27 (section 6 on ASTC weight compression and quality recovery adapters, Table 3) | arXiv:2507.13575 (https://arxiv.org/abs/2507.13575) | 2026-10-10 | graphics |
| S136 | code | Mesa RADV at bffd919: `src/amd/vulkan/radv_physical_device.c` (`radv_cooperative_matrix_enabled`, `radv_bfloat16_enabled`, extension table, `textureCompressionBC`, `textureCompressionASTC_LDR`, `emulate_astc`) and `src/amd/vulkan/radv_image.c` (`radv_image_get_plane_format`) | https://gitlab.freedesktop.org/mesa/mesa/-/raw/bffd91947efc1932da99aca69fc540d6d4201b0c/src/amd/vulkan/radv_physical_device.c and https://gitlab.freedesktop.org/mesa/mesa/-/raw/bffd91947efc1932da99aca69fc540d6d4201b0c/src/amd/vulkan/radv_image.c | 2026-10-10 | graphics |
| S137 | code | ROCm/rocm-systems at 71a246a: `projects/hip/include/hip/driver_types.h` (block-compressed `hipResourceViewFormat` values) and `projects/clr/hipamd/src/hip_conversions.hpp` (`getNumChannels`, `getArrayFormat`) | https://github.com/ROCm/rocm-systems/blob/71a246ac15a1c010d0c80de568c05eb206687f46/projects/clr/hipamd/src/hip_conversions.hpp and https://github.com/ROCm/rocm-systems/blob/71a246ac15a1c010d0c80de568c05eb206687f46/projects/hip/include/hip/driver_types.h | 2026-10-10 | graphics |
| S138 | PR | 0cc4m, "vulkan: int8 coopmat1 matmul implementation for AMD RDNA3 and RDNA4", ggml-org/llama.cpp #27952, opened 2026-08-29, merged 2026-09-24; description benchmarks (Radeon 8060S) and comments by huppiflupp of 2026-09-18 and 2026-09-20 | https://github.com/ggml-org/llama.cpp/pull/27952 | 2026-10-10 | graphics |
| S139 | issue | Ankk98, "Feature Request: Vulkan: retune coopmat/MMQ tiles for RDNA3.5 / Strix Halo prefill", ggml-org/llama.cpp #28843, 2026-09-13 (Radeon 8050S measurements and path checks) | https://github.com/ggml-org/llama.cpp/issues/28843 | 2026-10-10 | graphics |
| S140 | repository | nabe2030, `hip-vs-vulkan-evo-x2` README, "llama.cpp HIP vs Vulkan benchmark on AMD Strix Halo (gfx1151) with ROCm 7.2.2 - Ubuntu 26.04" (2026; setup and results tables) | https://github.com/nabe2030/hip-vs-vulkan-evo-x2 | 2026-10-10 | graphics |
| S141 | discussion | realugbun, "Known-Good Strix Halo ROCm + llama.cpp Stack", ggml-org/llama.cpp discussion #20856, 2026-03-22, update of 2026-07-23 and comment of 2026-07-31 | https://github.com/ggml-org/llama.cpp/discussions/20856 | 2026-10-10 | graphics |
| S142 | PR | 0cc4m, "vulkan: use graphics queue on AMD", ggml-org/llama.cpp #20551, merged 2026-03-15 (benchmark tables); and "vulkan: allow graphics queue only through env var", #20599, merged 2026-03-17 (comments of 2026-03-15 and 2026-03-17) | https://github.com/ggml-org/llama.cpp/pull/20551 and https://github.com/ggml-org/llama.cpp/pull/20599 | 2026-10-10 | graphics |
| S143 | code | llama.cpp `ggml/src/ggml-vulkan/ggml-vulkan.cpp` at 23b0202 (queue selection and `GGML_VK_ALLOW_GRAPHICS_QUEUE`, the `GGML_VK_DISABLE_*` switches, UMA handling in `ggml_vk_tensor_subbuffer` and `ggml_vk_mul_mat_q_f16`, training operators in `ggml_vk_op_get_pipeline`) | https://github.com/ggml-org/llama.cpp/blob/23b0202a189c44a54625aadcb37a946dd1d6278d/ggml/src/ggml-vulkan/ggml-vulkan.cpp | 2026-10-10 | graphics |
| S144 | code | llama.cpp `ggml/src/ggml-opencl/kernels/gemv_noshuffle_q8_0_f32.cl` at 23b0202 (`kernel_gemv_noshuffle_q8_0_f32`, weights and activations read through `image1d_buffer_t`) | https://github.com/ggml-org/llama.cpp/blob/23b0202a189c44a54625aadcb37a946dd1d6278d/ggml/src/ggml-opencl/kernels/gemv_noshuffle_q8_0_f32.cl | 2026-10-10 | graphics |
| S145 | vendor talk | Hongqiang Wang (Qualcomm), "Introduction to the Qualcomm Adreno Optimized OpenCL backend for Llama.cpp", IWOCL 2025, Heidelberg (slides 11, 13 and 16) | https://www.iwocl.org/wp-content/uploads/iwocl-2025-hongqiang-wang-lamacpp-backend-update.pdf | 2026-10-10 | graphics |
| S146 | paper | Zhaode Wang, Jingbang Yang, Xinyu Qian, Shiwen Xing, Xiaotang Jiang, Chengfei Lv, Shengyu Zhang, "MNN-LLM: A Generic Inference Engine for Fast Large Language Model Deployment on Mobile Devices", arXiv 2025-06-12 (section 5.1, Image objects for weights on Adreno) | arXiv:2506.10443 (https://arxiv.org/abs/2506.10443) | 2026-10-10 | graphics |
| S147 | paper | Jiuqiang Tang, Raman Sarokin, Ekaterina Ignasheva, Grant Jensen, Juhyun Lee, Andrei Kulik, Matthias Grundmann, "Scaling On-Device GPU Inference for Large Generative Models" (ML Drift), CVPR 2025 workshops, arXiv 2025-05-01 (section 3.2, tensor virtualization) | arXiv:2505.00232 (https://arxiv.org/abs/2505.00232) | 2026-10-10 | graphics |
| S148 | vendor doc | Khronos Vulkan Working Group, "Vulkan: Continuing to Forge Ahead" (SIGGRAPH 2025 update), 2025-08-05 (Vulkan ML bfloat16 and float8 extensions; llama.cpp's Vulkan extensions) | https://www.khronos.org/blog/vulkan-continuing-to-forge-ahead-siggraph-2025 | 2026-10-10 | graphics |
| S149 | project doc | Mesa 25.2.0 release notes, 2025-08-06, "New features" (VK_KHR_shader_bfloat16 on radv/gfx12+, VK_KHR_shader_float8 on RADV RDNA4+) | https://docs.mesa3d.org/relnotes/25.2.0.html | 2026-10-10 | graphics |
| S150 | project doc (background, 2023) | Mesa 23.3.0 release notes, 2023-11-29, "New features" (VK_KHR_cooperative_matrix on RADV/GFX11+); opened, not cited (S136 states the same from source) | https://docs.mesa3d.org/relnotes/23.3.0.html | 2026-10-10 | not cited |
| S151 | project doc | Mesa documentation, "RADV" (ACO as the default shader compiler, the LLVM backend for testing and bring-up) | https://docs.mesa3d.org/drivers/radv.html | 2026-10-10 | graphics |
| S152 | project doc | GPUOpen-Drivers/AMDVLK, discussion #416 "AMDVLK open-source project is discontinued" (jinjianrong, 2025-09-15) and README notice | https://github.com/GPUOpen-Drivers/AMDVLK/discussions/416 | 2026-10-10 | graphics |
| S153 | project doc | microsoft/DirectML README ("DirectML is in maintenance mode") and the commit "Update README to indicate DML is in maintenance mode (#710)" of 2025-07-19 | https://github.com/microsoft/DirectML | 2026-10-10 | graphics |
| S154 | paper | Reese Levine, Rithik Sharma, Nikhil Jain, Abhijit Ramesh, Zheyuan Chen, Neha Abbas, James Contini, Tyler Sorensen, "Llamas on the Web: Memory-Efficient, Performance-Portable, and Multi-Precision LLM Inference with WebGPU", arXiv 2026-05-20 (section 6.2) | arXiv:2605.20706 (https://arxiv.org/abs/2605.20706) | 2026-10-10 | graphics |
| S155 | paper | Charlie F. Ruan et al., "WebLLM: A High-Performance In-Browser LLM Inference Engine", arXiv 2024-12-20 (abstract: up to 80% of native performance on the same device); opened, not cited (no device named in the abstract) | arXiv:2412.15803 (https://arxiv.org/abs/2412.15803) | 2026-10-10 | not cited |
| S156 | vendor doc | AMD ROCm documentation, "AMD RDNA3.5 system optimization" (Strix Halo: GTT, TTM pages_limit, amd-ttm, BIOS VRAM reservation) | https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html | 2026-10-10 | graphics |
| S157 | issue | NickAllmende, "[gfx1151 / Strix Halo] ROCm reports VRAM + GTT as total available memory on APU ...", ROCm/ROCm #6004, 2026-02-27, with AMD replies of 2026-06-18 and 2026-08-10 (harkgill-amd) | https://github.com/ROCm/ROCm/issues/6004 | 2026-10-10 | graphics |
| S158 | blog (background, 2023) | Chester Lam, "AMD's RDNA 2: Shooting For the Top", Chips and Cheese, 2023-02-20 (raytracing in the texture unit) | https://chipsandcheese.com/p/amds-rdna-2-shooting-for-the-top | 2026-10-10 | graphics |
| S159 | paper | Enzo Meneses, Hugo Bec, Cristobal A. Navarro, Benoit Crespin, Felipe A. Quezada, Nancy Hitschfeld, Heinich Porro, Maxime Maria, "Advancing RT Core-Accelerated Fixed-Radius Nearest Neighbor Search", arXiv 2026-01-22 | arXiv:2601.15633 (https://arxiv.org/abs/2601.15633) | 2026-10-10 | graphics |
| S160 | vendor doc | AMD, rocDecode 1.10.0 documentation, "rocDecode supported codecs and hardware capabilities" | https://rocm.docs.amd.com/projects/rocDecode/en/latest/reference/rocDecode-formats-and-architectures.html | 2026-10-10 | graphics |
| S161 | blog | Tether Data QVAC team, "An Edge-First Generalized LLM LoRA Fine-Tuning Framework for Heterogeneous GPUs", Hugging Face blog, 2025-12-01 | https://huggingface.co/blog/qvac/fabric-llm-finetune | 2026-10-10 | graphics |
| S162 | paper | Laurent Belcour, Anis Benyoub, "Hardware Accelerated Neural Block Texture Compression with Cooperative Vectors", High-Performance Graphics 2025, arXiv 2025-06-06 | arXiv:2506.06040 (https://arxiv.org/abs/2506.06040) | 2026-10-10 | graphics |
| S163 | vendor doc | Khronos Group press release, "Khronos Group launches Slang initiative, hosting open source compiler contributed by NVIDIA", 2024-11-21 | https://www.khronos.org/news/press/khronos-group-launches-slang-initiative-hosting-open-source-compiler-contributed-by-nvidia | 2026-10-10 | graphics |
| S164 | vendor doc | Khronos Vulkan Working Group, "Vulkan introduces Roadmap 2026 and new descriptor heap extension", 2026-01-23 | https://www.khronos.org/blog/vulkan-introduces-roadmap-2026-and-new-descriptor-heap-extension | 2026-10-10 | graphics |
| S165 | blog | Bkpaine (MindStrix Build Studio), "I tuned llama.cpp on a Strix Halo mini-PC and it beats Ollama on the same weights", 2026-07-24 | https://msbs.com/writing/tuning-llama-cpp-strix-halo/ | 2026-10-10 | graphics |
| S166 | project doc | KhronosGroup/Vulkan-Loader, `docs/LoaderInterfaceArchitecture.md` (`VK_DRIVER_FILES`, `VK_ADD_DRIVER_FILES`, `VK_ICD_FILENAMES` deprecated) | https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderInterfaceArchitecture.md | 2026-10-10 | graphics |
| S167 | paper | Jedrzej Maczan, "Measuring and Reducing WebGPU Dispatch Overhead for LLM Inference", arXiv 2026-08-09; abstract only, opened, not cited | arXiv:2608.08730 (https://arxiv.org/abs/2608.08730) | 2026-10-10 | not cited |
| S168 | paper | "Silicon Showdown: Performance, Efficiency, and Ecosystem Barriers in Consumer-Grade LLM Inference", arXiv 2605.00519; abstract only (no AMD hardware), opened, not cited | arXiv:2605.00519 (https://arxiv.org/abs/2605.00519) | 2026-10-10 | not cited |
| S169 | repository | kyuz0, `amd-strix-halo-vllm-toolboxes` README (historical high-concurrency benchmarks only); opened, not cited | https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes | 2026-10-10 | not cited |

Seen in search results and not opened, so never cited: Steinkraus, Buck and Simard, "Using GPUs
for machine learning algorithms", ICDAR 2005; Fujieda and Harada, "Neural Texture Block
Compression", 2024; the Mesa merge request for ANV ASTC emulation (!25467) and the Phoronix
articles on RADV bfloat16 and descriptor heaps (bot checks); the rocJPEG hardware page (HTTP 404).
## Added by R51, the public Strix Halo benchmark dataset (2026-10-10)

Note `public_strix_halo_benchmarks.md` (public) also cites L4 above. Each entry below is one source group of
`data/public-strix-halo-benchmarks.csv`: a repository, gist, issue, PR or discussion. The row count and
quality flags are those of the CSV, and "dated" is the span of the dates the rows carry. Every page or file
was opened in this session on 2026-10-10 through github.com, gist.github.com or raw.githubusercontent.com;
the individual file or comment each row came from is its `source_url`.

| Id | Kind | Source | Link or location | Opened | Cited in |
| --- | --- | --- | --- | --- | --- |
| P1 | repository | strix-halo-testing llm-bench (lhl); 5257 rows (5257 reproducible-script); dated 2025-07-13 to 2025-10-28 | https://github.com/lhl/strix-halo-testing | 2026-10-10 | public |
| P2 | repository | amd-strix-halo-toolboxes benchmark results.json (kyuz0); 5222 rows (5220 reproducible-script, 2 forum-post); dated 2025-08-09 to 2026-10 | https://github.com/kyuz0/amd-strix-halo-toolboxes | 2026-10-10 | public |
| P3 | repository | strix-halo-guide structured benchmark data (hogeheer499-commits): moe_density_gate.csv; 709 rows (629 reproducible-script, 80 forum-post); dated 2026-03-20 to 2026-09-26 | https://github.com/hogeheer499-commits/strix-halo-guide | 2026-10-10 | public |
| P4 | repository | strix-halo-llamacpp glance run (Nathanw1014); 506 rows (506 reproducible-script); dated 2026-07-26 to 2026-08-19 | https://github.com/Nathanw1014/strix-halo-llamacpp | 2026-10-10 | public |
| P5 | repository | strix-halo-windows-llm-bench (ihanesman) phase C1; 276 rows (276 reproducible-script); dated 2026-09-24 to 2026-09-25 | https://github.com/ihanesman/strix-halo-windows-llm-bench | 2026-10-10 | public |
| P6 | repository | Local LLM benchmarks on AMD Strix Halo (slb350); 137 rows (137 reproducible-script); dated 2026-09 to 2026-09-22 | https://github.com/slb350/strix-benchmarks | 2026-10-10 | public |
| P7 | repository | kyuz0 amd-strix-halo-vllm-toolboxes benchmark results (vllm bench throughput); 110 rows (110 reproducible-script); dated 2025-12 | https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes | 2026-10-10 | public |
| P8 | issue | Vulkan batched decode throughput cliff at n_tokens=9 on many-expert MoE (ftoleedo) - density gate comment; 97 rows (97 reproducible-script); dated 2026-07-06 to 2026-08-30 | https://github.com/ggml-org/llama.cpp/issues/25356 | 2026-10-10 | public |
| P9 | repository | lessing-systems llm-bench-strixhalo: Local LLM shoot-out (tables.md, REPORT.md); 89 rows (89 reproducible-script); dated 2026-10-06 | https://github.com/lessing-systems/llm-bench-strixhalo | 2026-10-10 | public |
| P10 | repository | visorcraft strix-halo-llm-perf BENCHMARKS.md; 72 rows (72 reproducible-script); dated 2026-02-13 to 2026-03-03 | https://github.com/visorcraft/strix-halo-llm-perf | 2026-10-10 | public |
| P11 | repository | inkling-strix-halo reference run (markhuguet); 67 rows (67 reproducible-script); dated 2026-10-08 | https://github.com/markhuguet/inkling-strix-halo | 2026-10-10 | public |
| P12 | issue | Misc. bug: Inefficient defaults for gfx1151 cost substantial performance for prefill (ROCm), pedapudi; 56 rows (38 reproducible-script, 18 forum-post); dated 2026-04-02 to 2026-07-19 | https://github.com/ggml-org/llama.cpp/issues/21284 | 2026-10-10 | public |
| P13 | repository | strix-halo-optimize FINDINGS (randomm): -ub calibration; 39 rows (39 reproducible-script); dated 2026-08-28 to 2026-10-01 | https://github.com/randomm/strix-halo-optimize | 2026-10-10 | public |
| P14 | repository | strix-halo-sglang patch 7 ablation (JeremiahM37); 36 rows (36 reproducible-script); dated 2026-05 to 2026-08-18 | https://github.com/JeremiahM37/strix-halo-sglang | 2026-10-10 | public |
| P15 | repository | vllm-awq4-qwen bench matrix (hec-ovi); 32 rows (32 reproducible-script); dated 2026-05-02 | https://github.com/hec-ovi/vllm-awq4-qwen | 2026-10-10 | public |
| P16 | discussion | Llama-batched-bench speedups do not translate to llama-server (Mushoz); 30 rows (30 reproducible-script); dated 2025-08-13 to 2025-10-07 | https://github.com/ggml-org/llama.cpp/discussions/15290 | 2026-10-10 | public |
| P17 | discussion | llama-bench Strix Halo (gfx1151) ROCm 7.2.3 + Qwen3.5 122B A10B MTP UD-Q6_K_XL, muslimpribadi; 28 rows (28 reproducible-script); dated 2026-05-25 | https://github.com/ggml-org/llama.cpp/discussions/23659 | 2026-10-10 | public |
| P18 | discussion | Qwen3.8-Flash-Next on Strix Halo (gfx1151): 17 -> 47 tok/s, working MTP on ROCm, drluoto; 27 rows (27 forum-post); dated 2026-08-29 to 2026-09-20 | https://github.com/ggml-org/llama.cpp/discussions/27950 | 2026-10-10 | public |
| P19 | repository | Bosgame Strix Halo 128 GB local LLM guide (ddifofeu); 25 rows (25 reproducible-script); dated 2026-09 to 2026-09-27 | https://github.com/ddifofeu/strix-halo-128gb-llm-guide | 2026-10-10 | public |
| P20 | issue | Framework Desktop (Ryzen AI Max+ 395) AI benchmarks, geerlingguy; 25 rows (25 reproducible-script); dated 2025-08-07 | https://github.com/geerlingguy/ai-benchmarks/issues/21 | 2026-10-10 | public |
| P21 | issue | Vulkan/AMD performance: vendored llama.cpp b7437 missing Wave32 FA and graphics queue (ollama), sagar-kale; 25 rows (23 forum-post, 2 reproducible-script); dated 2026-04-15 to 2026-04-17 | https://github.com/ollama/ollama/issues/15601 | 2026-10-10 | public |
| P22 | repository | Qwen3.8-27B evaluation EVO-X2 (pablo-ross); 23 rows (23 reproducible-script); dated 2025-11-04 to 2026-08-17 | https://github.com/pablo-ross/strix-halo-gmktec-evo-x2 | 2026-10-10 | public |
| P23 | repository | Qwen 3.8 Flash-Next on AMD Strix Halo, tuned llama.cpp branch (Aristo94 EngramHalo); 22 rows (22 reproducible-script); dated 2026-08-27 | https://github.com/Aristo94/EngramHalo.cpp | 2026-10-10 | public |
| P24 | repository | halogen-flash-server README: Measured; 22 rows (22 reproducible-script); dated date not stated | https://github.com/peonist-ai/halogen-flash-server | 2026-10-10 | public |
| P25 | gist | Beelink GTR9Pro (AI Max+ 395) benchmarks (KMouratidis gist); 22 rows (22 reproducible-script); dated 2025-10-22 | https://gist.github.com/KMouratidis/88456bea439ea8d38f452bb6df289b58 | 2026-10-10 | public |
| P26 | repository | strix-halo-sglang Qwen3.8 single request vs prompt length (jvantuyl); 21 rows (21 reproducible-script); dated date not stated | https://github.com/jvantuyl/strix-halo-sglang | 2026-10-10 | public |
| P27 | gist | AMD Ryzen AI MAX+ 395 - LLM Benchmark Results (Ollama & Lemonade) (ricog gist); 21 rows (21 forum-post); dated 2026-04-13 | https://gist.github.com/ricog/96ba9cc295a4706f548ad0d2f7ab616b | 2026-10-10 | public |
| P28 | issue | MTP draft acceptance rate regression after v11.7.0 on gfx1151 (dup2, lemonade issue); 19 rows (19 reproducible-script); dated 2026-08-20 | https://github.com/lemonade-sdk/lemonade/issues/3269 | 2026-10-10 | public |
| P29 | repository | Qwen 3.6-35B-A3B quantization comparison on EVO-X2 (nabe2030); 16 rows (16 reproducible-script); dated 2026-04-19 | https://github.com/nabe2030/qwen36-evo-x2 | 2026-10-10 | public |
| P30 | issue | halogen-flash-server #125 decode regression report (kirkdis); 14 rows (14 forum-post); dated 2026-10-02 | https://github.com/peonist-ai/halogen-flash-server/issues/125 | 2026-10-10 | public |
| P31 | issue | lemonade #1642 MLX Engine ROCm backend (ghost); 13 rows (13 forum-post); dated 2026-04-15 | https://github.com/lemonade-sdk/lemonade/issues/1642 | 2026-10-10 | public |
| P32 | other | Strix Halo HomeLab Wiki: llama.cpp Performance (GitHub mirror of strixhalo.wiki/AI/llamacpp-performance); 12 rows (12 reproducible-script); dated 2025-11-01 | https://github.com/deseven/strixhalo-homelab | 2026-10-10 | public |
| P33 | discussion | Known-Good Strix Halo ROCm + llama.cpp Stack, realugbun (4-month update); 11 rows (11 forum-post); dated 2026-03-22 to 2026-07-23 | https://github.com/ggml-org/llama.cpp/discussions/20856 | 2026-10-10 | public |
| P34 | issue | [Performance]: Framework Desktop Ryzen AI Max+ 128GB - CPU Only (colibri); 11 rows (11 reproducible-script); dated 2026-07-10 to 2026-07-14 | https://github.com/JustVugg/colibri/issues/39 | 2026-10-10 | public |
| P35 | issue | Vulkan flash attention performance regression on Strix Halo, ddpasa; 10 rows (10 reproducible-script); dated 2026-07-01 to 2026-07-02 | https://github.com/ggml-org/llama.cpp/issues/25207 | 2026-10-10 | public |
| P36 | issue | Misc. bug: HIP backend performs poorly on AMD Ryzen AI MAX 395 (Strix Halo gfx1151), lhl; 9 rows (9 reproducible-script); dated 2025-05-15 to 2025-05-17 | https://github.com/ggml-org/llama.cpp/issues/13565 | 2026-10-10 | public |
| P37 | repository | Qwen3.8-Flash-Next on Strix Halo: 32 to 61 tokens per second (drluoto); 8 rows (8 reproducible-script); dated 2026-09-21 | https://github.com/drluoto/flash-next-strix-halo | 2026-10-10 | public |
| P38 | repository | Gemma 4 vs Qwen 3.5 on EVO-X2 (nabe2030); 8 rows (8 reproducible-script); dated 2026-05 | https://github.com/nabe2030/gemma4-vs-qwen35-evo-x2 | 2026-10-10 | public |
| P39 | repository | minis-s1max-llm-bench README reference table (cites r/LocalLLM post); 8 rows (6 forum-post, 2 reproducible-script); dated 2026-10-03 | https://github.com/InterfaceX-co-jp/minis-s1max-llm-bench | 2026-10-10 | public |
| P40 | repository | xCloudinfo-EnterpriseTeam ai395-strix-halo-llm README; 8 rows (8 reproducible-script); dated 2026-10-08 | https://github.com/xCloudinfo-EnterpriseTeam/ai395-strix-halo-llm | 2026-10-10 | public |
| P41 | issue | Test distributed-llama for clustered AI (geerlingguy); 7 rows (7 reproducible-script); dated 2025-08-03 to 2025-08-05 | https://github.com/geerlingguy/beowulf-ai-cluster/issues/4 | 2026-10-10 | public |
| P42 | issue | Beelink GTR9 Pro bench.sh results glm-5.3-exl3 (xXm-mXx); 7 rows (7 reproducible-script); dated 2026-10-09 | https://github.com/Yamz-Labs/kyojin/issues/48 | 2026-10-10 | public |
| P43 | issue | bundled llamacpp:rocm backend produces incorrect output above ~5K prompt tokens on Strix Halo (ken8848); 7 rows (7 forum-post); dated 2026-09-17 | https://github.com/lemonade-sdk/lemonade/issues/3610 | 2026-10-10 | public |
| P44 | issue | Performance regression: llama-server high CPU use when generating tokens (ollama issue) - comment by DTU-trave; 7 rows (7 forum-post); dated 2026-09-02 | https://github.com/ollama/ollama/issues/18038 | 2026-10-10 | public |
| P45 | discussion | gfx1151 throughput datapoints Qwen3.8-27B MTP+ngram (simongonzalezdc); 6 rows (6 reproducible-script); dated 2026-08-16 | https://github.com/ggml-org/llama.cpp/discussions/27154 | 2026-10-10 | public |
| P46 | issue | Increase llama.cpp performance - strategies for AMD Ryzen AI Max 395+ (comment by hogeheer499-commits); 6 rows (6 forum-post); dated 2026-06-01 | https://github.com/geerlingguy/beowulf-ai-cluster/issues/5 | 2026-10-10 | public |
| P47 | issue | Bug: --ollama-engine runner ignores GGML_CUDA_INIT=1, breaking ROCm bootstrap on gfx1151 APUs (ftmng); 6 rows (6 forum-post); dated 2026-04-08 | https://github.com/ollama/ollama/issues/15420 | 2026-10-10 | public |
| P48 | issue | vLLM #32180 (hogeheer499-commits comment); 5 rows (5 forum-post); dated 2026-06-01 to 2026-07-30 | https://github.com/vllm-project/vllm/issues/32180 | 2026-10-10 | public |
| P49 | repository | halomarchy: services, speculative prefill (kinncj); 5 rows (5 forum-post); dated 2026-09 to 2026-09-11 | https://github.com/kinncj/halomarchy | 2026-10-10 | public |
| P50 | PR | server: allow parallel requests for qwen35/qwen35moe (ollama PR) - comment by DTU-travelpals; 5 rows (5 forum-post); dated 2026-07-30 to 2026-09-25 | https://github.com/ollama/ollama/pull/17144 | 2026-10-10 | public |
| P51 | issue | [gfx1151] mul_mat_q<Q8_0,128,true> Memory Fault during large-ubatch prompt processing on b1292 (blackdeathdrow; 5 rows (5 forum-post); dated 2026-06-19 | https://github.com/lemonade-sdk/llamacpp-rocm/issues/116 | 2026-10-10 | public |
| P52 | discussion | Performance of llama.cpp on AMD ROCm (HIP) scoreboard; 4 rows (4 reproducible-script); dated date not stated | https://github.com/ggml-org/llama.cpp/discussions/15021 | 2026-10-10 | public |
| P53 | issue | For gfx1151, llama.cpp should be built with -DGGML_HIP_ROCWMMA_FATTN=ON, lhl; 4 rows (4 reproducible-script); dated 2025-08-10 | https://github.com/lemonade-sdk/llamacpp-rocm/issues/7 | 2026-10-10 | public |
| P54 | issue | ROCm build of llama.cpp is suboptimal for Strix Halo, pwilkin; 4 rows (4 reproducible-script); dated 2026-07-09 | https://github.com/lemonade-sdk/lemonade/issues/2624 | 2026-10-10 | public |
| P55 | issue | Hp Zbook G1a bench report (alanoo81); 4 rows (4 reproducible-script); dated 2026-10-08 | https://github.com/Yamz-Labs/kyojin/issues/45 | 2026-10-10 | public |
| P56 | issue | Vulkan on gfx1151 (Strix Halo): amdgpu compute-ring timeout / ErrorDeviceLost on long-prompt prefill (yonggony; 4 rows (4 forum-post); dated 2026-08-19 | https://github.com/ollama/ollama/issues/17870 | 2026-10-10 | public |
| P57 | issue | AMD Strix Halo (Radeon 8060S) VRAM detection regression: only partial GPU utilization on 0.30.x (bg8erz); 4 rows (4 forum-post); dated 2026-06-04 | https://github.com/ollama/ollama/issues/16529 | 2026-10-10 | public |
| P58 | issue | [Vulkan] Qwen3-30b-a3b - Severe degradation between 1.29.0 and 1.38.0/1.39.0 (waltercool, LM Studio bug tracke; 4 rows (4 forum-post); dated 2025-07-13 to 2025-07-14 | https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/761 | 2026-10-10 | public |
| P59 | discussion | RFC: Add Atlas Inference backend for native Windows/Linux ROCm inference (lemonade discussion) - comment by Az; 4 rows (4 forum-post); dated 2026-09-25 | https://github.com/lemonade-sdk/lemonade/discussions/3643 | 2026-10-10 | public |
| P60 | issue | Performance decrease on Beelink GTR9 Pro AMD Ryzen AI Max+ 395 (kevenboo, ollama issue); 4 rows (4 forum-post); dated 2026-06-02 | https://github.com/ollama/ollama/issues/16441 | 2026-10-10 | public |
| P61 | issue | pytorch #171687 gfx1151 decode hipMemcpy bound (BellaDoggie); 3 rows (3 forum-post); dated 2026-01-08 | https://github.com/pytorch/pytorch/issues/171687 | 2026-10-10 | public |
| P62 | vendor blog | AI Inference on AMD Ryzen AI Max Processor (AMD ROCm Blogs; Chengjia Huang, Daniel Huang, George Wang); 3 rows (3 vendor); dated 2026-05-25 | https://github.com/ROCm/rocm-blogs | 2026-10-10 | public |
| P63 | issue | ds4: forced --ssd-streaming costs 4.4x on 128 GB Strix Halo (iKyi, lemonade issue); 3 rows (3 forum-post); dated 2026-08-29 | https://github.com/lemonade-sdk/lemonade/issues/3431 | 2026-10-10 | public |
| P64 | issue | ROCm backend breaks tool discovery in real agent workloads (LM Studio bug tracker) - update comment by mlaihk; 3 rows (3 forum-post); dated 2026-09-12 | https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2376 | 2026-10-10 | public |
| P65 | repository | geerlingguy ai-benchmarks README Findings (obench.sh, Ollama eval rate); 3 rows (3 reproducible-script); dated 2025-08-07 | https://github.com/geerlingguy/ai-benchmarks | 2026-10-10 | public |
| P66 | discussion | Performance of llama.cpp with Vulkan scoreboard; 2 rows (2 reproducible-script); dated date not stated | https://github.com/ggml-org/llama.cpp/discussions/10879 | 2026-10-10 | public |
| P67 | PR | vllm-omni PR #6236 Document ROCm validation on gfx1151 (tangzzycc); 2 rows (2 reproducible-script); dated 2026-08-16 | https://github.com/vllm-project/vllm-omni/pull/6236 | 2026-10-10 | public |
| P68 | issue | AMD Strix Halo (gfx1151) - ROCm Working Guide: 40 tok/s on 30B models (queensone); 2 rows (2 forum-post); dated 2026-03-14 | https://github.com/ollama/ollama/issues/14855 | 2026-10-10 | public |
| P69 | issue | [llamacpp-hrx] Intermittent materialize host weight failed + GPU hang on gfx1151 (dg-w36, lemonade issue); 2 rows (2 forum-post); dated 2026-09-08 | https://github.com/lemonade-sdk/lemonade/issues/3517 | 2026-10-10 | public |
| P70 | issue | ROCm backend on Strix Halo (gfx1151 iGPU) bleeds KV state across sequential requests (timames, ollama issue); 2 rows (2 forum-post); dated 2026-08-18 | https://github.com/ollama/ollama/issues/17847 | 2026-10-10 | public |
| P71 | issue | gfx1151 (Radeon 8050S) silently falls back to CPU on Linux (ollama issue) - comment by blue-az; 2 rows (2 forum-post); dated 2026-01-01 to 2026-03-11 | https://github.com/ollama/ollama/issues/13589 | 2026-10-10 | public |
| P72 | issue | AMD Strix Halo VRAM Detection Regression in Ollama 0.30+ (ollama issue) - comment by rick-github; 2 rows (2 forum-post); dated 2026-08-14 | https://github.com/ollama/ollama/issues/16462 | 2026-10-10 | public |
| P73 | issue | serving model(llama4:scout) fails on gfx1151 (ollama issue) - comment by pnjacket; 2 rows (2 forum-post); dated 2025-05-18 | https://github.com/ollama/ollama/issues/10758 | 2026-10-10 | public |
| P74 | issue | When will minimax-m2 be supported? (ollama issue) - comment by rick-github; 2 rows (2 forum-post); dated 2025-12-10 | https://github.com/ollama/ollama/issues/13049 | 2026-10-10 | public |
| P75 | repository | patrickmfurbert ai-tools: llama.cpp build doc (r/LocalLLaMA community guide); 2 rows (2 forum-post); dated 2026-08-31 | https://github.com/patrickmfurbert/ai-tools | 2026-10-10 | public |
| P76 | repository | strix-halo-evo-x2-evidence (ciru-ai); 1 rows (1 forum-post); dated 2026-06-14 | https://github.com/ciru-ai/strix-halo-evo-x2-evidence | 2026-10-10 | public |
| P77 | issue | Test ROCm with llama.cpp on Fedora Rawhide (geerlingguy); 1 rows (1 reproducible-script); dated 2025-08-05 | https://github.com/geerlingguy/beowulf-ai-cluster/issues/7 | 2026-10-10 | public |
| P78 | repository | Strix Halo Cluster (mightystudios); 1 rows (1 forum-post); dated 2026-09 | https://github.com/mightystudios/StrixHaloCluster | 2026-10-10 | public |
| P79 | repository | Framework-strix-halo-llm-setup-V2 (Gygeek); 1 rows (1 forum-post); dated 2025-12-14 | https://github.com/Gygeek/Framework-strix-halo-llm-setup-V2 | 2026-10-10 | public |
| P80 | PR | Add support for ROCm gfx1151 (dhiltgen, Ollama maintainer); 1 rows (1 forum-post); dated 2025-03-14 | https://github.com/ollama/ollama/pull/9773 | 2026-10-10 | public |
| P81 | issue | ROCm backend does not detect GPU on Windows when AMD driver ships a newer HIP runtime than TheRock (blackdeath; 1 rows (1 forum-post); dated 2026-08-18 | https://github.com/lemonade-sdk/lemonade/issues/3213 | 2026-10-10 | public |
| P82 | vendor doc | GAIA v0.20.0 Release Notes (AMD amd/gaia); 1 rows (1 vendor); dated 2026-06-03 | https://github.com/amd/gaia | 2026-10-10 | public |
| P83 | issue | Nemotron3.5-lightning:30b stalling on AMD AI395+ (ollama issue) - comment by cipriancu; 1 rows (1 forum-post); dated 2026-08-12 | https://github.com/ollama/ollama/issues/17692 | 2026-10-10 | public |
| P84 | issue | qwen3.5:122b crashes with 0xc0000409 on Strix Halo + Adrenalin 32.0.31019 (TenshuA); 1 rows (1 forum-post); dated 2026-06-15 | https://github.com/ollama/ollama/issues/16734 | 2026-10-10 | public |
| P85 | PR | fix: name the gpt-oss-120b checkpoint instead of globbing (noamsto, lemonade PR); 1 rows (1 forum-post); dated 2026-09-03 | https://github.com/lemonade-sdk/lemonade/pull/3496 | 2026-10-10 | public |
| P86 | repository | ds4-rocm README (lemonade-sdk, AMD-maintained); 1 rows (1 vendor); dated date not stated | https://github.com/lemonade-sdk/ds4-rocm | 2026-10-10 | public |

Searched for R51 but not opened (each refused by the session's network policy, by DNS, or by the fetch tool;
no number from these is in the CSV):

- AMD pages (amd.com product and blog pages, rocm.blogs.amd.com, developer.amd.com, gpuopen.com), including AMD's Ryzen AI Max+ 395 LLM claims and their endnotes: not opened; searched 2026-10-10.
- Press and review sites (Phoronix, OpenBenchmarking.org, ServeTheHome, Notebookcheck, Tom's Hardware, The Register, Chips and Cheese, TechPowerUp, jeffgeerling.com): not opened; searched 2026-10-10.
- Forums and aggregators (reddit.com and old.reddit.com, news.ycombinator.com and hn.algolia.com, community.frame.work, forum.level1techs.com, llm-tracker.info, strixhalo.wiki, kyuz0.github.io): not opened; strixhalo.wiki was read through its GitHub mirror deseven/strixhalo-homelab instead; searched 2026-10-10.
- Vendor and tool sites (lmstudio.ai, ollama.com, lemonade-server.ai, OEM sites) and YouTube: not opened; searched 2026-10-10.
- Archives (web.archive.org, archive.ph): not opened; searched 2026-10-10.
- ggml-org/llama.cpp discussions 10879 and 15021 beyond their AI Max rows, the LaurentZuijdwijk llama.cpp fork README, and halo-box/strix-llama.cpp PR 150: not opened (truncated, timed out or 404); searched 2026-10-10.
## Added by R49, fast settings optimization note (2026-10-10)

Ids `B1` to `B60` belong to `fast_optimization.md` ("fast" in the Cited in column). They use the
letter B so that they cannot collide with the `S` and `L` ids that other open pull requests append
to this index. PDFs that could not be read in the browser fetch were saved and read with pdftotext.

| Id | Kind | Source | Link or location | Opened | Cited in |
| --- | --- | --- | --- | --- | --- |
| B1 | paper | Olson, Santorella, Tiao, Cakmak, Garrard, Daulton, Lin, Ament, Beckerman, Onofrey, Igusti, Lara, Letham, Cardoso, Shen, Lin, Grange, Kashtelyan, Eriksson, Balandat, Bakshy, "Ax: A Platform for Adaptive Experimentation", AutoML 2025, PMLR 293 (sections 1.1, 5.2, 5.3, 6, 7, Table 2, Appendix A.3; PDF at https://raw.githubusercontent.com/mlresearch/v293/main/assets/olson25a/olson25a.pdf) | https://proceedings.mlr.press/v293/olson25a.html | 2026-10-10 | fast |
| B2 | code | facebook/Ax `ax/api/client.py` at tag 1.3.1 (`get_next_trials`, `attach_baseline`, `attach_trial`, `attach_data`, `complete_trial`, `mark_trial_*`, `should_stop_trial_early`, `run_trials`) | https://github.com/facebook/Ax/blob/1.3.1/ax/api/client.py | 2026-10-10 | fast |
| B3 | code | facebook/Ax `ax/api/utils/generation_strategy_dispatch.py` at tag 1.3.1 (methods fast, quality, random_search; initialization budget and pause criterion) | https://github.com/facebook/Ax/blob/1.3.1/ax/api/utils/generation_strategy_dispatch.py | 2026-10-10 | fast |
| B4 | project doc | Ax docs, "Trial-level Early Stopping" tutorial, version 1.3.1 | https://ax.dev/docs/tutorials/early_stopping/ | 2026-10-10 | fast |
| B5 | project doc | PyPI, ax-platform release history (1.0.0 2025-05-08 to 1.3.1 2026-06-09; Python 3.11 or newer) | https://pypi.org/project/ax-platform/ | 2026-10-10 | fast |
| B6 | blog | Engineering at Meta, "Efficient Optimization With Ax, an Open Platform for Adaptive Experimentation", 2025-11-18 | https://engineering.fb.com/2025/11/18/open-source/efficient-optimization-ax-open-platform-adaptive-experimentation/ | 2026-10-10 | fast |
| B7 | paper | Benjamins, Graf, Segel, Deng, Ruhkopf, Hennig, Basu, Mallik, Bergman, Chen, Clement, Tornede, Feurer, Eggensperger, Hutter, Doerr, Lindauer, "carps: A Framework for Comparing N Hyperparameter Optimizers on M Benchmarks", v2 2025-09-18 (Tables 5 and 6) | arXiv:2506.06143 (https://arxiv.org/abs/2506.06143) | 2026-10-10 | fast |
| B8 | paper | Cheng, Wang, Hu, Yang, Li, Zhang, "SCOOT: SLO-Oriented Performance Tuning for LLM Inference Engines", WWW 2025 (doi 10.1145/3696410.3714930), sections 1, 4.6, 5.2, 5.3 | arXiv:2408.04323 (https://arxiv.org/abs/2408.04323) | 2026-10-10 | fast |
| B9 | paper | Lysenstoen, "SLO-Guard: Crash-Aware, Budget-Consistent Autotuning for SLO-Constrained LLM Serving", 2026-04-19 (abstract, section 5.5) | arXiv:2604.17627 (https://arxiv.org/abs/2604.17627) | 2026-10-10 | fast |
| B10 | paper | Turner, Eriksson, McCourt, Kiili, Laaksonen, Xu, Guyon, "Bayesian Optimization is Superior to Random Search for Machine Learning Hyperparameter Tuning: Analysis of the Black-Box Optimization Challenge 2020", v2 2021-08-31 (Table 2) | arXiv:2104.10201 (https://arxiv.org/abs/2104.10201) | 2026-10-10 | fast |
| B11 | paper | Cowen-Rivers, Lyu, Tutunov, Wang, Grosnit, Griffiths, Maraval, Hao, Wang, Peters, Bou Ammar, "HEBO: Pushing The Limits of Sample-Efficient Hyperparameter Optimisation", JAIR (v6 2022-05-25) | arXiv:2012.03826 (https://arxiv.org/abs/2012.03826) | 2026-10-10 | fast |
| B12 | repository doc | huawei-noah/HEBO, HEBO README (batch suggestion `opt.suggest(n_suggestions=4)`, NeurIPS 2020 winning submission) | https://github.com/huawei-noah/HEBO/tree/master/HEBO | 2026-10-10 | fast |
| B13 | paper | Song, Zhang, Lee, Fertig, Huang, Belenki, Kochanski, Ariafar, Vasudevan, Perel, Golovin, "The Vizier Gaussian Process Bandit Algorithm", v3 2024-12-06 (sections 3.6, 4.3, 4.4) | arXiv:2408.11527 (https://arxiv.org/abs/2408.11527) | 2026-10-10 | fast |
| B14 | project doc | Optuna docs (stable), `optuna.samplers` reference, sampler comparison table | https://optuna.readthedocs.io/en/stable/reference/samplers/index.html | 2026-10-10 | fast |
| B15 | project doc | Optuna docs (stable), `optuna.samplers.GPSampler` | https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GPSampler.html | 2026-10-10 | fast |
| B16 | project doc | Optuna docs (stable), `optuna.pruners.WilcoxonPruner` | https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.WilcoxonPruner.html | 2026-10-10 | fast |
| B17 | repository doc | automl/SMAC3 README (random forest default since 2.4.0, ask-and-tell, multi-threading) | https://github.com/automl/SMAC3 | 2026-10-10 | fast |
| B18 | project doc | Nevergrad docs, "How to perform optimization" (parallel evaluation, optimizer choice, discrete and noisy problems) | https://facebookresearch.github.io/nevergrad/optimization.html | 2026-10-10 | fast |
| B19 | paper | Ansel, Kamil, Veeramachaneni, Ragan-Kelley, Bosboom, O'Reilly, Amarasinghe, "OpenTuner: An Extensible Framework for Program Autotuning", PACT 2014 (abstract, section 3.2.1) | https://commit.csail.mit.edu/papers/2014/ansel-pact14-opentuner.pdf | 2026-10-10 | fast |
| B20 | paper | Li, Jamieson, Rostamizadeh, Gonina, Hardt, Recht, Talwalkar, "A System for Massively Parallel Hyperparameter Tuning", MLSys 2020 | arXiv:1810.05934 (https://arxiv.org/abs/1810.05934) | 2026-10-10 | fast |
| B21 | paper | Hvarfner, Hellsten, Nardi, "Vanilla Bayesian Optimization Performs Great in High Dimensions", ICML 2024 | arXiv:2402.02229 (https://arxiv.org/abs/2402.02229) | 2026-10-10 | fast |
| B22 | paper | Eriksson, Jankowiak, "High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces", UAI 2021 | arXiv:2103.00349 (https://arxiv.org/abs/2103.00349) | 2026-10-10 | fast |
| B23 | paper | Raponi, Rakotonirina Carraz, Rapin, Doerr, Teytaud, "Optimizing with Low Budgets: a Comparison on the Black-box Optimization Benchmarking Suite and OpenAI Gym", IEEE TEVC (v3 2024-01-02) | arXiv:2310.00077 (https://arxiv.org/abs/2310.00077) | 2026-10-10 | fast |
| B24 | paper | Hvarfner, Stoll, Souza, Lindauer, Hutter, Nardi, "piBO: Augmenting Acquisition Functions with User Beliefs for Bayesian Optimization", ICLR 2022 | arXiv:2204.11051 (https://arxiv.org/abs/2204.11051) | 2026-10-10 | fast |
| B25 | paper | Hvarfner, Hutter, Nardi, "A General Framework for User-Guided Bayesian Optimization" (ColaBO), ICLR 2024 | arXiv:2311.14645 (https://arxiv.org/abs/2311.14645) | 2026-10-10 | fast |
| B26 | paper | Fehring, Wever, Spliethoever, Hennig, Wachsmuth, Lindauer, "Dynamic Priors in Bayesian Optimization for Hyperparameter Optimization" (DynaBO), v1 2025-11-04, v3 2026-07-31 | arXiv:2511.02570 (https://arxiv.org/abs/2511.02570) | 2026-10-10 | fast |
| B27 | paper | Bai, Li, Shen, Zhang, Zhang, Cui, "Transfer Learning for Bayesian Optimization: A Survey", 2023-02-12 | arXiv:2302.05927 (https://arxiv.org/abs/2302.05927) | 2026-10-10 | fast |
| B28 | vendor doc | NIST/SEMATECH e-Handbook of Statistical Methods, section 5.3.3.5, Plackett-Burman designs | https://www.itl.nist.gov/div898/handbook/pri/section3/pri335.htm | 2026-10-10 | fast |
| B29 | code | vllm-project/vllm `vllm/config/cache.py` at b1388b1 (`gpu_memory_utilization`, `kv_cache_memory_bytes`) | https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/config/cache.py | 2026-10-10 | fast |
| B30 | code | vllm-project/vllm `vllm/config/compilation.py` at b1388b1 (`CompilationConfig.compute_hash`, `cudagraph_capture_sizes`, `max_cudagraph_capture_size`) | https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/config/compilation.py | 2026-10-10 | fast |
| B31 | code | vllm-project/vllm `vllm/envs.py` at b1388b1 (`compile_factors`, `VLLM_CACHE_ROOT`) | https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/envs.py | 2026-10-10 | fast |
| B32 | code | vllm-project/vllm `vllm/config/multimodal.py` at b1388b1 (`skip_mm_profiling`) | https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/config/multimodal.py | 2026-10-10 | fast |
| B33 | benchmark | llm-tracker, "Strix Halo" (lhl), page of 2025-08-08 with updates from 2025-05 (Fedora 43, kernel 6.15 rc7, HIP 6.4 to 6.5; llama.cpp backends, rocm_bandwidth_test, hgemm with and without hipBLASLt) | https://llm-tracker.info/_TOORG/Strix-Halo | 2026-10-10 | fast |
| B34 | issue | Foxlight-Foundation/Skulk issue 144, "spike: AMD ROCm support - Ryzen AI Max 395 (Strix Halo / gfx1151)", opened 2026-05-05, closed 2026-06-24 (community data, ROCm 6.4.4) | https://github.com/Foxlight-Foundation/Skulk/issues/144 | 2026-10-10 | fast |
| B35 | blog | Darren Soothill, "Strix Halo LLM upgrade: wins and regressions", Soot / Silicon, 2026-08-11 (Ubuntu 24.04.4, kernel 6.17, ROCm 7.14, vLLM 0.26 and 0.27, llama.cpp b10362) | https://www.soothill.io/blog/2026/08/11/strix-halo-llm-stack-upgrade-benchmarks/ | 2026-10-10 | fast |
| B36 | blog | shuzan, "Local LLM Performance Improved on Ryzen AI Max+ 395", zenn.dev, 2026-06-17 (ROCm 7.2.4, llama.cpp b9672, vLLM 0.23.0) | https://zenn.dev/shuzan/articles/72852bb9621b40 | 2026-10-10 | fast |
| B37 | code | triton-lang/triton `python/triton/runtime/autotuner.py` on main, read 2026-10-10 (`autotune` docstring, `prune_configs_by`, `Autotuner._bench`) | https://github.com/triton-lang/triton/blob/main/python/triton/runtime/autotuner.py | 2026-10-10 | fast |
| B38 | code | apache/tvm `gallery/how_to/tune_with_autoscheduler/tune_conv2d_layer_cuda.py` at v0.15.0 (`num_measure_trials`, `min_repeat_ms`) | https://github.com/apache/tvm/blob/v0.15.0/gallery/how_to/tune_with_autoscheduler/tune_conv2d_layer_cuda.py | 2026-10-10 | fast |
| B39 | code | apache/tvm `gallery/how_to/tune_with_autoscheduler/tune_network_cuda.py` at v0.15.0 (900 x len(tasks)) | https://github.com/apache/tvm/blob/v0.15.0/gallery/how_to/tune_with_autoscheduler/tune_network_cuda.py | 2026-10-10 | fast |
| B40 | paper | Zheng, Jia, Sun, Wu, Yu, Haj-Ali, Wang, Yang, Zhuo, Sen, Gonzalez, Stoica, "Ansor: Generating High-Performance Tensor Programs for Deep Learning", OSDI 2020 | arXiv:2006.06762 (https://arxiv.org/abs/2006.06762) | 2026-10-10 | fast |
| B41 | paper | Zheng, Liu, Shao, Chen, Gonzalez, Stoica, Haj-Ali, "TenSet: A Large-scale Program Performance Dataset for Learned Tensor Compilers", NeurIPS 2021 Datasets and Benchmarks | https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/a684eceee76fc522773286a895bc8436-Abstract-round1.html | 2026-10-10 | fast |
| B42 | vendor doc | AMD ROCm 6.1.0 documentation, "Optimizing Triton kernels" (MI300X: `num_stages`, `waves_per_eu`, `matrix_instr_nonkdim`, occupancy from VGPRs and LDS) | https://rocm.docs.amd.com/en/docs-6.1.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html | 2026-10-10 | fast |
| B43 | paper | Swann, Osama, Guo, Nelson, Zhang, Brown, Ong, Yazdani, Siddens, Dasika, Underwood, "tritonBLAS: Triton-based Analytical Approach for GEMM Kernel Parameter Selection", 2025-12-03 | arXiv:2512.04226 (https://arxiv.org/abs/2512.04226) | 2026-10-10 | fast |
| B44 | paper | Zhang, Ding, Qian, Wang, Cao, Xue, Huang, Yang, Zhang, "WaveTune: Wave-aware Bilinear Modeling for Efficient GPU Kernel Auto-tuning", 2026-04-11 | arXiv:2604.10187 (https://arxiv.org/abs/2604.10187) | 2026-10-10 | fast |
| B45 | paper | Ringlein, Parnell, Stoica, "GPU Performance Portability needs Autotuning", v3 2025-07-17 | arXiv:2505.03780 (https://arxiv.org/abs/2505.03780) | 2026-10-10 | fast |
| B46 | paper | Wang, Joshi, Majumder, Chao, Ding, Liu, Brahma, Li, Liu, Barsoum, "GEAK: Introducing Triton Kernel AI Agent & Evaluation Benchmarks", 2025-07-31 | arXiv:2507.23194 (https://arxiv.org/abs/2507.23194) | 2026-10-10 | fast |
| B47 | paper | Wiedemann, Leboutet, Paulitsch, Wofk, Ummenhofer, "KernelFoundry: Hardware-aware evolutionary GPU kernel optimization", ICML 2026 (v2 2026-09-03) | arXiv:2603.12440 (https://arxiv.org/abs/2603.12440) | 2026-10-10 | fast |
| B48 | paper | Liu, Majumder, Huang and others, "AMDKernelVault: Large-Scale Datasets and Agentic Training for AMD GPU Kernel Optimization", 2026-09-11 | arXiv:2609.12471 (https://arxiv.org/abs/2609.12471) | 2026-10-10 | fast |
| B49 | our measurement | GPU Queue row R30 (pod D, account D, 2026-10-10 21:47 to 21:57 UTC), Result: ready 523 s after a cold start | https://app.notion.com/p/3f596ef5fec58108a766cddbcb7ddf8d | 2026-10-10 | fast |
| B50 | our measurement | GPU Queue row R18 (pod D, 2026-10-10 22:15 to 22:21 UTC), Result: proxy 500 for 294 s while the teacher started | https://app.notion.com/p/3f596ef5fec581e98081c844ded7ea64 | 2026-10-10 | fast |
| B51 | our measurement | `lab/README.md`, sections "Teacher on SJC today (pod C, Strix Halo)", "The same build on pod D, from scripts alone (R30)" and "Idle baseline on the Strix Halo" | this repo, `lab/README.md` (main at 02db66b) | 2026-10-10 | fast |
| B52 | our files | `lab/bench.py`, module docstring, `Session.measure` and the `DEFAULT_*` constants | this repo, `lab/bench.py` (main at 02db66b) | 2026-10-10 | fast |
| B53 | our measurement | `lab/refine/metrics.jsonl` lines 13 and 14: smoke LoRA samples/s 0.51 and peak 8.7 GiB on one R9700 | this repo, `lab/refine/metrics.jsonl:13-14` (main at 02db66b) | 2026-10-10 | fast |
| B54 | our files | `lab/pod_a_teacher.sh`, `GPU_UTIL` default 0.90 (line 25) and the serve command | this repo, `lab/pod_a_teacher.sh:25` (main at 02db66b) | 2026-10-10 | fast |
| B55 | our files | `lab/machine/verdict.py`, `MIN_RUNS_PER_ARM = 5` (line 52) | this repo, `lab/machine/verdict.py:52` (main at 02db66b) | 2026-10-10 | fast |
| B56 | pr | PR #235, R35 inference settings sheet and `lab/refine/research/kernels/data/searchspace-inference-gfx1151.json` (merged 2026-10-10 as 99db20c) | https://github.com/raohemdutt/Knighthacks26/pull/235 | 2026-10-10 | fast |
| B57 | pr | PR #238, R36 training settings sheet and `lab/refine/research/kernels/data/searchspace-training-gfx1151.json` (merged 2026-10-10 as 86f7b8d) | https://github.com/raohemdutt/Knighthacks26/pull/238 | 2026-10-10 | fast |
| B58 | pr | PR #237, R32 Strix Halo hardware note `lab/refine/research/kernels/strix_halo_hardware.md` (sections 2, 3 and 5; merged 2026-10-10 as 6a8e4a0) | https://github.com/raohemdutt/Knighthacks26/pull/237 | 2026-10-10 | fast |
| B59 | benchmark | kyuz0, "AMD Strix Halo - Legacy Backend Benchmarks" (Framework Desktop, Fedora 42, captured 2025-11-14); the results grid did not load, so nothing was taken from it | https://kyuz0.github.io/amd-strix-halo-toolboxes/ | 2026-10-10 | not used |
| B60 | our files | `lab/refine/research/plumbing/placement.md` (R46's placement plan, merged in #243 as c39b242): "What the teacher's vLLM holds" (64.0 GiB visible to torch, 58 GB available, 17.98 GiB of weights), "Memory margin" (13.05 GiB LoRA budget, 4 GiB host reserve, 74.65 GiB at 0.90, a share of 0.63 or less) and "Measurements get a chip to themselves" | this repo, `lab/refine/research/plumbing/placement.md` (main at ab7f9f3) | 2026-10-10 | fast |

Seen but not opened, so not cited: AMD's blog post "AMD Ryzen AI MAX+ 395 Processor: Breakthrough AI
Performance in Thin and Light" (the fetch timed out), the Framework Community threads on Strix Halo
LLM performance and toolboxes, the hogeheer499 Strix Halo guide, FleetSieve (arXiv 2608.19659),
Optuna's journal-storage recipe, and the OpenReview page of the Ax paper (a browser check blocked
it; the PMLR copy was read instead).

## Added by kernels refinement pass 3 (2026-10-10)

Re-opened by this pass and cited under their existing ids: S13 (vLLM PR #40977), S18 (vLLM PR
#56301), S58 (vLLM `vllm/platforms/rocm.py` at b1388b1, `supports_fp8`), S65 (#6347, no comment
after 2026-09-25), S138 (ggml-org/llama.cpp#27952 and huppiflupp's comments) and S140 (nabe2030
`hip-vs-vulkan-evo-x2`).

| Id | Kind | Source | Link or location | Opened | Cited in |
| --- | --- | --- | --- | --- | --- |
| S170 | code | vLLM at 382bbd5 (the merge commit of PR #40977): `vllm/model_executor/kernels/linear/__init__.py` (the ROCm W4A16 list: RDNA3W4A16, RDNAHybridW4A16, TritonW4A16, Conch, Exllama) and `mixed_precision/rdna3_w4a16.py` (`can_implement` requires gfx1100) | https://github.com/vllm-project/vllm/blob/382bbd51448b2f58c73b3e51d051bc352166ba91/vllm/model_executor/kernels/linear/__init__.py and https://github.com/vllm-project/vllm/blob/382bbd51448b2f58c73b3e51d051bc352166ba91/vllm/model_executor/kernels/linear/mixed_precision/rdna3_w4a16.py | 2026-10-10 | main |
| S171 | issue | The same issue as S66 and S71, read through the GitHub API to its comment of 2026-10-10: the reproducer in the body ("Reproducer 1: plain HIP, no framework") and lobo235's fourth gfx1201 host (in-tree amdgpu, MES 0x89, HIP 7.15.26333: +6.7 us in graph mode at two waits, none at `GPU_MAX_HW_QUEUES` 1 or 2) | https://github.com/ROCm/legacy-rocm-build/issues/6685 | 2026-10-10 | ports |
| L10 | our measurement | `lab/refine/research/kernels/data/chipsheet-gfx1151-2026-10-11.md` (R33 on pod D, merged in PR #250 as fa0359e): device-to-device copy 207.8 to 212.1 GB/s from 256 MiB to 4 GiB, LDS launch limit 65,536 bytes, HIP probes built with `/opt/rocm/bin/hipcc` | this repo, `lab/refine/research/kernels/data/chipsheet-gfx1151-2026-10-11.md` (main at 4e084dc) | 2026-10-10 | main, ports |
| L11 | our files | `lab/public_prior.py` (R51, merged in PR #251 as fe7d0a6) run with `--gb 15.5 --backend vllm --batch 1, 4, 8, 16`: 7.49 tok/s single stream, aggregate 22.86, 34.73 and 46.91 tok/s, k 0.104 | this repo, `lab/public_prior.py` (main at 4e084dc) | 2026-10-10 | main |
| L12 | our files | `lab/README.md`, section "Benchmark harness: `lab/bench.py` (R11)": conditions on every row (environment read back from `/proc/<pid>/environ`, `GPU_MAX_HW_QUEUES` always present) and bands (largest-gap split, 10% of the median, at least 4 spawns) | this repo, `lab/README.md` (main at 4e084dc) | 2026-10-10 | ports |
| L13 | our page | R20 task page, line of Sat 8:27 PM ET (Ilan3's clock): R20 started on pod D with `python -m lab bench run`, 3 spawns per batch size at 1, 4, 8 and 16, reusing R30's build | https://app.notion.com/p/3f596ef5fec581a38be4e70c385da9dd | 2026-10-10 | ports |

## Added by R59, the vLLM upgrade audit for gfx1151 (2026-10-11)

Re-opened by R59 and cited under their existing ids: S13 (vLLM PR #40977, the gfx1151 benchmark table),
S16 (vLLM PR #52619), S29 (vLLM issue #48723, read to its comment of 2026-07-30), S31 (vLLM PR #58225),
S32 (vLLM PR #59132, the gfx1151 serving table), L3 (`lab/vllm-gfx1201.diff`), L10 (the gfx1151 chipsheet)
and L11 (`lab/public_prior.py`).
"upgrade" in the Cited in column means `vllm_upgrade_gfx1151.md`.

| Id | Kind | Source | Link or location | Opened | Cited in |
| --- | --- | --- | --- | --- | --- |
| S172 | code | vllm-project/vllm, blobless clone of origin/main at 3c062d391e (2026-10-11 01:41 UTC): `git tag --contains`, `git merge-base`, `git log` over the audited paths from the merge base d816834c, `git apply --cached --reject` of `lab/vllm-gfx1201.diff` against each tag's tree, and `git show` of `CMakeLists.txt`, `pyproject.toml`, `requirements/build/rocm.txt`, `tools/use_existing_torch.py`, `vllm/platforms/__init__.py`, `vllm/platforms/rocm.py`, `vllm/v1/attention/ops/triton_unified_attention.py`, `vllm/v1/attention/backends/rocm_segmented_attn.py`, `vllm/model_executor/kernels/linear/__init__.py` and `mixed_precision/rdna_hybrid_w4a16.py` at v0.19.1, v0.21.0, v0.27.0, v0.29.0, v0.31.0 and main | https://github.com/vllm-project/vllm | 2026-10-11 | upgrade |
| S173 | PR | mgehre-amd, "[ROCm][Perf] Tune wvSplitK on gfx1151", vLLM #40784, merged 2026-06-25 (page also shows the ROCm/vllm fork commit 7648cc8 reporting a gfx1151 decode regression) | https://github.com/vllm-project/vllm/pull/40784 | 2026-10-11 | upgrade |
| S174 | PR | yiz-liu, "[Fix] Improve ROCm detection in WSL environments", vLLM #38434, merged 2026-08-30 | https://github.com/vllm-project/vllm/pull/38434 | 2026-10-11 | upgrade |
| S175 | PR | pmaybank, "use HIP_VERSION variables to guard against duplicate atomicAdd definitions", vLLM #41802, merged 2026-05-08 (ROCm 7.13 build fix) | https://github.com/vllm-project/vllm/pull/41802 | 2026-10-11 | upgrade |
| S176 | PR | AIwork4me, "[ROCm][RDNA3] Fix W4A16 split-K accuracy and determinism", vLLM #54706, merged 2026-10-03, gfx1100 only | https://github.com/vllm-project/vllm/pull/54706 | 2026-10-11 | upgrade |
| S177 | PR | cofuente, "[Bugfix][Hardware][AMD] Handle AITER unified-attention LDS overflow with Triton fallback", vLLM #49264, opened 2026-07-21, open on 2026-10-11, verified on gfx1151 | https://github.com/vllm-project/vllm/pull/49264 | 2026-10-11 | upgrade |
| S178 | project doc | vLLM v0.31.0 release notes (released 2026-10-05 per the page; tag commit db9527a46 dated 2026-10-02): Quantization, AMD ROCm and Dependencies lines; first 100,000 characters read | https://github.com/vllm-project/vllm/releases/tag/v0.31.0 | 2026-10-11 | upgrade |
| S179 | project doc | ROCm/vllm releases page (v0.7.2+rocm to v0.9.2; "Navi" lines only, no RDNA or gfx1151 line) | https://github.com/ROCm/vllm/releases | 2026-10-11 | upgrade |
| S180 | project doc | triton-lang/triton releases page (3.4.0 to 3.8.0; AMD lines for gfx11, gfx12, RDNA 3, 3.5 and 4) | https://github.com/triton-lang/triton/releases | 2026-10-11 | upgrade |

## Added by R37, thinking sitting 1 (2026-10-11): `predictions/w4a16_decode_gfx1151.md`

Re-opened on 2026-10-11 and cited under their existing ids: S07 (vLLM `mixed_precision/conch.py` at b1388b1, `can_implement` and `apply_weights`) and S10 (Conch v1.2.1 `gemm.py`: `_get_tuning_parameters`, `_gemm_kernel`, `_dequantize`, `mixed_precision_gemm_launcher`, the grid `cdiv(m, 128) * cdiv(n, 64)` and `stride_mul = block_k / group_size`). Notes in `predictions/` are named by path in the "Cited in" column.

| Id | Kind | Source | Link or location | Opened | Cited in |
| --- | --- | --- | --- | --- | --- |
| L14 | our measurement | `lab/refine/metrics.jsonl` lines 24 and 29 (R29 on pod C: 2.88 tok/s, 120.3 W) and lines 67 to 96 (R26 `teacher_batching.*` at 1, 4, 8, 16 and 64 streams: per-stream and total tok/s, TTFT, 96.0 W average, 36.66 to 1.94 J/token, median of 3 repeats) | this repo, `lab/refine/metrics.jsonl` (main at e81bccd) | 2026-10-11 | predictions/w4a16_decode_gfx1151.md |
| L15 | our files | `lab/kernels/chipsheet/chipsheet.py`, `_w4a16_kernel`, `_dequant_body` and `DEQUANT_CONFIGS` (BLOCK_M 16 with BLOCK_N 64 and BLOCK_K 64, or BLOCK_N 32 and BLOCK_K 128; no `num_warps` or `num_stages` passed): the kernel behind the chip sheet's `dequant.w4a16.*` rows | this repo, `lab/kernels/chipsheet/chipsheet.py` (main at e81bccd) | 2026-10-11 | predictions/w4a16_decode_gfx1151.md |
| L16 | our files | `gpu/R12_T0_PREREG.md` on branch `ctx/runtime/kernel-experiment-gfx1151` at 3ecd427 (R12, not yet on main): attention about 0.3 percent of a batch-1 decode step, both unified-attention levers predicted at 0.0 percent J/token | this repo, branch `ctx/runtime/kernel-experiment-gfx1151`, `gpu/R12_T0_PREREG.md` (3ecd427) | 2026-10-11 | predictions/w4a16_decode_gfx1151.md |
| L17 | our files | `lab/README.md`, sections "Batching sweep on the pod C teacher (R26, Strix Halo, gfx1151)" (the 96 W flat package power, the 10.44 W loaded idle) and "Decode profile: `lab/profile_decode.py` (R47)" (the fields of the profile result; "Not yet run on a pod") | this repo, `lab/README.md` (main at e81bccd) | 2026-10-11 | predictions/w4a16_decode_gfx1151.md |

## Added by R37, thinking sitting 1 (2026-10-11): `predictions/train_step_gfx1151.md`

Re-opened on 2026-10-11 and cited under its existing id: S121 (transformers `modeling_qwen3_5.py` at 536ecc0: `torch_chunk_gated_delta_rule`, its float32 casts, 64-token chunks, `torch.linalg.solve_triangular` and the sequential chunk loop; `Qwen3_5GatedDeltaNet.__init__` and the attention projections).

| Id | Kind | Source | Link or location | Opened | Cited in |
| --- | --- | --- | --- | --- | --- |
| S181 | model files | Qwen, `Qwen/Qwen3.5-4B` `config.json` (`text_config`: 32 layers, `full_attention_interval` 4, hidden 2560, `intermediate_size` 9216, 16 query and 4 KV heads of 256 with `attn_output_gate`, linear-attention key heads 16 x 128 and value heads 32 x 128, vocabulary 248,320, tied embeddings, `mamba_ssm_dtype` float32) | https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json | 2026-10-11 | predictions/train_step_gfx1151.md |
| L18 | our files | `lab/train.py` (L3 trainer): module docstring ("Settings", "Per step"), `BASELINE`, `PINNED`, `VISION_MODULES` and the `SFTConfig` it builds (batch 4 x accumulation 4, checkpointing on, SDPA, fused AdamW, torch Gated DeltaNet, no torch.compile; `trainer.samples_per_s` and `trainer.energy_j`) | this repo, `lab/train.py` (main at e81bccd) | 2026-10-11 | predictions/train_step_gfx1151.md |
| L19 | our measurement | `lab/refine/metrics.jsonl` lines 13 and 14: the R9700 smoke step of Qwen3.5-4B (`lab/smoke_lora.py`, batch 4, 256 tokens, 20 steps, 0.51 samples/s, 8.7 GiB peak), another chip, never a gfx1151 baseline | this repo, `lab/refine/metrics.jsonl` (main at e81bccd) | 2026-10-11 | predictions/train_step_gfx1151.md |

## Added by R56, the gfx1151 energy model note (2026-10-11)

Re-opened by R56 and cited under their existing ids: S62 (the ML.ENERGY Benchmark paper) and L10 (the gfx1151
chip sheet). "energy" in the Cited in column means `energy_model_gfx1151.md`.

| Id | Kind | Source | Link or location | Opened | Cited in |
| --- | --- | --- | --- | --- | --- |
| S182 | project doc | ml-energy/zeus `docs/measure/index.md` at 42d76ac (2026-10-10), the GPU paragraphs and section "ROCm and AMDSMI Versions": AMD GPUs through AMDSMI with ROCm 6.3 or later, older versions returned wrong power and energy values, `current_socket_power` can read N/A and the average socket power is used instead | https://github.com/ml-energy/zeus/blob/42d76ac/docs/measure/index.md | 2026-10-11 | energy |
| L20 | our files | `lab/energy_model.py` (this task's branch `ctx/runtime/energy-model-gfx1151`): `chip_inputs`, `sheet_knees`, `knees`, `predict`, `check` and `OBJECTIVE`; every prediction in the note is its output on the committed chip sheet | this repo, `lab/energy_model.py` | 2026-10-11 | energy |
| L21 | our measurement | `lab/README.md`, section "Batching sweep on the pod C teacher (R26, Strix Halo, gfx1151)" (PR #269, 081c496): 2.62 to 49.53 tok/s total at 1 to 64 streams, flat 96.0 W, 36.66 to 1.94 J/token gross, the 82.4 W idle read caught an earlier attempt's requests, the real loaded idle 10.44 W | this repo, `lab/README.md` (main at 9ed11fc) | 2026-10-11 | energy |
| L22 | our measurement | `fixtures/lab/energy_model/attach-sweep-pod-c-2026-10-10.jsonl` with `source.json`: R26's `lab/bench.py` rows (session bench-20261010T215122Z-2c65) trimmed to the fields the checker reads, values as the harness wrote them | this repo, `fixtures/lab/energy_model/` | 2026-10-11 | energy |
| L23 | our files | `lab/README.md`, section "Request-side settings sweep: `lab/sweep_request.py` (R52)": the grid, its slot minutes and the collect command; no result committed on main at 9ed11fc | this repo, `lab/README.md` (main at 9ed11fc) | 2026-10-11 | energy |

Sources of `dispatch_by_batch_gfx1151.md` (R63). IDs in the 630s so they do not meet a parallel pass's next number.

| Id | Kind | Source | Link or location | Opened | Cited in |
| --- | --- | --- | --- | --- | --- |
| S630 | code | vllm-project/vllm at 5281e4990 (the R60 build's pin): `vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py` (`MAX_SKINNY_BATCH_SIZE` 5, `MEDIUM_SKINNY_LIMIT_ELEMENTS` 39,321, `_rdna_hybrid_w4a16_apply_impl`, `pack_skinny_int4`, the gfx1151 Triton tile table, `RDNAHybridW4A16LinearKernel.process_weights_after_loading` and `apply_weights`), `MPLinearKernel.py`, `vllm/model_executor/parameter.py::permute_param_layout_` and `layers/quantization/utils/quant_utils.py::pack_quantized_values_into_int32`, read as raw files | https://github.com/vllm-project/vllm/blob/5281e4990/vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py | 2026-10-11 | dispatch |
| L630 | our files | `lab/kernels/dispatch_bench.py` and `lab/kernels/pod_dispatch_bench.sh` (R63, branch `ctx/runtime/dispatch-by-batch`): the head-to-head, the rule (`decide`) and the metrics rows | this repo, `lab/kernels/dispatch_bench.py` | 2026-10-11 | dispatch |
| L631 | our files | `lab/kernels/lut_gemv.py` at R62's commit 3826ec3 (branch `ctx/kernels/lut-dequant`, carried unchanged on this branch): the lookup-table GEMV, its packing and its R62 result on pod-b | this repo, `lab/kernels/lut_gemv.py` | 2026-10-11 | dispatch |
