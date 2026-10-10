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
