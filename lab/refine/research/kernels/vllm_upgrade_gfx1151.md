# Can a newer vLLM close the teacher's decode gap on gfx1151, and what is the smallest upgrade?

Task R59. The teacher decodes at 2.88 tok/s on the Strix Halo (gfx1151) with vLLM at b1388b1 plus
`lab/vllm-gfx1201.diff`, against 7.5 tok/s from the public prior for vLLM and about 12 tok/s for
llama.cpp on the same chip. This note audits every vLLM commit after b1388b1 that touches the ROCm path,
the Triton unified attention kernel, gfx1151 or gfx12 targets, or the W4A16 (AWQ, GPTQ, Quark INT4)
decode kernels; says which of our nine hunks each one collides with; names the smallest upgrade that
keeps the patch applying; and predicts the decode gain. Nothing here ran on a pod. The patch checks and
the backport trial ran on a read-only clone of vLLM on a workstation, without a GPU or a HIP compiler.

## Summary

- **The pin.** b1388b1 is the tag v0.19.1, a patch release on a release branch that left main on
  2026-03-29. "0.19.2" is only the version string `pod_a_teacher.sh` passes to setuptools-scm. Main is
  7,486 commits ahead of that branch point (3c062d391, 2026-10-11).
- **What closes the gap.** One change matters for batch-1 decode on this chip: the RDNA hybrid W4A16
  kernel (vLLM #40977, merged 2026-07-14, first released in v0.26.0). It replaces the Conch Triton path
  that serves every 4-bit layer today. Its own benchmarks on a Strix Halo cut time per output token 3.2x
  to 4.8x on 4B to 8B models. The follow-ups move batch-1 decode on our symmetric checkpoint by a few
  percent at most.
- **No tag takes the patch as it is.** The nine hunks apply only at v0.19.1. From v0.20.0 on, the build
  hunks and the attention hunks fail. From v0.22.0 on, ROCm builds a stable-ABI extension pinned to the
  torch 2.11 C shim, and the pod has torch 2.9.1. Every release that carries the hybrid kernel therefore
  also needs a new torch and Triton on the pod.
- **Smallest path.** Backport the hybrid kernel onto b1388b1: two files copied from main and about 65
  added lines in six files (1,519 added lines in all). It stays inside the legacy `_rocm_C` extension, so
  the pod's torch 2.9.1 still serves. All nine hunks still apply on top of it. The rebuild is the
  existing recipe (about 5 min to build, about 9 min to a cold engine), so it fits one 30-minute GPU
  slot.
- **Prediction.** 2.88 to about 8 tok/s at batch 1 (range 7 to 9), at medium confidence (0.6). That
  reaches the vLLM prior but stays at about two thirds of llama.cpp's measured 12.4 tok/s for the same
  model class. Whether the teacher should change engines is a decision for its owner, once R58 and the
  backport have both been measured on a pod.

## The pin and the version label

- **[practitioner]** vLLM's v0.19.1 release points at commit b1388b1fbf5aaef47937fabe98931211684666a6
  and is described as "a patch release on top of v0.19.0". [S172] vLLM release page v0.19.1
  ([link](https://github.com/vllm-project/vllm/releases/tag/v0.19.1))
  - Implication: none for the metric set. The engine label in `lab/refine/metrics.jsonl` rows
    ("0.19.2+gfx1151") comes from `SETUPTOOLS_SCM_PRETEND_VERSION` in `lab/pod_a_teacher.sh::build`,
    not from a vLLM release. A ledger reader should read it as v0.19.1 plus the patch. No v0.19.2
    release exists. The only v0.19.2 tag is v0.19.2rc0 on main.
  - Candidate experiment: VU3, rerun the patch check on every new vLLM tag (defined in "Our nine hunks
    against every later tag"). Expected movement: none on decode; it keeps the engine label honest.
    Minimum sample size: 1 run per tag. Confidence: high (0.9).

- **[measured]** In a clone of vLLM, `git merge-base b1388b1 origin/main` is d816834c1 (2026-03-29).
  b1388b1 is 32 commits ahead of it on `releases/v0.19.1`, and main is 7,486 commits ahead of it at
  3c062d391 (2026-10-11). The releases since then are v0.20.0 (2026-04-27), v0.21.0, v0.22.0 and
  v0.22.1, v0.23.0, v0.24.0, v0.25.0 and v0.25.1, v0.26.0 (2026-07-26), v0.27.0 and v0.27.1, v0.28.0,
  v0.29.0, v0.30.0 (2026-09-21) and v0.31.0 (2026-10-02 commit, published 2026-10-05). [S173] vLLM git
  history read with `git log`, `git tag` and `git merge-base` at 3c062d391
  - Implication: none for the metric set. Every commit below is "after b1388b1" in the sense the task
    asks: it is on main after the branch point, and so it is missing from our build.
  - Candidate experiment: VU3. Expected movement: none. Minimum sample size: 1 run per tag.
    Confidence: high (0.9).

## The commits that touch our path

### W4A16 decode kernels

- **[measured]** vLLM #40977 (merged 2026-07-14 as 382bbd5, first released in v0.26.0) adds
  `RDNAHybridW4A16LinearKernel` for gfx11 and gfx12. It sends M <= 5 rows to the HIP skinny GEMM
  `wvSplitK_int4_g` and larger batches to a Triton fused-dequant GEMM, with Triton tiles tuned for
  gfx1151 and gfx1201. On a Strix Halo (gfx1151) at `--max-num-seqs 1`, 128 tokens in and 128 out,
  time per output token fell from 54.81 to 13.69 ms on the symmetric `RedHatAI/Qwen3-4B-quantized.w4a16`
  (4.00x) and from 116.99 to 24.55 ms on the asymmetric `RedHatAI/Qwen3-8B-quantized.w4a16` (4.77x).
  GSM8K on 200 samples went from 0.835 to 0.855. [S13] vLLM PR #40977, description and benchmark tables
  ([link](https://github.com/vllm-project/vllm/pull/40977)), re-opened 2026-10-11; release membership
  from [S182] and [S173]
  - Implication: this is the only commit in the audit that changes the bytes-per-second of the 4-bit
    layers, which hold 12.6 of the 15.5 GB a decode step reads
    ([w4a16_decode_gfx1201.md section 1](w4a16_decode_gfx1201.md#1-baseline-and-byte-budget)). It is
    the change that can close the gap to the prior. Our checkpoint is compressed-tensors, symmetric,
    group 128, bf16, which the kernel accepts (the same note, section 3).
  - Candidate experiment: VU1, b1388b1 plus the nine hunks plus the backport (section "The smallest
    upgrade path"), served on pod B (account D) in one GPU slot. Arm A is the default kernel choice
    (the hybrid kernel). Arm B sets `VLLM_DISABLED_KERNELS=RDNAHybridW4A16LinearKernel` on the same
    build (Conch). Both use the R29 text prompt at batch 1, with the serve log naming the kernel class
    for each spawn and the R21 correctness gate on arm A. Expected movement: teacher.decode_tok_s 2.88
    to 7 to 9 at batch 1, and teacher.joules_per_token about 42 to 13 to 17 at a socket power near
    120 W. Minimum sample size: 5 cold spawns per arm, 256 generated tokens each. Confidence: medium
    (0.6).

- **[measured]** vLLM #54965 (merged 2026-09-14, in v0.30.0) keeps asymmetric zero points packed 4-bit
  in the skinny and Triton paths. It cut TPOT 1.49% on the asymmetric Qwen3-8B w4a16 on gfx1151, and it
  moved the symmetric Qwen3-4B control by +0.15%, within noise, because "symmetric layers have no
  zero-point tensor". It also changes the op's zero-point argument format. [S17] vLLM PR #54965
  ([link](https://github.com/vllm-project/vllm/pull/54965)), re-opened 2026-10-11
  - Implication: no decode gain for our symmetric checkpoint. A backport should still copy the
    kernel, its binding and the Python module from one commit, because the op signature changed.
  - Candidate experiment: none beyond VU1. Expected movement: none on teacher.decode_tok_s. Minimum
    sample size: as VU1. Confidence: high (0.8).

- **[measured]** vLLM #56301 (merged 2026-10-06, on main only, not in any release) pads W4A16 weight
  rows whose stride is a multiple of 1024 bytes and activation rows whose stride is a multiple of
  2048 bytes, gated to gfx1151. It does not pad activations on decode. TPOT fell 3.0% on Qwen3-8B w4a16
  (128 in, 128 out) and 2.3% on Llama-3.1-8B w4a16, and did not change on Qwen3-4B. [S18] vLLM PR #56301
  ([link](https://github.com/vllm-project/vllm/pull/56301)), re-opened 2026-10-11
  - Implication: worth 0% to 3% at batch 1. Our K of 5120 and 17408 at 4 bits gives 2560- and
    8704-byte packed rows, which are not multiples of 1024, so the weight padding may not even fire on
    those layers. The 6144-wide layers (3072-byte rows) also miss it.
  - Candidate experiment: none beyond VU1. Expected movement: 0% to +3% decode, inside VU1's range.
    Minimum sample size: as VU1. Confidence: low (0.3) that it is measurable.

- **[measured]** vLLM #52619 (merged 2026-10-07, on main only) raises the HIP skinny dispatch limit
  from K x M <= 32,768 to 39,321, with no C++ change. On a Radeon 8060S at batch 3, Qwen3-8B GPTQ
  aggregate decode went from 41.70 to 68.16 tok/s (+63%). The PR reports no batch-1 number. [S16] vLLM
  PR #52619 ([link](https://github.com/vllm-project/vllm/pull/52619)), re-opened 2026-10-11
  - Implication: none at batch 1. It matters for labeling at batch 2 to 5, where our `down_proj`
    (K = 17,408) would otherwise fall to Triton at M >= 2. The backport takes the module from main, so
    it includes this change.
  - Candidate experiment: VU1 extended to batch 4. Expected movement: aggregate teacher.decode_tok_s at
    batch 4 above 3x the batch-1 rate on the hybrid arm. Minimum sample size: 3 spawns at batch 4 per
    arm. Confidence: low (0.4).

- **[practitioner]** vLLM #41394 (merged 2026-05-29, in v0.23.0) adds a native W4A16 kernel for RDNA3.
  The PR text says it is "gated to gfx11 only; CDNA / MI3xx / RDNA4 fall through" through `on_gfx11()`,
  and a reviewer noted that this also matches gfx115x. On main, `RDNA3W4A16LinearKernel.can_implement`
  returns "RDNA3 W4A16 kernel requires gfx1100" on any other GPU. The split-K accuracy fix #54706
  (2026-10-03, main only) touches only that kernel's files (`csrc/rocm/q_gemm_rdna3*.cu`,
  `qdq_4_rdna3.cuh`, `moe_q_gemm_rdna3.cu`), and #60388 (2026-10-08) only its MoE path. [S19] vLLM PR
  #41394 ([link](https://github.com/vllm-project/vllm/pull/41394)), re-opened 2026-10-11; [S192] vLLM
  `rdna3_w4a16.py` at 3c062d391, `can_implement`
  ([permalink](https://github.com/vllm-project/vllm/blob/3c062d391d15cfa41d3f321cad24f3030c892a97/vllm/model_executor/kernels/linear/mixed_precision/rdna3_w4a16.py));
  file lists from [S173]
  - Implication: on main none of the three runs on gfx1151, so none moves our decode. A tag between
    v0.23.0 and the gfx1100 gate would need checking, because there the RDNA3 kernel may claim gfx1151
    ahead of the hybrid kernel. The serve log's kernel class settles it per spawn. The hybrid kernel's
    accuracy is covered by its own tests, not by #54706.
  - Candidate experiment: none beyond VU1's correctness gate. Expected movement: none. Minimum sample
    size: as VU1. Confidence: high (0.8).

- **[practitioner]** The Triton W4A16 kernel for ROCm (#37352, 2026-04-10, v0.20.0; a fused int4
  dequantize-and-GEMM benchmarked on MI300X, with block sizes later tuned for gfx1151) and its fixes,
  #39705 (2026-04-13, v0.20.0), #47770 (2026-07-15, v0.26.0), #48998 (2026-08-17, v0.29.0) and #51453
  (2026-09-01, v0.30.0, registered as a custom op), sit below the hybrid kernel in the ROCm kernel list.
  #45072 (2026-06-10, v0.24.0) makes Conch skip layers that reorder by `g_idx`. [S12] vLLM PR #37352
  ([link](https://github.com/vllm-project/vllm/pull/37352)), re-opened 2026-10-11; the other titles,
  dates and releases from [S173]
  - Implication: on gfx1151 the hybrid kernel is picked first (the ROCm list at 382bbd5, [S170]), and
    its prefill path carries its own Triton GEMM, so none of these is needed for a backport. Our
    checkpoint has static activation order, so #45072 changes nothing for Conch either.
  - Candidate experiment: none beyond VU1. Expected movement: none. Minimum sample size: as VU1.
    Confidence: medium (0.6).

### Unquantized skinny GEMMs (the bf16 `lm_head`)

- **[measured]** vLLM #40784 (merged 2026-06-25, in v0.25.0) retunes the bf16 and fp16 `wvSplitK`
  skinny GEMM for gfx1151 only, using YTILE 1 for memory-bound decode shapes. Kernel bandwidth rose
  18.7% at N = 1 in bf16. End-to-end median TPOT on a Strix Halo (LPDDR5X-8000, 128 in and 128 out)
  fell 9.2% on Qwen3-8B, 5.6% on gemma-3-4b-it and 1.3% on Qwen3-4B. [S174] vLLM PR #40784
  ([link](https://github.com/vllm-project/vllm/pull/40784))
  - Implication: our bf16 `lm_head` (2.54 GB, about 16% of a step's bytes) runs this kernel at batch 1
    on b1388b1 ([settings_inference_gfx1151.md](settings_inference_gfx1151.md#unquantized-gemms-the-skinny-kernels-hipblas-and-hipblaslt-tunableop)).
    An 18.7% faster `lm_head` is worth about 2% to 4% of a step once the 4-bit layers are fast. It is
    a separate edit to `csrc/rocm/skinny_gemms.cu` and is not in the backport below.
  - Candidate experiment: VU1b, VU1's hybrid arm with and without #40784's `skinny_gemms.cu` change.
    Expected movement: teacher.decode_tok_s +2% to +4%. Minimum sample size: 5 cold spawns per arm.
    Confidence: low (0.35).

- **[measured]** vLLM #58566 (merged 2026-09-25, in v0.31.0) builds the activation's contiguous copy in
  `rocm_unquantized_gemm_impl` only on the skinny branches that use it. On 8x MI355X with Kimi-K3 it
  removed 69 copies per decode step (0.34 ms of a 23.5 ms step), and the author does not claim an
  end-to-end gain. [S175] vLLM PR #58566 ([link](https://github.com/vllm-project/vllm/pull/58566))
  - Implication: the copy is real work only for a non-contiguous input. On our model the bf16 GDN
    `in_proj_a` and `in_proj_b` take this path, so the saving is at most a few launches per layer,
    under 1% of a 125 ms step.
  - Candidate experiment: none beyond VU1. Expected movement: under 1%. Minimum sample size: as VU1.
    Confidence: medium (0.5).

### Attention and the rest of the ROCm path

- **[practitioner]** Upstream never took our `num_stages=1`. #40631 (merged 2026-04-24, in v0.20.0)
  merges the 2D and 3D unified attention kernels into one launch gated by `IS_3D`. #45163 (2026-06-11,
  v0.24.0) then adds `launch_num_warps` and `launch_num_stages`, which main sets only for head size 256
  prefill on device capability family 100 (Blackwell), and otherwise leaves to Triton. [S179] vLLM PR
  #40631 ([link](https://github.com/vllm-project/vllm/pull/40631)); [S188] vLLM
  `vllm/v1/attention/ops/triton_unified_attention.py` at 3c062d391, `unified_attention`
  ([permalink](https://github.com/vllm-project/vllm/blob/3c062d391d15cfa41d3f321cad24f3030c892a97/vllm/v1/attention/ops/triton_unified_attention.py))
  - Implication: on any tag from v0.20.0 our two attention hunks fail. They become one line,
    `launch_num_stages = 1` on gfx1x, in front of the existing `launch_kwargs` block. The chip sheet
    measured the same 65,536-byte LDS on gfx1151 as on gfx1201, so the hunk stays needed.
  - Candidate experiment: VU2's attention check: on a rebuilt tag, serve once without the line and
    look for Triton `OutOfResources` at warm-up. Expected movement: none on decode; it decides whether
    the line is needed on gfx1151. Minimum sample size: 1 spawn. Confidence: medium (0.6) that it is
    still needed.

- **[practitioner]** Three attention commits target RDNA after our pin: #58225 (2026-09-23, v0.31.0)
  narrows the Triton prefill KV tile on RDNA3 and RDNA4, #59132 (2026-10-06, main only) adds a
  `ROCM_SEGMENTED_ATTN` backend for RDNA3, 3.5 and 4, and #60156 (2026-10-08, main only) keeps small
  FP8 softmax weights in Triton attention. Titles, dates and files from [S173]; the PR pages of #58225
  and #59132 were opened by an earlier pass and not re-opened here
  - Implication: only 16 of 64 layers are full attention, and at a short context they read 64 KB of KV
    per token, so attention is a small share of a decode step. None of the three is a lever for
    batch-1 decode. #58225 may help TTFT on images.
  - Candidate experiment: none for decode. Expected movement: under 2% on teacher.decode_tok_s.
    Minimum sample size: none. Confidence: medium (0.5).

- **[practitioner]** Other ROCm-path commits after the pin: #38455 (2026-04-10, v0.20.0) adds amdsmi
  device names for gfx1150, gfx1151 and gfx1201 (`0x1586` is "AMD_Radeon_8060S"); #37712 (2026-04-20,
  v0.20.0) enables the FP8 `wvSplitK` path on RDNA; #45000 (2026-07-13, v0.26.0) fixes a GDN warm-up
  regression on RDNA by avoiding an fp32 `tl.dot`; #50907 (2026-08-11, v0.28.0) removes "stale SDPA
  and skinny GEMM workarounds"; and #57750 (2026-10-06, main only) disables AITER attention query
  quantization on RDNA3. Diffs of #38455 and #45000 and the titles of the rest from [S173]
  - Implication: none moves batch-1 decode on our checkpoint. #38455 helps only the amdsmi path, which
    our pods avoid. We run no FP8 and no AITER. #45000 may shorten engine start, which is the only
    item here a later full upgrade would bring for free.
  - Candidate experiment: none. Expected movement: none on teacher.decode_tok_s. Minimum sample size:
    none. Confidence: medium (0.5).

## Our nine hunks against every later tag

- **[measured]** `git apply --cached --check lab/vllm-gfx1201.diff` against each tag's tree applies
  cleanly only at v0.19.1. From v0.20.0 the pyproject, `requirements/build.txt`,
  `requirements/rocm-build.txt` and unified attention hunks fail. From v0.21.0 `compat.cuh` fails too,
  and from v0.22.0 the file is gone. From v0.27.1 `vllm/platforms/rocm.py` fails, and from v0.29.0
  `vllm/platforms/__init__.py` fails. At v0.29.0, v0.31.0 and main all seven files fail. [S191] the
  patch check, run on 2026-10-11 against v0.19.1, v0.20.0, v0.20.2, v0.21.0, v0.22.0, v0.22.1, v0.23.0,
  v0.24.0, v0.25.1, v0.26.0, v0.27.1, v0.28.0, v0.29.0 and main 3c062d391 of [S173]
  - Implication: no release after our pin keeps the patch applying, so "a newer tag with the nine hunks
    unchanged" does not exist. Any tag upgrade means rewriting the patch, which kernels rubric K2 treats
    as an engine change on top of the kernel change.
  - Candidate experiment: VU3, rerun the same check on each new tag and on the backport branch.
    Expected movement: none on decode. Minimum sample size: 1 run per tag. Confidence: high (0.9).

- **[practitioner]** The collisions have the following upstream causes, each read in the PR or in the
  file at the tag:

  | Hunk (file) | First tag that rejects it | Upstream cause | What a rebased patch does |
  | --- | --- | --- | --- |
  | 1-2 `pyproject.toml` (torch pin, uv) | v0.20.0 | #34644 (2026-04-08) pins torch 2.11.0 | drop; run `use_existing_torch.py` (at `tools/` from v0.31.0) |
  | 3 `requirements/build.txt` | v0.20.0 | #39024 (2026-04-10) moves it to `requirements/build/*.txt` | drop; same script |
  | 4 `requirements/rocm-build.txt` | v0.20.0 | #39024 | drop; same script |
  | 5 `vllm/platforms/__init__.py` (torch.version.hip fallback) | v0.29.0 | #38434 (2026-08-30) adds a fallback only under `in_wsl()` | keep, rewritten without the WSL gate |
  | 6 `vllm/platforms/rocm.py` (no `warning_once`) | v0.27.0 | #46516 (2026-07-31) removed the `warning_once` call | drop: upstream did it |
  | 7-8 `triton_unified_attention.py` (`num_stages=1`) | v0.20.0 | #40631 (2026-04-24) one launch for 2D and 3D | keep as one `launch_num_stages = 1` line on gfx1x |
  | 9 `csrc/quantization/gptq/compat.cuh` | v0.21.0 | #41802 (2026-05-08) guards the overloads on HIP < 7.13 | drop: upstream did it |

  Sources: [S181] vLLM PR #34644 ([link](https://github.com/vllm-project/vllm/pull/34644)); [S177] vLLM
  PR #38434 ([link](https://github.com/vllm-project/vllm/pull/38434)) and [S187] `rocm_platform_plugin`
  at 3c062d391 ([permalink](https://github.com/vllm-project/vllm/blob/3c062d391d15cfa41d3f321cad24f3030c892a97/vllm/platforms/__init__.py));
  [S178] vLLM PR #46516 ([link](https://github.com/vllm-project/vllm/pull/46516)) and its diff to
  `vllm/platforms/rocm.py` in [S173]; [S179] vLLM PR #40631; [S176] vLLM PR #41802
  ([link](https://github.com/vllm-project/vllm/pull/41802)); #39024 from [S173]
  - Implication: at v0.29.0 or later the rebased patch shrinks to two hunks: the non-WSL torch fallback
    and the attention line. Upstream's WSL-only fallback does not cover our pods, which are not WSL and
    have no `amdsmi` Python package (`lab/README.md`, "Teacher on the Strix Halo (pod A, account C)").
  - Candidate experiment: VU3. Expected movement: none on decode. Minimum sample size: 1 run per tag.
    Confidence: high (0.8).

## Torch and Triton: what a tag upgrade costs

- **[practitioner]** From v0.22.0 the ROCm build compiles `_C_stable_libtorch` with `USE_ROCM`. #42663
  (merged 2026-05-20) moved the GPTQ, activation and int8 kernels there and enabled the stable
  extension for ROCm. From v0.23.0 to v0.26.0 the extension is built with
  `TORCH_TARGET_VERSION=0x020B000000000000ULL`, which CMake explains as "only use C-shim APIs available
  in PyTorch 2.11". [S180] vLLM PR #42663 ([link](https://github.com/vllm-project/vllm/pull/42663));
  [S186] vLLM `CMakeLists.txt` at v0.26.0 (568afb3)
  ([permalink](https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/CMakeLists.txt))
  - Implication: the pod's torch 2.9.1+rocm7.13 predates that C shim. A build of v0.26.0 or later
    against it is expected to fail at compile or load time. So the first release with the hybrid kernel
    also needs torch 2.11 or later for ROCm 7.13 on gfx1151, which the image does not ship.
    `lab/README.md` already names `repo.radeon.com/rocm/manylinux` as the place such a torch would
    come from. That is inference from the code, not a measured build failure.
  - Candidate experiment: VU2, on pod D outside any serving slot: check out v0.26.0, run
    `use_existing_torch.py`, apply the two rewritten hunks, and run `pod_a_teacher.sh build` against
    torch 2.9.1. Expected movement: none on decode; the build fails in `_C_stable_libtorch` (or the
    import fails), which confirms that a tag upgrade includes a torch upgrade. Minimum sample size:
    1 build. Confidence: medium (0.7) that it fails.

- **[practitioner]** The ROCm pins move further. v0.26.0 pins torch 2.11.0 and Triton 3.6.0.
  v0.30.0, v0.31.0 and main pin `torch==2.12.0+rocm7.14.0` and `triton==3.7.1+git0263a6a6` in
  `requirements/build/rocm.txt`, with torch 2.13.0 in `pyproject.toml`. The v0.31.0 release notes list
  "torch 2.13, Triton 3.8" for ROCm, and the wheel index is `rocm723`. [S189] vLLM
  `requirements/build/rocm.txt` and `pyproject.toml` at v0.31.0 (db9527a)
  ([permalink](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/requirements/build/rocm.txt));
  [S183] vLLM release page v0.31.0 ([link](https://github.com/vllm-project/vllm/releases/tag/v0.31.0));
  [S182] vLLM release page v0.26.0 ([link](https://github.com/vllm-project/vllm/releases/tag/v0.26.0))
  - Implication: the pod runs Triton 3.5.1+rocm7.13.0 (`lab/README.md`). A tag upgrade replaces
    torch, Triton and maybe ROCm user space at once, and the vision encoder fix (`--mm-encoder-attn-backend
    TRITON_ATTN`, chosen because torch 2.9.1's AOTriton kernels were wrong on gfx1151) would need
    re-validating. Triton 3.8.0's release notes enable in-thread transpose on RDNA 3 and 3.5 ([S185]),
    which could change the Triton prefill path, but nothing there is a decode lever for the HIP kernel.
  - Candidate experiment: none before VU1 reads out. If VU1 lands, a tag upgrade buys nothing more for
    batch-1 decode. Expected movement: none. Minimum sample size: none. Confidence: medium (0.6).

- **[anecdote]** The ROCm/vllm fork's releases page shows only 2025 releases (v0.9.2 and older, with
  "Navi" build fixes) and no gfx1151, Strix Halo or W4A16 item. [S184] ROCm/vllm releases page
  ([link](https://github.com/ROCm/vllm/releases))
  - Implication: none for the metric set. The fork offers no newer packaged build for this chip; the
    upstream kernel is the one to take.
  - Candidate experiment: none. Expected movement: none. Minimum sample size: none. Confidence: high
    (0.8).

## The smallest upgrade path

- **[measured]** A backport of the hybrid kernel onto b1388b1 in a scratch worktree took
  `csrc/rocm/skinny_gemms_int4.cu` (823 lines) and
  `vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py` (631 lines) verbatim from
  main 3c062d391. It added the op declaration to `csrc/rocm/ops.h` (6 lines), its registration to
  `csrc/rocm/torch_bindings.cpp` (8), the `wvSplitK_int4_g` wrapper to `vllm/_custom_ops.py` (36),
  `on_gfx1151` and `on_gfx12x` to `vllm/platforms/rocm.py` (10), the kernel at the head of the ROCm
  list in `vllm/model_executor/kernels/linear/__init__.py` (4), and the source to `_rocm_C` in
  `CMakeLists.txt` (1): 8 files, 1,519 added lines, none removed. Every name the module imports exists
  at b1388b1, the four touched Python files byte-compile, and all nine hunks of
  `lab/vllm-gfx1201.diff` still pass `git apply --check` on top. The HIP code was not compiled, because
  the workstation has no `hipcc`. [S190] vLLM `rdna_hybrid_w4a16.py` and `skinny_gemms_int4.cu` at
  3c062d391
  ([permalink](https://github.com/vllm-project/vllm/blob/3c062d391d15cfa41d3f321cad24f3030c892a97/vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py));
  [S191] the patch check
  - Implication: this is the smallest change that keeps the nine-hunk patch applying. It stays on the
    pod's torch 2.9.1 because the kernel includes `torch/all.h` and registers on the legacy `_rocm_C`
    extension, which b1388b1 already builds. It ships as a second patch next to
    `lab/vllm-gfx1201.diff`, or as new hunks in it. The R40 owner decides which; this note edits no code.
  - Candidate experiment: VU1 (defined under #40977). Expected movement: as there. Minimum sample size:
    5 cold spawns per arm. Confidence: medium (0.6).

- **[measured]** The rebuild cost on a pod is the recipe that exists today. On pod C the build took
  4 min 18 s on local disk with 20 compile jobs, and the engine was ready about 5 min after launch with a
  warm torch.compile cache. On pod D the build took 292 s, dependencies included, and the engine was
  ready 523 s after a cold start. [L4] `lab/README.md`, sections "Teacher on the Strix Halo (pod A,
  account C)" and "The same build on pod D, from scripts alone (R30)"
  - Implication: a backport adds one 823-line HIP file to `_rocm_C`, so the build stays near 5 min.
    The changed kernel invalidates the compile cache, so the first serve is a cold one of about 9 min:
    about 15 min to a served teacher, inside one 30-minute GPU slot. A tag upgrade adds a new torch
    and Triton, the vision-encoder revalidation and the rewritten patch, which is hours of pod time with
    an uncertain outcome.
  - Candidate experiment: VU1's build and serve, with `### build wall_s` and the serve's ready time
    recorded. Expected movement: build 250 to 350 s; engine ready 450 to 600 s cold. Minimum sample
    size: 1 build, 5 spawns. Confidence: high (0.8).

## Predicted decode gain

- **[opinion]** The gain, by arithmetic on the byte budget and the PR's measurements. A step at
  2.88 tok/s takes 347 ms. The parts that are not 4-bit (the bf16 `lm_head` at 2.54 GB, the GDN state at
  about 0.3 GB, attention, norms and launches) take about 30 to 45 ms if they move at 70% to 90% of the
  chip's 209.5 GB/s copy rate, plus a fixed overhead. That leaves about 300 ms for the 12.6 GB of 4-bit
  layers, about 42 GB/s. If the hybrid kernel's 3.4x to 4.8x end-to-end gain on 4B to 8B models
  carries over to those layers, they take 65 to 90 ms, and a step takes 100 to 135 ms: 7.4 to 10 tok/s.
  If the kernel instead reaches 60% to 85% of the copy rate on our shapes, they take 70 to 100 ms: 7 to
  9 tok/s. Central estimate about 8 tok/s, below the 13.5 tok/s ceiling. Byte budget from
  [w4a16_decode_gfx1201.md section 1](w4a16_decode_gfx1201.md#1-baseline-and-byte-budget); ceiling and
  copy rate from [w4a16_decode_gfx1201.md section 9](w4a16_decode_gfx1201.md#9-pass-3-additions-the-gfx1151-ceiling-the-hybrid-kernel-on-this-chip-and-llamacpp-beside-it);
  speedups from [S13]; our 2.88 from `lab/refine/metrics.jsonl:24`
  - Implication: the backport is predicted to take the teacher from 38% to about 105% of the public
    vLLM prior (7.5 tok/s, [public_strix_halo_benchmarks.md](public_strix_halo_benchmarks.md#prediction-check-against-our-own-numbers)),
    and joules per token from about 42 to about 15 at 120 W. Every commit after the hybrid kernel adds
    at most about 5% at batch 1 on our symmetric checkpoint (#56301, #40784, #58566). So moving further
    forward in vLLM's history does not close the remaining gap to llama.cpp.
  - Candidate experiment: VU1. Expected movement: teacher.decode_tok_s 2.88 to 7 to 9 at batch 1.
    Minimum sample size: 5 cold spawns per arm. Confidence: medium (0.6).

- **[opinion]** Against llama.cpp: an earlier pass recorded Qwen3.8-27B in Q4_K_M at 12.4 tok/s
  (`tg128`, Vulkan) and a dense 27B Qwen at 12.05 tok/s (HIP) on a Strix Halo, which is the chip's copy
  rate. [w4a16_decode_gfx1201.md section 9](w4a16_decode_gfx1201.md#9-pass-3-additions-the-gfx1151-ceiling-the-hybrid-kernel-on-this-chip-and-llamacpp-beside-it)
  - Implication: even the best vLLM path found here (about 8 tok/s) stays at about 65% of llama.cpp at
    batch 1. Switching engines gains decode, but it costs the vLLM features the teacher uses today:
    the `qwen3_coder` tool-call parser, `--scheduling-policy priority` for the group brain, two served
    names, the vision encoder path, and continuous batching for labeling. The choice is the owner's,
    once R58 (llama.cpp Vulkan on the pod) and VU1 have both reported.
  - Candidate experiment: R58 as queued, read beside VU1 at the same prompt and batch. Expected
    movement: llama.cpp 11 to 13 tok/s against the hybrid kernel's 7 to 9 at batch 1. Minimum sample
    size: 5 cold spawns per arm. Confidence: medium (0.55).

## Candidate experiments, ranked

1. VU1: the backport of the hybrid kernel onto b1388b1, against Conch on the same build, at batch 1 and
   batch 4 (expected 2.88 to 7 to 9 tok/s; confidence 0.6; one 30-minute slot on pod D).
2. VU1b: VU1's hybrid arm with #40784's gfx1151 `wvSplitK` tuning added (expected +2% to +4%;
   confidence 0.35).
3. VU2: build v0.26.0 against torch 2.9.1 on pod D, without serving, to confirm that a tag upgrade
   needs a new torch (no decode movement; confidence 0.7).
4. VU3: rerun the patch check on every new vLLM tag and on the backport branch (bookkeeping; no pod).

## Open questions

- Does `skinny_gemms_int4.cu` from main compile under ROCm 7.13's `hipcc` for gfx1151 without the
  later `skinny_gemms.cu` changes? Not tested here, because there is no HIP compiler on the
  workstation. VU1's build answers it.
- Is a torch 2.11 or later wheel for ROCm 7.13 and gfx1151 available on `repo.radeon.com/rocm/manylinux`?
  Not opened.
- PR #40784's page mentions a ROCm fork commit (7648cc8) that restored the gfx1151 `wvSplitK` tile
  routing after an upstream dispatch change. Not opened; it may mean main's routing for gfx1151 regressed
  after June.
- The R47 decode profile has not run on a pod yet, so the split between 4-bit layers and everything
  else in the prediction is arithmetic, not a measurement.

## Search log

- 2026-10-11, R59. Budget about 3 h, used about 2 h. Starting points: `cloud/tasks/R59.md`,
  `lab/vllm-gfx1201.diff`, this folder's `w4a16_decode_gfx1201.md`, `public_strix_halo_benchmarks.md`,
  `settings_inference_gfx1151.md`, `triton_gfx12.md`, `rdna4_vllm_ports.md`, `lab/README.md` and
  `lab/pod_a_teacher.sh`. Cloned vLLM with `--filter=blob:none` and ran `git log` from the branch point
  d816834c1 to 3c062d391 with `--grep` for gfx1151, gfx115, gfx12, gfx1201, rdna, w4a16, conch,
  skinny, wvSplit and GDN, plus a path log of `triton_unified_attention.py` and `platforms/rocm.py`.
  Mapped each commit to its first release with `merge-base --is-ancestor`, checked the patch against 14
  trees, and tried the backport in a scratch worktree. Opened PRs #40977, #40784, #58566, #54965,
  #56301, #52619, #41802, #38434, #46516, #40631, #42663 and #34644, the v0.19.1, v0.26.0 and v0.31.0
  release pages, the ROCm/vllm releases page and the Triton releases page. Stopped when the last three
  sources opened (the ROCm/vllm releases, the Triton releases and #34644) added nothing that changes the
  recommendation.
