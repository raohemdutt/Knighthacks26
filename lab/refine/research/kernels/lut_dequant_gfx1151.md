# Does a lookup-table dequant GEMV make 3-bit and 2-bit weights pay on gfx1151?

R62 ported the LUT-GEMM and FLUTE idea (turn each weight code into a value by a table lookup on chip instead of
unpack-and-scale arithmetic) to one Triton kernel for the Strix Halo (gfx1151, Radeon 8060S) and measured it on pod-b
against the chip sheet's W4A16 probe, torch fp16 and the copy ceiling. It covers the decode GEMV at M 1 to 16 on the
27B teacher's shapes; it does not cover prefill, a full model, output quality of a real 3-bit checkpoint, or the vLLM
kernel `RDNAHybridW4A16LinearKernel`, which was not run here.

## Summary

The 3-bit lookup-table GEMV beats the chip sheet's W4A16 probe at M=1 in bytes per second and, narrowly, in time, so
the kill criterion is not met. The per-group table it was asked to use costs 16 bytes per 128 weights at 3 bits,
which makes a 3-bit layer exactly as large as the int4 layer it replaces, so the time saved per layer is about 8
percent with weights streamed from DRAM. The plain scale-and-zero arithmetic in the same packing runs at half the
speed of the table at 5120x6144, which is the paper's premise holding on this chip. The layout that turns 3 bits into a real byte
saving (one table per output channel plus a per-group scale, FLUTE's) is committed as mode 4 but not yet measured;
its predictions below are marked as such.

## Prior work and ports

- **[measured]** LUT-GEMM removes the dequantization step of weight-only quantized GEMM with lookup tables and group-wise quantization, and reports a 2.1x speedup over OPTQ for OPT-175B at 3 bits on a single GPU. Park et al., ICLR 2024, [S183](https://arxiv.org/abs/2206.09557)
  - Implication: the premise is that unpacking arithmetic, not DRAM, limits a sub-4-bit decode GEMV; our arith-against-table comparison below tests exactly that on gfx1151.
  - Candidate experiment: none beyond KL1 (below). Expected movement: none on its own. Minimum sample size: none. Confidence: not applicable.
- **[measured]** FLUTE runs 2 to 4x faster than existing GEMM kernels at batch below 32 and group 128 by restructuring the quantized weights offline and vectorizing and duplicating the lookup table in shared memory, with 1.5 to 2x end-to-end throughput on LLaMA3. Guo et al., EMNLP 2024 Findings, [S184](https://arxiv.org/abs/2407.10960)
  - Implication: FLUTE's table is shared across the matrix with a per-group scale, not one table per group; that is mode 4 here, and it is the layout that saves bytes at 3 bits.
  - Candidate experiment: KL1 (below).
- **[practitioner]** FLUTE's repository is CUDA only (A100, A6000, RTX 4090, H100 unoptimized), with 4- and 3-bit weights, groups 32 to 256 and a vLLM integration; it never mentions AMD, ROCm or HIP. [S185](https://github.com/HanGuo97/flute)
  - Implication: no ROCm, HIP or Triton port of LUT-GEMM or FLUTE was found; the closest AMD work is below, and this kernel is, as far as the search reached, the first on gfx1151.
  - Candidate experiment: none for the metric set; it is a search result.
- **[practitioner]** ROCm hip-ep PR #1017 (merged 2026-09-16) fuses a quantized Gemm chain with per-channel and 4-bit weights into one op with a dedicated M==1 decode kernel; it uses integer-domain dequantization, no table, and states no performance numbers. [S186](https://github.com/ROCm/hip-ep/pull/1017)
  - Implication: the nearest AMD kernel is a fused 4-bit dequant GEMM, not a lookup table, and it gives no bandwidth figure to compare with.
  - Candidate experiment: none for the metric set.
- **[measured]** On gfx1151 the bitsandbytes 4-bit HIP GEMV `kgemm_4bit_inference_naive` reads 117 MB in 1128 us, about 102 GB/s or 49 percent of a measured 210 GB/s, while rocBLAS bf16 reaches about 191 GB/s (91 percent). [S187](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1842)
  - Implication: a 4-bit decode GEMV at half the bandwidth is the known state of the art outside vLLM on this chip; our numbers below sit well above it.
  - Candidate experiment: none beyond KL1.
- **[anecdote]** Triton discussion #1851 (2023, unanswered) asked how to write a lookup-table matmul in Triton and found the element access it needed missing; a 2025 reply points at the then-new gather op, untested. [S188](https://github.com/triton-lang/triton/discussions/1851)
  - Implication: `tl.gather` is that op; it compiles and runs on gfx1151 with Triton 3.5.1 (all `group_lut` rows below), which answers the thread for this chip.
  - Candidate experiment: none for the metric set.

## Method

One Triton kernel, `lab/kernels/lut_gemv.py` [L26], with BITS as a constexpr (2, 3 or 4). Codes pack along K to
exactly BITS bits a weight: 4 bits as eight nibbles a word, 2 bits as sixteen pairs a word, 3 bits as a 2-bit field
plus a 1-bit plane. Groups are 128 along K. The modes turn a code into a weight as follows: `arith` (0), the per-group
scale and zero, `(c - z) * s`; `group_lut` (1), a per-group, per-column fp16 table of the 2^BITS values loaded into
registers and read with `tl.gather`; `group_lut_ld` (2), the same table read with a pointer gather; `shared_lut` (3),
one table for the matrix, then scale and zero; `channel_lut` (4, added after the run), one table per output channel and
one scale per group. The table holds a uniform asymmetric grid, so the reference is ordinary 3-bit quantization;
the torch reference gathers the table and multiplies. M=1 uses a vector reduction, M of 4 and 16 use `tl.dot` with
BLOCK_M 16. Timing, watts and the entry format are the chip sheet's own (`time_torch`, the `Power` thread over
`lab/pod_power.py`'s amd-smi reader, `chipsheet_format.entry`) [L24]. Weight GB/s counts every byte the kernel reads for
the weights: codes plus table, or codes plus scale and zero. "Hot" runs the same weights back to back (the chip
sheet's method); "cold" rotates enough copies to pass 256 MB, eight times the 32 MiB cache, so the weights come from
DRAM as they do in decode. Run: pod-b (account D, SJC hub, `jupyter-roscon02`), 2026-10-11 05:17 to 05:19 UTC, torch
2.9.1+rocm7.13.0, Triton 3.5.1, ROCm 7.13.0, 125 s of GPU inside one GPU Queue slot, 11 repeats each (5 in the sweep).

## Kill criterion (written 2026-10-11 1:17 AM ET, before the pod run)

The task ends as a measured failure, with the numbers, if after three tile configurations (BLOCK sizes, split-K and
num_warps; `M1_CONFIGS` in `lab/kernels/lut_gemv.py`, plus the chip sheet's own W4A16 tile) the 3-bit lookup-table
GEMV at M=1 on the 5120x6144 decode shape reads weight bytes slower than the chip sheet's `dequant.w4a16` probe
(`dequant.w4a16.decode27b.k5120_n6144.m1.bn64_bk64`: 0.594 TFLOP/s, which is 106 us for 16.2 MB of int4 codes and
scales, 153 GB/s) re-measured in the same slot. Weight bytes are everything the kernel must read for the weights:
the codes plus the per-group metadata (the table, or the scale and zero).

Outcome: not met. The best of the four tiles (`v_bn32_bk128_s1_w4`) read 212.0 GB/s in the sweep and 206.6 GB/s in the
table run, against 157.0 GB/s for the W4A16 probe in the same slot; with cold weights 149.0 against 140.7 GB/s.

## Measured at M=1

- **[measured]** At M=1 with cold weights on pod-b, weight GB/s (time per GEMV): 3-bit `group_lut` 149.0 (105.6 us), 139.2 (113.0 us), 144.2 (309.1 us), 146.3 (304.6 us) on the 5120x6144, 6144x5120, 5120x17408 and 17408x5120 shapes; 2-bit 103.1 (95.4 us), 95.2 (103.3), 106.4 (261.7), 101.4 (274.7); 4-bit through the same kernel 181.0 (130.3 us), 173.6 (135.9), 184.4 (362.4), 179.2 (373.1); the chip sheet's W4A16 probe in the same slot 140.7 (115.3 us), 134.4 (120.7), 150.2 (305.9), 141.7 (324.3); torch fp16 155.4 (404.8 us), 129.0 (487.7), 140.8 (1266.1), 145.9 (1221.7). The 1 GiB copy in the same slot moved 209 GB/s of DRAM traffic, the chip sheet's 207.8 to 216.3 GB/s ceiling. [L25] `fixtures/lab/lut_gemv/lut_gemv-gfx1151-2026-10-11.json`, rows `lut_gemv.*.cold.*` in `lab/refine/metrics.jsonl`
  - Implication: 3-bit `group_lut` reaches 67 to 71 percent of the copy ceiling with cold weights and is faster per GEMV than the W4A16 probe by 6 to 8 percent on three shapes and 1 percent slower on 5120x17408; 2-bit is faster per GEMV than 3-bit by only 9 to 15 percent while reading 37 percent fewer bytes, so at 2 bits the kernel is no longer bandwidth bound.
  - Candidate experiment: KL1, mode 4 (one table per output channel, per-group scale) at the same points, cold and hot. Expected movement: 3-bit time per GEMV down about 19 percent against `group_lut` (105.6 to about 86 us at 5120x6144) if its weight GB/s holds at 145. Minimum sample size: 11 repeats per point, 4 shapes, one slot. Confidence: medium (0.5): the hot sweep put the matrix-shared table (mode 3, similar arithmetic) 7 percent behind `group_lut` per GEMV.
- **[measured]** With hot weights (the chip sheet's method) the same kernel reads 206.6 GB/s at 3 bits and 307.1 GB/s at 4 bits on 5120x6144, above the 208 GB/s DRAM ceiling, because a 15 to 25 MB layer stays in the 32 MiB cache that the chip sheet measures at 943 GB/s for a 16 MiB copy. [L25]; [L24] `lab/refine/research/kernels/data/chipsheet-gfx1151-2026-10-11.md`
  - Implication: a hot GEMV number on this chip is a cache number for every decode shape up to about 30 MB; decode reads 12 GB a token, so the cold rows are the ones to predict from, and the chip sheet's `dequant.*` rows should be read as hot.
  - Candidate experiment: rerun the chip sheet's dequant section with rotation (the `copies` argument of this bench). Expected movement: `dequant.w4a16` at M=1 down from about 157 to about 141 GB/s. Minimum sample size: 11 repeats per point. Confidence: high (0.8), measured here for the four shapes.
- **[measured]** The plain arithmetic path in the same packing (`arith`, `(c - z) * s` per weight) is slower than the table at 3 bits with hot weights although it reads fewer bytes (3.25 against 4 bits a weight): 141.9 against 76.1 us on 5120x6144 (90.1 against 206.6 GB/s) and 394.4 against 305.3 us on 17408x5120. In the tile sweep the best tiles of the pointer-gather table (`group_lut_ld`) and the matrix-shared table (`shared_lut`) were 4 and 7 percent behind `group_lut` per GEMV. [L25], rows `lut_gemv.b3.arith.*` and the sweep entries in the result file
  - Implication: on gfx1151 the unpack-and-scale arithmetic, not DRAM, limits a 3-bit decode GEMV, which is LUT-GEMM's premise; any 3-bit path in vLLM should dequantize through a table or with fewer vector ops per weight.
  - Candidate experiment: none beyond KL1.
- **[measured]** Max abs error of the kernel's fp16 output against the fp32 torch reference on the same quantized weights was 0.0019 at K 5120 and 6144 and 0.0038 at K 17408 for every mode and BITS (outputs of size about 1.4, so this is fp16 rounding); the W4A16 probe showed 0.0056 to 0.0124 against its own reference. Loaded socket power was 97 to 113 W on average over the cold rows (idle 27 to 67 W read before each section). [L25], rows `*.max_abs_error` and `*.watts`
  - Implication: the table changes no output beyond fp16 rounding; quality is a property of the grid (R55's checkpoint), not of the kernel.
  - Candidate experiment: none for the metric set.

## Measured at M=4 and M=16, and the 4-bit rows stated plainly

- **[measured]** With the `tl.dot` tile (BLOCK_M 16, BLOCK_N 64, BLOCK_K 64), hot weights, 5120x6144: 3-bit `group_lut` 119.4 GB/s at M=4 (131.7 us) and 117.7 GB/s, 7.53 TFLOP/s at M=16 (133.6 us); 2-bit 89.7 and 89.4 GB/s (109.6 and 110.0 us, 9.15 TFLOP/s at M=16); 4-bit 165.2 and 163.1 GB/s (142.8 and 144.6 us); the chip sheet's W4A16 probe 161.9 and 145.8 GB/s (100.2 and 111.2 us, 9.05 TFLOP/s at M=16); torch fp16 116.1 and 113.5 GB/s (542.1 and 554.2 us). [L25], rows `lut_gemv.*.m4.*` and `*.m16.*`
  - Implication: at M=4 the LUT kernel is slower per GEMV than the W4A16 probe at every BITS, and at M=16 only 2 bits matches it (110.0 against 111.2 us), so batched decode should keep a fused-dequant `tl.dot` path; the LUT path is a batch-1 lever.
  - Candidate experiment: KL2, a dot-tile sweep for the LUT kernel at M=16 (BLOCK_K 128 and 256 over two groups, num_warps 8, split-K 2). Expected movement: 3-bit M=16 time per GEMV down to the W4A16 probe's 111 us or below. Minimum sample size: 11 repeats per tile, 2 shapes. Confidence: low (0.3).
- **[measured]** At 4 bits the LUT kernel read weights at 181 GB/s against the W4A16 probe's 141 GB/s with cold weights at M=1 (about 28 percent more bytes per second) and at 163 against 146 GB/s at M=16 (hot). Because the per-group 16-entry table adds 2 bits a weight (6 bits against the probe's 4.125), it was slower per GEMV in both cases: 130.3 against 115.3 us cold at M=1, 144.6 against 111.2 us at M=16. The probe is the chip sheet's plain Triton W4A16 kernel (`chipsheet._w4a16_kernel`, int4 symmetric, group 128, two `tl.dot` calls per K block, BLOCK_M 16, BLOCK_N 64, BLOCK_K 64), not vLLM's `RDNAHybridW4A16LinearKernel`, whose HIP skinny GEMM its author reports 3.4 to 4.8x faster than the Conch fallback on gfx1151 and which was not run here. [L25]; [L15] `lab/kernels/chipsheet/chipsheet.py`; [S13](https://github.com/vllm-project/vllm/pull/40977)
  - Implication: no number in this note says the LUT kernel beats what vLLM serves today; the head-to-head with the hybrid kernel is KL3.
  - Candidate experiment: KL3, time `RDNAHybridW4A16LinearKernel`'s `wvSplitK_int4_g` path at M=1 on the four shapes with cold weights on the R60 build, beside this kernel. Expected movement: none on the metric set; it decides whether a 3-bit LUT path can beat the served int4 path per layer. Minimum sample size: 11 repeats per point. Confidence: medium (0.5) that the hybrid kernel is faster than 115 us at 5120x6144.

## Predicted teacher decode at 3 and 2 bits

- **[opinion]** Arithmetic, GEMV time only, batch 1. The int4 teacher reads 12.60 GB of int4 layers with bf16 group-128 scales a token, which is 12.60e9 / (0.5 + 2/128) = 24.44 billion quantized weights; the fp16 parts are 2.54 GB of `lm_head` and about 0.30 GB of GDN state. [L26] `lab/refine/research/kernels/w4a16_decode_gfx1201.md` section 1.2. The fixed part is the `lm_head` at the chip sheet's measured `matmul.fp16.decode27b.k5120_n248320.m1` (0.22 TFLOP/s, 11.56 ms) plus the state at 208 GB/s (1.44 ms): 13.0 ms. The quantized layers take 24.44e9 times the measured cold time per weight, averaged over the four shapes. Measured paths: 3-bit `group_lut` 3.46 ps a weight, 84.5 ms of layers, 97.5 ms a token, 10.3 tok/s, 9.5 J/token at its 97.5 W; 2-bit `group_lut` 3.08 ps, 75.3 ms, 11.3 tok/s, 9.0 J/token at 102.3 W; the W4A16 probe 3.64 ps, 89.0 ms, 9.8 tok/s, 11.5 J/token at 113.0 W; 4-bit `group_lut` 8.7 tok/s. Predicted, not measured, mode 4 (`channel_lut`): 3 bits are 3.125 bits a weight, 9.55 GB, 66.0 ms at the 3-bit cold rate of 144.7 GB/s (above the 61.7 ms that the hot `shared_lut` sweep time implies as an arithmetic floor), 79.0 ms a token, 12.7 tok/s and 8.3 J/token at 105 W; 2 bits stay compute bound at the 2-bit `group_lut` time, about 11.3 tok/s. The brief's 11.7 GB for 27B at 3 bits counts all 27 billion weights; this arithmetic counts only the 24.44 billion the checkpoint quantizes and keeps `lm_head` in 16 bits. [L25]; [L24]
  - Implication: on GEMV time alone a 3-bit per-channel-table checkpoint would decode about 29 percent faster than the same model through the chip sheet's int4 probe (12.7 against 9.8 tok/s) and the per-group table only about 5 percent faster; these are ceilings that leave out attention, GDN compute, norms, sampling and launch gaps, so they bound, not predict, served tok/s (R29 measured 2.88 tok/s on the Conch path, [L14]).
  - Candidate experiment: KL1 first (measure mode 4), then KL4 below. Expected movement: KL1 confirms or cuts the 12.7 tok/s ceiling. Minimum sample size: as KL1. Confidence: medium (0.5).

## Conclusions

The kill criterion is not met: the 3-bit table GEMV reads weight bytes faster than the W4A16 probe in the same slot,
hot and cold. The useful finding is narrower than "3-bit wins": a table beats unpack arithmetic by about 2x on this
chip, but a per-group table erases the byte saving of 3 bits, and at M above 1 the probe's fused-dequant `tl.dot` is
faster. The next measurement is KL1 (mode 4) in one 15-minute slot, then KL3 against the vLLM hybrid kernel.

## Next steps: wiring it into vLLM at `RDNAHybridW4A16LinearKernel`

The vLLM upgrade note puts the ROCm W4A16 kernels in `vllm/model_executor/kernels/linear/__init__.py` in the order
`RDNA3W4A16LinearKernel, RDNAHybridW4A16LinearKernel, TritonW4A16LinearKernel, ConchLinearKernel,
ExllamaLinearKernel`, with the hybrid kernel picked by default on gfx1151 (`vllm_upgrade_gfx1151.md`, S172). The plan,
for R55's 3-bit checkpoint as the input:

1. Read R55's checkpoint format first: bit width, group size, symmetric or asymmetric, and whether its quantization
   config names a 3-bit weight type that the compressed-tensors (or GPTQ) loader at our vLLM commit accepts. If the
   loader refuses 3 bits, the first patch is the config path, not the kernel.
2. Add `RDNALutWxA16LinearKernel` beside the hybrid kernel, listed before it, whose `can_implement` accepts gfx11 and
   gfx12, 2- or 3-bit weights, group 128, K a multiple of 128 and N a multiple of the tile, and refuses everything else
   so the hybrid kernel keeps every 4-bit layer.
3. In `process_weights_after_loading`, repack once on load: the checkpoint's codes into `main` and `hi` (the layout in
   `lut_gemv.pack`), and the scales and zeros into either the per-group table (mode 1, works for any asymmetric grid)
   or, when the checkpoint is symmetric with one grid, the per-channel table and scale (mode 4). This is FLUTE's
   offline restructuring step; it costs load time, not decode time.
4. In `apply_weights`, call `lut_gemv` for M up to 2 (where the LUT path wins) and, above that, dequantize through the
   table into a bf16 buffer per layer and use hipBLASLt or the Triton fused path that the hybrid kernel already uses
   for prefill. The activations arrive as bf16 in the teacher; the kernel takes fp16 today, so a bf16 variant (one
   dtype constexpr) comes with this step.
5. Gate it behind an environment variable and `VLLM_DISABLED_KERNELS` for the A/B, as the upgrade note does for the
   Conch control, and check outputs with `python -m lab correctness compare` against a 3-bit baseline, holding the
   K4 self-match floor of 0.5 that R21 measured.
6. Measure on the R60 build: `lab/bench.py` at batch 1, 4, 8 and 16, three spawns each, tok/s and J/token beside the
   int4 teacher, in one 30-minute slot on pod D.

Candidate experiments, ranked: KL1 (mode 4 measured, one 15-minute slot, highest value per minute), KL3 (the hybrid
kernel at the same points), KL2 (the dot tile at M=16), KL4 (the vLLM wiring above with R55's checkpoint, after KL1
and KL3 say it can win).

## Open questions

- Does R55's 3-bit checkpoint load at our vLLM commit at all, and with which weight type?
- Is the per-channel table's grid (mid-quantiles of w / s here) good enough at 3 bits, or does it need the NF3 or
  learned codebook that FLUTE's NormalFloat extension uses? On random Gaussian weights the quantile grid gave a
  relative error of 0.24 at 3 bits against 0.22 for the uniform asymmetric grid (`tests/test_lab_lut_gemv.py` setup,
  CPU), which says nothing about a real layer.
- Why is the 2-bit path compute bound near 100 GB/s: the 16-codes-a-word unpack, the gather, or occupancy at BLOCK_N 32?

## Search log

- 2026-10-11, budget 20 min inside a 4-hour task, used about 15 min. Opened: arXiv 2206.09557 and 2407.10960, the
  FLUTE repository, ROCm/hip-ep PR #1017, bitsandbytes issue #1842, Triton discussion #1851; searched for a ROCm,
  HIP or Triton port of LUT-GEMM or FLUTE and found none. Stopped when the searches returned only the CUDA kernels
  and the fused int4 work above.
