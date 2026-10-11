# Does the lookup-table GEMV beat vLLM's hybrid W4A16 kernel at batch 1 on gfx1151?

R63 measures the R62 lookup-table GEMV against the kernel vLLM (main at 5281e4990, the R60 build) selects for the teacher's compressed-tensors W4A16 layers, `RDNAHybridW4A16LinearKernel`, on the same packed 4-bit weights, at batch 1, 4 and 16, cold and hot, on one Strix Halo (pod D, pod-b, gfx1151). If the lookup path wins, a batch-size switch inside that kernel class is the chip profile as working code; if not, this note records why and the work stops. 3-bit end to end is out of scope.

## Question and rule (written 2026-10-11 1:55 AM ET, before the pod run)

The question is whether, at M = 1 with the weights of a real decoder layer that does not fit the cache, the lookup path finishes a GEMV sooner than the hybrid kernel's HIP skinny GEMM (`wvSplitK_int4_g`), which vLLM routes every M of at most 5 to when K x M is at most 39,321 (`lab/kernels/dispatch_bench.py::hybrid_route`, the constants of `rdna_hybrid_w4a16.py` at 5281e4990, [S630]).

Both kernels get the same codes and scales: layer 3 of RedHatAI/Qwen3.8-27B-INT4 (symmetric int4, group 128, no zero points, read with safetensors on the pod; the 5120 x 6144 shape is the first 6144 rows of `q_proj`, the other three are `o_proj`, `gate_proj` and `down_proj`). The vLLM side is the kernel class itself, built the way compressed-tensors builds it (`MPLinearLayerConfig` with `uint4b8`, `process_weights_after_loading`, then `apply_weights` per call); the lookup side is `lab/kernels/lut_gemv.py` at R62's commit, in its per-group table mode and in its plain arithmetic mode, with the tile picked hot at the first shape. Each point is timed from a CUDA graph of about 2 ms of calls, which is how vLLM replays decode, with the chip sheet's power reader beside it.

The value compared is the common weight GB/s: the bytes the checkpoint holds for the layer (int4 codes plus one fp16 scale per 128 weights) over the time per call. It is a time measure in GB/s units, so it cannot repeat R62's accounting, where the table's extra bytes made the lookup kernel read faster per byte while each call took longer.

The rule: the lookup path wins if, with cold weights at M = 1, its better mode takes at most 1 / 1.10 of the hybrid kernel's time summed over the four shapes (it reads the common weight bytes at least 10 percent faster) and every one of its M = 1 outputs is at fp16 rounding: relative L2 error at most 1e-3 and max abs error at most 2^-8 of the largest output, both against the fp32 product with the dequantized weights. If it wins, the switch threshold is the largest M in 1, 4, 16 at which it is still faster cold at every shape. The rule is coded as `lab/kernels/dispatch_bench.py::decide` and tested in `tests/test_lab_dispatch_bench.py`.

## Search log

- 2026-10-11, budget 4 h end to end, starting points `lab/kernels/lut_gemv.py` and its result at R62's commit, `lab/vllm_upgrade.sh` on branch `ctx/runtime/vllm-upgrade` (3239f0d), `lab/refine/research/kernels/vllm_upgrade_gfx1151.md`, and vLLM's `rdna_hybrid_w4a16.py`, `MPLinearKernel.py`, `parameter.py` and `quant_utils.py` read at 5281e4990.
