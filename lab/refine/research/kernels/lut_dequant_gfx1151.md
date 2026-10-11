# Lookup-table dequant GEMV for 3-bit and 2-bit weights on gfx1151 (R62)

## Kill criterion (written 2026-10-11 1:17 AM ET, before the pod run)

The task ends as a measured failure, with the numbers, if after three tile configurations (BLOCK sizes, split-K and
num_warps; `M1_CONFIGS` in `lab/kernels/lut_gemv.py`, plus the chip sheet's own W4A16 tile) the 3-bit lookup-table
GEMV at M=1 on the 5120x6144 decode shape reads weight bytes slower than the chip sheet's `dequant.w4a16` probe
(`dequant.w4a16.decode27b.k5120_n6144.m1.bn64_bk64`: 0.594 TFLOP/s, which is 106 us for 16.2 MB of int4 codes and
scales, 153 GB/s) re-measured in the same slot. Weight bytes are everything the kernel must read for the weights:
the codes plus the per-group metadata (the table, or the scale and zero).
