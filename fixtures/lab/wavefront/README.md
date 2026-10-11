# Wavefront harness fixtures (synthetic)

Every file here is made up to exercise the parsers in `lab/kernels/wavefront/wavefront_format.py`; none is a
measurement and none was recorded on a pod. `wavefront.py self-check` reads them, along with the chip sheet's
synthetic hipcc remarks in `fixtures/lab/chipsheet/remarks_w32_synthetic.txt`.

| File | Shape | Stands for |
| --- | --- | --- |
| `triton_compile_synthetic.json` | row id to `{n_regs, n_spills, shared, num_warps, num_stages}` | what a compiled Triton kernel handle and its metadata expose, for the `tile` and `lora` quick cases |
| `rocprof_kernel_trace_synthetic.csv` | rocprofv3 `kernel_trace.csv` columns (the ones `lab/trace_profile.py` reads) | a short trace of four kernel families: a bf16 Tensile GEMM, an elementwise op, a batched float32 triangular solve, a MIOpen conv; durations, VGPRs and LDS invented |

Replace them with recorded output (a real `kernel_trace.csv` from R34's profile, the compile records of a pod run)
once those exist, keeping the synthetic ones for the tests that pin exact numbers.
