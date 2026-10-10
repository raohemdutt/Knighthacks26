# Strix Halo (Radeon 8060S, gfx1151): what the documentation says about the chip

Research note for the kernels track, written for Task Board task R32 and searched on 2026-10-10. It
collects what AMD's own documentation, the LLVM AMDGPU backend, the Linux amdgpu driver and the
ROCm libraries say about the GPU in the two SJC lab pods: AMD Ryzen AI Max+ 395, Radeon 8060S,
LLVM target gfx1151, RDNA 3.5, 64 GB of LPDDR5X shared with the CPU, ROCm 7.13 (HIP 7.13.99004),
torch 2.9.1+rocm7.13.0, triton 3.5.1. It replaces the R9700 (gfx1201) chip facts of
`triton_gfx12.md` section 1 for anything that runs on the pods; the R9700 numbers stay as another
chip, never as a gfx1151 baseline. The measured half of the chip sheet is task R33; the last
section of this note lists what R33 must measure because the documents are silent or disagree.

Labels follow `w4a16_decode_gfx1201.md` (the main note): **[measured]** is a number with its setup,
**[practitioner]** is documented behavior from the people who build the thing (vendor docs, a
project's own source read at a pinned commit), **[anecdote]** is a single report, **[opinion]** is
argument or arithmetic, ours included. Every number in a finding is also tagged by where it comes
from: *vendor-stated* (AMD's own documents or code, including the ROCm documentation source),
*upstream-code* (LLVM, Linux, PyTorch, Triton source at a pinned commit), *measured-ours* (our pods),
*derived* (our arithmetic, shown) or *unknown* (no opened source states it). Experiment ids `SH1`
to `SH12` are local to this note; they name the R33 measurement that settles each gap (section 9).

How the sources were reached. The session's network policy refused amd.com, rocm.docs.amd.com,
docs.amd.com, llvm.org, docs.kernel.org and gpuopen.com (HTTP 403 at the proxy), so the rendered
pages could not be opened. Every vendor and upstream fact below is therefore read from the
published source of those same pages on GitHub, pinned to a commit: the ROCm documentation source
(`ROCm/ROCm`, which builds rocm.docs.amd.com), `llvm/docs/AMDGPUUsage.rst` (which builds the
llvm.org AMDGPUUsage page), and the Linux tree whose kernel-doc comments build docs.kernel.org's
amdgpu pages. AMD's Ryzen AI Max product pages and specification sheets, the RDNA 3.5 ISA reference
guide and the GPUOpen articles could not be opened at all; what only they would state is marked
unknown below and listed in "Searched but not opened" at the end.

## 1. Identity and support status

- **[practitioner]** AMD's GPU specification table lists one gfx1151 row: name "AMD Ryzen AI Max+
  PRO 395", graphics model "Radeon 8060S", architecture "RDNA3.5", LLVM target gfx1151, VRAM
  "Dynamic + carveout", GFXIP major 11, minor 5 (all vendor-stated). The Linux driver's APU table
  names "Ryzen AI Max 300 series, Strix Halo" with graphics core IP 11.5.1 and SMU (MP1) 14.0.1
  (upstream-code), and the LLVM processor table lists gfx1151 as an APU with target features
  `cumode` and `wavefrontsize64` (upstream-code). Source: [S74] ROCm documentation source,
  `docs/reference/gpu-specs.rst` lines 876 to 895; [S86] Linux
  `Documentation/gpu/amdgpu/apu-asic-info-table.csv` line 18; [S79] `llvm/docs/AMDGPUUsage.rst`
  line 606.

  Implication: our pods (Ryzen AI Max+ 395, not the PRO part, per the R30 hour-0 facts [L5]) are
  the same gfx1151 die as the documented row; GC 11.5.1 and SMU 14.0.1 identify which driver paths
  (the SMU 14.0.x APU driver of the telemetry note's PS-37) report power on the pods.

  Experiment: SH1 records `rocminfo` (gfx target, compute unit count, max clock, wavefront size,
  LDS size per workgroup) and the driver's IP versions once per pod; expected to match this row
  exactly on both pods; minimum 1 read per pod (2 reads); confidence high (0.9).

- **[practitioner]** AMD documents a minimum Linux kernel for Ryzen AI Max (gfx1151) compute:
  6.18.4 or later on distributions other than Ubuntu 24.04 HWE (6.17.0-19.19~24.04.2) or OEM
  (6.14.0-1018), because KFD fixes for "queue creation and memory availability checks" are needed;
  without them "GPU compute workloads might fail to initialize or exhibit unpredictable behavior".
  Its per-release table rates ROCm 7.2.0 to 7.2.3 and 7.11.0 to 7.12.0 stable on those kernels,
  7.1.x and 7.9.0 to 7.10.0 unsupported, and has no row for 7.13 (vendor-stated). The pods run
  Ubuntu 24.04.4 with kernel 6.18.6-061806-generic (measured-ours), which is 6.18.4 or later
  (derived: 6.18.6 >= 6.18.4). Source: [S75] ROCm documentation source,
  `docs/reference/system-optimization/rdna3-5.rst` lines 180 to 268; [L5] R30 task page, hour-0
  facts table (pod D) and R29's column (pod C).

  Implication: the pods' kernel meets AMD's stated floor, but the pods' ROCm 7.13 is newer than
  the newest release AMD rates for gfx1151 in that table, so 7.13 on gfx1151 is unrated, not
  "supported"; an intermittent failure on the pods is not ruled out by the documents.

  Experiment: SH1 also records `uname -r`, the HIP and ROCm versions, and runs the teacher for 10 minutes
  watching its log for HIP queue-creation or allocation errors; expected no queue or
  memory-check failures; minimum 1 run per pod; confidence medium (0.7).

- **[practitioner]** ROCm's release notes first enable gfx1151 in libraries at 7.0.2 (hipBLAS
  3.0.2: "Enabled support for gfx1150, gfx1151, gfx1200, and gfx1201"; rocBLAS 5.0.2: "Enabled
  gfx1150 and gfx1151"), and the development notes add "Improved roofline support on gfx1150
  (Strix Point), gfx1151 (Strix Halo and Gorgon Halo)", reporting "the correct set of supported
  precisions" (vendor-stated). TheRock's support table marks gfx1151 as build passing, sanity
  tested and release ready (vendor-stated). The main 7.0.2 to 7.2.0 compatibility matrices do not
  list gfx1151 and refer Radeon and Ryzen users to a separate page, which could not be opened.
  Source: [S78] ROCm `RELEASE.md` at tags rocm-7.0.2 (lines 633, 646) and rocm-7.1.0 (line 811);
  [S77] ROCm documentation source, `docs/about/release-notes.md` line 178; [S93] ROCm/TheRock
  `SUPPORTED_GPUS.md` line 63.

  Implication: sources disagree on how supported gfx1151 is (enabled in 7.0.2 libraries, unrated
  for 7.13 in the RDNA 3.5 page, release ready in TheRock); the lab treats every library path on
  gfx1151 as needing its own correctness check (the correctness note's V-tests), not as a given.

  Experiment: SH9 also runs the rocprofiler-compute roofline benchmark (if the pod's ROCm ships it) to
  learn which precisions it reports for gfx1151; expected fp16, bf16 and int8 present, fp8 absent;
  minimum 1 run per pod; confidence medium (0.6).

## 2. Compute units, wavefronts and registers

- **[practitioner]** The Radeon 8060S has 40 compute units and wavefront size "32 or 64"
  (vendor-stated). In WGP mode "the wavefronts of a work-group are executed on the SIMDs of both CUs
  of the WGP", and WGP is the native mode on GFX10 and later, with `-mcumode` selecting CU mode
  (upstream-code). Source: [S74] `gpu-specs.rst` gfx1151 row; [S79] `AMDGPUUsage.rst`, target
  feature `cumode` (lines 913 to 917) and "Memory Model GFX10-GFX11".

  Implication: 40 CUs at two CUs per WGP is 20 WGPs (derived: 40 / 2 = 20); a grid needs at least
  20 workgroups in WGP mode, or 40 in CU mode, to put work on every unit, which matters for the
  skinny decode GEMMs where N/tile can fall below that.

  Experiment: SH2 sweeps a bandwidth-bound kernel over grid sizes 10 to 160 workgroups in WGP and
  CU mode; expected bandwidth to plateau at about 20 workgroups (WGP) or 40 (CU); minimum 3 runs
  per point; confidence medium (0.6).

- **[practitioner]** For GFX10 and later the compiler defaults to wave32 ("For gfx10+ set wave32 as
  a default"), and wave64 is selected with `-mwavefrontsize64` (target feature `wavefrontsize64`:
  "When disabled native wavefront size 32 is used, when enabled wavefront size 64 is used")
  (upstream-code). How the hardware executes a wave64 instruction (one pass with dual issue or two
  passes) is not stated in any opened source (unknown; the RDNA 3.5 ISA guide was not opened).
  Source: [S83] LLVM `llvm/lib/Target/AMDGPU/GCNSubtarget.cpp` lines 143 to 147 and clang
  `clang/lib/Basic/Targets/AMDGPU.cpp` lines 219 to 222; [S79] `AMDGPUUsage.rst` lines 935 to 938.

  Implication: HIP and Triton kernels on the pods are wave32 unless a kernel asks otherwise;
  register and occupancy arithmetic below uses wave32 first.

  Experiment: SH3 times one FMA-bound and one bandwidth-bound microkernel compiled wave32 and wave64
  with the same work per thread; expected wave64 within 10% on the bandwidth-bound kernel and no
  faster on the FMA-bound one; minimum 5 runs per variant; confidence low (0.4).

- **[practitioner]** gfx1151 has the large register file: LLVM gives `FeatureISAVersion11_5_1` the
  feature `Feature1536VGPRs` ("Has 1536 physical VGPRs per SIMD"), which gfx1150 and gfx1152 lack;
  total VGPRs are "IsWave32 ? 1536 : 768" with the feature and 1024 or 512 without; the allocation
  granule is 24 VGPRs in wave32 and 12 in wave64 with the feature (16 and 8 without); at most 256
  VGPRs per wave; at most 16 waves per SIMD (`FeatureMaxWavesPerEU16`) (all upstream-code). AMD's
  table gives a "VGPR File (KiB)" of 768 for gfx1151 against 512 for gfx1150 and gfx1103
  (vendor-stated). Source: [S80] LLVM `llvm/lib/Target/AMDGPU/AMDGPU.td` lines 1288 to 1289, 1873
  and 2370 to 2392; [S81] `llvm/lib/TargetParser/AMDGPUTargetParser.cpp` lines 447 to 463 and
  `llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp` lines 1294 to 1298; [S74] `gpu-specs.rst`.

  Implication: register pressure tuned for gfx1201 or for gfx1100 transfers to gfx1151 (gfx1100
  has the same 1536-VGPR file), but configs tuned on gfx1150 or gfx1103 do not; the 768 versus 512
  KiB ratio in AMD's table is the same 1.5x as LLVM's 1536 versus 1024 (derived: 768 / 512 = 1.5 =
  1536 / 1024). The table does not say whether its column is per CU or per WGP: 1536 VGPRs x 32
  lanes x 4 bytes is 192 KiB per SIMD (derived), so 768 KiB is four SIMDs' files, which fits a WGP
  of two CUs with two SIMDs each, but the SIMD count per CU is not in any opened source (unknown).

  Experiment: SH4 compiles one kernel at 96, 120, 144, 192 and 256 VGPRs and reads the occupancy the
  compiler reports and the measured time; expected the occupancy to follow the table in the next
  finding exactly; minimum 3 runs per point; confidence high (0.85).

- **[opinion]** Occupancy table for gfx1151 in wave32, derived from LLVM's formula
  ("RoundedRegs = alignTo(NumVGPRs, Granule); return min(max(TotalNumVGPRs / RoundedRegs, 1),
  MaxWaves)") with total 1536, granule 24 and 16 waves maximum: at most 96 VGPRs for 16 waves per
  SIMD (1536 / 96 = 16), 120 for 12, 144 for 10, 168 for 9, 192 for 8, 216 for 7, 240 for 6, and
  256 gives 5 (256 rounds up to 264; 1536 / 264 = 5.8, floor 5). In wave64 the same formula with
  768 and granule 12 gives 16 waves at 48 VGPRs or fewer (768 / 48 = 16). SGPRs do not limit
  occupancy: "From GFX10 on the SGPR file is large enough that SGPRs never limit occupancy", with
  106 addressable SGPRs per wave (upstream-code). Source: arithmetic on [S81]
  `AMDGPUBaseInfo.cpp` lines 1239 to 1242, 1354 to 1357 and 1432 to 1436 and
  `AMDGPUTargetParser.cpp` lines 371 to 389.

  Implication: on gfx1151 a kernel can use up to 96 VGPRs at full occupancy, against 64 on gfx1150
  (derived: 1024 / 16 = 64); a decode GEMM that spills at 128 VGPRs drops to 10 waves, not 8 as it
  would on a 1024-VGPR part (derived: 1024 / 128 = 8).

  Experiment: SH4 (above) checks this table against the compiler's reported occupancy; expected
  exact agreement; minimum 1 compile per point; confidence high (0.9).

## 3. LDS

- **[practitioner]** LDS is 128 KiB per WGP (AMD's table: "LDS (KiB)" 128, vendor-stated). LLVM
  gives GFX11 a 65,536-byte addressable LDS with `FeatureHalfAddressablePhysicalLocalMemory` ("A
  work-group can address only half of the physical LDS block"), so one workgroup gets at most 64
  KiB, and in CU mode the workgroup reaches only half of the block; LDS is allocated in 1,024-byte
  granules; "Each WGP has a single LDS memory shared by the wavefronts"; GFX11 has 32 LDS banks
  (`FeatureLDSBankCount32`) (all upstream-code). The bank width is not in any opened source
  (unknown). Source: [S74] `gpu-specs.rst`; [S80] `AMDGPU.td` lines 301 to 302, 435, 1841 to 1845
  and 2301; [S81] `AMDGPUTargetParser.cpp` lines 482 to 489; [S79] `AMDGPUUsage.rst`, "Memory Model
  GFX10-GFX11".

  Implication: the 65,536-byte per-workgroup LDS limit that `triton_gfx12.md` section 1 recorded on
  gfx1201 holds on gfx1151 too, so `lab/vllm-gfx1201.diff`'s `num_stages=1` attention fix (66,560
  bytes requested by default, [L3]) is still needed; two 64 KiB workgroups fill one WGP, so a
  kernel using more than 32 KiB (derived: 128 / 4) fits at most three workgroups per WGP.

  Experiment: SH5 measures LDS bandwidth with stride-1, stride-2 and stride-32 four-byte accesses
  per lane to infer bank width and conflict cost; expected stride-32 at about 1/32 of stride-1 if
  banks are four bytes wide; minimum 5 runs per stride; confidence medium (0.6).

## 4. Caches

- **[practitioner]** AMD's table gives gfx1151: Infinity Cache 32 MiB, L2 2 MiB, graphics L1 256
  KiB, L0 vector 32 KiB, L0 scalar 16 KiB, L0 instruction 32 KiB, SGPR file 32 KiB; it is the only
  APU in the table with an Infinity Cache, the others reading "N/A" (vendor-stated). LLVM's memory
  model describes the hierarchy: "a single L0 cache per CU", a "scalar L0 cache shared by all
  wavefronts on a WGP", an "L1 cache shared by all WGPs on the same SA", an "L2 cache shared by all
  SAs", and "On GFX10.3 and GFX11 a memory attached last level (MALL) cache exists"
  (upstream-code). The table does not say whether the L0 and L1 sizes are per CU, per WGP or per
  shader array, or how many shader arrays gfx1151 has (unknown). Source: [S74] `gpu-specs.rst`
  lines 876 to 895; [S79] `AMDGPUUsage.rst`, "Memory Model GFX10-GFX11".

  Implication: the int4 weights of one large GEMM of the teacher (for example the MLP up
  projection at 5120 x 17408 x 0.5 bytes, about 44.6 MB, derived) no longer fit the last-level
  cache as they did in the R9700's 64 MiB, so microbenchmarks on gfx1151 overstate bandwidth only
  for the smaller GEMMs (K = 5120 and 6144 projections), and the cache-aware benchmarking rule of
  `triton_gfx12.md` section 4 still applies with a 32 MiB threshold.

  Experiment: SH6 runs a pointer-chase and a streaming read over working sets from 16 KiB to 1 GiB;
  expected latency and bandwidth steps near 32 KiB, 256 KiB, 2 MiB and 32 MiB; minimum 3 runs per
  size; confidence medium (0.7).

## 5. Memory and the CPU-GPU split

- **[practitioner]** AMD's RDNA 3.5 system page says these APUs "support LPDDR5X-8000 or DDR5
  memory" (vendor-stated, for the RDNA 3.5 APU family, not per SKU). The memory bus width and an
  AMD-stated bandwidth for the Ryzen AI Max+ 395 are on AMD's product page and specification
  sheet, which could not be opened (unknown). The theoretical peak is therefore stated as a
  formula: bandwidth = 8,000 x 10^6 transfers per second x (bus width in bits / 8) bytes (derived;
  the width is unknown here). Source: [S75] `rdna3-5.rst` (opening section); AMD Ryzen AI Max+ 395
  product page, not opened (see "Searched but not opened").

  Implication: no bandwidth roofline for gfx1151 can be drawn from opened sources yet; the decode
  roofline of the main note (640 GB/s on the R9700) has no gfx1151 counterpart until SH7 measures
  it or the product page is read.

  Experiment: SH7 measures sustained read, write and copy bandwidth from the GPU over a 4 GiB
  buffer (and the same from the CPU cores for comparison); expected the GPU's read bandwidth to be
  the decode ceiling; minimum 5 runs per mode; confidence high (0.8) that the number is stable
  within 3%.

- **[opinion]** Our measured teacher decode moves about 44.6 GB/s: 2.88 tok/s x about 15.5 GB read
  per decode step = 44.6 GB/s (derived), with 2.88 tok/s measured on pod C (gfx1151, vLLM 0.19.2
  at b1388b1 plus `lab/vllm-gfx1201.diff`, `RedHatAI/Qwen3.8-27B-INT4`, batch 1, 16k context,
  thinking off, text prompt of 31 tokens in and 77 out, TTFT 0.39 s, 120 W average and 129 W peak
  socket power while decoding) and 15.5 GB per step from the main note's byte budget. Source: [L4]
  `lab/README.md`, section "Teacher on SJC today (pod C, Strix Halo)", merged in PR #194 (commit
  498e161); the byte budget in `w4a16_decode_gfx1201.md` section 1.2, arithmetic on [S25].

  Implication: until SH7 gives the chip's sustained bandwidth, the fraction of peak the 4-bit path
  reaches on gfx1151 is unknown; the R9700's figure (about 119 GB/s, 19% of 640 GB/s) cannot be
  carried over. If SH7 lands near the product's rated bandwidth, the Conch path's share on gfx1151
  can be compared with the R9700's directly.

  Experiment: SH7's read bandwidth divides the 44.6 GB/s to give the 4-bit path's fraction of peak;
  expected well under 50%, as on the R9700; minimum 5 bandwidth runs plus 5 decode runs of 256
  tokens; confidence medium (0.5).

- **[practitioner]** On RDNA 3.5 APUs, AMD says GPU memory is "mapped rather than physically
  partitioned": the GTT pool used by "PyTorch and other AI/compute workloads" is dynamic, "By
  default, the GTT limit is set to approximately 50 percent of total system RAM", and when "the GTT
  size is larger than the VRAM, the AMD GPU driver performs VRAM allocations using GTT"; AMD
  recommends keeping the BIOS VRAM reservation small (for example 0.5 GB) and raising the TTM limit
  instead, at `/sys/module/ttm/parameters/pages_limit`, "expressed in pages", with `amd-ttm --set`
  writing `/etc/modprobe.d/ttm.conf` and a reboot (vendor-stated). The kernel agrees: TTM limits its
  pool to "about 50% of the total system memory", `pages_limit` is a module parameter of mode 0644,
  and amdgpu's `gttsize` (MiB, mode 0600) "is deprecated and will be removed" (upstream-code).
  amd-smi's `--gtt` and `--mem-carveout` need root and a reboot, the carveout a kernel 7.0 or later
  and an ATCS 0xA VBIOS (vendor-stated). Source: [S75] `rdna3-5.rst` lines 20 to 170; [S85] Linux
  `drivers/gpu/drm/ttm/ttm_tt.c` lines 52 to 55, `drivers/gpu/drm/ttm/ttm_device.c` lines 88 to
  92, `drivers/gpu/drm/amd/amdgpu/amdgpu_drv.c` lines 295 to 303; [S88] amd-smi CLI documentation,
  lines 1740 to 1752.

  Implication: the split is set by the host's administrator (BIOS carveout plus the TTM page
  limit, both needing root and a reboot), never from a pod; the pods show 64.0 GiB to torch on a
  host with 62 GB of RAM (measured-ours, [L5]), which is more than the documented 50% default, so
  the hub's hosts have a raised limit or a large carveout, and which of the two is not visible from
  these facts (unknown).

  Experiment: SH8 reads `mem_info_vram_total`, `mem_info_gtt_total` and their `_used` files,
  `/sys/module/ttm/parameters/pages_limit` and `torch.cuda.mem_get_info` on each pod, then allocates
  in 1 GiB steps until failure; expected the largest single allocation near the larger of VRAM and
  GTT; minimum 1 sweep per pod; confidence medium (0.6).

## 6. Arithmetic paths: WMMA and dot products

- **[practitioner]** gfx1151 has the RDNA 3 style WMMA (`FeatureWMMA256bInsts` in
  `FeatureISAVersion11_5_Common`): `v_wmma_f32_16x16x16_f16`, `v_wmma_f32_16x16x16_bf16`,
  `v_wmma_f16_16x16x16_f16`, `v_wmma_bf16_16x16x16_bf16`, `v_wmma_i32_16x16x16_iu8` and
  `v_wmma_i32_16x16x16_iu4`, each in wave32 and wave64 forms, all 16x16x16 (upstream-code). The fp8
  WMMA, the 16x16x32 iu4 form and the sparse SWMMAC instructions need `HasWMMA128bInsts` or
  `HasSWMMACGfx1200Insts`, which only gfx11.7 and gfx12 carry, not gfx1151 (upstream-code). Source:
  [S80] `AMDGPU.td` lines 994 to 995, 2370 to 2392 and 2403 to 2404; [S82] LLVM
  `llvm/lib/Target/AMDGPU/VOP3PInstructions.td` lines 1718 to 1737 and 2374 to 2388.

  Implication: on the pods, fp16, bf16, int8 and int4 have matrix instructions, fp8 does not, so
  every fp8 lever in `precision_kv_batching.md` that relies on gfx12 fp8 WMMA (FP8 weights, FP8 KV
  dequant inside WMMA) has no hardware path on gfx1151; a W4A16 kernel still dequantizes int4 to
  fp16 or bf16 before WMMA, while a W4A8 or W8A8 int path could use `iu4` or `iu8` WMMA directly.

  Experiment: SH9 measures sustained WMMA throughput per type (f16, bf16, iu8, iu4) with a
  register-resident microkernel; expected iu8 and iu4 at or above f16's rate; minimum 5 runs per
  type; confidence low (0.4), because no opened source states the per-type rates.

- **[practitioner]** AMD's precision table lists matrix-core support for "RDNA3" as int8, float16
  and bfloat16 yes, and float8 (E4M3, E5M2), float6, float4, tensorfloat32, float32 and float64 no;
  it has no RDNA 3.5 row and no int4 column (vendor-stated). The same table marks RDNA2 int8,
  float16 and bfloat16 matrix-core support as yes, although LLVM gives no WMMA to GFX10.3, so the
  table's notion of "matrix core" is looser than WMMA (a disagreement between sources). Source:
  [S76] ROCm documentation source, `docs/reference/precision-support.rst`, "Matrix core support"
  (lines 589 to 797); [S80] `AMDGPU.td`.

  Implication: AMD's documentation and LLVM agree that gfx1151 has no fp8 matrix path; library
  type support (hipBLASLt fp8 on gfx1151) should be expected to fall back to slower paths or fail.

  Experiment: SH9 (above) includes one hipBLASLt fp8 GEMM attempt; expected unsupported or slower
  than bf16; minimum 1 attempt; confidence medium (0.6).

- **[practitioner]** gfx1151 has the dot-product instructions of GFX11: `v_dot4_i32_iu8` and
  `v_dot8_i32_iu4` (Dot8), `v_dot4_u32_u8` and `v_dot8_u32_u4` (Dot7), `v_dot2_f16_f16` and
  `v_dot2_bf16_bf16` (Dot9), `v_dot2_f32_f16` (Dot10) and `v_dot2_f32_bf16` (Dot12), plus Dot5 from
  the 11.5 common set (upstream-code). Source: [S80] `AMDGPU.td` lines 916 to 917 and 2302 to 2307;
  [S82] `VOP3PInstructions.td` lines 2957 to 2958.

  Implication: the skinny decode GEMMs (one to a few tokens) that do not fill a 16x16 WMMA tile can
  use packed dot products instead, which is how the HIP skinny W4A16 kernels in the main note's
  section 3.1 work on RDNA; their availability on gfx1151 is confirmed by the compiler.

  Experiment: SH10 compares a dot2-based and a WMMA-based int4 dequant GEMV at N = 1, 4 and 16;
  expected dot2 faster at N = 1 and WMMA faster at N = 16; minimum 5 runs per point; confidence
  medium (0.5).

- **[opinion]** No opened source states a peak fp16, bf16, int8 or int4 rate for the Radeon 8060S,
  the GPU's maximum clock, or the WMMA operations per clock per CU (unknown). The peak would be
  derived as 40 CUs x (dense matrix operations per clock per CU) x (sustained clock); both factors
  are missing. Source: [S74] (40 CUs, no clock or rate column); AMD product page and GPUOpen WMMA
  article, not opened (see "Searched but not opened").

  Implication: the compute side of the roofline is unknown; decode is bandwidth-bound regardless,
  but prefill time to first token (3.51 s for a 902-token caption prompt on pod C, [L4]) depends on
  it.

  Experiment: SH9 (above) gives the measured matrix peak per type at the clock the pod sustains,
  read alongside SH11's clock; expected a stable peak within 5% across runs; minimum 5 runs per
  type; confidence medium (0.5).

## 7. Power and clocks a non-root pod can read

- **[practitioner]** In the amdgpu driver, `power1_average` is "average power used by the SoC in
  microWatts. On APUs this includes the CPU", `power1_input` is the instantaneous equivalent with
  the same APU note, and both, with `power1_cap_min`, `power1_cap_max` and `freq1_input` (the gfx
  clock in hertz), are `S_IRUGO` (readable by every user); `power1_cap` and `power2_cap` are
  `S_IRUGO | S_IWUSR` (written only by root); `pp_dpm_sclk`, `pp_dpm_mclk` and
  `power_dpm_force_performance_level` are read-write device attributes (root writes) and
  `gpu_metrics` is read-only (upstream-code). The driver documents the memory clock hwmon file as
  "dGPU only". amd-smi's Python API documents, for gfx1151, `apu_metrics.*` fields including
  `apu_metrics.average_socket_power` and `apu_metrics.average_gfxclk_frequency` (vendor-stated).
  Source: [S84] Linux `drivers/gpu/drm/amd/pm/amdgpu_pm.c` lines 3585 to 3600, 3660 to 3680, 2583
  to 2621 and `drivers/gpu/drm/amd/pm/inc/amdgpu_pm.h` lines 112 to 117; [S88] amd-smi Python API
  reference line 3962.

  Implication: a pod can read socket power, the power cap and the gfx clock, but cannot set a power
  cap or pin clocks, so every energy lever in the kernels and telemetry notes that needs
  `power1_cap` or `pp_dpm_sclk` writes is out of reach on the pods; the power a pod reads is the
  whole APU, CPU included, as the telemetry note's PS-37 found from the SMU 14.0.x driver.

  Experiment: SH11 reads every hwmon file and `gpu_metrics` once as the pod user and logs which
  return values, `N/A` or permission errors, then samples `freq1_input` and socket power at 10 Hz
  during a 60 s decode; expected power readable, clock readable through hwmon even though amd-smi
  reports CLOCK N/A, caps readable but not writable; minimum 1 read pass and 3 decode windows per
  pod; confidence medium (0.6).

- **[measured]** On pod C, `amd-smi metric --power --clock --temperature --usage --energy` answered
  in 0.21 s a read with SOCKET_POWER and GFX_ACTIVITY, while CLOCK, ENERGY, HOTSPOT and the other
  sensors read N/A and EDGE was the only temperature; idle was 8 W with no server and 9.2 W with
  the teacher loaded (10.2 W over 60 s in the sampler run), and decode drew 120 W average and 129 W
  peak; on pod D ten idle reads one second apart gave 19, 12, 10, 9, 9, 9, 9, 8, 9 and 9 W after a
  first read of 24 W on pod C (all measured-ours). Source: [L4] `lab/README.md`, section "Teacher
  on SJC today (pod C, Strix Halo)", including the T6 sampler paragraph (PR #194, commit 498e161);
  [L5] R30 task page, hour-0 facts table.

  Implication: the "about 24 W idle" quoted for the pods is a first read after launch, not a
  settled idle; settled idle is 8 to 10 W, so decode's 120 W is about 110 W above idle (derived: 120 W
  minus 9.2 W is 110.8 W with the teacher loaded). amd-smi's CLOCK N/A means clocks must come from hwmon
  `freq1_input` (SH11) or not at all.

  Experiment: SH11 (above) records the settled idle after 120 s with no process on the pod's CPU;
  expected 8 to 10 W; minimum 3 windows of 60 s per pod; confidence high (0.8).

- **[practitioner]** The Ryzen AI Max+ 395's default and configurable TDP, its power limits (the
  PPT values the driver labels `power1_label` "PPT") and the GPU's boost clock are not in any opened
  source (unknown); the driver documents the labels but not the values, and AMD's product page was
  not opened. Source: [S84] `amdgpu_pm.c` (power labels); AMD product page, not opened (see
  "Searched but not opened").

  Implication: whether 129 W peak is the chip's limit, the OEM's setting on the hub's hosts, or
  neither is unknown; any statement that decode is "power-limited" on the pods needs SH11's
  `power1_cap` read first.

  Experiment: SH11 (above) reads `power1_cap`, `power1_cap_max` and `power1_cap_default`; expected
  the cap to sit above 129 W if decode is not power-limited; minimum 1 read per pod; confidence low
  (0.4).

## 8. ROCm settings and library paths documented for gfx115x

- **[practitioner]** None of the ROCm, HIP or ROCr environment-variable references opened carries
  a note specific to gfx115x or to APUs. The documented settings relevant to the pods are:
  `GPU_MAX_HW_QUEUES` (default 4, "The recommended maximum value for this setting is 4"),
  `HIP_HOST_COHERENT` (default 0; 1 makes host memory coherent), `HSA_XNACK`, `HSA_ENABLE_SDMA`
  (default 1), `HSA_ENABLE_PEER_SDMA`, `HSA_NO_SCRATCH_RECLAIM`, `HSA_SCRATCH_SINGLE_LIMIT_ASYNC`
  (default 3 GB), and `ROCR_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES`, with "On Linux, use
  ROCR_VISIBLE_DEVICES" (vendor-stated). `HSA_OVERRIDE_GFX_VERSION` and `PYTORCH_ROCM_ARCH` appear
  in none of these references (unknown as documented settings). Source: [S87] ROCm/rocm-systems,
  HIP `docs/reference/env_variables/debug_hip_env.rst` lines 95 to 99,
  `memory_management_hip_env.rst` lines 24 to 28, `gpu_isolation_hip_env.rst` lines 14 to 31, and
  ROCr `runtime/docs/data/env_variables.rst` lines 19 to 68; [S67] ROCm 7.13.0 environment
  variables page (`GPU_MAX_HW_QUEUES`).

  Implication: no environment variable is documented as needed for gfx1151, so a pod launch script
  that sets `HSA_OVERRIDE_GFX_VERSION` is working around something undocumented and should say
  what; `GPU_MAX_HW_QUEUES` is the one queue knob that `rdna4_vllm_ports.md` found to matter on
  gfx1201 and is worth a check on gfx1151.

  Experiment: SH12 runs the teacher decode with `GPU_MAX_HW_QUEUES` at 1, 2 and 4; expected no
  difference beyond noise on gfx1151; minimum 5 spawns per value; confidence low (0.35).

- **[practitioner]** PyTorch treats hipBLASLt as supported on gfx1151 from ROCm 7.0 and as the
  preferred BLAS backend on gfx1151 only from ROCm 7.13 (`#if ROCM_VERSION >= 71300 ... "gfx1100",
  "gfx1101", "gfx1151"`); rocBLAS lists `gfx1150;gfx1151` in its ROCm 7.0 target list, documents
  `ROCBLAS_USE_HIPBLASLT` (0 always Tensile, 1 prefer hipBLASLt with fallback), and hipBLASLt ships
  tuned kernel logic under `Tensile/Logic/asm_full/gfx1151/` (upstream-code). Source: [S90]
  pytorch/pytorch `aten/src/ATen/cuda/detail/CUDAHooks.cpp` lines 555 to 572; [S89]
  ROCm/rocm-libraries `projects/rocblas/docs/reference/env-variables.rst` lines 23 to 29,
  `projects/rocblas/CMakeLists.txt` line 90, and the hipBLASLt `Tensile/Logic/asm_full/gfx1151`
  directory.

  Implication: on the pods (ROCm 7.13) torch's dense bf16 GEMMs go to hipBLASLt by default, the
  path to profile first for the training lane (R34); the pinned PyTorch commit is newer than the
  pods' torch 2.9.1, so whether 2.9.1+rocm7.13.0 has the same preference is unknown until read on
  the pod.

  Experiment: SH12 (above) also reads `torch.backends.cuda.preferred_blas_library()` on the pod and
  times a 4096 x 4096 bf16 GEMM with hipBLASLt and with rocBLAS; expected hipBLASLt preferred and
  not slower; minimum 5 runs per backend; confidence medium (0.6).

- **[practitioner]** AOTriton at its current commit maps gfx1151 to gfx1100's tuning
  (`'gfx1151_mod0' : 'gfx1100_mod0'`) and does not list gfx1151 among the experimental targets
  (0x1150, 0x1152, 0x1153, 0x1101 to 0x1103, 0x1200), and PyTorch caps gfx11 SDPA head dimension at
  256 from AOTriton 0.11 (upstream-code). Our own build found the opposite on the pods: torch 2.9.1
  "keeps the AOTriton SDPA kernels off as experimental on gfx1151", and with
  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` those kernels went wrong on a 720p image's about 3.5k
  tokens (max error 0.08 to 0.12, NaN at 4096), so the server passes
  `--mm-encoder-attn-backend TRITON_ATTN` on gfx115x (measured-ours). Source: [S91] ROCm/aotriton
  `v3src/hip-dependent/util.cc` lines 97 to 103 and `python/gpu_targets.py` lines 17 and 33; [S90]
  pytorch `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp` lines 165 to 175 and 525 to 528;
  [L4] `lab/README.md`, section "Teacher on SJC today (pod C, Strix Halo)".

  Implication: the sources disagree because they are different versions: upstream has moved
  gfx1151 out of the experimental list, while the pods' torch 2.9.1 bundles an older AOTriton that
  still gates it and is wrong at long sequence; flash attention from torch is not usable on the
  pods today, and Triton or vLLM's own attention stays the path.

  Experiment: SH12 (above) also runs one SDPA correctness check against the math kernel at 1k, 2k
  and 4k tokens on the pod's torch; expected failure at 4k with the experimental flag, as recorded;
  minimum 3 runs per length; confidence high (0.8).

- **[practitioner]** Triton's AMD backend classes every GFX11 target, gfx1151 included, as the
  RDNA3 ISA family (`if (major == 11) return ISAFamily::RDNA3;`) and enables an in-thread transpose
  path when `"gfx115" in arch` (upstream-code). vLLM's own tuned configs for gfx1151 exist but were
  reported unreachable because the lookup keys off an unstable device name. Source: [S92]
  triton-lang/triton `third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp` lines 104 to
  105 and `third_party/amd/backend/compiler.py` line 174; [S35] vLLM issue #50436.

  Implication: Triton kernels tuned for gfx1100 are the closest starting configs for gfx1151 (same
  family, same 1536-VGPR file per section 2); vLLM's shipped gfx1151 configs should be checked for
  actually loading before any tuning result is trusted.

  Experiment: SH12 (above) also logs which tuned-config file vLLM loads for one fused-MoE or W4A16
  call on the pod; expected the default (untuned) config, per the issue; minimum 1 run; confidence
  medium (0.5).

## Open questions for R33 (the measured chip sheet)

Each question is where the opened documents are silent or disagree. The `SH` id is the experiment
above that answers it.

- Q1 (SH7): the GPU's sustained read, write and copy bandwidth on the pods, the number every decode
  roofline needs; the bus width and AMD's rated bandwidth were not opened.
- Q2 (SH9, SH11): the GPU clock under sustained matrix load, and the measured WMMA throughput for
  f16, bf16, iu8 and iu4; no opened source gives the boost clock or operations per clock.
- Q3 (SH6): where the cache steps fall (L0, L1, L2 2 MiB, Infinity Cache 32 MiB) and the latency
  of each, and whether the table's L0 and L1 sizes are per CU, per WGP or per shader array.
- Q4 (SH5): LDS bank width and the cost of a bank conflict on gfx1151.
- Q5 (SH4): whether the compiler-reported occupancy follows the derived 1536 / 24 / 16 table, and
  what the 768 KiB "VGPR File" column counts.
- Q6 (SH3): wave32 against wave64 speed for FMA-bound and bandwidth-bound kernels.
- Q7 (SH8): the VRAM carveout, GTT total and TTM page limit on the hub's hosts, and the largest
  single allocation torch can make (64.0 GiB visible on 62 GB of host RAM).
- Q8 (SH11): which power, clock and cap files a pod user can read, the cap's value, the settled
  idle, and whether hwmon gives the gfx clock that amd-smi reports as N/A.
- Q9 (SH2): how many workgroups fill the GPU in WGP and CU mode.
- Q10 (SH12): whether `GPU_MAX_HW_QUEUES`, the BLAS backend choice and the attention backend
  change decode or prefill on gfx1151, and which tuned configs vLLM actually loads.
- Q11 (SH1): that both pods match the documented gfx1151 row and that ROCm 7.13, unrated by AMD
  for gfx1151, runs queue creation and allocation without errors.

## Search notes: pages refused by the network policy

- AMD, "AMD Ryzen AI Max+ 395" product page
  (https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html):
  refused by the session's network policy (DNS failure, then HTTP 403 at the proxy). It would
  state the GPU clock, the memory bus width and rated bandwidth, the maximum memory and the TDP
  range.
- AMD, "RDNA 3.5 Instruction Set Architecture" reference guide (docs.amd.com) and GPUOpen's "How to
  accelerate AI applications on RDNA 3 using WMMA" and "RDNA performance guide" (gpuopen.com):
  refused. They would state wave64 execution, the LDS bank width and the WMMA rate per clock.
- The rendered pages rocm.docs.amd.com (GPU specifications, compatibility matrix, Radeon and Ryzen
  pages), llvm.org/docs/AMDGPUUsage.html and docs.kernel.org/gpu/amdgpu: refused; their sources were
  read on GitHub instead ([S74] to [S88]).
- techpowerup.com, chipsandcheese.com, notebookcheck.net and phoronix.com secondary pages: refused.

## Search log

- 2026-10-10, R32. Budget about 2 hours; used about 1 hour 15 minutes, 6:35 to 7:50 PM ET. Three
  parallel searches (product specifications; ISA, registers and LLVM; ROCm, driver and libraries),
  each told to cite only pages it opened. Every vendor host was refused by the network policy, so
  the search moved to the GitHub sources of the same documents, cloned shallow and pinned to the
  commits in `SOURCES.md`; the main facts (the gfx1151 row, the 1536-VGPR feature, the granule,
  the WMMA list, the driver permissions, the PyTorch hipBLASLt preference) were re-read by hand at
  those commits. Our own pod numbers come from `lab/README.md` and the R30 task page. Stopped
  because what is left (clock, bus width, rated bandwidth and TDP, wave64 execution, LDS bank
  width, WMMA rate) is only on the refused vendor pages or must be measured by R33.
