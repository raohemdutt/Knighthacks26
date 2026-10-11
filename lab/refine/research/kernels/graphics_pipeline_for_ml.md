# Can the graphics pipeline speed up the teacher's inference or the student's fine-tuning on the Strix Halo (gfx1151)?

Research note for the kernels track, written for Task Board task R48 and searched on 2026-10-10
from online sources only (no GPU and no pod was used). `SOURCES.md` in this folder indexes every
source; `[Sxx]` ids point into it and `[L8]` and `[L9]` are our own measurement and page. Labels,
the Implication lines and the Experiment lines follow the conventions at the top of
`w4a16_decode_gfx1201.md`; the experiment ids `G1` to `G3` are local to this note and are written
out as T0 queue rows in the section "Candidate experiments, ranked".

What this note covers: the history and mechanism of computing through the graphics pipeline (Brook
and the shader era), which graphics-only hardware on an RDNA 3.5 APU a compute program can still
reach and what it is good for, how inference engines use graphics APIs today and what they
measured against compute-only paths, and which of these deserve one experiment on the lab's two
SJC Strix Halo pods (Radeon 8060S, gfx1151, 40 CUs, 64 GB of LPDDR5X shared with the CPU, ROCm
7.13), for the teacher (vLLM serving `RedHatAI/Qwen3.8-27B-INT4`) and for the student
(`Qwen3.5-4B` trained with LoRA). Sources from 2025 and 2026 come first; anything older than 2024
is marked background. Chip numbers come from R32's datasheet note (`strix_halo_hardware.md`,
merged while this note was written), which leaves the memory bus width and bandwidth unknown, and
from R33 (the measured chip sheet, not yet run); where this note needs a number neither has, it
says TBD. Whether Vulkan and Mesa are installed on the pod image is unknown and is a question for
R33.

## Summary

The teacher's 2.88 tok/s on gfx1151 [L8] is a compute-kernel gap, not a missing graphics trick.
The same 27B model family at 4 bits decodes at about 12.4 tok/s on the same chip in llama.cpp's
Vulkan backend [S138], and vLLM's own HIP skinny W4A16 decode kernel, merged in July 2026 after our
vLLM commit, cut time per output token 4.0x on a Strix Halo [S13]. That kernel is already queued as
E19 [L9]; nothing in the graphics pipeline competes with it as a lever for the teacher.

Through a graphics API, the 8060S reaches the same compute units by another driver and compiler
(Mesa's RADV with its ACO compiler) [S151], the same WMMA matrix units through
`VK_KHR_cooperative_matrix` [S136], and the graphics queue [S142]. In a 2026 community head-to-head
this gave 5 to 8% more decode than llama.cpp's HIP build on dense 27B and 31B models, while HIP kept
the lead on long prompts [S140]. The gain comes from kernels and the driver, not from texture or raster units.

The graphics-only units help less than the 2004 story suggests. On RDNA 3.5 every vector load
already goes through the texture cache [S132]. HIP cannot sample block-compressed textures [S137].
AMD hardware has no ASTC decoder, which RADV emulates by keeping a decoded copy [S136]. Apple's 2025
server model stores its weights as ASTC at 3.56 bits per weight and decodes them in fixed-function
hardware [S135]; on AMD the only comparable path is BC4 at 4 bits per weight through Vulkan [S133],
which saves ALU work but no bytes. For fine-tuning, the graphics paths reach nothing the student's
PyTorch-on-ROCm LoRA run can use: compressed textures are read-only [S133], RADV turns off bf16
matrix math on GFX11 to 11.5 [S136], and Vulkan LoRA training in llama.cpp exists but has no AMD
number against PyTorch [S161].

Three experiments are proposed: G1, llama.cpp Vulkan against llama.cpp HIP on pod B (a reference
for the fallback engine and for E20); G2, the Vulkan graphics queue against the compute queue
(rides in G1's slot); G3, BC4 texture-decoded weights, gated by a CPU-only quality check on the
student before any GPU time.

## 1. History and mechanism: Brook and shader-era GPGPU

- **[measured]** Brook for GPUs (Buck, Foley, Horn, Sugerman, Fatahalian, Houston, Hanrahan;
  SIGGRAPH 2004, ACM TOG 23(3)) ran general programs on graphics hardware. It stored each stream
  as a floating-point texture and compiled each kernel to a Cg fragment shader. A kernel ran by
  drawing one quad over the output and rendering into the output stream's texture, and gathers
  became dependent texture reads. Reductions took O(log n) render passes, and scatter ran on the
  CPU "due to GPU hardware limitations". On an ATI Radeon X800 XT against a 3 GHz Pentium 4, SAXPY
  ran about 7x and image segmentation 4.7x faster, FFT 0.99x, and SGEMV 2251 against 1407 MFLOPS.
  Summing 2^20 float4 values took 0.79 ms (ATI, DirectX) against 14.6 ms on the CPU. Background
  (2004). Source: [S128] sections 3.3, 3.4, 4.1 to 4.3 and 5.1, Figure 3 and its table.

  Implication: inference and fine-tuning alike. This is the mechanism Ilan's question refers to:
  weights in textures, the fragment processor as the kernel engine, render targets as outputs. The
  four limits that ended it (fixed texture sizes, float-precision addressing, no scatter, multipass
  reductions) are what compute APIs removed. Brook's cost model, where the GPU wins when the work per
  byte moved is high, is the same bandwidth argument that bounds decode today.

  Experiment: none; background that frames sections 3 and 4.

- **[measured]** Brook's own measurements tied the speedups to memory bandwidth. The paper reports
  "1.2 Gfloats/sec of floating point texture bandwidth on NVIDIA compared to ATI's 4.5
  Gfloats/sec", against a peak of 40 billion multiplies per second on NVIDIA and 33 billion on ATI.
  It also predicted that "future streaming hardware will share the same memory as the CPU,
  eliminating the need for data transfer altogether". Background (2004). Source: [S128] sections 5.1
  and 6.

  Implication: the APU we run on is that prediction. On the pods the transfer cost Brook modelled is
  gone, and what is left is the bandwidth of one shared LPDDR5X pool serving both the CPU and the
  GPU, which R32's note could not state from opened sources (`strix_halo_hardware.md`, section 5;
  TBD until R33's SH7).

  Experiment: none beyond R33 measuring that bandwidth.

- **[practitioner]** The 2005 GPGPU survey describes the shader-era recipe: kernels as fragment
  programs, arrays in textures, an invocation as a screen-aligned quad, and output only at the
  fragment's own pixel. That last rule is why "fragment processors are incapable of memory scatter".
  The survey also lists fixed-function units borrowed for compute. Z-cull skipped masked work,
  occlusion queries counted surviving fragments without a stall, depth and stencil tests evaluated
  database predicates, and blending implemented sorting comparisons. It notes that Bohn et al. used
  fixed-function graphics hardware to compute artificial neural networks. Background (2005).
  Source: [S129] sections 2.3, 2.4.2.3, 2.4.2.4, 4.1.5, 4.5 and 5.1.

  Implication: inference: each of these units is still on gfx1151, but a compute kernel now gets
  scatter, atomics, local memory and early exit without them. Fine-tuning: neural networks ran on
  graphics hardware before CUDA, but only as small fixed-function tricks; nothing in the survey
  scales to a 4B model's backward pass.

  Experiment: none; sections 3 and 5 check what is still reachable.

## 2. The teacher's gap on gfx1151 is in the compute kernels

- **[measured]** The stock teacher (vLLM at b1388b1 with `lab/vllm-gfx1201.diff`, built for
  gfx1151, `RedHatAI/Qwen3.8-27B-INT4`, batch 1, temperature 0, text prompt, 77 output tokens)
  decoded at 2.88 tok/s on pod C at 120.3 W average over the decode window. That is about 41.8 J
  per token (120.3 W / 2.88 tok/s, derived), and the engine took 297 s to start. Source: [L8]
  `lab/refine/metrics.jsonl:24`, `lab/refine/metrics.jsonl:29` and `lab/refine/metrics.jsonl:34`
  (R29 on pod C).

  Implication: this is the before value for every inference proposal below. With about 15.5 GB
  read per decode step (the main note's byte budget), 2.88 tok/s is about 44.6 GB/s, as R32's note
  derives (`strix_halo_hardware.md`, section 5; R47 will measure the bytes per step). That is far
  below the roughly 212 GB/s that section 4 shows llama.cpp sustaining on the same chip.

  Experiment: G1, as specified in section "Candidate experiments, ranked".

- **[measured]** vLLM PR #40977 (merged 2026-07-14) added a hybrid W4A16 linear kernel for gfx11
  and gfx12. It routes at most 5 rows to a HIP skinny GEMM for decode and larger batches to a Triton
  fused-dequant GEMM, with tile heuristics tuned for gfx1151. Its decode kernel uses "a
  marlin-style magic-number dequantization trick for bf16 models, avoiding an expensive fp32
  round-trip on gfx1151". On a Strix Halo with LPDDR5X-8000 at `--max-num-seqs 1`, time per output
  token for `RedHatAI/Qwen3-4B-quantized.w4a16` fell from 54.81 to 13.69 ms (4.00x), and for
  `Qwen2.5-7B-Instruct-AWQ` from 82.04 to 21.03 ms (3.90x). Source: [S13] vLLM PR #40977,
  description, benchmark tables (re-opened 2026-10-10 for this note).

  Implication: inference: the software analogue of a texture unit's free format conversion already
  exists as a bit trick in this kernel, and it is the lever for the teacher. Our b1388b1
  (2026-04-18) predates it, and E19 on T0 tests it against Conch [L9]. Carries to our chip: yes,
  it was measured on gfx1151. Does not carry: the 27B shapes, which E19 measures. Fine-tuning: the
  same kernel family is what a QLoRA-style student run with a 4-bit frozen base would call; the
  student's LoRA today loads a bf16 base (`lab/smoke_lora.py`), so nothing changes there.

  Experiment: none beyond E19 [L9], which this note does not duplicate.

- **[measured]** vLLM PR #56301 (merged 2026-10-06) found that on gfx11 (RDNA3 and 3.5) a W4A16
  GEMM "loses a large share of its achievable read bandwidth when an operand's row stride lands on a
  multiple of 1024 bytes", because the rows alias into the same L2 and DDR channels. Padding the
  stride by one 128-byte cache line improved time to first token by up to 23% and time per output
  token by up to 3%, and the PR gates the fix on gfx1151, "where the effect was measured". Source:
  [S18] vLLM PR #56301, description (re-opened 2026-10-10 for this note).

  Implication: inference: the memory channels of this APU punish power-of-two strides. That matters
  for any layout this note proposes, BC4 texture rows included (G3 stage 2 pads image rows the same
  way). Fine-tuning: the student's LoRA run loads its base in bf16 (`lab/smoke_lora.py`), and
  whether its GEMM strides hit the same cliff is something R34's training profile can check.

  Experiment: none beyond G3, which applies the padding rule to its image layout.

## 3. Graphics-only hardware on an RDNA 3.5 APU, as reachable from compute

### 3.1 Texture units and the texture cache path

- **[practitioner]** The ROCm Compute Profiler documentation for RDNA 3.5 (it names gfx1151) counts
  "global, buffer, and texture memory operations" together as vector memory instructions. It defines
  TCP, "Texture Cache Per-pipe", as "the L0 vector cache in RDNA3.5, one per WGP", and counts
  texture and image loads separately (`SQ_INSTS_TEX_LOAD`). Source: [S132] ROCm Compute Profiler 3.6.0
  (ROCm 7.13.0 preview), "Workgroup processor (WGP)".

  Implication: inference: on this chip an ordinary buffer load of packed weights already travels
  through the texture cache, so the old "route reads through the texture cache" trick has nothing
  left to win for plain weight streaming. A texture fetch can only add the sampler's work: format
  conversion, filtering and block decode. Fine-tuning: the same holds for the student's weight and
  activation reads in PyTorch.

  Experiment: none beyond G3, which is the one test of the sampler's own work.

- **[practitioner]** HIP still exposes texture objects to compute kernels (`hipTextureObject_t`,
  `hipCreateTextureObject`), with point or linear filtering and border, clamp or wrap addressing.
  Its documentation says texture reads "are routed through a special read-only texture cache"
  "optimized for logical spatial locality". Source: [S131] HIP 7.16.0 documentation, "Texture
  fetching".

  Implication: inference: the Brook-era path survives in HIP as an API, but [S132] shows that on
  RDNA 3.5 the "special" cache is the same L0 that buffer loads use. On this chip the texture-object
  path offers filtering and normalized reads, not extra bandwidth. Fine-tuning: texture objects are
  read-only, so they could only hold frozen weights.

  Experiment: none; no movement is expected from re-routing reads, for the reason in [S132].

- **[practitioner]** Mobile engines store LLM weights in images on purpose. Qualcomm's Adreno
  OpenCL backend in llama.cpp reads quantized weights and activations through `image1d_buffer_t`
  (`read_imageui`, `read_imagef`) in its matrix-vector kernels. Qualcomm's IWOCL 2025 talk lists
  "use of image vs buffer" among its optimizations. Alibaba's MNN-LLM says Adreno GPUs "possess
  powerful texture engines and dedicated L1 caches" and that Image objects "offer higher bandwidth",
  so it stores rearranged 4-bit weights in RGBA images. Google's ML Drift maps tensors to textures,
  image buffers and buffers per device. Sources: [S144] llama.cpp `ggml/src/ggml-opencl/kernels/gemv_noshuffle_q8_0_f32.cl`
  at 23b0202, `kernel_gemv_noshuffle_q8_0_f32`; [S145] slide 11; [S146] section 5.1; [S147] section 3.2, "Tensor Virtualization".

  Implication: inference: the 2025 texture-storage results are about GPUs whose texture path has
  its own L1. On RDNA 3.5 it does not [S132], so we expect no bandwidth gain from moving weights into
  images. What carries is the layout discipline: 128-bit loads matched to the fetch width
  ([S146]), and tensors laid out per device. Fine-tuning: none of these engines trains.

  Experiment: none beyond G3; a buffer-versus-image test without block decode is not proposed,
  because [S132] predicts no movement.

### 3.2 Block-compressed formats as weight storage

- **[practitioner]** BC4 stores a 4x4 block of one channel in 8 bytes: two 8-bit endpoints and
  sixteen 3-bit indices into an 8-entry table interpolated between them, which is 4 bits per value.
  BC4_UNORM covers 0 to 1 and BC4_SNORM covers -1 to 1, the hardware decodes it automatically, and
  "a block-compressed texture ... cannot be used as an output of the pipeline". BC1 also stores 4
  bits per texel but with RGB565 endpoints and 2-bit indices. Source: [S133] Microsoft Learn, "Block
  Compression (Direct3D 10)", sections BC1, BC4, BC4_UNORM and BC4_SNORM (page updated 2025-04-15).

  Implication: inference: BC4 is a fixed 3-bit codebook per 16 weights with 8-bit endpoints relative
  to one global range. A per-group fp16 scale and offset stored outside the block, as Apple does
  for ASTC [S135], would make it about 4.1 bits per weight, no smaller than 4-bit group quantization.
  Any gain is ALU work moved to the texture unit, not bytes saved. Fine-tuning: read-only, so only a
  frozen base could live in it.

  Experiment: G3, as specified in section "Candidate experiments, ranked".

- **[anecdote]** A 2021 reverse-engineering post reports that BC4 and BC5 "decode to more than 8
  bits of precision" on AMD, Intel and NVIDIA. AMD appears to use an internal 14-bit format, and on
  AMD 13 of 16,077 possible BC4/BC5 results round differently in the last float bit. Decoders also
  differ between vendors for BC1 to BC5, while the D3D11 spec requires bit-exact BC6H and BC7.
  The post names no GPU models or drivers. Background (2021). Source: [S134].

  Implication: inference: a BC4 weight decoded by an RDNA texture unit has more precision than its
  8-bit endpoints suggest. It may not match a CPU reference bit for bit, so G3 stage 1 must emulate
  the decoder at float precision and stage 2 must check the GPU's decode against it. Fine-tuning: a
  LoRA trained against a CPU-decoded BC4 base would see weights that differ slightly from the
  GPU's, which K4-style output checks would have to absorb.

  Experiment: G3, stage 2 compares the GPU's decoded values with the numpy decoder before timing.

- **[practitioner]** Apple's 2025 server foundation model stores its weights with ASTC, a GPU
  texture format, applied after training. It uses 6x6 blocks of 36 weights in 128 bits (about 3.56 bits per weight) in
  ASTC's HDR-ch mode, which takes no negative values. So each block's minimum is subtracted first,
  stored as one fp16 value, and added back inside the following matrix multiply. "The fixed-function
  hardware for ASTC decompression within Apple GPUs enables these blocks to be decoded on-the-fly
  with effectively zero compute cost". Before compression, the most significant singular vectors
  are moved into LoRA adapters, and the adapters are fine-tuned to recover quality. Source: [S135]
  Apple, "Apple Intelligence Foundation Language Models: Tech Report 2025", arXiv:2507.13575,
  section 6.

  Implication: inference: this is the strongest 2025 evidence that texture formats work as LLM
  weight storage in production, on hardware that decodes them. Carries to gfx1151: the per-block
  minimum outside the block, and adapters to recover quality. Does not carry: ASTC itself, which
  AMD hardware does not decode (next finding), and the 3.56 bits per weight. Fine-tuning: it is
  also the precedent for training LoRA adapters against a frozen, texture-compressed base.

  Experiment: G3, as specified in section "Candidate experiments, ranked".

- **[measured]** Apple reports MMLU 80.0 and IFEval 89.1 for its server model at 16 bits per weight,
  and 79.2 and 90.2 after its whole optimization stack (ASTC weights at about 3.6 bits per weight,
  4-bit embeddings, 8-bit KV cache and the recovery adapters). Source: [S135] Table 3.

  Implication: inference: with adapters, an ASTC-compressed base lost 0.8 MMLU points. The number
  is for the whole stack, so it does not isolate the texture format, and G3 stage 1 measures that
  part on our student without adapters. Fine-tuning: it bounds what a recovery LoRA can buy back.

  Experiment: G3 stage 1, as specified in section "Candidate experiments, ranked".

- **[practitioner]** In Mesa's RADV driver at main (bffd919, 2026-10-10), `textureCompressionBC` is
  always true. `textureCompressionASTC_LDR` is set only when ASTC emulation is on (a drirc option,
  or always on Android), `textureCompressionASTC_HDR` is false, and an emulated ASTC image carries a
  second plane in an uncompressed emulation format (`radv_image_get_plane_format`). Source: [S136]
  `src/amd/vulkan/radv_physical_device.c` (`radv_physical_device_get_features`,
  `radv_physical_device_try_create`) and `src/amd/vulkan/radv_image.c`.

  Implication: inference: on our chip, BC1 to BC7 are the only compressed formats the texture unit
  decodes. Apple's ASTC route does not carry, because an emulated ASTC image is read back from its
  decoded plane and moves full-size bytes (our inference from the second plane). Fine-tuning: none.

  Experiment: none beyond G3, which uses BC4 for this reason.

- **[practitioner]** HIP declares block-compressed resource-view formats
  (`hipResViewFormatUnsignedBlockCompressed1` to `hipResViewFormatUnsignedBlockCompressed7`), but the
  runtime's helpers that map a view format to channels and an array format (`getNumChannels`,
  `getArrayFormat`) have no case for them. Those helpers fall through to the branch commented "error
  scenario". Source: [S137] ROCm/rocm-systems at 71a246a, `projects/hip/include/hip/driver_types.h`
  and `projects/clr/hipamd/src/hip_conversions.hpp::getNumChannels`.

  Implication: inference: a HIP or Triton kernel, and so vLLM, cannot sample a BC texture through any
  path we found. The block-decode hardware is reachable only through a graphics API (Vulkan via
  RADV), which is why G3 stage 2 is a Vulkan compute shader. Fine-tuning: PyTorch on ROCm cannot
  use BC-stored weights at all. A one-line HIP probe (create a texture object with a BC4 view and
  check the error) would confirm this, and is an R33 question.

  Experiment: none beyond the probe listed under Open questions.

- **[measured]** An HPG 2025 paper stores a neural representation of PBR texture sets in standard
  BC formats, which keeps hardware texture filtering, and runs the network on the GPU's "hardware
  matrix multiplication engine" (cooperative vectors, in its title). Rendering 4K texture sets (9 channels)
  with anisotropic filtering at 1080p took 0.55 ms on an Intel B580, with 28 MB of VRAM per texture
  set. Source: [S162] Belcour and Benyoub, "Hardware Accelerated Neural Block Texture Compression
  with Cooperative Vectors", HPG 2025, arXiv:2506.06040, abstract.

  Implication: inference: graphics research in 2025 runs the BC-decode-then-matrix-engine pipeline
  for small per-material networks, not for billions of weights. It shows the two units can be chained from
  shaders, not that the chain beats a matrix-vector kernel for LLM decode. Fine-tuning: none.

  Experiment: none beyond G3.

### 3.3 Sampler interpolation, rasterizer, render targets, ray and media engines

- **[practitioner]** HIP's linear filtering "does a linear interpolation between values" of
  neighbouring texels. The 2005 survey notes that GPGPU data structures kept 2D locality so they
  could use the hardware's "native bilinear filtering". Sources: [S131] "Texture fetching", filter
  modes; [S129] section 4.2.1, Figure 7.

  Implication: inference: a 1D texture with linear filtering is a hardware piecewise-linear lookup
  table, usable for an activation function or a dequantization table. At batch-1 decode, activation
  math is a small share of the step next to weight reads, so the unit has nothing to remove. The
  precision of AMD's interpolation weights is undocumented in what we opened (R33 question).
  Fine-tuning: a filtered lookup has no useful gradient path in PyTorch.

  Experiment: none; no movement is expected for decode or training.

- **[practitioner]** The shader-era uses of the rasterizer and render targets (quads to launch
  kernels, Z-cull to skip masked work, occlusion queries to count, blending to combine, depth and
  stencil tests for predicates) are described in the 2005 survey. No 2025 or 2026 source in our
  search uses them for LLM inference or training (search log). Background (2005). Source: [S129]
  sections 2.3, 2.4.2.3, 2.4.2.4, 4.1.5 and 4.5.

  Implication: inference and fine-tuning: on gfx1151 a compute kernel has atomics, LDS reductions
  and early exit, which do these jobs without a graphics pipeline state. Driving the rasterizer
  would also need a graphics pipeline in Vulkan, adding binding cost for no unit the decode step
  lacks.

  Experiment: none; not worth one.

- **[anecdote]** On RDNA 2, AMD's ray accelerators sit in the texture unit. They run only an
  intersection test, at four box tests or one triangle test per cycle, and "regular compute shader
  code deals with traversing the BVH". Background (2023, RDNA 2, not our chip). Source: [S158] Chips
  and Cheese, "AMD's RDNA 2: Shooting For the Top", raytracing section.

  Implication: inference and fine-tuning: RT-core nearest-neighbour methods built on NVIDIA's
  full-traversal hardware would not carry one for one, because traversal here is shader code.
  Atlas's vector search runs on the Jetson in sqlite-vec, not on the pods, so no pod metric moves.

  Experiment: none; not worth one.

- **[opinion]** A January 2026 study of RT-core fixed-radius neighbour search reports gains for its
  RT-core pipeline and concludes that in some simulation cases "regular GPU computation should still
  be preferred". Source: [S159] Meneses et al., "Advancing RT Core-Accelerated Fixed-Radius Nearest
  Neighbor Search", arXiv:2601.15633, abstract.

  Implication: none for the metric set. Even on hardware built for it, the ray path wins only on
  some workloads, and none of ours is neighbour search on the pods.

  Experiment: none.

- **[practitioner]** rocDecode, ROCm's video decode library on the VCN media engine, lists gfx908,
  gfx90a, gfx942, gfx1030 to gfx1032 and gfx1100 to gfx1102 in its hardware table, and no RDNA 3.5
  APU. Source: [S160] rocDecode 1.10.0 documentation, "rocDecode supported codecs and hardware
  capabilities".

  Implication: inference and fine-tuning: the media engine is not documented as reachable from ROCm
  on gfx1151. It would not matter anyway, because raw frames never leave the Jetson and the pods
  receive only blurred keyframes and text. The display engine has no compute use in any source we
  found.

  Experiment: none; not worth one.

### 3.4 Shared CPU-GPU memory

- **[practitioner]** AMD's ROCm guide for RDNA 3.5 APUs says memory "is mapped rather than
  physically partitioned", GTT allocations "are dynamic and not permanently reserved" with a
  default limit of about half of system RAM (raised through `/sys/module/ttm/parameters/pages_limit`
  or `amd-ttm`), the BIOS VRAM reservation should be small (for example 0.5 GB), and "AI frameworks
  work more efficiently with GTT-backed allocations". Source: [S156] ROCm documentation, "AMD
  RDNA3.5 system optimization".

  Implication: inference and fine-tuning: on the pods (no root, BIOS fixed by AMD's cloud) the
  carve-out and the TTM limit are given, not levers. What a job controls is the allocator and
  whether weights are mapped from files. Both draw on the same LPDDR5X, so neither changes decode
  bandwidth. The pod's VRAM and GTT totals are an R33 item.

  Experiment: none beyond R33 recording the VRAM and GTT totals that `rocm-smi` reports.

- **[practitioner]** On a gfx1151 report that ROCm added a carve-out VRAM pool and a GTT pool as
  if they were separate memory, AMD's engineers replied that the two pools are carved from one
  physical memory. An upstream Linux driver change now "prevents GTT from ever exceeding the amount
  of physical RAM backing it". Source: [S157] ROCm/ROCm#6004, comments of 2026-06-18 and 2026-08-10
  by harkgill-amd.

  Implication: inference and fine-tuning: a memory number that adds VRAM and GTT on these pods
  overstates what fits. R33 and the placement module (R46) should read physical memory, not the sum,
  when sizing the teacher, a replica and the student together.

  Experiment: none beyond R33.

- **[practitioner]** llama.cpp's Vulkan backend marks an integrated GPU as UMA
  (`device->uma` from `ePhysicalDeviceType::eIntegratedGpu`). On UMA it looks up host-mapped
  buffers for tensors (`ggml_vk_host_get`), so weights and activations in host memory are bound to
  shaders without a copy. Source: [S143] llama.cpp `ggml/src/ggml-vulkan/ggml-vulkan.cpp` at
  23b0202, `ggml_vk_tensor_subbuffer` and `ggml_vk_mul_mat_q_f16`.

  Implication: inference: a graphics-API engine already uses the APU's shared memory as zero-copy
  storage. For a 16 GB model that removes a host-to-device copy at load, which is a start-up
  time lever (plumbing), not a decode lever. Fine-tuning: none.

  Experiment: none for the kernels metric; G1 records model load seconds per arm as a side metric.

## 4. How inference engines use graphics APIs today

- **[measured]** llama.cpp PR #27952 (merged 2026-09-24) added an int8 cooperative-matrix matmul
  path for RDNA3 and RDNA4 in the Vulkan backend. On a Radeon 8060S, `Qwen3.8-27B Q4_K_M` prompt
  processing at 512 tokens went from 186.5 to 254.1 tok/s, against 237.6 tok/s for the ROCm build of
  the same master. On the same chip with Mesa 26.2.1 RADV, `-fa 1`, 3 repeats, a user measured the
  dense model at pp512 349 to 380 tok/s and tg128 12.4 tok/s before and after the PR, with wikitext-2
  perplexity unchanged within error (5.7401 and 5.7225, plus or minus 0.18). Source: [S138] llama.cpp
  PR #27952, description (table "llama-bench pp512") and comment by huppiflupp of 2026-09-18.

  Implication: inference: the same model as our teacher, at 4 bits, decodes at 12.4 tok/s on our chip
  through Vulkan, 4.3x the vLLM build's 2.88 [L8]. Carries: same chip and model family. Does not
  carry: a 128 GB machine with its own BIOS split, Mesa 26.2.1 rather than whatever the pod has, and
  Q4_K_M against our INT4 checkpoint. Fine-tuning: none.

  Experiment: G1, as specified in section "Candidate experiments, ranked".

- **[measured]** A 2026 head-to-head on a 128 GB Strix Halo (Radeon 8060S, Ubuntu 26.04, kernel 7.0,
  ROCm 7.2.2, llama.cpp b8966, `-ngl 99 -fa 1 -mmp 0 -ub 2048 -r 3`, HIP built with rocWMMA flash
  attention off) measured a dense-plus-Gated-DeltaNet `qwen35-27b` Q4_K_M (15.58 GiB). Vulkan gave
  tg128 12.67 tok/s against HIP's 12.05, with pp512 365 against 366 and pp16384 247 against 293 (HIP
  ahead). A dense `gemma4-31b` gave tg128 11.25 against 10.44 for Vulkan, and output quality matched
  (96.16% against 96.07% on the author's test). Mesa's version is not stated. Source: [S140]
  nabe2030/hip-vs-vulkan-evo-x2, setup and results tables.

  Implication: inference: Vulkan leads decode by 5 to 8% on dense 27B to 31B models, and HIP leads
  long prompts by up to 19% on the 27B. At 12.67 tok/s, a 16.7 GB file streams at about 212 GB/s
  (derived), which is the effective bandwidth this chip sustains in a good decode kernel until R33
  measures it. For the teacher's caption workload (short prompts, short outputs) decode dominates.
  Fine-tuning: none.

  Experiment: G1, as specified in section "Candidate experiments, ranked".

- **[measured]** On a Radeon 8050S (32 CUs, RADV and Mesa 26.1.8, `uma:1`), `Qwen3.8-27B UD-Q4_K_S`
  ran pp128 194.13 plus or minus 1.45 and tg32 12.86 plus or minus 0.10 tok/s.
  `GGML_VK_DISABLE_COOPMAT=1` cut prefill 45% and left decode unchanged.
  `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` cut prefill 10% and left decode unchanged. The device
  reports only 16x16x16 fp16 and int8 cooperative-matrix shapes. Source: [S139] llama.cpp issue
  #28843, "Measurements / observations".

  Implication: inference: in the Vulkan backend the matrix units (the same WMMA HIP uses) carry
  prefill, and decode runs on matrix-vector shaders that use neither switch. A smaller sibling chip
  decodes at the same 12.9 tok/s, which says decode is bandwidth-bound there. G1's decode result
  should not depend on the cooperative-matrix path, and the two environment switches give G1 a free
  check of which path ran. Fine-tuning: none.

  Experiment: G1, which records both switches' effect on prefill as a path check.

- **[anecdote]** One Strix Halo user moved generation of a Qwen3.6-35B-A3B MoE from llama.cpp's HIP
  build to mainline Vulkan and RADV and reports single-stream decode rising from 47.7 to 60.9 tok/s,
  and about 59 against 80 tok/s at build b9870. The same user reports that Ubuntu 24.04's glslc
  2023.8 fails the `integer_dot.comp` feature probe, "silently" dropping the q8_1 integer-dot
  kernels, and that a 2026 glslc passes. They measured vLLM through a community gfx1151 toolbox at
  about 20 tok/s single-stream on the same model. Approximate numbers, one machine, a prefill
  statement that contradicts itself. Source: [S141] llama.cpp discussion #20856, update of 2026-07-23
  and comment of 2026-07-31.

  Implication: inference: the Vulkan lead is larger on a MoE than on a dense model, and a Vulkan
  build can lose its fast kernels without an error if the shader compiler is old. G1 must record the
  glslc version and check that the integer-dot kernels were built. Fine-tuning: none.

  Experiment: G1, which pins the glslc version and checks the build log for the integer-dot variants.

- **[practitioner]** Mesa's RADV documentation says "ACO is the default shader-compiler used in
  RADV", and keeps an LLVM backend "solely used for testing and hardware bringup". RADV enables
  `VK_KHR_cooperative_matrix` on GFX11 and newer only when ACO compiles (`radv_cooperative_matrix_enabled`).
  Sources: [S151] Mesa documentation, "RADV"; [S136] `radv_physical_device.c` at bffd919.

  Implication: inference: a Vulkan build on gfx1151 runs its kernels through a different compiler
  (ACO) from HIP and Triton (LLVM AMDGPU), on the same compute units and matrix instructions.
  Differences between G1's arms come from compilers and kernel design, not from graphics hardware.
  Fine-tuning: none.

  Experiment: none beyond G1.

- **[practitioner]** In RADV at main (2026-10-10), `VK_KHR_shader_bfloat16` is exposed only on GFX12
  or with an experimental flag, with the comment "GFX11-11.5 has precision issues".
  `VK_EXT_shader_float8` needs GFX11.7 or newer, while `VK_KHR_shader_integer_dot_product`,
  `VK_KHR_shader_float_controls2`, `VK_EXT_descriptor_buffer` and `VK_EXT_descriptor_heap` are
  exposed. Mesa 25.2.0 (2025-08-06) first listed bfloat16 for radv/gfx12+ and float8 for RDNA4+.
  Sources: [S136] `radv_physical_device.c` at bffd919, `radv_bfloat16_enabled` and the extension
  table; [S149] Mesa 25.2.0 release notes, "New features".

  Implication: inference: on gfx1151 (GFX11.5) a Vulkan engine gets fp16 and int8 matrix math but
  no bf16 or fp8. That is enough for 4-bit weights with fp16 activations. Fine-tuning: bf16 training,
  which is what the student's LoRA uses on ROCm, has no Vulkan matrix path on this chip.

  Experiment: none; a fact for section 5.

- **[practitioner]** Khronos's SIGGRAPH 2025 update (2025-08-05) says the Vulkan ML Task Sub Group
  released `VK_KHR_shader_bfloat16` and `VK_EXT_shader_float8`. It names llama.cpp's Vulkan backend
  as a user of `VK_KHR_cooperative_matrix`, `VK_NV_cooperative_matrix2`,
  `VK_KHR_shader_integer_dot_product` and `VK_KHR_shader_bfloat16`. Khronos's Roadmap 2026 post
  (2026-01-23) adds `VK_EXT_descriptor_heap`, "a complete overhaul of the current Vulkan descriptor
  system". Sources: [S148]; [S164] Khronos blog, "Vulkan introduces Roadmap 2026 and new descriptor
  heap extension".

  Implication: inference: the Vulkan ML additions of 2025 and 2026 are new data types and cheaper
  binding. None adds a weight-storage path, and the data types do not reach GFX11.5 (previous
  finding). Descriptor heaps lower per-dispatch cost, which matters at batch-1 decode with hundreds
  of dispatches per token. They reach us only if llama.cpp adopts them. Fine-tuning: none.

  Experiment: none.

- **[measured]** llama.cpp PR #20551 (merged 2026-03-15) made the Vulkan backend submit work on the
  graphics queue on AMD, after a user saw decode rise with `RADV_DEBUG=nocompute`. On an RX 9070 XT,
  tg128 rose 4.8% (Llama 8B Q4_0) and 10.0% (Qwen3 MoE 30B-A3B Q2_K) at full offload, while a Radeon
  Pro VII moved between -2.1% and +3.0%. Two days later PR #20599 made it opt-in
  (`GGML_VK_ALLOW_GRAPHICS_QUEUE=1`), after desktop stutter on small GPUs, with the maintainer
  calling it "only really safe when headless". Sources: [S142] llama.cpp PRs #20551 (benchmark
  tables) and #20599 (comments of 2026-03-15); [S143] `ggml-vulkan.cpp` at 23b0202, queue selection.

  Implication: inference: this is the one place where the graphics part of the GPU front end itself
  speeds up LLM decode, through the queue the dispatches go to. Carries: RADV is our driver, and a
  pod is headless. Does not carry: RDNA4 discrete cards, whose sizes differ. Fine-tuning: none.

  Experiment: G2, as specified in section "Candidate experiments, ranked".

- **[anecdote]** A user on the same PR wrote that the graphics queue "measurably helps performance
  on strix halo's 8060s with 40 CUs" but made the desktop "unusable", with no numbers. Source: [S142]
  comment by lemmi on PR #20599, 2026-03-15.

  Implication: inference: the only Strix Halo report on the graphics queue is positive but
  unquantified, which is why G2's confidence is low. Fine-tuning: none.

  Experiment: G2, as specified in section "Candidate experiments, ranked".

- **[measured]** "Llamas on the Web" (May 2026) built a WebGPU backend for llama.cpp and compared it
  with the native backends on 16 devices. On an AMD RX 7900 XT, "the native HIP backend does always
  outperform the WebGPU backend", while "the Vulkan backend underperforms the WebGPU backend during
  prefill by 3x" but "outperforms even the native HIP backend during decode by 38% on the q4_k_m
  model". WebGPU's compiler-inserted safety checks cost an average 14% and 23% of prefill and 5%
  and 1% of decode. Source: [S154] Levine et al., arXiv:2605.20706, section 6.2.

  Implication: inference: on a discrete RDNA3 card, Vulkan's decode lead over HIP was 38%, larger
  than the 5% on Strix Halo [S140], so the size of the lead varies by chip and build. Browser engines
  (WebGPU, WebLLM) run below native and need a browser, which a headless pod does not have, so they
  are out of scope. Fine-tuning: none.

  Experiment: none beyond G1.

- **[practitioner]** Microsoft's DirectML repository says "DirectML is in maintenance mode", with "no
  new functionality or feature updates" planned, and points Windows 11 24H2 users to Windows ML
  (README change of 2025-07-19). AMD discontinued its AMDVLK open-source Vulkan driver on 2025-09-15,
  "throwing our full support behind the RADV driver". Sources: [S153] microsoft/DirectML README and
  commit history; [S152] GPUOpen-Drivers/AMDVLK discussion #416 and README.

  Implication: inference: DirectML is a Windows-only path in maintenance and does not apply to Linux
  pods. On Linux, Vulkan on AMD now means RADV, so G1 needs no driver choice beyond RADV's version.
  Fine-tuning: none.

  Experiment: none.

- **[opinion]** One 2026 Strix Halo write-up argues that "Vulkan beats ROCm" comparisons tested
  untuned HIP, and that HIP with `ROCBLAS_USE_HIPBLASLT=1` and rocWMMA flash attention wins except
  for short-context decode of mid-size dense models. Source: [S165] msbs.com, "I tuned llama.cpp on a
  Strix Halo mini-PC", 2026-07-24.

  Implication: inference: the one case it concedes to Vulkan, short-context decode of a mid-size
  dense model, is the teacher's caption case. G1's HIP arm must be the tuned build, with these flags,
  so the comparison is fair. Fine-tuning: none.

  Experiment: G1, whose arm A uses the tuned HIP build.

## 5. Fine-tuning through graphics paths

- **[anecdote]** Tether's QVAC team (December 2025) added LoRA fine-tuning to llama.cpp with Vulkan
  as the cross-vendor backend. They added an `OUT_PROD` operator for Vulkan and CUDA, optimizer and
  backward operators (`OPT_STEP_ADAM`, `SILU_BACK`, `RMS_NORM_BACK`, `SOFT_MAX_BACK`, `ROPE_BACK`),
  and masked cross-entropy. They report 13 minutes per epoch on an AMD 7900 XTX for Qwen3-1.7B at
  Q8, against 5.5 on an RTX 4090, and quality "near-parity" with PyTorch on their tasks. They give no
  dataset size, sequence length or PyTorch run on the same card. Source: [S161] Hugging Face blog,
  "An Edge-First Generalized LLM LoRA Fine-Tuning Framework for Heterogeneous GPUs", 2025-12-01.

  Implication: fine-tuning: LoRA training through a graphics API now exists in an open engine, with
  the base kept quantized (the QLoRA shape). Nothing in it tells us whether it beats PyTorch on ROCm
  for the student, and the student's Gated DeltaNet layers need backward operators we did not
  verify (Open questions). Inference: none.

  Experiment: none now; not worth one before an AMD number against PyTorch exists (section 6).

- **[practitioner]** llama.cpp's Vulkan backend at 23b0202 dispatches the training operators
  `GGML_OP_OPT_STEP_ADAMW`, `GGML_OP_OPT_STEP_SGD`, `GGML_OP_CROSS_ENTROPY_LOSS_BACK`,
  `GGML_OP_RMS_NORM_BACK`, `GGML_OP_SOFT_MAX_BACK`, `GGML_OP_ROPE_BACK` and `GGML_OP_SILU_BACK`.
  Source: [S143] `ggml/src/ggml-vulkan/ggml-vulkan.cpp` at 23b0202, `ggml_vk_op_get_pipeline`.

  Implication: fine-tuning: the operators a LoRA step needs are on upstream main, so a Vulkan LoRA
  run on gfx1151 is buildable. It would run its matrix math in fp16 or fp32, because RADV gives
  GFX11.5 no bf16 matrix path [S136]. That is a precision change from the student's bf16 run, not
  only a speed one. Inference: none.

  Experiment: none; recorded so a later learning pass can size such a test.

- **[practitioner]** Block-compressed textures cannot be render targets or pipeline outputs [S133],
  and HIP cannot sample them [S137]. Apple trained recovery adapters while the ASTC-compressed base
  stayed frozen [S135]. Sources: [S133] "Block Compression (Direct3D 10)", introduction; [S137]
  `hip_conversions.hpp::getNumChannels`; [S135] section 6, "Quality Recovery Adapters".

  Implication: fine-tuning: a texture format can hold only the frozen base of a LoRA run. The
  adapters, gradients and optimizer state stay in buffers, and on ROCm the base would have to be
  decoded in software. The only route where texture storage meets the student's training is
  Apple's: compress a frozen base, then train a LoRA against it to recover quality. That is G3's
  follow-up, a learning task, only if G3 stage 1 comes close. Inference: see section 3.2.

  Experiment: G3 stage 1, as specified in section "Candidate experiments, ranked".

- **[practitioner]** Khronos took over the Slang shading language on 2024-11-21. It has "automatic
  differentiation as a first-class language feature", generating forward and backward derivative
  code, compiles to SPIR-V for Vulkan among other targets, and can serve as the kernel language for
  PyTorch through slangtorch. Source: [S163] Khronos press release, "Khronos Group launches Slang
  initiative".

  Implication: fine-tuning: a graphics-API route to writing differentiable kernels exists and is
  aimed at neural graphics. We found no LLM-scale training with it, and the student's training is
  bound by its matrix math, which ROCm's libraries already cover. Inference: none.

  Experiment: none; not worth one.

## 6. Worth one experiment, and not worth one

The verdict per technique, inference and fine-tuning separately. "G" rows are the proposals in
the next section; every other row names the finding that rules it out.

| Technique | Inference (teacher, 27B INT4, vLLM today) | Fine-tuning (student, 4B, LoRA in PyTorch on ROCm) |
| --- | --- | --- |
| Vulkan through RADV instead of HIP (graphics API, same compute units) | G1, as a reference: only llama.cpp has the backend, and llama.cpp is the fallback engine (kernels ruling 2 in `lab/refine/kernels/OWNER_RULINGS.md`) | Not worth it now: no bf16 matrix path on GFX11.5 [S136], no AMD number against PyTorch [S161] |
| Graphics queue for compute dispatches | G2, inside G1's slot | Not applicable (no Vulkan trainer in use) |
| Block-compressed weights decoded by the texture unit (BC4) | G3, gated by a CPU quality check | Only as a frozen base under a recovery LoRA (Apple's route [S135]); follow-up to G3, not before |
| ASTC weights (Apple's 2025 trick) | Not on AMD: no hardware decoder, RADV emulates with a decoded copy [S136] | Same |
| Texture objects or images for plain weight streaming | Not worth it: RDNA 3.5 buffer loads already use the texture cache [S132] | Same |
| Sampler interpolation as a lookup table | Not worth it: activation math is not the decode bottleneck | Not worth it: no gradient path |
| Rasterizer, depth and stencil, blending, render targets | Not worth it: compute has atomics, LDS and early exit [S129] | Not worth it: compressed formats cannot be outputs [S133] |
| Ray accelerators | Not worth it: intersection only, traversal in shaders [S158]; no pod workload | Not applicable |
| Video and display engines | Not reachable from ROCm on gfx1151 per rocDecode [S160]; no frames reach the pods | Not applicable |
| APU shared memory as zero-copy storage | A start-up lever, not a decode lever [S143]; G1 records load time | Sizing only: read physical memory, not VRAM plus GTT [S157] |
| The compute-path fix the graphics ideas compete with | E19 on T0 (RDNA hybrid W4A16, 4.0x TPOT on Strix Halo [S13]) | None needed from this note |

## Candidate experiments, ranked (proposed T0 rows)

The kernels rubric ranks by expected joules-per-token gain times confidence, and by that rule
none of the three scores for the teacher. G1 and G2 change the backend inside llama.cpp, which
kernels ruling 2 keeps out of the gain ranking as a reference, and G3 expects 0 to +5% at
confidence 0.15. So the order below is by cost and by what each one settles. G1 is cheap, the most
likely to move a number on this chip, and decides which llama.cpp build E20's arm B and the
fallback engine use on gfx1151. G2 rides in G1's slot. G3 spends no GPU time unless its CPU gate
passes. None of the three is on T0 yet. A kernels pass that queues one takes the next free E number
from a fresh fetch of T0, and each needs Ilan's go and a GPU Queue slot on pod B.

- G1 (kernels) Graphics API against compute API in one engine: llama.cpp Vulkan (RADV) against
  llama.cpp HIP on gfx1151, a reference measurement. Hypothesis: On the Strix Halo the Vulkan build
  of llama.cpp (Mesa RADV, ACO compiler, `KHR_cooperative_matrix` for prefill, integer-dot
  matrix-vector shaders for decode) decodes a 4-bit GGUF of the teacher's model about 5% faster than
  the tuned HIP build of the same commit, while HIP holds or wins long prefill. Both run about 4x the
  stock vLLM build's 2.88 tok/s, because the gap to vLLM is in the 4-bit kernels, not in the API.
  Lever: the llama.cpp GPU backend (`GGML_VULKAN` against `GGML_HIP`), same commit, same GGUF, same
  flags. Arms: A (control): HIP build (`GGML_HIP=ON`, `GPU_TARGETS=gfx1151`, rocWMMA flash attention
  off, `ROCBLAS_USE_HIPBLASLT=1`); B: Vulkan build (`GGML_VULKAN=ON`, RADV, compute queue, a glslc
  that passes the integer-dot probe). Metric: teacher.decode_tok_s, higher is better (also
  teacher.joules_per_token, teacher.ttft_s, teacher.watts_avg, model load seconds). Expected: B over A
  +5% decode at batch 1 (12.67 against 12.05 tok/s for a dense Qwen3.5-27B Q4_K_M in one 2026 run
  [S140]; 12.4 tok/s for Qwen3.8-27B Q4_K_M on Vulkan in another [S138]); pp512 within 10%, HIP ahead
  at pp8192; both arms near 12 tok/s against 2.88 for vLLM b1388b1, so about 10 J per token if the
  pod draws the same 120 W (derived; watts TBD); delta +5%; confidence 0.55; a reference, outside the
  gain ranking by kernels ruling 2. Minimum 5 runs per arm (`llama-bench -r 5` at tg128 and pp512,
  one pp8192 per arm). Predicted effect from the chip sheet: decode ceiling = R33's measured read
  bandwidth / bytes per token; at the 212 GB/s the community Vulkan run implies [S140], a 16.7 GB
  GGUF gives 12.7 tok/s; the rated bandwidth is TBD (R32's Q1, R33's SH7). Cost: 1 day (user-space Mesa RADV
  through `VK_DRIVER_FILES` [S166] if the image has none, two builds, one GGUF of about 17 GB) and
  one 15-minute slot. Kill criterion: step 0 fails, meaning `vulkaninfo` enumerates no device on pod
  B (no RADV, no render-node access): stop, B is void, and the finding goes to R33. Or B sits inside
  both arms' run spread at batch 1: Vulkan brings nothing and the fallback stays HIP. Or the two
  arms' `llama-perplexity` on one fixed wikitext slice differs by more than its reported error, as
  the check [S138] used: stop and explain first. Source: lab/refine/research/kernels/graphics_pipeline_for_ml.md, section 4
  ([S138], [S139], [S140], [S141], [S165]); `lab/refine/metrics.jsonl:24`; kernels ruling 2. Node: pod-b
  (account D, SJC Strix Halo). Note: same GGUF for both arms, named with its quantization type; record
  the Mesa, glslc, ROCm and llama.cpp versions, the device line llama.cpp prints (`uma`, `int dot`,
  `matrix cores`), and both `GGML_VK_DISABLE_COOPMAT=1` and `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1`
  once each on arm B as a path check [S139]. Status: proposed in this note. Metric now: none for
  llama.cpp on our pods; teacher.decode_tok_s 2.88 tok/s for vLLM b1388b1 on pod-c
  (`lab/refine/metrics.jsonl:24`).

- G2 (kernels) Graphics queue against compute queue for llama.cpp Vulkan decode on a headless pod.
  Hypothesis: Submitting the Vulkan backend's dispatches on RADV's graphics queue instead of its
  compute-only queue raised RDNA4 decode by 1% to 10% [S142], and one user reports it "measurably
  helps" on an 8060S [S142]. On a headless pod the desktop stutter that made it opt-in does not apply.
  Lever: `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` (unset is the control), on G1's arm B build unchanged.
  Arms: A (control): unset; B: `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`. Metric: teacher.decode_tok_s, higher
  is better (also teacher.joules_per_token, teacher.ttft_s). Expected: +4% decode at batch 1 (range
  0 to +10%), prefill within 3%; delta +4%; confidence 0.3; a reference like G1. Minimum 5 runs per
  arm. Predicted effect from the chip sheet: none; the mechanism is submission and scheduling, which
  R33 does not measure. Cost: under 0.1 day, in G1's slot after G1's arms (about 10 more minutes).
  Kill criterion: G1 killed at step 0, or B inside the run spread, or any hang or timeout in the 5
  runs. Source: section 4 ([S142], [S143]). Node: pod-b. Note: run only after G1, never on a node with
  a desktop session. Status: proposed in this note. Metric now: none.

- G3 (kernels, with a learning gate) BC4 weights decoded by the texture unit, against packed INT4
  dequantized in the shader. Hypothesis: BC4 stores 16 weights in 8 bytes (two 8-bit endpoints and
  3-bit indices, 4 bits per weight [S133]) and the texture unit decodes it in flight. That is the AMD
  counterpart of Apple's ASTC weights [S135], since ASTC has no AMD hardware [S136]. With a per-group
  fp16 scale and offset stored outside the blocks (as Apple stores a per-block minimum), quality may
  sit near 4-bit group quantization. The bytes per weight do not fall below INT4's, and RDNA 3.5
  already reads buffers through the same cache [S132], so the decode gain is at most the dequant ALU
  work removed. Lever: the weight storage format and the unit that decodes it (BC4 sampled image
  decoded by the texture unit, against packed INT4 dequantized in shader ALU), same shapes, same
  API (Vulkan compute). Arms: stage 1 (CPU only, no GPU): A (control): INT4 group-128 asymmetric
  round-to-nearest of the student's (Qwen3.5-4B) linear weights; B: BC4_SNORM 4x4 tiles of the same
  weights with an fp16 scale and offset per 256 weights, decoded with the D3D rule [S133] in numpy at
  float precision. Stage 2 (pod B, only if stage 1 passes): A (control): a Vulkan compute GEMV
  reading packed INT4 from a storage buffer; B: the same GEMV reading BC4 through a sampled image,
  row pitch padded off multiples of 1024 bytes [S18], at the teacher's four linear shapes. Metric:
  stage 1: student KL divergence to bf16 on a fixed text set, lower is better (also perplexity);
  stage 2: GEMV time per call, lower is better (also achieved GB/s and agreement of decoded values
  with the numpy decoder). Expected: stage 1, B's KL within 1.5x of A's, unknown because no source
  measures BC4 on LLM weights; stage 2, B within 10% of A, so at most 0 to +5% teacher decode if ever
  ported; delta 0%; confidence 0.15. Minimum: stage 1, 3 text sets per arm (deterministic); stage 2,
  5 runs per arm per shape. Predicted effect from the chip sheet: stage 2's ceiling is the lower of
  R33's read bandwidth / 4 bits per weight and the texture unit's BC4 texel rate. The texel rate is
  TBD (not in R32's note): a 12 tok/s decode of 27 billion weights needs about 3.2e11 weights per second, and if
  the texel rate falls below that, B loses. Cost: stage 1, 1 day on a CPU (no slot); stage 2, 2 days
  (shader, image upload, harness) and one 15-minute slot. Kill criterion: stage 1, B's KL above 1.5x
  A's: stop, no GPU time spent; stage 2, `vulkaninfo` reports no sampled-image support for
  `VK_FORMAT_BC4_SNORM_BLOCK`, or the GPU's decoded values disagree with the numpy decoder beyond the
  last float bit [S134], or B is slower than A on any shape: stop. Source: section 3.2 ([S133], [S134],
  [S135], [S136], [S137]). Node: CPU for stage 1; pod-b for stage 2. Note: if stage 1 lands between 1.5x
  and 3x, Apple's recovery adapters [S135] (train a LoRA on the student against the frozen BC4 base)
  become a learning-track follow-up, not part of this row. Status: proposed in this note. Metric
  now: none.

## Open questions

- Does the pod image (SJC Strix Halo, ROCm 7.13) ship a Vulkan loader and Mesa RADV, and can a
  non-root process open the render node (`/dev/dri/renderD*`)? A question for the chip sheet (R33);
  G1 and G2 depend on it.
- What are gfx1151's texture-unit rates for BC4 point sampling and for gathers, and its texture
  count per CU? R32's note does not give them; needed for G3 stage 2's ceiling (R33).
- How precise are AMD's linear-filtering interpolation weights? No AMD source we opened states it.
- Does HIP really refuse a BC4 resource view at `hipCreateTextureObject`? A one-call probe on a pod
  settles what [S137] shows in source.
- What are the pod's carve-out VRAM and GTT totals (`rocm-smi`), and what does
  `torch.cuda.mem_get_info()` report against physical memory [S157]?
- Do ggml's Vulkan training operators cover the Gated DeltaNet layers of Qwen3.5-4B (forward and
  backward)? Not checked; it decides whether a Vulkan LoRA test on the student is even buildable.
- Does rocJPEG (VCN JPEG decode) support gfx1151? Its hardware page returned 404; low stakes,
  because the pods receive only blurred keyframes and text.
- What does R47's decode profile show as the split between weight streaming, dequant ALU and launch
  overhead for the teacher on gfx1151? If launch overhead dominates, descriptor heaps and fewer
  dispatches matter more than any format (section 4).

## Search log

- 2026-10-10, budget 3 h (kernels track), used about 35 min of wall-clock time (Sat 6:47 to 7:20 PM ET), no GPU. Order: papers and vendor
  documentation first (Brook 2004, the 2005 GPGPU survey, Apple's 2025 report, ROCm, HIP and Mesa
  documentation, Microsoft Learn, Khronos), then engine source and pull requests (llama.cpp Vulkan
  and OpenCL at 23b0202, vLLM PRs #40977, #52619 and #56301, Mesa RADV at bffd919, ROCm/rocm-systems
  at 71a246a), then community benchmarks and forums last. Queries (WebSearch, extended mode where
  marked in the session): Brook for GPUs; GPGPU survey; texture units for LLM dequantization 2026;
  BC4 and BC7 for neural network weights; Strix Halo llama.cpp Vulkan RADV against ROCm 2026; llama.cpp
  cooperative matrix PRs; AMDVLK discontinued; Vulkan bfloat16 and float8; RADV bfloat16 on GFX11;
  DirectML status; WebGPU LLM gap 2025 and 2026; Adreno `image1d_buffer_t`; Apple Metal texture
  weights (which led to the ASTC section of Apple's report); ROCm Strix Halo GTT; GTT against
  carve-out benchmarks; ray-tracing hardware for neighbour search 2025 and 2026; texture-sampler
  interpolation for activations; rocDecode and rocJPEG on gfx1151; Vulkan LoRA training; mobile
  GPU texture layouts; Slang autodiff; descriptor heap. Nothing found in 2025 or 2026 that uses AMD
  texture units, the rasterizer, render targets or blending for LLM inference or training. The
  search stopped when the last three opened sources (the Strix Halo GTT write-ups, the descriptor
  heap posts, the RT-core paper) repeated what was already indexed, well inside the budget.
  Searched but set aside: the Wikipedia page on AMD's Close to Metal (opened, not cited: secondary);
  the 2026 WebGPU dispatch-overhead paper (abstract only); the 2026 "Silicon Showdown" paper
  (abstract has no AMD); the community vLLM toolbox for Strix Halo (no dense-model numbers for the
  current image); Steinkraus, Buck and Simard 2005 on GPUs for machine learning (not opened). The
  Mesa merge request for ASTC emulation and the Phoronix articles were blocked by bot checks, so
  emulation is cited from RADV's source instead.
