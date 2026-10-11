// Chip sheet probes for one AMD GPU: device properties, the wavefront size the compiler chose, the largest LDS
// allocation a workgroup can launch with, and the kernels of the occupancy curve. Every answer is one JSON object
// on one line of stdout; chipsheet_format.parse_probe_lines reads them back.
//
//   probe props                               hipDeviceProp_t fields
//   probe wave                                __builtin_amdgcn_wavefrontsize() and the lanes seen in one wave
//   probe lds                                 largest dynamic LDS bytes a 256-thread workgroup launches with
//   probe occ_lds  <lds_bytes> <reps> <min_ms>  pointer chase with that much LDS per workgroup
//   probe occ_vgpr <live_regs> <reps> <min_ms>  pointer chase with that many float registers kept live
//
// Build: hipcc -O3 --offload-arch=gfx1151 [-mwavefrontsize64] -Rpass-analysis=kernel-resource-usage probe.hip

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

#define CHECK(call)                                                                                   \
  do {                                                                                                \
    hipError_t err_ = (call);                                                                         \
    if (err_ != hipSuccess) {                                                                         \
      std::printf("{\"probe\":\"error\",\"line\":%d,\"error\":\"%s\"}\n", __LINE__, hipGetErrorString(err_)); \
      return 1;                                                                                       \
    }                                                                                                 \
  } while (0)

static const int kThreads = 256;
static const int kChaseLog2 = 26;  // 2^26 four-byte entries: 256 MiB, larger than every cache on the chip
static const int kIters = 512;

// One full-period cycle over 2^n entries (an LCG with a % 4 == 1 and odd c), so every chase step is a dependent
// load at a far-away address.
extern "C" __global__ void chase_init(unsigned* next, unsigned mask) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= mask) next[i] = (i * 1664525u + 1013904223u) & mask;
}

extern "C" __global__ void wave_probe(int* out) {
  atomicMax(&out[1], (int)__lane_id() + 1);
  if (threadIdx.x == 0) out[0] = __builtin_amdgcn_wavefrontsize();
}

extern "C" __global__ void lds_probe(unsigned* out, int bytes) {
  extern __shared__ unsigned char smem[];
  for (int k = threadIdx.x; k < bytes; k += blockDim.x) smem[k] = (unsigned char)(k * 7 + 1);
  __syncthreads();
  if (threadIdx.x == 0) out[0] = bytes > 0 ? (unsigned)smem[bytes - 1] + 1u : 1u;
}

extern "C" __global__ void occ_lds(const unsigned* __restrict__ next, unsigned* out, int iters, unsigned mask,
                                   int lds_bytes) {
  extern __shared__ unsigned smem_words[];
  unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned i = (gid * 2654435761u) & mask;
  int words = lds_bytes / 4;
  if (words > 0) smem_words[threadIdx.x % words] = i;
  __syncthreads();
  for (int k = 0; k < iters; ++k) i = next[i];
  if (words > 0) i ^= smem_words[(threadIdx.x + 1) % words];
  out[gid] = i;
}

// R floats stay live across the chase loop: the empty asm pins each one in a VGPR without adding instructions.
#define OCC_VGPR(R)                                                                                   \
  extern "C" __global__ void occ_vgpr_##R(const unsigned* __restrict__ next, unsigned* out, int iters, \
                                          unsigned mask) {                                            \
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;                                             \
    unsigned i = (gid * 2654435761u) & mask;                                                          \
    float acc[R];                                                                                     \
    _Pragma("unroll") for (int r = 0; r < R; ++r) acc[r] = (float)(gid + r);                          \
    for (int k = 0; k < iters; ++k) {                                                                 \
      i = next[i];                                                                                    \
      _Pragma("unroll") for (int r = 0; r < R; ++r) asm volatile("" : "+v"(acc[r]));                 \
    }                                                                                                 \
    float s = 0.f;                                                                                    \
    _Pragma("unroll") for (int r = 0; r < R; ++r) s += acc[r];                                        \
    out[gid] = i ^ __float_as_uint(s);                                                                \
  }

OCC_VGPR(4)
OCC_VGPR(16)
OCC_VGPR(32)
OCC_VGPR(64)
OCC_VGPR(96)
OCC_VGPR(128)
OCC_VGPR(192)
OCC_VGPR(240)

typedef void (*VgprKernel)(const unsigned*, unsigned*, int, unsigned);

static VgprKernel vgpr_kernel(int r, const char** name) {
  switch (r) {
#define CASE(R) case R: *name = "occ_vgpr_" #R; return occ_vgpr_##R;
    CASE(4) CASE(16) CASE(32) CASE(64) CASE(96) CASE(128) CASE(192) CASE(240)
#undef CASE
    default: *name = ""; return nullptr;
  }
}

static double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int props() {
  hipDeviceProp_t p;
  CHECK(hipGetDeviceProperties(&p, 0));
  int clock_khz = 0, mem_clock_khz = 0;
  (void)hipDeviceGetAttribute(&clock_khz, hipDeviceAttributeClockRate, 0);
  (void)hipDeviceGetAttribute(&mem_clock_khz, hipDeviceAttributeMemoryClockRate, 0);
  std::printf("{\"probe\":\"props\",\"name\":\"%s\",\"gcnArchName\":\"%s\",\"totalGlobalMem\":%zu,"
              "\"sharedMemPerBlock\":%zu,\"maxSharedMemoryPerMultiProcessor\":%zu,\"regsPerBlock\":%d,"
              "\"warpSize\":%d,\"maxThreadsPerBlock\":%d,\"maxThreadsPerMultiProcessor\":%d,"
              "\"multiProcessorCount\":%d,\"l2CacheSize\":%d,\"clockRate_khz\":%d,\"memoryClockRate_khz\":%d,"
              "\"memoryBusWidth\":%d}\n",
              p.name, p.gcnArchName, (size_t)p.totalGlobalMem, (size_t)p.sharedMemPerBlock,
              (size_t)p.maxSharedMemoryPerMultiProcessor, p.regsPerBlock, p.warpSize, p.maxThreadsPerBlock,
              p.maxThreadsPerMultiProcessor, p.multiProcessorCount, p.l2CacheSize, clock_khz, mem_clock_khz,
              p.memoryBusWidth);
  return 0;
}

static int wave() {
  int* d = nullptr;
  int h[2] = {0, 0};
  CHECK(hipMalloc(&d, sizeof(h)));
  CHECK(hipMemset(d, 0, sizeof(h)));
  hipLaunchKernelGGL(wave_probe, dim3(1), dim3(kThreads), 0, 0, d);
  CHECK(hipGetLastError());
  CHECK(hipMemcpy(h, d, sizeof(h), hipMemcpyDeviceToHost));
  int host_warp = 0;
  CHECK(hipDeviceGetAttribute(&host_warp, hipDeviceAttributeWarpSize, 0));
  std::printf("{\"probe\":\"wave\",\"compiled_wave\":%d,\"lanes\":%d,\"device_warp_size\":%d}\n", h[0], h[1],
              host_warp);
  CHECK(hipFree(d));
  return 0;
}

// 0 when a 256-thread workgroup with `bytes` of dynamic LDS launched and ran; otherwise the error.
static hipError_t try_lds(unsigned* d, int bytes) {
  (void)hipFuncSetAttribute((const void*)lds_probe, hipFuncAttributeMaxDynamicSharedMemorySize, bytes);
  (void)hipGetLastError();
  hipError_t err = hipMemset(d, 0, sizeof(unsigned));
  if (err != hipSuccess) return err;
  hipLaunchKernelGGL(lds_probe, dim3(1), dim3(kThreads), (size_t)bytes, 0, d, bytes);
  err = hipGetLastError();
  if (err != hipSuccess) return err;
  err = hipDeviceSynchronize();
  if (err != hipSuccess) return err;
  unsigned h = 0;
  err = hipMemcpy(&h, d, sizeof(h), hipMemcpyDeviceToHost);
  if (err != hipSuccess) return err;
  unsigned want = bytes > 0 ? (unsigned)(unsigned char)((bytes - 1) * 7 + 1) + 1u : 1u;
  return h == want ? hipSuccess : hipErrorLaunchFailure;
}

static int lds() {
  unsigned* d = nullptr;
  CHECK(hipMalloc(&d, sizeof(unsigned)));
  int ok = 0, fail = -1, tries = 0;
  hipError_t fail_err = hipSuccess;
  for (int bytes = 1024; bytes <= (1 << 20); bytes *= 2) {
    ++tries;
    hipError_t err = try_lds(d, bytes);
    if (err == hipSuccess) {
      ok = bytes;
    } else {
      fail = bytes;
      fail_err = err;
      break;
    }
  }
  while (fail > 0 && fail - ok > 128) {
    int mid = ok + (fail - ok) / 2;
    mid -= mid % 128;
    if (mid <= ok) break;
    ++tries;
    hipError_t err = try_lds(d, mid);
    if (err == hipSuccess) {
      ok = mid;
    } else {
      fail = mid;
      fail_err = err;
    }
  }
  int attr = 0;
  (void)hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxSharedMemoryPerBlock, 0);
  std::printf("{\"probe\":\"lds\",\"max_ok_bytes\":%d,\"first_fail_bytes\":%d,\"fail_error\":\"%s\","
              "\"attribute_max_shared_per_block\":%d,\"tries\":%d,\"threads\":%d}\n",
              ok, fail, fail > 0 ? hipGetErrorString(fail_err) : "", attr, tries, kThreads);
  CHECK(hipFree(d));
  return 0;
}

// Time one occupancy kernel: `reps` timed launches, then more untimed ones until `min_ms` has passed, so the
// power reader sees the load.
template <typename Launch>
static int occupancy(const char* kind, const char* kernel, int param, int blocks_per_cu, int reps, double min_ms,
                     Launch launch) {
  hipDeviceProp_t p;
  CHECK(hipGetDeviceProperties(&p, 0));
  int blocks = p.multiProcessorCount * 32;
  launch(blocks);  // warm-up
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  hipEvent_t a, b;
  CHECK(hipEventCreate(&a));
  CHECK(hipEventCreate(&b));
  double start = now_ms();
  std::printf("{\"probe\":\"%s\",\"kernel\":\"%s\",\"param\":%d,\"blocks\":%d,\"threads\":%d,\"iters\":%d,"
              "\"blocks_per_cu\":%d,\"ms\":[",
              kind, kernel, param, blocks, kThreads, kIters, blocks_per_cu);
  for (int r = 0; r < reps; ++r) {
    CHECK(hipEventRecord(a, 0));
    launch(blocks);
    CHECK(hipEventRecord(b, 0));
    CHECK(hipEventSynchronize(b));
    float ms = 0.f;
    CHECK(hipEventElapsedTime(&ms, a, b));
    std::printf("%s%.6f", r ? "," : "", ms);
  }
  while (now_ms() - start < min_ms) {
    launch(blocks);
    CHECK(hipDeviceSynchronize());
  }
  std::printf("]}\n");
  CHECK(hipEventDestroy(a));
  CHECK(hipEventDestroy(b));
  return 0;
}

static int occ(const char* kind, int param, int reps, double min_ms) {
  unsigned mask = (1u << kChaseLog2) - 1u;
  unsigned *next = nullptr, *out = nullptr;
  hipDeviceProp_t p;
  CHECK(hipGetDeviceProperties(&p, 0));
  size_t out_n = (size_t)p.multiProcessorCount * 32 * kThreads;
  CHECK(hipMalloc(&next, sizeof(unsigned) * ((size_t)mask + 1)));
  CHECK(hipMalloc(&out, sizeof(unsigned) * out_n));
  hipLaunchKernelGGL(chase_init, dim3((mask + 1) / kThreads), dim3(kThreads), 0, 0, next, mask);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  int rc = 0;
  int per_cu = 0;
  if (std::strcmp(kind, "occ_lds") == 0) {
    int bytes = param;
    (void)hipFuncSetAttribute((const void*)occ_lds, hipFuncAttributeMaxDynamicSharedMemorySize, bytes);
    (void)hipGetLastError();
    CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, (const void*)occ_lds, kThreads, (size_t)bytes));
    rc = occupancy(kind, "occ_lds", bytes, per_cu, reps, min_ms, [&](int blocks) {
      hipLaunchKernelGGL(occ_lds, dim3(blocks), dim3(kThreads), (size_t)bytes, 0, next, out, kIters, mask, bytes);
    });
  } else {
    const char* name = "";
    VgprKernel k = vgpr_kernel(param, &name);
    if (k == nullptr) {
      std::printf("{\"probe\":\"error\",\"error\":\"no occ_vgpr kernel keeps %d registers live\"}\n", param);
      return 2;
    }
    CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, (const void*)k, kThreads, 0));
    rc = occupancy(kind, name, param, per_cu, reps, min_ms, [&](int blocks) {
      hipLaunchKernelGGL(k, dim3(blocks), dim3(kThreads), 0, 0, next, out, kIters, mask);
    });
  }
  CHECK(hipFree(next));
  CHECK(hipFree(out));
  return rc;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: probe props|wave|lds|occ_lds BYTES REPS MIN_MS|occ_vgpr REGS REPS MIN_MS\n");
    return 2;
  }
  const char* mode = argv[1];
  if (std::strcmp(mode, "props") == 0) return props();
  if (std::strcmp(mode, "wave") == 0) return wave();
  if (std::strcmp(mode, "lds") == 0) return lds();
  if ((std::strcmp(mode, "occ_lds") == 0 || std::strcmp(mode, "occ_vgpr") == 0) && argc >= 5) {
    return occ(mode, std::atoi(argv[2]), std::atoi(argv[3]), std::atof(argv[4]));
  }
  std::fprintf(stderr, "unknown mode %s\n", mode);
  return 2;
}
