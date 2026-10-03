// Added by Wavefill (2026). See docs/wavefill/technical-report.zh-CN.md §1.2 / §3.3.
// Can kernel-to-kernel overlap (PDL) + next-weight L2 prefetch close the gap between a chain of weight-streaming
// kernels and the DRAM ceiling?  A synthetic "decode token": for each of L layers a list of big streaming kernels
// (sizes mimic Qwen3.8-27B decode: 84 / 31.5 / 100 / 50 MB) each followed by S small latency-bound kernels
// (pointer-chase ~2.5 us) that produce the input of the next big kernel.  Captured into a CUDA graph.
//   plain  : ordinary launches
//   pdl    : programmatic stream serialization, producers trigger at start, consumers wait at start
//   pdl+pf : as pdl, but every big kernel first issues cp.async.bulk.prefetch.L2 for the head of ITS weights
//            (weights do not depend on the previous kernel) and only then waits for the dependency
//   pdl-late     : big kernels trigger their dependents AFTER their streaming loop, small kernels trigger at start
//   pdl-late+pf  : pdl-late plus an evict_last L2 prefetch of the head of the next big kernel's weights, issued in
//                  its prologue, i.e. while the small kernel runs and DRAM would otherwise be idle
//   pdl-late+ld4 / +ld8 : pdl-late, but the prologue issues the FIRST 4 / 8 weight loads of every thread (real loads into
//                  registers, the way a PDL GEMV would) before waiting for the dependency
// Usage: ./pdl_chain [--layers 64] [--small 1] [--chase 24] [--prefetch-kib 64]
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)

__device__ __forceinline__ uint4 ldnc(const uint4* p) {
    uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}

// MODE 0: plain, 1: pdl (early trigger), 2: pdl early trigger + prefetch, 3: pdl late trigger, 4: pdl late trigger + evict_last prefetch
template <int MODE>
__global__ void k_big(const uint4* __restrict__ w, size_t n4, const float* s_in, float* s_out, unsigned prefetch_bytes) {
    if (MODE == 1 || MODE == 2) cudaTriggerProgrammaticLaunchCompletion();
    if ((MODE == 2 || MODE == 4) && threadIdx.x == 0) {          // one bulk prefetch per CTA of ITS first bytes of the (grid-stride) stream
        size_t per = (size_t)prefetch_bytes;
        const char* base = reinterpret_cast<const char*>(w) + (size_t)blockIdx.x * per;
        if (MODE == 4) {
            unsigned long long pol;
            asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
            asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" :: "l"(base), "r"((unsigned)per), "l"(pol) : "memory");
        } else {
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(base), "r"((unsigned)per) : "memory");
        }
    }
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x, stride = (size_t)gridDim.x * blockDim.x;
    constexpr int PRE = MODE == 5 ? 4 : (MODE == 6 ? 8 : 0);
    uint4 pv[PRE > 0 ? PRE : 1];
    bool have = false;
    if (PRE > 0 && tid + (PRE - 1) * stride < n4) {              // weights do not depend on the previous kernel: load them before waiting
#pragma unroll
        for (int u = 0; u < PRE; ++u) pv[u] = ldnc(w + tid + u * stride);
        have = true;
    }
    if (MODE >= 1) cudaGridDependencySynchronize();
    const float a = s_in[0];                                     // the (tiny) dependent input
    uint4 s = make_uint4(0, 0, 0, 0);
    size_t i = tid;
    if (PRE > 0 && have) {
#pragma unroll
        for (int u = 0; u < PRE; ++u) { s.x += pv[u].x; s.y += pv[u].y; s.z += pv[u].z; s.w += pv[u].w; }
        i = tid + PRE * stride;
    }
    for (; i + 3 * stride < n4; i += 4 * stride) {
        uint4 v0 = ldnc(w + i), v1 = ldnc(w + i + stride), v2 = ldnc(w + i + 2 * stride), v3 = ldnc(w + i + 3 * stride);
        s.x += v0.x + v1.x + v2.x + v3.x; s.y += v0.y + v1.y + v2.y + v3.y; s.z += v0.z + v1.z + v2.z + v3.z; s.w += v0.w + v1.w + v2.w + v3.w;
    }
    for (; i < n4; i += stride) { uint4 v = ldnc(w + i); s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w; }
    if (MODE >= 3) cudaTriggerProgrammaticLaunchCompletion();   // late: the stream is (nearly) done, DRAM is about to go idle
    unsigned r = s.x ^ s.y ^ s.z ^ s.w;
    for (int o = 16; o; o >>= 1) r += __shfl_xor_sync(0xffffffffu, r, o);
    if (threadIdx.x == 0) atomicAdd(s_out, (r == 0x9E3779B9u ? 1.f : 0.f) + 0.f * a);   // negligible tiny output
}

// small latency-bound kernel: dependent pointer chase in an L2-resident table, then publish the next input
template <int MODE>
__global__ void k_small(const unsigned* __restrict__ table, int chase, const float* s_in, float* s_out) {
    if (MODE >= 1) { cudaTriggerProgrammaticLaunchCompletion(); cudaGridDependencySynchronize(); }
    unsigned p = threadIdx.x + blockIdx.x * 7;
    for (int i = 0; i < chase; ++i) p = table[p & 0xFFFF];
    if (threadIdx.x == 0 && blockIdx.x == 0) s_out[0] = s_in[0] * 0.f + (float)(p & 1);
}

static void launch_big(int mode, cudaStream_t st, int grid, const uint4* w, size_t n4, const float* si, float* so, unsigned pf) {
    cudaLaunchConfig_t cfg{}; cfg.gridDim = dim3(grid); cfg.blockDim = dim3(256); cfg.stream = st;
    cudaLaunchAttribute at{}; at.id = cudaLaunchAttributeProgrammaticStreamSerialization; at.val.programmaticStreamSerializationAllowed = 1;
    if (mode >= 1) { cfg.attrs = &at; cfg.numAttrs = 1; }
    if (mode == 0) CK(cudaLaunchKernelEx(&cfg, k_big<0>, w, n4, si, so, pf));
    else if (mode == 1) CK(cudaLaunchKernelEx(&cfg, k_big<1>, w, n4, si, so, pf));
    else if (mode == 2) CK(cudaLaunchKernelEx(&cfg, k_big<2>, w, n4, si, so, pf));
    else if (mode == 3) CK(cudaLaunchKernelEx(&cfg, k_big<3>, w, n4, si, so, pf));
    else if (mode == 4) CK(cudaLaunchKernelEx(&cfg, k_big<4>, w, n4, si, so, pf));
    else if (mode == 5) CK(cudaLaunchKernelEx(&cfg, k_big<5>, w, n4, si, so, pf));
    else CK(cudaLaunchKernelEx(&cfg, k_big<6>, w, n4, si, so, pf));
}
static void launch_small(int mode, cudaStream_t st, const unsigned* t, int chase, const float* si, float* so) {
    cudaLaunchConfig_t cfg{}; cfg.gridDim = dim3(8); cfg.blockDim = dim3(128); cfg.stream = st;
    cudaLaunchAttribute at{}; at.id = cudaLaunchAttributeProgrammaticStreamSerialization; at.val.programmaticStreamSerializationAllowed = 1;
    if (mode >= 1) { cfg.attrs = &at; cfg.numAttrs = 1; }
    if (mode == 0) CK(cudaLaunchKernelEx(&cfg, k_small<0>, t, chase, si, so)); else CK(cudaLaunchKernelEx(&cfg, k_small<1>, t, chase, si, so));
}

int main(int argc, char** argv) {
    int layers = 64, small = 1, chase = 24, pf_kib = 64;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--layers")) layers = std::atoi(argv[++i]); else if (!std::strcmp(argv[i], "--small")) small = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--chase")) chase = std::atoi(argv[++i]); else if (!std::strcmp(argv[i], "--prefetch-kib")) pf_kib = std::atoi(argv[++i]);
    }
    const size_t MB = 1 << 20;
    const size_t sizes[4] = {84 * MB, 32 * MB, 100 * MB, 50 * MB};          // GDN-in, out-proj, gate/up, down (rounded)
    size_t total = 0; for (int l = 0; l < layers; ++l) for (int k = 0; k < 4; ++k) total += sizes[k];
    char* buf; CK(cudaMalloc(&buf, total)); CK(cudaMemset(buf, 1, total));
    unsigned* table; CK(cudaMalloc(&table, 65536 * 4)); { std::vector<unsigned> h(65536); for (int i = 0; i < 65536; ++i) h[i] = (i * 2654435761u + 12345u) & 0xFFFF; CK(cudaMemcpy(table, h.data(), 65536 * 4, cudaMemcpyHostToDevice)); }
    float* s; CK(cudaMalloc(&s, 64)); CK(cudaMemset(s, 0, 64));
    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    const int grid = 340;                                                    // 2 CTAs/SM like NInfer's GEMV grids (320)
    const double ideal_us = total / 1.700e6;                                 // at the measured 1.70 TB/s read ceiling
    std::printf("# %d layers x 4 big kernels (%.2f GB), %d small kernel(s) after each big one (chase=%d), ideal at 1.70 TB/s = %.0f us\n",
                layers, total / 1e9, small, chase, ideal_us);
    std::printf("%-12s %10s %10s %12s %13s\n", "mode", "token_us", "GB/s", "vs_ideal", "extra_us/big");
    double plain_us = 0;
    for (int mode = 0; mode < 7; ++mode) {
        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
        size_t off = 0; int nb = 0;
        for (int l = 0; l < layers; ++l) for (int k = 0; k < 4; ++k) {
            launch_big(mode, st, grid, reinterpret_cast<const uint4*>(buf + off), sizes[k] / 16, s, s + 16, (unsigned)pf_kib * 1024); ++nb;
            off += sizes[k];
            for (int q = 0; q < small; ++q) launch_small(mode, st, table, chase, s + 16, s);
        }
        CK(cudaStreamEndCapture(st, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
        for (int w = 0; w < 3; ++w) CK(cudaGraphLaunch(ge, st));
        CK(cudaStreamSynchronize(st));
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        std::vector<float> ms;
        for (int it = 0; it < 15; ++it) { CK(cudaEventRecord(a, st)); CK(cudaGraphLaunch(ge, st)); CK(cudaEventRecord(b, st)); CK(cudaEventSynchronize(b)); float m; CK(cudaEventElapsedTime(&m, a, b)); ms.push_back(m); }
        std::sort(ms.begin(), ms.end());
        double us = ms[ms.size() / 2] * 1e3;
        if (mode == 0) plain_us = us;
        const char* nm[] = {"plain", "pdl", "pdl+pf", "pdl-late", "pdl-late+pf", "pdl-late+ld4", "pdl-late+ld8"};
        std::printf("%-12s %10.0f %10.1f %11.1f%% %13.2f\n", nm[mode], us, total / (us * 1e-6) / 1e9, 100.0 * (us - ideal_us) / ideal_us, (us - ideal_us) / nb);
        if (mode > 0) std::printf("             -> %+.1f%% vs plain\n", 100.0 * (plain_us - us) / us);
    }
    return 0;
}
