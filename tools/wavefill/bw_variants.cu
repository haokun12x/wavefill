// Added by Wavefill (2026). See docs/wavefill/technical-report.zh-CN.md §1.2 / §3.3.
// How close to the DRAM pins can a read stream get on this RTX 5090? Variants of a pure streaming read
// (values are discarded / checksummed), 4 GiB working set (42x L2), best and median of N timed launches.
//   ld      : ld.global.nc.v4, grid-stride, unroll U, CTAs = ctas_per_sm * 170            (the repo probe's pattern)
//   slice   : every CTA streams its OWN contiguous slice (GEMV-like row split)
//   hint    : slice + ld.global.nc.L1::no_allocate.L2::256B
//   tma     : cp.async.bulk 1D global->smem, S stages x B bytes per CTA, mbarrier pipeline, data discarded
// Usage: ./bw_variants [--gib 4] [--iters 20]
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)

__device__ __forceinline__ uint4 ldnc(const uint4* p) {
    uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ uint4 ldnc_hint(const uint4* p) {
    uint4 v; asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}

template <int U>
__global__ void k_ld(const uint4* __restrict__ src, size_t n, uint4* out) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x, stride = (size_t)gridDim.x * blockDim.x;
    uint4 s = make_uint4(0, 0, 0, 0);
    size_t i = tid;
    for (; i + (U - 1) * stride < n; i += U * stride) {
        uint4 v[U];
#pragma unroll
        for (int u = 0; u < U; ++u) v[u] = ldnc(src + i + u * stride);
#pragma unroll
        for (int u = 0; u < U; ++u) { s.x += v[u].x; s.y += v[u].y; s.z += v[u].z; s.w += v[u].w; }
    }
    for (; i < n; i += stride) { uint4 v = ldnc(src + i); s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w; }
    if ((s.x ^ s.y ^ s.z ^ s.w) == 0x9E3779B9u) out[blockIdx.x] = s;   // keeps the loads alive, essentially never stores
}

template <int U, bool HINT>
__global__ void k_slice(const uint4* __restrict__ src, size_t n, uint4* out) {
    size_t per = (n + gridDim.x - 1) / gridDim.x, begin = per * blockIdx.x, end = min(begin + per, n);
    uint4 s = make_uint4(0, 0, 0, 0);
    size_t i = begin + threadIdx.x;
    for (; i + (U - 1) * (size_t)blockDim.x < end; i += (size_t)U * blockDim.x) {
        uint4 v[U];
#pragma unroll
        for (int u = 0; u < U; ++u) v[u] = HINT ? ldnc_hint(src + i + u * blockDim.x) : ldnc(src + i + u * blockDim.x);
#pragma unroll
        for (int u = 0; u < U; ++u) { s.x += v[u].x; s.y += v[u].y; s.z += v[u].z; s.w += v[u].w; }
    }
    for (; i < end; i += blockDim.x) { uint4 v = ldnc(src + i); s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w; }
    if ((s.x ^ s.y ^ s.z ^ s.w) == 0x9E3779B9u) out[blockIdx.x] = s;
}

// TMA 1D bulk pipeline: each CTA owns a contiguous slice; S stages of B bytes; thread 0 issues, everyone waits.
template <int S, int B>
__global__ void k_tma(const char* __restrict__ src, size_t bytes, uint4* out) {
    extern __shared__ __align__(128) char smem[];
    __shared__ __align__(8) unsigned long long bar[S];
    size_t per = ((bytes / gridDim.x) / B) * B, begin = per * blockIdx.x;
    if (blockIdx.x == gridDim.x - 1) per = bytes - begin;            // last CTA takes the remainder (multiple of B is not required)
    const int chunks = (int)((per + B - 1) / B);
    const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem), bb = (uint32_t)__cvta_generic_to_shared(bar);
    if (threadIdx.x == 0) {
        for (int s = 0; s < S; ++s) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bb + 8 * s));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    auto issue = [&](int c, int s) {
        uint32_t n = (uint32_t)min((size_t)B, per - (size_t)c * B);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bb + 8 * s), "r"(n) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(sb + (uint32_t)s * B), "l"(src + begin + (size_t)c * B), "r"(n), "r"(bb + 8 * s) : "memory");
    };
    if (threadIdx.x == 0) for (int s = 0; s < S && s < chunks; ++s) issue(s, s);
    unsigned acc = 0;
    for (int c = 0; c < chunks; ++c) {
        int s = c % S; unsigned par = (c / S) & 1;
        asm volatile("{\n.reg .pred p;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WAIT_%=;\n}" :: "r"(bb + 8 * s), "r"(par) : "memory");
        acc += *(volatile unsigned*)(smem + (size_t)s * B + (threadIdx.x % 32) * 4);          // touch smem so the slot is really consumed
        __syncthreads();                                                                    // everyone is done with slot s before it is refilled
        if (threadIdx.x == 0 && c + S < chunks) issue(c + S, s);
    }
    if (acc == 0x9E3779B9u) out[blockIdx.x] = make_uint4(acc, 0, 0, 0);
}

// L2 prefetch only: one thread per CTA walks its slice issuing cp.async.bulk.prefetch.L2 of B bytes
template <int B>
__global__ void k_prefetch(const char* __restrict__ src, size_t bytes) {
    size_t per = (bytes / gridDim.x / B) * B, begin = per * blockIdx.x;
    for (size_t o = threadIdx.x * (size_t)B; o < per; o += (size_t)blockDim.x * B)
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src + begin + o), "r"((uint32_t)B) : "memory");
}

struct Timer {
    cudaEvent_t a, b; Timer() { CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); }
    template <class F> std::vector<float> run(int iters, F&& f) {
        std::vector<float> ms; f(); CK(cudaDeviceSynchronize());
        for (int i = 0; i < iters; ++i) { CK(cudaEventRecord(a)); f(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); float m; CK(cudaEventElapsedTime(&m, a, b)); ms.push_back(m); }
        std::sort(ms.begin(), ms.end()); return ms;
    }
};

int main(int argc, char** argv) {
    double gib = 4; int iters = 20;
    for (int i = 1; i < argc; ++i) { if (!std::strcmp(argv[i], "--gib")) gib = std::atof(argv[++i]); else if (!std::strcmp(argv[i], "--iters")) iters = std::atoi(argv[++i]); }
    size_t bytes = (size_t)(gib * (1ull << 30)); bytes -= bytes % (1 << 20);
    char* buf; uint4* out; CK(cudaMalloc(&buf, bytes)); CK(cudaMalloc(&out, 1 << 20)); CK(cudaMemset(buf, 1, bytes));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); int sms = p.multiProcessorCount;
    std::printf("# %s, %d SMs, %.2f GiB working set; nominal 1792 GB/s; GB/s = bytes/time (read only)\n", p.name, sms, bytes / double(1 << 30));
    std::printf("%-46s %9s %9s %8s\n", "variant", "best", "median", "%of1792");
    Timer t;
    auto report = [&](const std::string& name, const std::vector<float>& ms) {
        double best = bytes / (ms.front() * 1e-3) / 1e9, med = bytes / (ms[ms.size() / 2] * 1e-3) / 1e9;
        std::printf("%-46s %9.1f %9.1f %7.1f%%\n", name.c_str(), best, med, 100 * best / 1792.0); std::fflush(stdout);
    };
    const uint4* s4 = reinterpret_cast<const uint4*>(buf); size_t n4 = bytes / 16;
    for (int cps : {1, 2, 4, 6}) for (int U : {1, 4, 8}) {
        int grid = cps * sms; auto f = [&]() { if (U == 1) k_ld<1><<<grid, 256>>>(s4, n4, out); else if (U == 4) k_ld<4><<<grid, 256>>>(s4, n4, out); else k_ld<8><<<grid, 256>>>(s4, n4, out); };
        report("ld   grid-stride  ctas=" + std::to_string(grid) + " unroll=" + std::to_string(U), t.run(iters, f));
    }
    for (int cps : {2, 4, 6}) for (int U : {4, 8}) {
        int grid = cps * sms;
        auto f = [&]() { if (U == 4) k_slice<4, false><<<grid, 256>>>(s4, n4, out); else k_slice<8, false><<<grid, 256>>>(s4, n4, out); };
        report("slice contiguous  ctas=" + std::to_string(grid) + " unroll=" + std::to_string(U), t.run(iters, f));
    }
    for (int cps : {2, 4}) { int grid = cps * sms; auto f = [&]() { k_slice<8, true><<<grid, 256>>>(s4, n4, out); };
        report("hint  slice+L1::no_allocate.L2::256B ctas=" + std::to_string(grid), t.run(iters, f)); }
    CK(cudaFuncSetAttribute(k_tma<3, 32768>, cudaFuncAttributeMaxDynamicSharedMemorySize, 3 * 32768));
    CK(cudaFuncSetAttribute(k_tma<4, 16384>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 16384));
    CK(cudaFuncSetAttribute(k_tma<6, 8192>, cudaFuncAttributeMaxDynamicSharedMemorySize, 6 * 8192));
    for (int cps : {1, 2}) {
        int g = cps * sms;
        report("tma   3 x 32 KiB  ctas=" + std::to_string(g), t.run(iters, [&]() { k_tma<3, 32768><<<g, 128, 3 * 32768>>>(buf, bytes, out); }));
        report("tma   4 x 16 KiB  ctas=" + std::to_string(g), t.run(iters, [&]() { k_tma<4, 16384><<<g, 128, 4 * 16384>>>(buf, bytes, out); }));
        report("tma   6 x  8 KiB  ctas=" + std::to_string(g), t.run(iters, [&]() { k_tma<6, 8192><<<g, 128, 6 * 8192>>>(buf, bytes, out); }));
    }
    return 0;
}
