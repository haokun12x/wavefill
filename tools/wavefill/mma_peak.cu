// Tensor-core throughput of the mma.sync shapes the causal prompt attention kernels can use, on
// this GPU. Every warp runs kChains independent accumulator chains so the pipe, not latency, is the
// limit. Prints dense TFLOP/s (2*M*N*K per mma).
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>

constexpr int kChains = 8;
constexpr int kIters  = 4096;

#define MMA_KERNEL(NAME, M, N, K, ACC_T, NACC, ASM)                                                \
    __global__ void NAME(ACC_T* out, unsigned seed) {                                              \
        unsigned a0 = seed ^ threadIdx.x, a1 = a0 * 3u, a2 = a0 * 5u, a3 = a0 * 7u;                \
        unsigned b0 = a0 * 11u, b1 = a0 * 13u;                                                     \
        ACC_T c[kChains][NACC];                                                                    \
        _Pragma("unroll") for (int i = 0; i < kChains; ++i) {                                      \
            _Pragma("unroll") for (int j = 0; j < NACC; ++j) c[i][j] = 0;                          \
        }                                                                                          \
        for (int it = 0; it < kIters; ++it) {                                                      \
            _Pragma("unroll") for (int i = 0; i < kChains; ++i) { ASM; }                           \
        }                                                                                          \
        ACC_T s = 0;                                                                               \
        _Pragma("unroll") for (int i = 0; i < kChains; ++i) {                                      \
            _Pragma("unroll") for (int j = 0; j < NACC; ++j) s += c[i][j];                         \
        }                                                                                          \
        out[blockIdx.x * blockDim.x + threadIdx.x] = s;                                            \
    }

// bf16 x bf16 -> f32, m16n8k16
MMA_KERNEL(k_bf16_f32, 16, 8, 16, float, 4,
           asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, "
                        "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)))
// f16 x f16 -> f32
MMA_KERNEL(k_f16_f32, 16, 8, 16, float, 4,
           asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
                        "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)))
// f16 x f16 -> f16 (accumulators are 2 x f16x2 registers)
MMA_KERNEL(k_f16_f16, 16, 8, 16, unsigned, 2,
           asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, "
                        "{%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                        : "+r"(c[i][0]), "+r"(c[i][1])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)))
// s8 x s8 -> s32, m16n8k32
MMA_KERNEL(k_s8_s32, 16, 8, 32, int, 4,
           asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, "
                        "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(c[i][0]), "+r"(c[i][1]), "+r"(c[i][2]), "+r"(c[i][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)))
// e4m3 x e4m3 -> f32, m16n8k32
MMA_KERNEL(k_e4m3_f32, 16, 8, 32, float, 4,
           asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, "
                        "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)))
// e4m3 x e4m3 -> f16, m16n8k32
MMA_KERNEL(k_e4m3_f16, 16, 8, 32, unsigned, 2,
           asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, "
                        "{%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                        : "+r"(c[i][0]), "+r"(c[i][1])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)))

template <typename T>
double run(void (*kernel)(T*, unsigned), double flops_per_mma, int blocks, int threads,
           T* out) {
    kernel<<<blocks, threads>>>(out, 1u);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float best = 1e30f;
    for (int rep = 0; rep < 5; ++rep) {
        cudaEventRecord(start);
        kernel<<<blocks, threads>>>(out, rep + 2u);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        best = ms < best ? ms : best;
    }
    const double mmas = double(blocks) * (threads / 32) * kIters * kChains;
    return mmas * flops_per_mma / (best * 1e-3) / 1e12;
}

int main() {
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    const int blocks = sms * 4, threads = 256;
    void* out = nullptr;
    cudaMalloc(&out, size_t(blocks) * threads * 8);
    std::printf("SMs=%d blocks=%d threads=%d chains=%d\n", sms, blocks, threads, kChains);
    std::printf("bf16 m16n8k16 f32acc : %7.1f TFLOP/s\n",
                run(k_bf16_f32, 2.0 * 16 * 8 * 16, blocks, threads, (float*)out));
    std::printf("f16  m16n8k16 f32acc : %7.1f TFLOP/s\n",
                run(k_f16_f32, 2.0 * 16 * 8 * 16, blocks, threads, (float*)out));
    std::printf("f16  m16n8k16 f16acc : %7.1f TFLOP/s\n",
                run(k_f16_f16, 2.0 * 16 * 8 * 16, blocks, threads, (unsigned*)out));
    std::printf("s8   m16n8k32 s32acc : %7.1f TOP/s\n",
                run(k_s8_s32, 2.0 * 16 * 8 * 32, blocks, threads, (int*)out));
    std::printf("e4m3 m16n8k32 f32acc : %7.1f TFLOP/s\n",
                run(k_e4m3_f32, 2.0 * 16 * 8 * 32, blocks, threads, (float*)out));
    std::printf("e4m3 m16n8k32 f16acc : %7.1f TFLOP/s\n",
                run(k_e4m3_f16, 2.0 * 16 * 8 * 32, blocks, threads, (unsigned*)out));
    const cudaError_t err = cudaDeviceSynchronize();
    std::printf("status: %s\n", cudaGetErrorString(err));
    return err == cudaSuccess ? 0 : 1;
}
