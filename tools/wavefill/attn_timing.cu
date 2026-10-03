// Kernel-time comparison of NInfer's BF16-cache causal prompt attention at very long context:
// the HEAD kernel (one CTA per tile, FP32 PV accumulation) against the working-tree kernel
// (cluster key split, FP16 PV accumulation). K/V are generated in place in the paged layout, so
// 256k keys fit next to another process on the GPU. Times are CUDA-event medians of graph replays.
//
// usage: attn_timing [--tokens T] [--contexts C,...] [--splits S,...] [--repeat N]
#include "ops/softmax_attention/dense/causal_cache/prompt_bf16.cuh"
#include "prompt_bf16_head.cuh"
#include "prompt_bf16_wide.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;
using Geometry = CausalD256H24Kv4;
using Meta     = PagedKVDirectMetadata;

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

template <typename T>
__global__ void fill(T* p, size_t n, unsigned seed, float scale) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned v = unsigned(i) * 2654435761u + seed;
    v ^= v >> 15; v *= 0x2c1b3c6du; v ^= v >> 12;
    p[i] = T((float(v >> 8) * (2.0f / 16777216.0f) - 1.0f) * scale);
}

std::vector<int> parse(const std::string& s) {
    std::vector<int> r;
    for (size_t p = 0; p < s.size();) {
        size_t c = s.find(',', p);
        r.push_back(std::stoi(s.substr(p, c - p)));
        p = c == std::string::npos ? s.size() : c + 1;
    }
    return r;
}

int main(int argc, char** argv) {
    int tokens = 1024, repeat = 5;
    bool wide_only = false;
    std::vector<int> contexts{65536, 131072, 196608, 258048}, splits{2};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--tokens") tokens = std::stoi(argv[++i]);
        else if (a == "--contexts") contexts = parse(argv[++i]);
        else if (a == "--splits") splits = parse(argv[++i]);
        else if (a == "--repeat") repeat = std::stoi(argv[++i]);
        else if (a == "--wide") wide_only = true;
    }
    auto* ref_kernel = ref::ref_causal_attention_prompt_bf16_kernel<Geometry, Meta>;
    auto* new_kernel = causal_attention_prompt_bf16_kernel<Geometry, Meta>;
    CHECK(cudaFuncSetAttribute(ref_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               kCausalPromptSmemBytes));
    CHECK(cudaFuncSetAttribute(new_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               kCausalPromptSmemBytes));
    const int max_keys  = *std::max_element(contexts.begin(), contexts.end()) + tokens;
    const int max_pages = (max_keys + kPagedKVPageSize - 1) / kPagedKVPageSize;
    const size_t kv     = size_t(max_pages) * kPagedKVPageSize * 4 * 256;
    __nv_bfloat16 *q, *k, *out;
    __half* v;
    int *table, *pos;
    CHECK(cudaMalloc(&q, size_t(tokens) * 24 * 256 * 2));
    CHECK(cudaMalloc(&out, size_t(tokens) * 24 * 256 * 2));
    CHECK(cudaMalloc(&k, kv * 2));
    CHECK(cudaMalloc(&v, kv * 2));
    CHECK(cudaMalloc(&table, max_pages * 4));
    CHECK(cudaMalloc(&pos, tokens * 4));
    fill<<<unsigned((size_t(tokens) * 24 * 256 + 255) / 256), 256>>>(q, size_t(tokens) * 24 * 256, 1u, 1.0f);
    fill<<<unsigned((kv + 255) / 256), 256>>>(k, kv, 2u, 1.0f);
    fill<<<unsigned((kv + 255) / 256), 256>>>(v, kv, 3u, 1.0f);
    std::vector<int> ht(max_pages);
    for (int i = 0; i < max_pages; ++i) ht[i] = i;
    CHECK(cudaMemcpy(table, ht.data(), max_pages * 4, cudaMemcpyHostToDevice));
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    const Meta meta{table};
    const dim3 grid(unsigned((tokens + kCausalPromptBr - 1) / kCausalPromptBr), 24, 1);

    auto time_graph = [&](auto&& launch) {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        launch();
        CHECK(cudaStreamEndCapture(stream, &g));
        CHECK(cudaGraphInstantiate(&ge, g, 0));
        CHECK(cudaGraphLaunch(ge, stream));
        std::vector<float> ms;
        for (int r = 0; r < repeat; ++r) {
            CHECK(cudaEventRecord(e0, stream));
            CHECK(cudaGraphLaunch(ge, stream));
            CHECK(cudaEventRecord(e1, stream));
            CHECK(cudaEventSynchronize(e1));
            float t = 0;
            CHECK(cudaEventElapsedTime(&t, e0, e1));
            ms.push_back(t);
        }
        CHECK(cudaGraphExecDestroy(ge));
        CHECK(cudaGraphDestroy(g));
        std::sort(ms.begin(), ms.end());
        return ms[ms.size() / 2];
    };

    std::printf("tokens=%d (24 q heads, 4 kv heads, D=256, BF16 K / FP16 V cache)\n", tokens);
    std::printf("%8s %-10s %10s %9s %8s\n", "context", "kernel", "ms", "TFLOP/s", "speedup");
    for (int context : contexts) {
        std::vector<int> hp(tokens);
        for (int t = 0; t < tokens; ++t) hp[t] = context + t;
        CHECK(cudaMemcpy(pos, hp.data(), tokens * 4, cudaMemcpyHostToDevice));
        double flops = 0;
        for (int t = 0; t < tokens; ++t) flops += 4.0 * 256 * 24 * (context + t + 1);
        if (wide_only) {
            const float base = time_graph([&] {
                ref_kernel<<<grid, kCausalPromptThreads, kCausalPromptSmemBytes, stream>>>(
                    q, k, v, meta, pos, 0.0625f, out, tokens);
            });
            std::printf("%8d %-14s %10.3f %9.1f %8s\n", context, "head 64x64", base, flops / base / 1e9,
                        "1.00x");
            auto run_wide = [&](auto kernel, int br, int bc) {
                const int smem = (br + 2 * bc) * 256 * 2;
                CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
                const dim3 g(unsigned((tokens + br - 1) / br), 24, 1);
                const float t = time_graph([&] {
                    kernel<<<g, br * 2, smem, stream>>>(q, k, v, meta, pos, 0.0625f, out, tokens);
                });
                char name[32];
                std::snprintf(name, sizeof name, "wide %dx%d", br, bc);
                std::printf("%8d %-14s %10.3f %9.1f %7.2fx\n", context, name, t, flops / t / 1e9,
                            base / t);
            };
            run_wide(wide::wide_prompt_bf16_kernel<Geometry, Meta, 64, 64>, 64, 64);
            run_wide(wide::wide_prompt_bf16_kernel<Geometry, Meta, 64, 32>, 64, 32);
            run_wide(wide::wide_prompt_bf16_kernel<Geometry, Meta, 96, 32>, 96, 32);
            run_wide(wide::wide_prompt_bf16_kernel<Geometry, Meta, 128, 32>, 128, 32);
            continue;
        }
        const float base = time_graph([&] {
            ref_kernel<<<grid, kCausalPromptThreads, kCausalPromptSmemBytes, stream>>>(
                q, k, v, meta, pos, 0.0625f, out, tokens);
        });
        std::printf("%8d %-10s %10.3f %9.1f %8s\n", context, "head", base, flops / base / 1e9, "1.00x");
        for (int s : splits) {
            const float t = time_graph([&] {
                cudaLaunchAttribute attr{};
                attr.id               = cudaLaunchAttributeClusterDimension;
                attr.val.clusterDim.x = unsigned(s);
                attr.val.clusterDim.y = 1;
                attr.val.clusterDim.z = 1;
                cudaLaunchConfig_t cfg{};
                cfg.gridDim          = dim3(grid.x * unsigned(s), grid.y, 1);
                cfg.blockDim         = dim3(kCausalPromptThreads);
                cfg.dynamicSmemBytes = kCausalPromptSmemBytes;
                cfg.stream           = stream;
                cfg.attrs            = &attr;
                cfg.numAttrs         = 1;
                CHECK(cudaLaunchKernelEx(&cfg, new_kernel, (const __nv_bfloat16*)q,
                                         (const __nv_bfloat16*)k, (const __half*)v, meta,
                                         (const int*)pos, 0.0625f, out, tokens));
            });
            char name[16];
            std::snprintf(name, sizeof name, "new S=%d", s);
            std::printf("%8d %-10s %10.3f %9.1f %7.2fx\n", context, name, t, flops / t / 1e9, base / t);
        }
    }
    return 0;
}
