// Numerical stress test of NInfer's BF16-cache causal prompt attention: the HEAD kernel (FP32 PV
// accumulation, one CTA per tile) against the working-tree kernel (cluster key split, FP16 PV
// accumulation per 64-key tile), both against an FP64 oracle on sampled rows, on activation
// regimes the op test does not reach: long context, peaked scores, an attention sink, and V
// outliers.
//
// usage: attn_numerics [--context C] [--tokens T] [--splits S,...] [--rows N] [--regime name|all]
#include "ops/softmax_attention/dense/causal_cache/prompt_bf16.cuh"
#include "prompt_bf16_head.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;
using Geometry = CausalD256H24Kv4;

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

constexpr int D = 256, HQ = 24, HKV = 4;

__global__ void scatter_kv(const __nv_bfloat16* k_logical, const __half* v_logical,
                           __nv_bfloat16* k_cache, __half* v_cache, int keys) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::int64_t(keys) * HKV * D) return;
    const int d   = int(i % D);
    const int h   = int((i / D) % HKV);
    const int key = int(i / (D * HKV));
    const std::int64_t off =
        paged_kv_element_offset<D, HKV>(key >> kPagedKVPageShift, h, key & kPagedKVPageMask, d);
    k_cache[off] = k_logical[i];
    v_cache[off] = v_logical[i];
}

struct Data {
    std::vector<float> q, k, v; // rounded values: q/k to bf16, v to bf16 then fp16
};

float bf16_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
float fp16_round(float x) { return __half2float(__float2half_rn(x)); }

Data make(const std::string& regime, int keys, int tokens, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> n01(0.0f, 1.0f);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    Data d;
    d.q.resize(size_t(tokens) * HQ * D);
    d.k.resize(size_t(keys) * HKV * D);
    d.v.resize(size_t(keys) * HKV * D);
    const bool peaked = regime != "uniform";
    const float qs    = regime == "uniform" ? 0.25f : 3.0f;
    for (auto& x : d.q) x = peaked ? qs * n01(rng) : qs * u(rng);
    for (auto& x : d.k) x = peaked ? n01(rng) : 0.25f * u(rng);
    for (auto& x : d.v) x = peaked ? n01(rng) : u(rng);
    if (regime == "sink" || regime == "sink_outlier") {
        // Key 0 of every KV head aligns with a direction every query carries: score ~ 12-15 while
        // the rest are N(0, 9) -> most probabilities are tiny next to the sink.
        std::vector<float> dir(D);
        float norm = 0.0f;
        for (auto& x : dir) { x = n01(rng); norm += x * x; }
        norm = std::sqrt(norm);
        for (auto& x : dir) x /= norm;
        for (int t = 0; t < tokens; ++t)
            for (int h = 0; h < HQ; ++h)
                for (int i = 0; i < D; ++i) d.q[(size_t(t) * HQ + h) * D + i] = n01(rng) + 16.0f * dir[i];
        for (int h = 0; h < HKV; ++h)
            for (int i = 0; i < D; ++i) d.k[size_t(h) * D + i] = 14.0f * dir[i];
    }
    if (regime == "outlier_v" || regime == "sink_outlier") {
        // A few channels carry 300x values and every 97th key is 20x louder: max|V| ~ 2e4.
        for (int key = 0; key < keys; ++key)
            for (int h = 0; h < HKV; ++h)
                for (int i = 0; i < D; ++i) {
                    float& x = d.v[(size_t(key) * HKV + h) * D + i];
                    if (i < 4) x *= 300.0f;
                    if (key % 97 == 0) x *= 20.0f;
                    x = std::clamp(x, -30000.0f, 30000.0f);
                }
    }
    for (auto& x : d.q) x = bf16_round(x);
    for (auto& x : d.k) x = bf16_round(x);
    for (auto& x : d.v) x = fp16_round(bf16_round(x));
    return d;
}

std::vector<double> oracle_row(const Data& d, int context, int t, int h) {
    const int visible = context + t + 1;
    const int kvh     = h / (HQ / HKV);
    const float* q    = &d.q[(size_t(t) * HQ + h) * D];
    std::vector<double> s(visible);
    double m = -INFINITY;
    for (int j = 0; j < visible; ++j) {
        const float* k = &d.k[(size_t(j) * HKV + kvh) * D];
        double dot     = 0.0;
        for (int i = 0; i < D; ++i) dot += double(q[i]) * double(k[i]);
        s[j] = dot / 16.0;
        m    = std::max(m, s[j]);
    }
    double l = 0.0;
    std::vector<double> out(D, 0.0);
    for (int j = 0; j < visible; ++j) {
        const double p = std::exp(s[j] - m);
        l += p;
        const float* v = &d.v[(size_t(j) * HKV + kvh) * D];
        for (int i = 0; i < D; ++i) out[i] += p * double(v[i]);
    }
    for (auto& x : out) x /= l;
    return out;
}

struct Err {
    double num = 0, den = 0, max_abs = 0, max_ref = 0;
    void add(double got, double ref) {
        num += (got - ref) * (got - ref);
        den += ref * ref;
        max_abs = std::max(max_abs, std::abs(got - ref));
        max_ref = std::max(max_ref, std::abs(ref));
    }
    double rel_l2() const { return std::sqrt(num / den); }
};

int main(int argc, char** argv) {
    int context = 65536, tokens = 1024, rows = 48;
    std::vector<int> splits{1, 2, 4};
    std::string regime_arg = "all";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next           = [&] { return std::string(argv[++i]); };
        if (a == "--context") context = std::stoi(next());
        else if (a == "--tokens") tokens = std::stoi(next());
        else if (a == "--rows") rows = std::stoi(next());
        else if (a == "--regime") regime_arg = next();
        else if (a == "--splits") {
            splits.clear();
            std::string s = next();
            for (size_t p = 0; p < s.size();) {
                size_t c = s.find(',', p);
                splits.push_back(std::stoi(s.substr(p, c - p)));
                p = c == std::string::npos ? s.size() : c + 1;
            }
        }
    }
    const std::vector<std::string> regimes =
        regime_arg == "all"
            ? std::vector<std::string>{"uniform", "peaked", "sink", "outlier_v", "sink_outlier"}
            : std::vector<std::string>{regime_arg};
    const int keys  = context + tokens;
    const int pages = (keys + kPagedKVPageSize - 1) / kPagedKVPageSize;

    using NewMeta = PagedKVDirectMetadata;
    auto* ref_kernel = ref::ref_causal_attention_prompt_bf16_kernel<Geometry, NewMeta>;
    auto* new_kernel = causal_attention_prompt_bf16_kernel<Geometry, NewMeta>;
    CHECK(cudaFuncSetAttribute(ref_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               kCausalPromptSmemBytes));
    CHECK(cudaFuncSetAttribute(new_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               kCausalPromptSmemBytes));

    __nv_bfloat16 *dq, *dk_log, *dk;
    __half *dv_log, *dv;
    __nv_bfloat16* dout;
    int *dtable, *dpos;
    const size_t kv_elems = size_t(pages) * kPagedKVPageSize * HKV * D;
    CHECK(cudaMalloc(&dq, size_t(tokens) * HQ * D * 2));
    CHECK(cudaMalloc(&dk_log, size_t(keys) * HKV * D * 2));
    CHECK(cudaMalloc(&dv_log, size_t(keys) * HKV * D * 2));
    CHECK(cudaMalloc(&dk, kv_elems * 2));
    CHECK(cudaMalloc(&dv, kv_elems * 2));
    CHECK(cudaMalloc(&dout, size_t(tokens) * HQ * D * 2));
    CHECK(cudaMalloc(&dtable, pages * 4));
    CHECK(cudaMalloc(&dpos, tokens * 4));
    std::vector<int> table(pages), pos(tokens);
    for (int i = 0; i < pages; ++i) table[i] = i;
    for (int t = 0; t < tokens; ++t) pos[t] = context + t;
    CHECK(cudaMemcpy(dtable, table.data(), pages * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dpos, pos.data(), tokens * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dk, 0, kv_elems * 2));
    CHECK(cudaMemset(dv, 0, kv_elems * 2));

    std::mt19937 pick(7);
    std::vector<std::pair<int, int>> sample; // (token, head), always including the last token
    for (int r = 0; r < rows; ++r)
        sample.push_back({r == 0 ? tokens - 1 : int(pick() % tokens), int(pick() % HQ)});

    std::printf("context=%d tokens=%d sampled rows=%d (x256 values)\n", context, tokens, rows);
    std::printf("%-13s %-12s %11s %11s %13s %9s\n", "regime", "kernel", "rel_l2", "max_abs",
                "max_abs/maxref", "nonfinite");
    for (const auto& regime : regimes) {
        const Data data = make(regime, keys, tokens, 1234u);
        std::vector<__nv_bfloat16> hq(data.q.size()), hk(data.k.size());
        std::vector<__half> hv(data.v.size());
        for (size_t i = 0; i < hq.size(); ++i) hq[i] = __float2bfloat16_rn(data.q[i]);
        for (size_t i = 0; i < hk.size(); ++i) hk[i] = __float2bfloat16_rn(data.k[i]);
        for (size_t i = 0; i < hv.size(); ++i) hv[i] = __float2half_rn(data.v[i]);
        CHECK(cudaMemcpy(dq, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dk_log, hk.data(), hk.size() * 2, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dv_log, hv.data(), hv.size() * 2, cudaMemcpyHostToDevice));
        const std::int64_t n = std::int64_t(keys) * HKV * D;
        scatter_kv<<<unsigned((n + 255) / 256), 256>>>(dk_log, dv_log, dk, dv, keys);
        CHECK(cudaDeviceSynchronize());

        std::vector<std::vector<double>> oracle;
        for (auto [t, h] : sample) oracle.push_back(oracle_row(data, context, t, h));
        Err floor_err;
        for (size_t r = 0; r < sample.size(); ++r)
            for (int i = 0; i < D; ++i) floor_err.add(bf16_round(float(oracle[r][i])), oracle[r][i]);
        std::printf("%-13s %-12s %11.3e %11.3e %13.3e %9s\n", regime.c_str(), "bf16(oracle)",
                    floor_err.rel_l2(), floor_err.max_abs, floor_err.max_abs / floor_err.max_ref, "-");

        const dim3 grid(unsigned((tokens + kCausalPromptBr - 1) / kCausalPromptBr), HQ, 1);
        auto evaluate = [&](const char* name) {
            std::vector<__nv_bfloat16> out(size_t(tokens) * HQ * D);
            CHECK(cudaMemcpy(out.data(), dout, out.size() * 2, cudaMemcpyDeviceToHost));
            Err e;
            long nonfinite = 0;
            for (size_t r = 0; r < sample.size(); ++r) {
                auto [t, h] = sample[r];
                for (int i = 0; i < D; ++i) {
                    const double got = __bfloat162float(out[(size_t(t) * HQ + h) * D + i]);
                    if (!std::isfinite(got)) { ++nonfinite; continue; }
                    e.add(got, oracle[r][i]);
                }
            }
            std::printf("%-13s %-12s %11.3e %11.3e %13.3e %9ld\n", regime.c_str(), name, e.rel_l2(),
                        e.max_abs, e.max_abs / e.max_ref, nonfinite);
        };
        const NewMeta meta{dtable};
        ref_kernel<<<grid, kCausalPromptThreads, kCausalPromptSmemBytes>>>(dq, dk, dv, meta, dpos,
                                                                           0.0625f, dout, tokens);
        CHECK(cudaDeviceSynchronize());
        evaluate("head fp32");
        for (int s : splits) {
            CHECK(cudaMemset(dout, 0xff, size_t(tokens) * HQ * D * 2));
            cudaLaunchAttribute attr{};
            attr.id               = cudaLaunchAttributeClusterDimension;
            attr.val.clusterDim.x = unsigned(s);
            attr.val.clusterDim.y = 1;
            attr.val.clusterDim.z = 1;
            cudaLaunchConfig_t cfg{};
            cfg.gridDim          = dim3(grid.x * unsigned(s), grid.y, 1);
            cfg.blockDim         = dim3(kCausalPromptThreads);
            cfg.dynamicSmemBytes = kCausalPromptSmemBytes;
            cfg.attrs            = &attr;
            cfg.numAttrs         = 1;
            CHECK(cudaLaunchKernelEx(&cfg, new_kernel, (const __nv_bfloat16*)dq,
                                     (const __nv_bfloat16*)dk, (const __half*)dv, meta,
                                     (const int*)dpos, 0.0625f, dout, tokens));
            CHECK(cudaDeviceSynchronize());
            char name[32];
            std::snprintf(name, sizeof name, "new S=%d", s);
            evaluate(name);
        }
    }
    return 0;
}
