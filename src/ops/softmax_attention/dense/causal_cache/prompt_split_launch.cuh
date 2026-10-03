#pragma once

// Host side of the cluster key-range split of causal prompt tiles (see prompt_split.cuh): how many
// CTAs share a tile, and the launch that groups them into clusters.

#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_split.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cuda_runtime.h>

#include <array>
#include <cstdint>

namespace ninfer::ops::detail {

// Per-rank cost of a split tile, in key blocks: the Q stage and first K/V tile of every rank plus
// the cluster merge.
inline constexpr int kCausalPromptSplitOverheadBlocks = 2;

// Clusters of S whole-SM CTAs that the device holds at once, for S = 0..kCausalPromptMaxSplits
// (0 where S cannot launch).
template <auto Kernel>
const std::array<int, kCausalPromptMaxSplits + 1>& causal_prompt_active_clusters(int threads,
                                                                                 int smem_bytes) {
    static const std::array<int, kCausalPromptMaxSplits + 1> clusters = [&] {
        std::array<int, kCausalPromptMaxSplits + 1> result{};
        for (int splits = 1; splits <= kCausalPromptMaxSplits; ++splits) {
            cudaLaunchAttribute attribute{};
            attribute.id               = cudaLaunchAttributeClusterDimension;
            attribute.val.clusterDim.x = static_cast<unsigned>(splits);
            attribute.val.clusterDim.y = 1;
            attribute.val.clusterDim.z = 1;
            cudaLaunchConfig_t config{};
            config.gridDim          = dim3(static_cast<unsigned>(splits), 1u, 1u);
            config.blockDim         = dim3(static_cast<unsigned>(threads), 1u, 1u);
            config.dynamicSmemBytes = static_cast<std::size_t>(smem_bytes);
            config.attrs            = &attribute;
            config.numAttrs         = 1;
            int count               = 0;
            if (cudaOccupancyMaxActiveClusters(&count, Kernel, &config) != cudaSuccess) {
                (void)cudaGetLastError();
                count = 0;
            }
            result[splits] = count;
        }
        return result;
    }();
    return clusters;
}

// Splits per tile for a chunk of `tokens` rows whose last row sees the envelope's visible keys:
// the one that minimizes waves x per-rank work, with every rank keeping at least
// kCausalPromptMinSplitBlocks key blocks of the shortest tile the envelope admits.
inline int causal_prompt_choose_splits(const std::array<int, kCausalPromptMaxSplits + 1>& clusters,
                                       int units, int tokens,
                                       CausalAttentionExecutionEnvelope envelope) {
    const auto context = [&](std::uint32_t visible) -> std::int64_t {
        return visible > static_cast<std::uint32_t>(tokens) ? visible - tokens : 0;
    };
    const std::int64_t shortest = context(envelope.min_visible_keys) / kCausalPromptBc + 1;
    const std::int64_t average =
        (context(envelope.max_visible_keys) + tokens / 2) / kCausalPromptBc + 1;
    int best               = 1;
    std::int64_t best_cost = 0;
    for (int splits = 1; splits <= kCausalPromptMaxSplits; ++splits) {
        if (clusters[splits] <= 0) continue;
        if (splits > 1 &&
            shortest < static_cast<std::int64_t>(splits) * kCausalPromptMinSplitBlocks)
            break;
        const std::int64_t waves = (units + clusters[splits] - 1) / clusters[splits];
        const std::int64_t work =
            (average + splits - 1) / splits + (splits > 1 ? kCausalPromptSplitOverheadBlocks : 0);
        const std::int64_t cost = waves * work;
        if (splits == 1 || cost < best_cost) {
            best      = splits;
            best_cost = cost;
        }
    }
    return best;
}

template <auto Kernel, typename... Args>
void causal_prompt_launch_split(dim3 grid, int threads, int smem_bytes, int splits,
                                cudaStream_t stream, Args... args) {
    if (splits == 1) {
        Kernel<<<grid, threads, smem_bytes, stream>>>(args...);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    cudaLaunchAttribute attribute{};
    attribute.id               = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = static_cast<unsigned>(splits);
    attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    cudaLaunchConfig_t config{};
    config.gridDim          = dim3(grid.x * static_cast<unsigned>(splits), grid.y, grid.z);
    config.blockDim         = dim3(static_cast<unsigned>(threads), 1u, 1u);
    config.dynamicSmemBytes = static_cast<std::size_t>(smem_bytes);
    config.stream           = stream;
    config.attrs            = &attribute;
    config.numAttrs         = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&config, Kernel, args...));
}

} // namespace ninfer::ops::detail
