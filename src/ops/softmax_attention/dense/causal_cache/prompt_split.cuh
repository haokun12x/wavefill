#pragma once

// Key-range split of one causal prompt tile across the CTAs of a thread-block cluster.
//
// A prompt chunk launches one CTA per (64-row query block, query head) and every CTA occupies a
// whole SM, so a 1024-token chunk is 384 equal CTAs on 170 SMs: three waves, the last one a quarter
// full. Launched as clusters of S CTAs, rank r of a cluster runs key blocks
// [r * n / S, (r + 1) * n / S) of the same tile, and the ranks merge their partial softmax states
// through distributed shared memory before one bf16 store. No global workspace is involved.
//
// Merge protocol, after the key loop, over the CTA's own dynamic shared memory (the Q/K/V staging
// buffers are dead by then):
//   1. __syncthreads(); each row owner stores {m, l} (raw score max, row sum relative to it).
//   2. cluster.sync(); every thread derives, for its own rows, the factor that maps this rank's
//      accumulator row into the normalized output and stores the scaled accumulator.
//   3. cluster.sync(); rank r sums slice r of the tile over all ranks in rank order and stores it.
//   4. cluster.sync(), so no rank exits while another still reads its shared memory.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <math_constants.h>

#include "ops/softmax_attention/dense/causal_cache/prompt_common.cuh"

namespace ninfer::ops {

inline constexpr int kCausalPromptMaxSplits = 8;
// Fewer key blocks per rank than this and the tile is not split (device-side guard; the launcher
// already avoids it for the chunk it knows about).
inline constexpr int kCausalPromptMinSplitBlocks = 4;
inline constexpr int kCausalPromptSplitRows      = 64;
// Padded fp32 row: 8 extra floats move each row by eight banks for the fragment-layout stores.
inline constexpr int kCausalPromptSplitAccStride = kCausalPromptHeadDim + 8;
inline constexpr int kCausalPromptSplitAccBytes =
    kCausalPromptSplitRows * kCausalPromptSplitAccStride * static_cast<int>(sizeof(float));
inline constexpr int kCausalPromptSplitSmemBytes =
    kCausalPromptSplitAccBytes + kCausalPromptSplitRows * static_cast<int>(sizeof(float2));

struct CausalPromptSplit {
    int rank;
    int splits;
    int kb_begin;
    int kb_end;
};

__device__ __forceinline__ int causal_prompt_cluster_splits() {
    return static_cast<int>(cooperative_groups::this_cluster().num_blocks());
}

__device__ __forceinline__ int causal_prompt_cluster_rank() {
    return static_cast<int>(cooperative_groups::this_cluster().block_rank());
}

// Key-block range of this rank. `splits` is 1 when the tile runs whole; ranks other than 0 then
// have an empty range and must return before touching shared memory.
__device__ __forceinline__ CausalPromptSplit causal_prompt_split(int key_blocks) {
    const int splits = causal_prompt_cluster_splits();
    const int rank   = causal_prompt_cluster_rank();
    if (splits == 1 || key_blocks < splits * kCausalPromptMinSplitBlocks) {
        return {rank, 1, 0, rank == 0 ? key_blocks : 0};
    }
    return {rank, splits, key_blocks * rank / splits, key_blocks * (rank + 1) / splits};
}

__device__ __forceinline__ float* causal_prompt_split_acc(unsigned char* smem) {
    return reinterpret_cast<float*>(smem);
}

__device__ __forceinline__ float2* causal_prompt_split_stats(unsigned char* smem) {
    return reinterpret_cast<float2*>(smem + kCausalPromptSplitAccBytes);
}

// Step 2: factor for one row of this rank's accumulator (includes 1 / l of the merged row).
__device__ __forceinline__ float
causal_prompt_split_row_factor(unsigned char* smem, int row, float scale_l2, int splits, int rank) {
    auto cluster        = cooperative_groups::this_cluster();
    float2* local_stats = causal_prompt_split_stats(smem);
    float2 stats[kCausalPromptMaxSplits];
    float m = -CUDART_INF_F;
#pragma unroll
    for (int r = 0; r < kCausalPromptMaxSplits; ++r) {
        if (r < splits) {
            stats[r] = *cluster.map_shared_rank(local_stats + row, r);
            m        = fmaxf(m, stats[r].x);
        }
    }
    const float m_scaled = m * scale_l2;
    float l              = 0.0f;
    float own            = 0.0f;
#pragma unroll
    for (int r = 0; r < kCausalPromptMaxSplits; ++r) {
        if (r < splits) {
            const float f = stats[r].x == -CUDART_INF_F
                                ? 0.0f
                                : exp2_approx(__fmaf_rn(stats[r].x, scale_l2, -m_scaled));
            l             = __fmaf_rn(stats[r].y, f, l);
            if (r == rank) { own = f; }
        }
    }
    return l > 0.0f ? own * __frcp_rn(l) : 0.0f;
}

__device__ __forceinline__ void causal_prompt_split_store_acc(unsigned char* smem, int row, int d,
                                                              float x0, float x1) {
    *reinterpret_cast<float2*>(causal_prompt_split_acc(smem) + row * kCausalPromptSplitAccStride +
                               d) = make_float2(x0, x1);
}

// Step 3: rank r reduces and stores its slice of the [rows, D] tile.
template <typename Geometry>
__device__ __forceinline__ void
causal_prompt_split_reduce_store(unsigned char* smem, int splits, int rank, __nv_bfloat16* out,
                                 int q_head, int q0, int tile_rows, int tid, int threads) {
    constexpr int VecPerRow = kCausalPromptHeadDim / 4;
    constexpr int Total     = kCausalPromptSplitRows * VecPerRow;
    auto cluster            = cooperative_groups::this_cluster();
    float* local_acc        = causal_prompt_split_acc(smem);
    const int begin         = Total * rank / splits;
    const int end           = Total * (rank + 1) / splits;
    for (int i = begin + tid; i < end; i += threads) {
        const int row = i / VecPerRow;
        if (row >= tile_rows) { continue; }
        const int d      = (i - row * VecPerRow) * 4;
        const int offset = row * kCausalPromptSplitAccStride + d;
        float4 sum =
            *reinterpret_cast<const float4*>(cluster.map_shared_rank(local_acc, 0) + offset);
        for (int r = 1; r < splits; ++r) {
            const float4 part =
                *reinterpret_cast<const float4*>(cluster.map_shared_rank(local_acc, r) + offset);
            sum.x += part.x;
            sum.y += part.y;
            sum.z += part.z;
            sum.w += part.w;
        }
        uint2 packed;
        packed.x = pack_bf16x2(sum.x, sum.y);
        packed.y = pack_bf16x2(sum.z, sum.w);
        *reinterpret_cast<uint2*>(&out[causal_prompt_q_index<Geometry>(q_head, d, q0 + row)]) =
            packed;
    }
}

} // namespace ninfer::ops
