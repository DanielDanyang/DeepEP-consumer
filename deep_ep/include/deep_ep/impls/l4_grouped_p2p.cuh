#pragma once

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>

namespace deep_ep::elastic {

template <int kNumThreads,
          int kNumRanks,
          int kHiddenBytes,
          int kNumSFPacks,
          int kNumTopk,
          int kTokenBytes>
__global__ void l4_grouped_pack_fp8_dispatch_impl(
    const void* x,
    const sf_pack_t* sf,
    const topk_idx_t* topk_idx,
    const float* topk_weights,
    uint8_t* packed,
    int* packed_count,
    int* packed_expert_count,
    int* dst_buffer_slot_idx,
    const int num_tokens,
    const int num_max_tokens_per_rank,
    const int rank_idx,
    const int num_experts) {
    constexpr int kNumWarps = kNumThreads / 32;
    constexpr int kHiddenOffset = 0;
    constexpr int kSFOffset = math::constexpr_align(kHiddenBytes, ptx::kNumTMAAlignBytes);
    constexpr int kMetadataOffset = kSFOffset + math::constexpr_align(kNumSFPacks * static_cast<int>(sizeof(sf_pack_t)),
                                                                      ptx::kNumTMAAlignBytes);
    constexpr int kMetadataBytes = math::constexpr_align((2 * kNumTopk + 1) * static_cast<int>(sizeof(int)),
                                                         ptx::kNumTMAAlignBytes);

    EP_STATIC_ASSERT(kNumThreads % 32 == 0, "L4 grouped pack expects full warps");
    EP_STATIC_ASSERT(kNumRanks <= 32, "L4 grouped pack stores destination ranks in a 32-bit mask");
    EP_STATIC_ASSERT(kHiddenBytes % sizeof(int4) == 0, "L4 grouped pack payload must be int4-aligned");
    EP_STATIC_ASSERT(kTokenBytes == kMetadataOffset + kMetadataBytes, "Invalid grouped token layout");

    const int warp_idx = ptx::get_warp_idx();
    const int lane_idx = ptx::get_lane_idx();
    const int global_warp_idx = static_cast<int>(blockIdx.x) * kNumWarps + warp_idx;
    const int num_global_warps = static_cast<int>(gridDim.x) * kNumWarps;
    const int num_experts_per_rank = num_experts / kNumRanks;

    for (int token_idx = global_warp_idx; token_idx < num_tokens; token_idx += num_global_warps) {
        unsigned dst_rank_mask = 0;
        int stored_expert_idx = -1;
        int stored_dst_rank_idx = -1;
        if (lane_idx < kNumTopk) {
            stored_expert_idx = static_cast<int>(__ldg(topk_idx + token_idx * kNumTopk + lane_idx));
            stored_dst_rank_idx = stored_expert_idx >= 0 ? stored_expert_idx / num_experts_per_rank : -1;
            if (stored_dst_rank_idx >= 0)
                dst_rank_mask = 1u << stored_dst_rank_idx;
            dst_buffer_slot_idx[token_idx * kNumTopk + lane_idx] = -1;
        }
        dst_rank_mask = ptx::reduce_or(dst_rank_mask);

        while (dst_rank_mask != 0) {
            const int dst_rank_idx = __ffs(dst_rank_mask) - 1;
            int packed_idx = 0;
            if (lane_idx == 0)
                packed_idx = atomicAdd(packed_count + dst_rank_idx, 1);
            packed_idx = ptx::exchange(packed_idx, 0);

            auto token_base = packed + (static_cast<int64_t>(dst_rank_idx) * num_max_tokens_per_rank + packed_idx) * kTokenBytes;

            const auto src_x = reinterpret_cast<const int4*>(
                static_cast<const uint8_t*>(x) + static_cast<int64_t>(token_idx) * kHiddenBytes);
            const auto dst_x = reinterpret_cast<int4*>(token_base + kHiddenOffset);
            constexpr int kNumHiddenVecs = kHiddenBytes / static_cast<int>(sizeof(int4));
            for (int i = lane_idx; i < kNumHiddenVecs; i += 32)
                dst_x[i] = src_x[i];

            if constexpr (kNumSFPacks > 0) {
                const auto src_sf = sf + static_cast<int64_t>(token_idx) * kNumSFPacks;
                const auto dst_sf = reinterpret_cast<sf_pack_t*>(token_base + kSFOffset);
                for (int i = lane_idx; i < kNumSFPacks; i += 32)
                    dst_sf[i] = src_sf[i];
            }

            auto metadata = reinterpret_cast<int*>(token_base + kMetadataOffset);
            auto packed_weights = reinterpret_cast<float*>(metadata + kNumTopk);
            if (lane_idx < kNumTopk) {
                metadata[lane_idx] = stored_expert_idx;
                packed_weights[lane_idx] = topk_weights == nullptr ?
                    0.0f : __ldg(topk_weights + token_idx * kNumTopk + lane_idx);
                if (stored_dst_rank_idx == dst_rank_idx)
                    dst_buffer_slot_idx[token_idx * kNumTopk + lane_idx] =
                        rank_idx * num_max_tokens_per_rank + packed_idx;
                if (stored_dst_rank_idx == dst_rank_idx)
                    atomicAdd(packed_expert_count + dst_rank_idx * num_experts_per_rank +
                              (stored_expert_idx - dst_rank_idx * num_experts_per_rank), 1);
            }
            if (lane_idx == 0)
                metadata[2 * kNumTopk] = rank_idx * num_max_tokens_per_rank + token_idx;

            dst_rank_mask &= dst_rank_mask - 1;
        }
    }
}

template <int kNumThreads,
          int kNumRanks,
          int kHiddenBytes,
          int kNumSFPacks,
          int kNumTopk,
          int kTokenBytes>
__global__ void l4_grouped_unpack_fp8_dispatch_impl(
    const uint8_t* recv_buffer,
    const int* recv_rank_prefix,
    void* recv_x,
    sf_pack_t* recv_sf,
    topk_idx_t* recv_topk_idx,
    float* recv_topk_weights,
    int* recv_src_metadata,
    int* recv_expert_count,
    const int num_recv_tokens,
    const int num_tokens_per_rank,
    const int rank_idx,
    const int num_experts) {
    constexpr int kNumWarps = kNumThreads / 32;
    constexpr int kHiddenOffset = 0;
    constexpr int kSFOffset = math::constexpr_align(kHiddenBytes, ptx::kNumTMAAlignBytes);
    constexpr int kMetadataOffset = kSFOffset + math::constexpr_align(kNumSFPacks * static_cast<int>(sizeof(sf_pack_t)),
                                                                      ptx::kNumTMAAlignBytes);
    constexpr int kMetadataBytes = math::constexpr_align((2 * kNumTopk + 1) * static_cast<int>(sizeof(int)),
                                                         ptx::kNumTMAAlignBytes);

    EP_STATIC_ASSERT(kNumThreads % 32 == 0, "L4 grouped unpack expects full warps");
    EP_STATIC_ASSERT(kHiddenBytes % sizeof(int4) == 0, "L4 grouped unpack payload must be int4-aligned");
    EP_STATIC_ASSERT(kTokenBytes == kMetadataOffset + kMetadataBytes, "Invalid grouped token layout");

    const int warp_idx = ptx::get_warp_idx();
    const int lane_idx = ptx::get_lane_idx();
    const int global_warp_idx = static_cast<int>(blockIdx.x) * kNumWarps + warp_idx;
    const int num_global_warps = static_cast<int>(gridDim.x) * kNumWarps;
    constexpr int kMetadataStride = kNumTopk + 2;
    const int num_experts_per_rank = num_experts / kNumRanks;
    const int expert_start_idx = rank_idx * num_experts_per_rank;
    const int expert_end_idx = expert_start_idx + num_experts_per_rank;

    for (int out_idx = global_warp_idx; out_idx < num_recv_tokens; out_idx += num_global_warps) {
        int src_rank_idx = 0;
        #pragma unroll
        for (int rank_idx = 0; rank_idx < kNumRanks; ++rank_idx) {
            const int begin = __ldg(recv_rank_prefix + rank_idx);
            const int end = __ldg(recv_rank_prefix + rank_idx + 1);
            if (out_idx >= begin and out_idx < end)
                src_rank_idx = rank_idx;
        }
        const int local_idx = out_idx - __ldg(recv_rank_prefix + src_rank_idx);
        const auto token_base = recv_buffer + (static_cast<int64_t>(src_rank_idx) * num_tokens_per_rank + local_idx) * kTokenBytes;

        const auto src_x = reinterpret_cast<const int4*>(token_base + kHiddenOffset);
        const auto dst_x = reinterpret_cast<int4*>(
            static_cast<uint8_t*>(recv_x) + static_cast<int64_t>(out_idx) * kHiddenBytes);
        constexpr int kNumHiddenVecs = kHiddenBytes / static_cast<int>(sizeof(int4));
        for (int i = lane_idx; i < kNumHiddenVecs; i += 32)
            dst_x[i] = src_x[i];

        if constexpr (kNumSFPacks > 0) {
            const auto src_sf = reinterpret_cast<const sf_pack_t*>(token_base + kSFOffset);
            const auto dst_sf = recv_sf + static_cast<int64_t>(out_idx) * kNumSFPacks;
            for (int i = lane_idx; i < kNumSFPacks; i += 32)
                dst_sf[i] = src_sf[i];
        }

        const auto metadata = reinterpret_cast<const int*>(token_base + kMetadataOffset);
        const auto packed_weights = reinterpret_cast<const float*>(metadata + kNumTopk);
        int stored_expert_idx = -1;
        bool in_range = false;
        if (lane_idx < kNumTopk) {
            stored_expert_idx = __ldg(metadata + lane_idx);
            in_range = expert_start_idx <= stored_expert_idx and stored_expert_idx < expert_end_idx;
            recv_topk_idx[out_idx * kNumTopk + lane_idx] = static_cast<topk_idx_t>(
                in_range ? stored_expert_idx - expert_start_idx : -1);
            recv_topk_weights[out_idx * kNumTopk + lane_idx] = __ldg(packed_weights + lane_idx);
            recv_src_metadata[out_idx * kMetadataStride + 2 + lane_idx] = -1;
        }
        const int master_src_topk_idx = ptx::get_master_lane_idx(ptx::gather(in_range));
        if (lane_idx == 0) {
            recv_src_metadata[out_idx * kMetadataStride + 0] = __ldg(metadata + 2 * kNumTopk);
            recv_src_metadata[out_idx * kMetadataStride + 1] = src_rank_idx * kNumTopk + master_src_topk_idx;
        }
    }
}

}  // namespace deep_ep::elastic
