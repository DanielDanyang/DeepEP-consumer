#pragma once

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>

namespace deep_ep::elastic {

template <int kNumThreads,
          int kHiddenBytes,
          int kNumSFPacks,
          int kNumTopk>
__global__ void host_staging_pack_fp8_dispatch_impl(
    const void* x,
    const sf_pack_t* sf,
    const topk_idx_t* topk_idx,
    const float* topk_weights,
    void* packed_x,
    sf_pack_t* packed_sf,
    topk_idx_t* packed_topk_idx,
    float* packed_topk_weights,
    int* packed_src_token_idx,
    int* packed_count,
    const int num_tokens,
    const int rank_idx,
    const int num_ranks,
    const int num_scaleup_ranks,
    const int num_experts) {
    constexpr int kNumWarps = kNumThreads / 32;
    EP_STATIC_ASSERT(kNumThreads % 32 == 0, "Host-staging pack expects full warps");
    EP_STATIC_ASSERT(kHiddenBytes % sizeof(int4) == 0, "Host-staging payload must be int4-aligned");

    const int warp_idx = ptx::get_warp_idx();
    const int lane_idx = ptx::get_lane_idx();
    const int global_warp_idx = static_cast<int>(blockIdx.x) * kNumWarps + warp_idx;
    const int num_global_warps = static_cast<int>(gridDim.x) * kNumWarps;

    const int num_scaleout_ranks = num_ranks / num_scaleup_ranks;
    const int local_scaleout_rank_idx = rank_idx / num_scaleup_ranks;
    const int num_experts_per_scaleout_rank = num_experts / num_scaleout_ranks;

    for (int token_idx = global_warp_idx; token_idx < num_tokens; token_idx += num_global_warps) {
        bool has_remote_target = false;
        if (lane_idx < kNumTopk) {
            const int expert_idx = static_cast<int>(__ldg(topk_idx + token_idx * kNumTopk + lane_idx));
            const int dst_scaleout_rank_idx = expert_idx >= 0 ? expert_idx / num_experts_per_scaleout_rank : -1;
            has_remote_target = dst_scaleout_rank_idx >= 0 and dst_scaleout_rank_idx != local_scaleout_rank_idx;
        }
        const uint32_t remote_mask = ptx::gather(has_remote_target);
        if (remote_mask == 0)
            continue;

        int packed_idx = 0;
        if (lane_idx == 0) {
            packed_idx = atomicAdd(packed_count, 1);
            packed_src_token_idx[packed_idx] = token_idx;
        }
        packed_idx = ptx::exchange(packed_idx, 0);

        const auto src_x = reinterpret_cast<const int4*>(
            static_cast<const int8_t*>(x) + static_cast<int64_t>(token_idx) * kHiddenBytes);
        const auto dst_x = math::advance_ptr<int4>(packed_x, static_cast<int64_t>(packed_idx) * kHiddenBytes);
        constexpr int kNumHiddenVecs = kHiddenBytes / static_cast<int>(sizeof(int4));
        for (int i = lane_idx; i < kNumHiddenVecs; i += 32)
            dst_x[i] = src_x[i];

        if constexpr (kNumSFPacks > 0) {
            const auto src_sf = sf + static_cast<int64_t>(token_idx) * kNumSFPacks;
            const auto dst_sf = packed_sf + static_cast<int64_t>(packed_idx) * kNumSFPacks;
            for (int i = lane_idx; i < kNumSFPacks; i += 32)
                dst_sf[i] = src_sf[i];
        }

        if (lane_idx < kNumTopk) {
            packed_topk_idx[packed_idx * kNumTopk + lane_idx] =
                __ldg(topk_idx + token_idx * kNumTopk + lane_idx);
            if (topk_weights != nullptr) {
                packed_topk_weights[packed_idx * kNumTopk + lane_idx] =
                    __ldg(topk_weights + token_idx * kNumTopk + lane_idx);
            }
        }
    }
}

template <int kNumThreads,
          int kHiddenBytes,
          int kNumTopk>
__global__ void host_staging_pack_bf16_combine_impl(
    const nv_bfloat16* x,
    const topk_idx_t* topk_idx,
    nv_bfloat16* packed_x,
    int* packed_src_token_idx,
    int* packed_count,
    const int num_tokens,
    const int rank_idx,
    const int num_ranks,
    const int num_scaleup_ranks,
    const int num_experts) {
    constexpr int kNumWarps = kNumThreads / 32;
    EP_STATIC_ASSERT(kNumThreads % 32 == 0, "Host-staging combine pack expects full warps");
    EP_STATIC_ASSERT(kHiddenBytes % sizeof(int4) == 0, "Host-staging combine payload must be int4-aligned");

    const int warp_idx = ptx::get_warp_idx();
    const int lane_idx = ptx::get_lane_idx();
    const int global_warp_idx = static_cast<int>(blockIdx.x) * kNumWarps + warp_idx;
    const int num_global_warps = static_cast<int>(gridDim.x) * kNumWarps;

    const int num_scaleout_ranks = num_ranks / num_scaleup_ranks;
    const int local_scaleout_rank_idx = rank_idx / num_scaleup_ranks;
    const int num_experts_per_scaleout_rank = num_experts / num_scaleout_ranks;

    for (int token_idx = global_warp_idx; token_idx < num_tokens; token_idx += num_global_warps) {
        bool has_remote_target = false;
        if (lane_idx < kNumTopk) {
            const int expert_idx = static_cast<int>(__ldg(topk_idx + token_idx * kNumTopk + lane_idx));
            const int dst_scaleout_rank_idx = expert_idx >= 0 ? expert_idx / num_experts_per_scaleout_rank : -1;
            has_remote_target = dst_scaleout_rank_idx >= 0 and dst_scaleout_rank_idx != local_scaleout_rank_idx;
        }
        const uint32_t remote_mask = ptx::gather(has_remote_target);
        if (remote_mask == 0)
            continue;

        int packed_idx = 0;
        if (lane_idx == 0) {
            packed_idx = atomicAdd(packed_count, 1);
            packed_src_token_idx[packed_idx] = token_idx;
        }
        packed_idx = ptx::exchange(packed_idx, 0);

        const auto src_x = reinterpret_cast<const int4*>(
            reinterpret_cast<const int8_t*>(x) + static_cast<int64_t>(token_idx) * kHiddenBytes);
        const auto dst_x = reinterpret_cast<int4*>(
            reinterpret_cast<int8_t*>(packed_x) + static_cast<int64_t>(packed_idx) * kHiddenBytes);
        constexpr int kNumHiddenVecs = kHiddenBytes / static_cast<int>(sizeof(int4));
        for (int i = lane_idx; i < kNumHiddenVecs; i += 32)
            dst_x[i] = src_x[i];
    }
}

}  // namespace deep_ep::elastic
