#pragma once

#include <cooperative_groups.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>


namespace deep_ep::elastic {

/*
 * deterministic dispatch prologue。
 *
 * 目的:
 *   在主 dispatch kernel 之前，为每个 token/top-k 计算稳定的 dst_buffer_slot_idx。
 *   这样 main dispatch 不再依赖 atomicAdd 的非确定性顺序。
 *
 * 两遍扫描:
 *
 *     pass 1: 每个 warp 统计自己负责 token 会发给各 rank 的去重数量
 *             -> rank_count_buffer[sm][rank]
 *
 *     grid sync
 *
 *     pass 2: 计算当前 SM/warp 之前的 prefix sum
 *             -> dst_buffer_slot_idx[token][topk]
 *
 * lane 分组:
 *
 *     one warp lanes
 *       token0 topk lanes | token1 topk lanes | ... | tokenN topk lanes
 *
 * 同一个 token 多个 expert 落同一 rank 时只计一次，避免向同 rank 重复发送。
 */

// TODO: support scale-out
template <int kNumSMs, int kNumWarps,
          int kNumScaleupRanks,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk,
          int kNumThreads = kNumWarps * 32>
__global__ void __launch_bounds__(kNumThreads, 1)
dispatch_deterministic_prologue_impl(
    topk_idx_t* topk_idx,
    int* rank_count_buffer,
    int* dst_buffer_slot_idx,
    const int num_tokens,
    const int scaleup_rank_idx
) {
    constexpr int kNumExpertsPerRank = kNumExperts / kNumScaleupRanks;
    EP_STATIC_ASSERT(kNumExperts % kNumScaleupRanks == 0, "Invalid number of experts or ranks");

    // Utils
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();
    const auto global_warp_idx = sm_idx * kNumWarps + warp_idx;

    // 当前 warp 负责一段连续 token。
    const auto num_tokens_per_warp = math::ceil_div(num_tokens, kNumSMs * kNumWarps);
    const auto start_token_idx = global_warp_idx * num_tokens_per_warp;
    const auto end_token_idx = min(start_token_idx + num_tokens_per_warp, num_tokens);

    // 一个 warp 同时处理 kNumTokensPerGroup 个 token；每个 token 占 kNumTopk 条 lane。
    constexpr int kNumTokensPerGroup = 32 / kNumTopk;
    const auto token_idx_offset = lane_idx / kNumTopk;
    const unsigned token_mask = ((1u << kNumTopk) - 1) << (token_idx_offset * kNumTopk);
    EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k");

    // shared memory:
    //
    //     rank_count_global_psum[num_ranks]
    //     rank_count_warp_sum[num_warps][num_ranks]
    //     rank_count_warp_psum[num_warps][num_ranks]
    //
    // 每个 warp 先独立计数，之后 block 内规约。
    extern __shared__ int8_t smem[];
    const auto rank_count_global_psum = math::advance_ptr<int>(smem, 0);
    const auto rank_count_warp_sum = math::advance_ptr<int>(rank_count_global_psum, (kNumScaleupRanks + warp_idx * kNumScaleupRanks) * sizeof(int));
    const auto rank_count_warp_psum = math::advance_ptr<int>(rank_count_warp_sum, kNumWarps * kNumScaleupRanks * sizeof(int));

    // Initialize to zero before reduce
    for (int i = thread_idx; i < kNumScaleupRanks * (1 + 2 * kNumWarps); i += kNumThreads)
        reinterpret_cast<int*>(smem)[i] = 0;
    __syncthreads();

    // Util functions
    const auto map_expert_to_rank_idx = [&](const int& expert_idx) {
        return expert_idx >= 0 ? expert_idx / kNumExpertsPerRank : -1;
    };
    const auto is_unique = [&](const int& rank_idx) {
        return ((ptx::match(rank_idx) & token_mask) >> lane_idx) == 1;
    };
    const auto count_ones_before = [&](const unsigned& mask, const int& bit_idx) {
        return __popc(mask & ((1u << bit_idx) - 1));
    };
    const auto get_other_rank_count_warp_sum = [&](const int& other_warp_idx) {
        // NOTES: pass negative num_bytes to advance pointer
        return math::advance_ptr<int>(rank_count_warp_sum, (other_warp_idx - warp_idx) * kNumScaleupRanks * sizeof(int));
    };

    // pass 1: 统计本 warp 对每个 rank 的去重发送数量。
    for (int i = start_token_idx; i < end_token_idx; i += kNumTokensPerGroup) {
        const auto token_idx = i + token_idx_offset;
        const auto is_active_thread = lane_idx < kNumTopk * kNumTokensPerGroup and token_idx < end_token_idx;
        const int expert_idx = is_active_thread ? static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx)) : -1;
        const auto rank_idx = map_expert_to_rank_idx(expert_idx);

        // Avoid duplicate messages to a single rank
        const auto deduped_rank_idx = is_unique(rank_idx) ? rank_idx : -1;
        const auto rank_idx_mask = ptx::match(deduped_rank_idx);

        // Let the one with the largest lane index send the count
        if ((rank_idx_mask >> lane_idx) == 1 and deduped_rank_idx >= 0)
            rank_count_warp_sum[deduped_rank_idx] += __popc(rank_idx_mask);
    }
    __syncthreads();

    // block sum 写到全局 rank_count_buffer，供所有 SM 做 prefix。
    for (int rank_idx = thread_idx; rank_idx < kNumScaleupRanks; rank_idx += kNumThreads) {
        int rank_count_block_sum = 0;
        for (int i = 0; i < kNumWarps; i++)
            rank_count_block_sum += get_other_rank_count_warp_sum(i)[rank_idx];
        rank_count_buffer[sm_idx * kNumScaleupRanks + rank_idx] = rank_count_block_sum;
    }
    cooperative_groups::this_grid().sync();

    // 计算当前 SM 之前所有 SM 的 rank count 前缀。
    for (int rank_idx = lane_idx; rank_idx < kNumScaleupRanks; rank_idx += 32) {
        int rank_count = 0;
        for (int i = warp_idx; i < sm_idx; i += kNumWarps)
            rank_count += rank_count_buffer[i * kNumScaleupRanks + rank_idx];
        atomicAdd_block(rank_count_global_psum + rank_idx, rank_count);
    }
    __syncthreads();

    // 再加上当前 SM 内更早 warp 的 count，得到本 warp 的 slot 起点。
    for (int rank_idx = lane_idx; rank_idx < kNumScaleupRanks; rank_idx += 32) {
        int rank_count = rank_count_global_psum[rank_idx];
        for (int i = 0; i < warp_idx; i++)
            rank_count += get_other_rank_count_warp_sum(i)[rank_idx];
        rank_count_warp_psum[rank_idx] = rank_count;
    }
    __syncwarp();

    // pass 2: 用前缀和为每个 token/top-k lane 写稳定 slot。
    for (int i = start_token_idx; i < end_token_idx; i += kNumTokensPerGroup) {
        const auto token_idx = i + token_idx_offset;
        const auto is_active_thread = lane_idx < kNumTopk * kNumTokensPerGroup and token_idx < end_token_idx;
        const auto expert_idx = is_active_thread ? static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx)) : -1;
        const auto rank_idx = map_expert_to_rank_idx(expert_idx);

        // Avoid duplicate messages to a single rank
        const auto deduped_rank_idx = is_unique(rank_idx) ? rank_idx : -1;
        const auto rank_idx_mask = ptx::match(deduped_rank_idx);

        // Store to target buffer
        const auto stored_dst_slot_idx = deduped_rank_idx >= 0 ?
                                         rank_count_warp_psum[deduped_rank_idx] + count_ones_before(rank_idx_mask, lane_idx) : -1;
        const auto value = stored_dst_slot_idx >= 0 ?
                           scaleup_rank_idx * kNumMaxTokensPerRank + stored_dst_slot_idx : -1;
        if (is_active_thread)
            dst_buffer_slot_idx[i * kNumTopk + lane_idx] = value;

        // Let the one with the largest lane index send the count
        if ((rank_idx_mask >> lane_idx) == 1 and deduped_rank_idx >= 0)
            rank_count_warp_psum[deduped_rank_idx] += __popc(rank_idx_mask);
        __syncwarp();
    }
}

}  // namespace deep_ep::elastic
