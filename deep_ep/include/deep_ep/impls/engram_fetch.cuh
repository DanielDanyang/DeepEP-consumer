#pragma once

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/ptx.cuh>


namespace deep_ep::elastic {

/*
 * Engram fetch device kernels。
 *
 * fetch kernel:
 *
 *     blockIdx.x = qp_idx
 *     each warp picks token indices
 *     global_idx -> (src_rank_idx, src_entry_idx)
 *     issue aggregated gin.get
 *     flush per touched rank and save request
 *
 * wait kernel:
 *
 *     one block per QP
 *     reload saved requests
 *     gin.wait(request) for non-empty request
 *
 * 这样 Python API 可以先发起 fetch，继续排别的 compute，再调用 hook 等待。
 */

template <int kNumQPs,
          int kNumEntriesPerRank,
          int kHidden,
          int kNumRanks,
          int kNumThreads,
          int kNumWarps = kNumThreads / 32,
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16),
          typename team_t = ncclTeamTagWorld>
__global__ void __launch_bounds__(kNumThreads, 1)
engram_fetch_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
                  void* storage, void* fetched, int* indices,
                  ncclGinRequest_t* last_gin_requests,
                  const int num_tokens) {
    const auto qp_idx = static_cast<int>(blockIdx.x);
    const auto warp_idx = ptx::get_warp_idx();
    const auto global_warp_idx = qp_idx * kNumWarps + warp_idx;
    const auto thread_idx = static_cast<int>(threadIdx.x);

    // Gin handle
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, NCCL_GIN_RESOURCE_SHARING_CTA);

    __shared__ bool sent_to_rank[kNumRanks];
    EP_STATIC_ASSERT(kNumRanks <= kNumThreads, "Too many ranks");
    if (thread_idx < kNumRanks)
        sent_to_rank[thread_idx] = false;
    __syncthreads();

    // token 的 global entry id 被拆成远端 rank 和该 rank 内 entry offset。
    const auto issue_rdma_get = [=](const int& token_idx, const int& src_rank_idx, const int& src_entry_idx,
                                    const int& extra_options = 0) {
        gin.get<team_t>(math::advance_ptr(storage, static_cast<int64_t>(src_entry_idx) * kNumHiddenBytes),
                        math::advance_ptr(fetched, static_cast<int64_t>(token_idx) * kNumHiddenBytes),
                        kNumHiddenBytes, src_rank_idx, extra_options);
    };

    // Each warp fetches one token cooperatively via RDMA gin.get
    // TODO: deal with padded tokens
    if (ptx::elect_one_sync()) {
        #pragma unroll 4
        for (int i = global_warp_idx; i < num_tokens; i += kNumQPs * kNumWarps) {
            const auto global_idx = __ldg(indices + i);
            const auto src_rank_idx = global_idx / kNumEntriesPerRank;
            const auto src_entry_idx = global_idx % kNumEntriesPerRank;

            // Delay ring DB
            issue_rdma_get(i, src_rank_idx, src_entry_idx, ncclGinOptFlagsAggregateRequests);
            sent_to_rank[src_rank_idx] = true;
        }
    }
    __syncthreads();

    // 对每个 touched peer 发 flushAsync；一个 flush 会提交同 QP 上之前聚合的 get。
    if (ptx::elect_one_sync()) {
        for (int i = warp_idx; i < kNumRanks; i += kNumWarps) {
            const auto request_ptr = last_gin_requests + qp_idx * kNumRanks + i;
            if (sent_to_rank[i]) {
                gin.flush_async<team_t>(i, request_ptr);
            } else {
                EP_STATIC_ASSERT(sizeof(ncclGinRequest_t) == sizeof(int4), "Invalid request size");
                *reinterpret_cast<int4*>(request_ptr) = make_int4(0, 0, 0, 0);
            }
        }
    }
}

template <int kNumRanks, int kNumThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
engram_fetch_wait_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
                       ncclGinRequest_t* last_gin_requests) {
    const auto qp_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);

    // Gin handle
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, NCCL_GIN_RESOURCE_SHARING_CTA);

    // 等 fetch kernel 里保存的所有非空 request 完成。
    for (int i = thread_idx; i < kNumRanks; i += kNumThreads) {
        EP_STATIC_ASSERT(sizeof(ncclGinRequest_t) == sizeof(int4), "Invalid request size");
        auto last_gin_req_int4 = __ldg(reinterpret_cast<int4*>(last_gin_requests + qp_idx * kNumRanks + i));
        if (last_gin_req_int4.x != 0 or last_gin_req_int4.y != 0 or
            last_gin_req_int4.z != 0 or last_gin_req_int4.w != 0) {
            auto last_gin_req = *reinterpret_cast<ncclGinRequest_t*>(&last_gin_req_int4);
            gin.wait(last_gin_req);
        }
    }
}

} // namespace deep_ep::elastic
