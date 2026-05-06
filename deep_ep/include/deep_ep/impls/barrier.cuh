#pragma once

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/ptx.cuh>


namespace deep_ep::elastic {

/*
 * GPU barrier kernel 主体。
 *
 * launch wrapper 已经把 topology 固定成模板参数。kernel 内只做一件事:
 *
 *     construct WorkspaceLayout + NCCLGin
 *              |
 *              v
 *     comm::gpu_barrier(...)
 *
 * WorkspaceLayout num_experts 传 0 是安全的，因为 barrier 只使用 workspace 开头的
 * NVLink/Gin signal 区域，不访问 expert count 区。
 */

template <bool kIsScaleupNVLink,
          int kNumSMs, int kNumThreads,
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int64_t kNumTimeoutCycles>
__global__ void __launch_bounds__(kNumThreads, 1)
barrier_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window, void* workspace,
             const int scaleout_rank_idx, const int scaleup_rank_idx) {
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);

    // Barrier only uses the first part of workspace, so making `num_experts` as 0 is fine
    const auto workspace_layout = layout::WorkspaceLayout(workspace, kNumScaleoutRanks, kNumScaleupRanks, 0);
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, 0);
    comm::gpu_barrier<kIsScaleupNVLink, kNumScaleoutRanks, kNumScaleupRanks,
                      kNumSMs, kNumThreads, comm::kFlushAllAllocatedQPs, kNumTimeoutCycles, comm::kKernelBarrierTag, false, false, false>(
            gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);
}

} // namespace deep_ep::elastic
