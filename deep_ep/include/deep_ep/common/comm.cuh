#pragma once

#include <cooperative_groups.h>
#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/handle.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/common/layout.cuh>

namespace deep_ep::elastic::comm {

/*
 * 通信同步公共逻辑。
 *
 * v2 kernel 里的跨 rank 同步通常走两层:
 *
 *     local/grid sync
 *          |
 *          v
 *     scale-up barrier   (NVLink LSA 或 World Gin)
 *     scale-out barrier  (Rail Gin)
 *          |
 *          v
 *     local/grid sync
 *
 * 关键设计:
 *   - QP 分配由 get_qp_mode 决定，notify warp 可独占 QP0。
 *   - barrier 前可 flush TMA stores / Gin QPs，保证 peer 看到数据。
 *   - timeout_while 在 GPU 上打印定位信息后 trap，避免 silent hang。
 *
 * hybrid barrier 示意:
 *
 *     SM0              SM1..SMn
 *      |                  |
 *      v                  v
 *   scale-up          scale-out
 *   barrier           barrier
 *      \              /
 *       \            /
 *        grid sync all SMs
 */

static constexpr int64_t kNumOneSecCycles = 2000000000;  // An approximation of the GPU clock at 2000 MHz

// Some reserved tags
static constexpr int kDeviceBarrierTag = 0;
static constexpr int kKernelBarrierTag = 1;
static constexpr int kDispatchTag0 = 2;
static constexpr int kDispatchTag1 = 3;
static constexpr int kCombineTag0 = 4;
static constexpr int kCombineTag1 = 5;
static constexpr int kHybridDispatchTag0 = 6;
static constexpr int kHybridDispatchTag1 = 7;
static constexpr int kHybridCombineTag0 = 8;
static constexpr int kHybridCombineTag1 = 9;

// Some reserved count
static constexpr int kFlushAllAllocatedQPs = -1;

template <int64_t kNumTimeoutCycles, typename func_t>
__device__ __forceinline__ void timeout_while(const bool& condition, const func_t& func,
                                              int64_t start_clock = 0) {
    // func(is_last_check) 返回 true 表示等待完成。超时最后一次会让调用方打印上下文，
    // 然后额外等待 1 秒，让其他线程也有机会打印，再 trap。
    // User may share a start clock for multiple waits
    if (start_clock == 0)
        start_clock = clock64();

    while (condition) {
        const bool timeout = clock64() - start_clock >= kNumTimeoutCycles;
        if (func(timeout))
            break;

        if (timeout) {
            // Wait another 1 second to let all threads print information and trap
            start_clock = clock64();
            while (clock64() - start_clock < kNumOneSecCycles) {}
            ptx::trap();
        }
    }
}

template <int64_t kNumTimeoutCycles, typename func_t>
__device__ __forceinline__ void timeout_while(const func_t& func, const int64_t& start_clock = 0) {
    timeout_while<kNumTimeoutCycles, func_t>(true, func, start_clock);
}

template <int kNumSMs, int kNumQPs, int kNumChannelsPerSM, bool kWithNotifyWarps = false>
__device__ __forceinline__ std::pair<int, ncclGinResourceSharingMode> get_qp_mode(
    const int& sm_idx, const int& channel_in_sm_idx, const bool& is_notify_warp = false) {
    // QP 映射策略:
    //
    //     notify warp -> QP0 + CTA sharing
    //     QP >= SM    -> 每个 SM 拿一组 QP，CTA sharing
    //     QP < SM     -> 所有 SM/channel round-robin 共享 QP，GPU sharing
    //
    // 目标是在 doorbell 开销和 channel 并行度之间折中。
    constexpr auto kSharingCTA = NCCL_GIN_RESOURCE_SHARING_CTA;
    constexpr auto kSharingGrid = kNumSMs == 1 ? NCCL_GIN_RESOURCE_SHARING_CTA : NCCL_GIN_RESOURCE_SHARING_GPU;

    // Only one QP
    if constexpr (kNumQPs == 1)
        return {0, kSharingGrid};

    // The notify warp always use 1 SM and 1 QP
    if (is_notify_warp)
        return {0, kSharingCTA};

    // Data channels
    constexpr int kQPStartIdx = static_cast<int>(kWithNotifyWarps);
    constexpr int kNumAvailableQPs = kNumQPs - kQPStartIdx;
    if constexpr (kNumSMs <= kNumAvailableQPs) {
        // A single SM uses an entire QP
        // e.g., 3 SMs and 10 QPs
        // SM 0: 0 3 6 9
        // SM 1: 1 4 7
        // SM 2: 2 5 8
        const int num_qps_in_sm = (kNumAvailableQPs / kNumSMs) + (sm_idx < (kNumAvailableQPs % kNumSMs));
        return {kQPStartIdx + sm_idx + (channel_in_sm_idx % num_qps_in_sm) * kNumSMs, kSharingCTA};
    } else {
        // All SMs share all QPs
        const auto global_channel_idx = sm_idx * kNumChannelsPerSM + channel_in_sm_idx;
        return {kQPStartIdx + (global_channel_idx % kNumAvailableQPs), kSharingGrid};
    }
}

template <int kNumRanks, int kNumSMs, int kNumThreads, int64_t kNumTimeoutCycles, int kTag = kDeviceBarrierTag>
__forceinline__ __device__ void nvlink_barrier_wo_local_sync(
    const handle::NCCLGin& gin,
    const layout::WorkspaceLayout& workspace,
    const int& rank_idx, const int& sm_idx, const int& thread_idx) {
    // NVLink barrier 只使用 SM0:
    //
    //     phase/sign 从 counter 低 bit 读取
    //     each rank atomically adds +1 or -1 to peer signal
    //     wait signal reaches target
    //     counter++ toggles next phase/sign
    //
    // 使用正负交替可以复用两组 signal，避免每次 memset。
    // This barrier only uses 1 SM
    if (kNumSMs > 1 and sm_idx > 0)
        return;

    // Read the current barrier phase first
    const int status = static_cast<int>((*workspace.get_nvl_barrier_counter_ptr()) & 3);
    const int phase = status & 1, sign = status >> 1;

    EP_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");
    if (thread_idx < kNumRanks) {
        const auto dst_ptr =
            gin.get_sym_ptr<ncclTeamTagLsa>(workspace.get_nvl_barrier_signal_ptr(phase), thread_idx);
        ptx::red_add_rel_sys(dst_ptr, sign ? -1 : 1);
    }
    __syncthreads();

    // NOTES: we need `2^64 / 1e6 / 3600 / 24 / 365 = 571000` years to make the counter overflow (1 barrier per us)
    // Add the phase counter
    if (thread_idx == 0)
        atomicAdd(workspace.get_nvl_barrier_counter_ptr(), 1);

    // Check timeout
    const auto target = sign ? 0 : kNumRanks;
    timeout_while<kNumTimeoutCycles>(thread_idx == 0, [=](const bool& is_last_check) {
        const auto signal = ptx::ld_acquire_sys<int>(workspace.get_nvl_barrier_signal_ptr(phase));
        if (signal == target)
            return true;

        if (is_last_check) {
            printf("DeepEP NVLink barrier timeout, tag: %d, nvl: %d, thread: %d, "
                   "status: %d, signal: %d, phase: %d, target: %d, counter: %llu\n",
                   kTag, rank_idx, thread_idx, status, signal, phase, target,
                   *workspace.get_nvl_barrier_counter_ptr());
        }
        return false;
    });
}

template <int kNumRanks, int kNumSMs, int kNumThreads, int kNumQPs, int64_t kNumTimeoutCycles,
          typename team_t, int kTag = kDeviceBarrierTag,
          bool kFlushStores = true,
          int kNumWarps = kNumThreads / 32>
__forceinline__ __device__ void gin_barrier_wo_local_sync(
    const ncclDevComm_t& nccl_dev_comm,
    const int& scaleout_rank_idx, const int& scaleup_rank_idx, 
    const int& sm_idx, const int& thread_idx) {
    // Gin barrier 分两步:
    //
    //   1. 可选 flush 所有 QP，保证之前 put/get/store 的 release 语义
    //   2. QP0 对 team 内每个 rank 发 signal，并轮询 signal shadow/global table
    const auto global_warp_idx = sm_idx * kNumWarps + (thread_idx / 32);
    const int& rank_idx = (std::is_same_v<team_t, ncclTeamTagWorld>) ? scaleup_rank_idx : scaleout_rank_idx;
    const int num_qps = kNumQPs == kFlushAllAllocatedQPs ? nccl_dev_comm.ginContextCount : kNumQPs;

    // Flush all QPs by all SMs (only needed for release semantics)
    if constexpr (kFlushStores) {
        for (int i = global_warp_idx; i < num_qps; i += kNumSMs * kNumWarps) {
            ncclGin(nccl_dev_comm, i, NCCL_GIN_RESOURCE_SHARING_CTA).flush(ncclCoopWarp());
        }
        // NOTES: we can not use `kNumSMs` to judge, as maybe only part of the SMs will call this function
        (gridDim.x > 1) ? cooperative_groups::this_grid().sync() : __syncthreads();
    }

    if (sm_idx == 0) {
        // Use QP 0 to do barrier
        const auto team = (std::is_same_v<team_t, ncclTeamTagWorld>) ?
            ncclTeamWorld(nccl_dev_comm) : ncclTeamRail(nccl_dev_comm);
        const ncclGin gin(nccl_dev_comm, 0, NCCL_GIN_RESOURCE_SHARING_CTA);
        for (int i = thread_idx; i < kNumRanks; i += kNumThreads)
            gin.signal(team, i, ncclGin_SignalInc{static_cast<ncclGinSignal_t>(rank_idx)});

        // TODO(NCCL): Using the official NCCL wait signal API, after they added timeout check.
        for (int i = thread_idx; i < kNumRanks; i += kNumThreads) {
            const auto signal_idx = static_cast<ncclGinSignal_t>(i);
            const auto shadow_ptr = gin.getSignalShadowPtr(signal_idx);
            const auto target = ++(*shadow_ptr);

            const auto gdaki = static_cast<struct ncclGinGdakiGPUContext*>(gin._ginHandle) + gin.contextId;
            const auto signal_ptr = reinterpret_cast<uint64_t*>(__ldg(reinterpret_cast<uint64_t*>(&gdaki->signals_table.buffer))) + signal_idx;
            timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                const auto signal = ptx::ld_acquire_sys<uint64_t>(signal_ptr);
                if (signal >= target)
                    return true;

                if (is_last_check) {
                    printf("DeepEP Gin barrier timeout, tag: %d, scaleout: %d, scaleup: %d, thread: %d, "
                           "signal: %lu, target: %lu\n", kTag, scaleout_rank_idx, scaleup_rank_idx, thread_idx, signal, target);
                }
                return false;
            });
        }
    }
}

template <bool kIsScaleupNVLink, int kNumRanks, int kNumSMs, int kNumThreads, int kNumQPs,
          int64_t kNumTimeoutCycles, int kTag = kDeviceBarrierTag, bool kFlushStores = true>
__forceinline__ __device__ void scaleup_barrier_wo_local_sync(
    const handle::NCCLGin& gin,
    const layout::WorkspaceLayout& workspace,
    const int& rank_idx, const int& sm_idx, const int& thread_idx) {
    if constexpr (kIsScaleupNVLink) {
        nvlink_barrier_wo_local_sync<kNumRanks, kNumSMs, kNumThreads, kNumTimeoutCycles, kTag>(
            gin, workspace, rank_idx, sm_idx, thread_idx);
    } else {
        gin_barrier_wo_local_sync<kNumRanks, kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, ncclTeamTagWorld, kTag, kFlushStores>(
            gin.nccl_dev_comm, 1, rank_idx, sm_idx, thread_idx);
    }
}

template <int kNumRanks, int kNumSMs, int kNumThreads, int kNumQPs, int64_t kNumTimeoutCycles, int kTag = kDeviceBarrierTag,
          bool kFlushStores = true>
__forceinline__ __device__ void scaleout_barrier_wo_local_sync(
    const handle::NCCLGin& gin,
    const int& scaleout_rank_idx, const int& scaleup_rank_idx,
    const int& sm_idx, const int& thread_idx) {
    gin_barrier_wo_local_sync<kNumRanks, kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, ncclTeamTagRail, kTag, kFlushStores>(
        gin.nccl_dev_comm, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);
}

template <bool kIsScaleupNVLink,
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int kNumSMs, int kNumThreads, int kNumQPs,
          int64_t kNumTimeoutCycles, int kTag = kDeviceBarrierTag,
          bool kFlushStores = true, bool kSyncAtStart = true, bool kSyncAtEnd = true>
__forceinline__ __device__ void gpu_barrier(const handle::NCCLGin& gin,
                                            const layout::WorkspaceLayout& workspace,
                                            const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                                            const int& sm_idx, const int& thread_idx,
                                            bool do_scaleout = true, bool do_scaleup = true) {
    // 通用 barrier 编排器。调用方可控制:
    //   kFlushStores  是否等待 TMA store / flush Gin QP
    //   kSyncAtStart  barrier 前是否 grid sync
    //   kSyncAtEnd    barrier 后是否 grid sync
    //   do_scaleout/do_scaleup 运行时跳过某个子域
    // A general TMA store wait to prevent proxy memory issues
    if constexpr (kFlushStores) {
        ptx::tma_store_commit();
        ptx::tma_store_wait();
        __syncwarp();
    }

    // All the SMs should wait
    if constexpr (kSyncAtStart) {
        cooperative_groups::this_grid().sync();
    } else {
        EP_STATIC_ASSERT(not kFlushStores, "No data to be flushed");
    }

    do_scaleout &= kNumScaleoutRanks > 1;
    do_scaleup &= kNumScaleupRanks > 1;
    if (do_scaleup and do_scaleout) {
        // Do scaleup and scaleout barrier in parallel
        EP_DEVICE_ASSERT(kNumSMs >= 2 and "At least 2 SMs for a hybrid barrier");
        if (sm_idx == 0) {
            // First SM do the scaleup barrier
            scaleup_barrier_wo_local_sync<kIsScaleupNVLink, kNumScaleupRanks, kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, kTag, kFlushStores>(
                gin, workspace, scaleup_rank_idx, sm_idx, thread_idx);

            // We need an extra grid sync, as the scaleout barrier will do a sync after flush, before the barrier
            // NOTES: this is kind of hacky
            if constexpr (kFlushStores) 
                cooperative_groups::this_grid().sync();
        } else {
            // The remaining SMs do the scaleout barrier
            scaleout_barrier_wo_local_sync<kNumScaleoutRanks, kNumSMs - 1, kNumThreads, kNumQPs, kNumTimeoutCycles, kTag, kFlushStores>(
                gin, scaleout_rank_idx, scaleup_rank_idx, sm_idx - 1, thread_idx);
        }
    } else if (do_scaleup) {
        // Scaleup only
        scaleup_barrier_wo_local_sync<kIsScaleupNVLink, kNumScaleupRanks, kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, kTag, kFlushStores>(
            gin, workspace, scaleup_rank_idx, sm_idx, thread_idx);
    } else if (do_scaleout) {
        // Scaleout only
        scaleout_barrier_wo_local_sync<kNumScaleoutRanks, kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, kTag, kFlushStores>(
            gin, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);
    }

    // All the SMs should wait
    if constexpr (kSyncAtEnd)
        cooperative_groups::this_grid().sync();
}

}  // namespace deep_ep::elastic::comm
