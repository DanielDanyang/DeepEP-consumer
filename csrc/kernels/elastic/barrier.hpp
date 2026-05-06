#pragma once

#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"

namespace deep_ep::elastic {

/*
 * Barrier JIT wrapper。
 *
 * v2 所有跨 rank buffer 复用都依赖 GPU-side barrier 保证可见性:
 *
 *     rank local work
 *          |
 *          v
 *     launch_barrier
 *          |
 *          +-- scale-up NVLink: workspace signal + system atomics
 *          +-- scale-up non-NVLink / scale-out: NCCL Gin signal
 *          |
 *          v
 *     all peers see previous stores
 *
 * hybrid mode 同时有 scale-out 与 scale-up 两个子域，所以 barrier kernel 用 2 个 SM；
 * direct/single-domain barrier 只用 1 个 SM。
 */

class BarrierRuntime final : public jit::LaunchRuntime<BarrierRuntime> {
public:
    struct Args {
        // Templated arguments
        bool is_scaleup_nvlink;
        int num_scaleout_ranks, num_scaleup_ranks;
        int64_t num_timeout_cycles;

        // Parameters
        ncclDevComm_t nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* workspace;
        int scaleout_rank_idx, scaleup_rank_idx;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/barrier.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&barrier_impl<{}, {}, {}, {}, {}, {}>);
}}
)",                        args.is_scaleup_nvlink,
                           args.launch_args.grid_dim.first, args.launch_args.num_threads,
                           args.num_scaleout_ranks, args.num_scaleup_ranks,
                           args.num_timeout_cycles);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.nccl_dev_comm, args.nccl_window,
            args.workspace, args.scaleout_rank_idx, args.scaleup_rank_idx
        ));
    }
};

static void launch_barrier(const ncclDevComm_t& nccl_dev_comm, const ncclWindow_t& nccl_window,
                           void* workspace,
                           const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                           const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                           const int64_t& num_timeout_cycles,
                           const bool& is_scaleup_nvlink,
                           const at::cuda::CUDAStream& stream) {
    // 线程数固定 512，覆盖当前 WorkspaceLayout 支持的常见 rank fanout。
    constexpr auto kNumThreads = 512;

    // 只有 hybrid barrier 需要 2 个 SM，分别推进 scale-up 和 scale-out 子域。
    const auto num_sms = num_scaleout_ranks > 1 ? 2 : 1;
    const BarrierRuntime::Args args = {
        .is_scaleup_nvlink = is_scaleup_nvlink,
        .num_scaleout_ranks = num_scaleout_ranks, .num_scaleup_ranks = num_scaleup_ranks,
        .num_timeout_cycles = num_timeout_cycles,
        .nccl_dev_comm = nccl_dev_comm,
        .nccl_window = nccl_window,
        .workspace = workspace,
        .scaleout_rank_idx = scaleout_rank_idx, .scaleup_rank_idx = scaleup_rank_idx,
        .launch_args = jit::LaunchArgs(num_sms, kNumThreads, 0, 1, true)};
    const auto code = BarrierRuntime::generate(args);
    const auto runtime = jit::compiler->build("barrier", code);
    BarrierRuntime::launch(runtime, args, stream);
}

}  // namespace deep_ep::elastic
