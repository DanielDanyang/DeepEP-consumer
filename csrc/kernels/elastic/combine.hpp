#pragma once

#include <nccl.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"

namespace deep_ep::elastic {

/*
 * Combine JIT wrapper。
 *
 * combine 是 dispatch 的反向通信:
 *
 *     expert output x
 *          |
 *          v
 *     launch_combine main kernel
 *          |
 *          +-- direct: combine.cuh
 *          +-- hybrid: hybrid_combine.cuh
 *          |
 *          v
 *     reduce buffer
 *          |
 *          v
 *     launch_combine_reduce_epilogue
 *          |
 *          v
 *     combined_x in original token order
 *
 * main kernel 负责跨 rank 把数据推回 reduce buffer；epilogue 负责最终 reduce、bias、
 * top-k weights 输出以及 expanded layout 的收口。
 */

class CombineRuntime final : public jit::LaunchRuntime<CombineRuntime> {
public:
    struct Args {
        // Templated arguments
        bool is_scaleup_nvlink;
        bool use_expanded_layout, allow_multiple_reduction;
        int num_scaleup_warps, num_forward_warps;
        int num_scaleout_ranks, num_scaleup_ranks;
        int hidden;
        int num_max_tokens_per_rank;
        int num_experts;
        int num_topk;
        int num_qps;
        int64_t num_timeout_cycles;

        // Parameters
        nv_bfloat16* x;
        float* topk_weights;
        int* src_metadata;
        int* psum_num_recv_tokens_per_scaleup_rank;
        int* token_metadata_at_forward;
        int* channel_linked_list;
        ncclDevComm_t nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* buffer;
        void* workspace;
        int scaleout_rank_idx, scaleup_rank_idx;
        int num_reduced_tokens;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        std::string header_name, func_name;
        if (args.num_scaleout_ranks == 1) {
            // direct combine: 只有一个 scale-out 域，目标 rank 可通过 NVLink/World Gin 直接写 reduce buffer。
            header_name = "combine";
            func_name = fmt::format("combine_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
                                    args.is_scaleup_nvlink,
                                    args.use_expanded_layout, args.allow_multiple_reduction,
                                    args.launch_args.grid_dim.first,
                                    args.launch_args.num_threads / 32,
                                    args.num_scaleup_ranks * args.num_scaleout_ranks,
                                    args.hidden,
                                    args.num_max_tokens_per_rank,
                                    args.num_experts,
                                    args.num_topk,
                                    args.num_qps, args.num_timeout_cycles);
        } else {
            // hybrid combine: 先在目标节点内 scale-up 域收集，再由 forward warps 跨 scale-out 返回。
            header_name = "hybrid_combine";
            func_name = fmt::format("hybrid_combine_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
                                    args.use_expanded_layout, args.allow_multiple_reduction,
                                    args.launch_args.grid_dim.first,
                                    args.num_scaleup_warps, args.num_forward_warps,
                                    args.num_scaleout_ranks, args.num_scaleup_ranks,
                                    args.hidden,
                                    args.num_max_tokens_per_rank,
                                    args.num_experts,
                                    args.num_topk,
                                    args.num_qps,
                                    args.num_timeout_cycles);
        }
        return fmt::format(R"(
#include <deep_ep/impls/{}.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{});
}}
)", header_name, func_name);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        if (args.num_scaleout_ranks == 1) {
            EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config,
                                                     args.x, args.topk_weights,
                                                     args.src_metadata, args.psum_num_recv_tokens_per_scaleup_rank,
                                                     args.nccl_dev_comm, args.nccl_window,
                                                     args.buffer, args.workspace,
                                                     args.scaleup_rank_idx,
                                                     args.num_reduced_tokens));
        } else {
            EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config,
                                                     args.x, args.topk_weights,
                                                     args.src_metadata,
                                                     args.psum_num_recv_tokens_per_scaleup_rank,
                                                     args.token_metadata_at_forward,
                                                     args.channel_linked_list,
                                                     args.nccl_dev_comm, args.nccl_window,
                                                     args.buffer, args.workspace,
                                                     args.scaleout_rank_idx, args.scaleup_rank_idx,
                                                     args.num_reduced_tokens));
        }
    }
};

static layout::TokenLayout get_combine_token_layout(
    const int& hidden, const int& elem_size, const int& num_topk) {
    // combine buffer 不需要 dispatch 的 src metadata，但仍需要 topk weights metadata。
    return layout::TokenLayout(hidden * elem_size, 0, num_topk, false);
}

static void* launch_combine(void* x,
                            void* topk_weights,
                            int* src_metadata,
                            int* psum_num_recv_tokens_per_scaleup_rank,
                            int* token_metadata_at_forward,
                            int* channel_linked_list,
                            const ncclDevComm_t& nccl_dev_comm, const ncclWindow_t& nccl_window,
                            void* buffer, void* workspace,
                            const int& num_reduced_tokens, const int& num_max_tokens_per_rank,
                            const int& hidden,
                            const int& num_experts, const int& num_topk,
                            const int& num_qps, const int64_t& num_timeout_cycles,
                            const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                            const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                            const bool& is_scaleup_nvlink,
                            const int& num_sms, const int& num_smem_bytes,
                            const int& num_channels,
                            const bool& use_expanded_layout, const bool& allow_multiple_reduction,
                            const at::cuda::CUDAStream& stream) {
    // 每个 warp 一个 TMA staging token。direct mode 尽量把 shared memory 填满；
    // hybrid mode 必须与 dispatch 的 channel 数匹配，所以 warps = scaleup + forward。
    const auto token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);
    auto num_warps = std::min(num_smem_bytes / token_layout.get_num_bytes<true>(), 32);

    // hybrid combine 中:
    //
    //     scaleup_warps:  从 expert output 写入节点内 scale-up buffer
    //     forward_warps:  等待 scale-up tail，规约/转发到 scale-out buffer
    //
    // num_channels 必须来自 dispatch handle，保证 linked-list/channel metadata 对齐。
    int num_scaleup_warps = 0, num_forward_warps = 0;
    if (num_scaleout_ranks > 1) {
        EP_HOST_ASSERT(num_channels % num_sms == 0 and
                       "Invalid number of channels or SMs, you may use a different SM count than dispatch");
        EP_HOST_ASSERT(num_channels / num_sms <= 16);

        num_scaleup_warps = num_forward_warps = num_channels / num_sms;
        num_warps = num_scaleup_warps + num_forward_warps;
        EP_HOST_ASSERT(num_warps * token_layout.get_num_bytes<true>() <= num_smem_bytes and
                       "Invalid combine SM count, please try to match your dispatch config");
    }

    // Generate, build and launch
    const auto num_threads = num_warps * 32;
    const CombineRuntime::Args args = {
        .is_scaleup_nvlink = is_scaleup_nvlink,
        .use_expanded_layout = use_expanded_layout,
        .allow_multiple_reduction = allow_multiple_reduction,
        .num_scaleup_warps = num_scaleup_warps, .num_forward_warps = num_forward_warps,
        .num_scaleout_ranks = num_scaleout_ranks, .num_scaleup_ranks = num_scaleup_ranks,
        .hidden = hidden,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_experts = num_experts,
        .num_topk = num_topk,
        .num_qps = num_qps, .num_timeout_cycles = num_timeout_cycles,
        .x = static_cast<nv_bfloat16*>(x),
        .topk_weights = static_cast<float*>(topk_weights),
        .src_metadata = src_metadata,
        .psum_num_recv_tokens_per_scaleup_rank = psum_num_recv_tokens_per_scaleup_rank,
        .token_metadata_at_forward = token_metadata_at_forward,
        .channel_linked_list = channel_linked_list,
        .nccl_dev_comm = nccl_dev_comm, .nccl_window = nccl_window,
        .buffer = buffer, .workspace = workspace,
        .scaleout_rank_idx = scaleout_rank_idx, .scaleup_rank_idx = scaleup_rank_idx,
        .num_reduced_tokens = num_reduced_tokens,
        // NOTES: make cluster dim 2 to overlap with clustered computation kernels
        .launch_args = jit::LaunchArgs(num_sms, num_threads, num_smem_bytes, 2 - (num_sms % 2), true)
    };
    const auto code = CombineRuntime::generate(args);
    const auto runtime = jit::compiler->build("combine", code);
    CombineRuntime::launch(runtime, args, stream);

    // 返回 epilogue 需要读取的 reduce buffer 起始地址。
    // hybrid 下 buffer 前缀是 scale-up 中间区，最终 reduce buffer 在其后。
    if (num_scaleout_ranks == 1)
        return buffer;

    // For hybrid mode, we have to skip the scale-up buffer
    const bool is_scaleup_buffer_rank_layout =
        allow_multiple_reduction ? (num_scaleup_ranks <= num_topk) : false;
    const auto scaleup_buffer = layout::BufferLayout<false>(
        token_layout, 
        is_scaleup_buffer_rank_layout ? num_scaleup_ranks : num_topk,
        num_scaleout_ranks * num_max_tokens_per_rank,
        buffer);
    return scaleup_buffer.get_buffer_end_ptr();
}

class CombineReduceEpilogueRuntime final : public jit::LaunchRuntime<CombineReduceEpilogueRuntime> {
public:
    struct Args {
        // Templated arguments
        bool use_expanded_layout, allow_multiple_reduction;
        int num_scaleout_ranks, num_scaleup_ranks;
        int hidden;
        int num_max_tokens_per_rank;
        int num_experts, num_topk;

        // Parameters
        nv_bfloat16* combined_x;
        float* combined_topk_weights;
        topk_idx_t* combined_topk_idx;
        void* reduce_buffer;
        void* bias_0;
        void* bias_1;
        int num_combined_tokens;
        int scaleout_rank_idx, scaleup_rank_idx;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        // reduce epilogue 只依赖输出 token 数、hidden/topk/topology/layout 策略，
        // 不需要 NCCL handle，因为所有跨 rank 数据在 main combine kernel 后已经落在本地 buffer。
        return fmt::format(R"(
#include <deep_ep/impls/combine_reduce_epilogue.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&combine_reduce_epilogue_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)",                        args.use_expanded_layout, args.allow_multiple_reduction,
                           args.launch_args.grid_dim.first,
                           args.launch_args.num_threads / 32,
                           args.num_scaleout_ranks, args.num_scaleup_ranks,
                           args.hidden,
                           args.num_max_tokens_per_rank,
                           args.num_experts, args.num_topk);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config,
                                                 args.combined_x,
                                                 args.combined_topk_weights,
                                                 args.combined_topk_idx,
                                                 args.reduce_buffer,
                                                 args.bias_0, args.bias_1,
                                                 args.num_combined_tokens,
                                                 args.scaleout_rank_idx, args.scaleup_rank_idx));
    }
};

static void launch_combine_reduce_epilogue(void* combined_x,
                                           float* combined_topk_weights,
                                           topk_idx_t* combined_topk_idx,
                                           const int& num_combined_tokens, const int& num_max_tokens_per_rank,
                                           const int& hidden,
                                           const int& num_experts, const int& num_topk,
                                           void* reduce_buffer,
                                           void* bias_0, void* bias_1,
                                           const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                           const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                                           const int& num_sms, const int& num_smem_bytes,
                                           const bool& use_expanded_layout, const bool& allow_multiple_reduction,
                                           const at::cuda::CUDAStream& stream) {
    // epilogue 做纯本地 reduce/copy，使用全 SM；每个 warp 在 shared memory 放一个输出 token。
    const auto token_layout = layout::TokenLayout(hidden * sizeof(nv_bfloat16), 0, 0, false);
    const auto num_warps = std::min<int>(num_smem_bytes / token_layout.get_num_bytes<false>(), 32);
    const auto num_threads = num_warps * 32;

    // Generate, build and launch
    const CombineReduceEpilogueRuntime::Args args = {
        .use_expanded_layout = use_expanded_layout,
        .allow_multiple_reduction = allow_multiple_reduction,
        .num_scaleout_ranks = num_scaleout_ranks, .num_scaleup_ranks = num_scaleup_ranks,
        .hidden = hidden,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_experts = num_experts, .num_topk = num_topk,
        .combined_x = static_cast<nv_bfloat16*>(combined_x),
        .combined_topk_weights = combined_topk_weights,
        .combined_topk_idx = combined_topk_idx,
        .reduce_buffer = reduce_buffer,
        .bias_0 = bias_0, .bias_1 = bias_1,
        .num_combined_tokens = num_combined_tokens,
        .scaleout_rank_idx = scaleout_rank_idx, .scaleup_rank_idx = scaleup_rank_idx,
        .launch_args = jit::LaunchArgs(num_sms, num_threads, num_smem_bytes, 1, false, true)
    };
    const auto code = CombineReduceEpilogueRuntime::generate(args);
    const auto runtime = jit::compiler->build("combine_reduce_epilogue", code);
    CombineReduceEpilogueRuntime::launch(runtime, args, stream);
}

}  // namespace deep_ep::elastic
