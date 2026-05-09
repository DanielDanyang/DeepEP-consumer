#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <torch/python.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"

namespace deep_ep::elastic {

static int l4_grouped_fp8_dispatch_token_bytes_impl(const int& hidden_bytes,
                                                    const int& num_sf_packs,
                                                    const int& num_topk) {
    return math::align(hidden_bytes, ptx::kNumTMAAlignBytes) +
           math::align(num_sf_packs * static_cast<int>(sizeof(sf_pack_t)), ptx::kNumTMAAlignBytes) +
           math::align((2 * num_topk + 1) * static_cast<int>(sizeof(int)), ptx::kNumTMAAlignBytes);
}

class L4GroupedPackFP8DispatchRuntime final : public jit::LaunchRuntime<L4GroupedPackFP8DispatchRuntime> {
public:
    struct Args {
        int num_ranks;
        int hidden_bytes;
        int num_sf_packs;
        int num_topk;
        int token_bytes;

        const void* x;
        const sf_pack_t* sf;
        const topk_idx_t* topk_idx;
        const float* topk_weights;
        uint8_t* packed;
        int* packed_count;
        int num_tokens;
        int rank_idx;
        int num_experts;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/l4_grouped_p2p.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&l4_grouped_pack_fp8_dispatch_impl<{}, {}, {}, {}, {}, {}>);
}}
)",
                           args.launch_args.num_threads,
                           args.num_ranks,
                           args.hidden_bytes,
                           args.num_sf_packs,
                           args.num_topk,
                           args.token_bytes);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.x,
            args.sf,
            args.topk_idx,
            args.topk_weights,
            args.packed,
            args.packed_count,
            args.num_tokens,
            args.rank_idx,
            args.num_experts));
    }
};

class L4GroupedUnpackFP8DispatchRuntime final : public jit::LaunchRuntime<L4GroupedUnpackFP8DispatchRuntime> {
public:
    struct Args {
        int num_ranks;
        int hidden_bytes;
        int num_sf_packs;
        int num_topk;
        int token_bytes;

        const uint8_t* recv_buffer;
        const int* recv_rank_prefix;
        void* recv_x;
        sf_pack_t* recv_sf;
        topk_idx_t* recv_topk_idx;
        float* recv_topk_weights;
        int* recv_src_token_idx;
        int num_recv_tokens;
        int num_tokens_per_rank;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/l4_grouped_p2p.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&l4_grouped_unpack_fp8_dispatch_impl<{}, {}, {}, {}, {}, {}>);
}}
)",
                           args.launch_args.num_threads,
                           args.num_ranks,
                           args.hidden_bytes,
                           args.num_sf_packs,
                           args.num_topk,
                           args.token_bytes);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.recv_buffer,
            args.recv_rank_prefix,
            args.recv_x,
            args.recv_sf,
            args.recv_topk_idx,
            args.recv_topk_weights,
            args.recv_src_token_idx,
            args.num_recv_tokens,
            args.num_tokens_per_rank));
    }
};

static void l4_grouped_check_fp8_dispatch_pack_args(const torch::Tensor& x,
                                                    const torch::Tensor& sf,
                                                    const torch::Tensor& topk_idx,
                                                    const std::optional<torch::Tensor>& topk_weights,
                                                    const torch::Tensor& packed,
                                                    const torch::Tensor& packed_count,
                                                    const int& rank_idx,
                                                    const int& num_ranks,
                                                    const int& num_experts,
                                                    const int& num_sms) {
    EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous());
    EP_HOST_ASSERT(sf.is_cuda() and sf.is_contiguous());
    EP_HOST_ASSERT(topk_idx.is_cuda() and topk_idx.is_contiguous());
    EP_HOST_ASSERT(packed.is_cuda() and packed.is_contiguous());
    EP_HOST_ASSERT(packed_count.is_cuda() and packed_count.is_contiguous());
    EP_HOST_ASSERT(x.dim() == 2 and sf.dim() == 2 and topk_idx.dim() == 2);
    EP_HOST_ASSERT(x.element_size() == 1 and "L4 grouped dispatch pack currently targets FP8 payloads");
    EP_HOST_ASSERT(sf.element_size() == sizeof(sf_pack_t));
    EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
    EP_HOST_ASSERT(packed.scalar_type() == torch::kByte);
    EP_HOST_ASSERT(packed_count.scalar_type() == torch::kInt32 and packed_count.numel() == num_ranks);
    EP_HOST_ASSERT(rank_idx >= 0 and rank_idx < num_ranks);
    EP_HOST_ASSERT(num_ranks > 0 and num_ranks <= 32);
    EP_HOST_ASSERT(num_experts > 0 and num_experts % num_ranks == 0);
    EP_HOST_ASSERT(num_sms > 0);

    const int num_tokens = static_cast<int>(x.size(0));
    const int hidden_bytes = static_cast<int>(x.size(1) * x.element_size());
    const int num_sf_packs = static_cast<int>(sf.size(1));
    const int num_topk = static_cast<int>(topk_idx.size(1));
    const int token_bytes = l4_grouped_fp8_dispatch_token_bytes_impl(hidden_bytes, num_sf_packs, num_topk);
    EP_HOST_ASSERT(sf.size(0) == num_tokens and topk_idx.size(0) == num_tokens);
    EP_HOST_ASSERT(hidden_bytes % sizeof(int4) == 0);
    EP_HOST_ASSERT(packed.numel() >= static_cast<int64_t>(num_ranks) * num_tokens * token_bytes);

    if (topk_weights.has_value()) {
        EP_HOST_ASSERT(topk_weights->is_cuda() and topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        EP_HOST_ASSERT(topk_weights->size(0) == num_tokens and topk_weights->size(1) == num_topk);
    }
}

static int l4_grouped_fp8_dispatch_token_bytes(const int& hidden,
                                               const int& num_sf_packs,
                                               const int& num_topk) {
    return l4_grouped_fp8_dispatch_token_bytes_impl(hidden, num_sf_packs, num_topk);
}

static void l4_grouped_pack_fp8_dispatch_out(const torch::Tensor& x,
                                             const torch::Tensor& sf,
                                             const torch::Tensor& topk_idx,
                                             const std::optional<torch::Tensor>& topk_weights,
                                             const torch::Tensor& packed,
                                             const torch::Tensor& packed_count,
                                             const int& rank_idx,
                                             const int& num_ranks,
                                             const int& num_experts,
                                             const int& num_sms) {
    l4_grouped_check_fp8_dispatch_pack_args(
        x, sf, topk_idx, topk_weights, packed, packed_count,
        rank_idx, num_ranks, num_experts, num_sms);

    constexpr int num_threads = 256;
    const int hidden_bytes = static_cast<int>(x.size(1) * x.element_size());
    const int num_sf_packs = static_cast<int>(sf.size(1));
    const int num_topk = static_cast<int>(topk_idx.size(1));
    const int token_bytes = l4_grouped_fp8_dispatch_token_bytes_impl(hidden_bytes, num_sf_packs, num_topk);
    const float* topk_weights_ptr = topk_weights.has_value() ? topk_weights->data_ptr<float>() : nullptr;
    const L4GroupedPackFP8DispatchRuntime::Args args = {
        .num_ranks = num_ranks,
        .hidden_bytes = hidden_bytes,
        .num_sf_packs = num_sf_packs,
        .num_topk = num_topk,
        .token_bytes = token_bytes,
        .x = x.data_ptr(),
        .sf = reinterpret_cast<sf_pack_t*>(sf.data_ptr()),
        .topk_idx = topk_idx.data_ptr<topk_idx_t>(),
        .topk_weights = topk_weights_ptr,
        .packed = packed.data_ptr<uint8_t>(),
        .packed_count = packed_count.data_ptr<int>(),
        .num_tokens = static_cast<int>(x.size(0)),
        .rank_idx = rank_idx,
        .num_experts = num_experts,
        .launch_args = jit::LaunchArgs(num_sms, num_threads, 0, 1, false)};
    const auto code = L4GroupedPackFP8DispatchRuntime::generate(args);
    const auto runtime = jit::compiler->build("l4_grouped_pack_fp8_dispatch", code);
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(packed_count.data_ptr<int>(), 0, num_ranks * sizeof(int),
                                       at::cuda::getCurrentCUDAStream()));
    L4GroupedPackFP8DispatchRuntime::launch(runtime, args, at::cuda::getCurrentCUDAStream());
}

static void l4_grouped_check_fp8_dispatch_unpack_args(const torch::Tensor& recv_buffer,
                                                      const torch::Tensor& recv_rank_prefix,
                                                      const torch::Tensor& recv_x,
                                                      const torch::Tensor& recv_sf,
                                                      const torch::Tensor& recv_topk_idx,
                                                      const torch::Tensor& recv_topk_weights,
                                                      const torch::Tensor& recv_src_token_idx,
                                                      const int& num_tokens_per_rank,
                                                      const int& num_ranks,
                                                      const int& num_sms) {
    EP_HOST_ASSERT(recv_buffer.is_cuda() and recv_buffer.is_contiguous());
    EP_HOST_ASSERT(recv_rank_prefix.is_cuda() and recv_rank_prefix.is_contiguous());
    EP_HOST_ASSERT(recv_x.is_cuda() and recv_x.is_contiguous());
    EP_HOST_ASSERT(recv_sf.is_cuda() and recv_sf.is_contiguous());
    EP_HOST_ASSERT(recv_topk_idx.is_cuda() and recv_topk_idx.is_contiguous());
    EP_HOST_ASSERT(recv_topk_weights.is_cuda() and recv_topk_weights.is_contiguous());
    EP_HOST_ASSERT(recv_src_token_idx.is_cuda() and recv_src_token_idx.is_contiguous());
    EP_HOST_ASSERT(recv_buffer.scalar_type() == torch::kByte);
    EP_HOST_ASSERT(recv_rank_prefix.scalar_type() == torch::kInt32 and recv_rank_prefix.numel() == num_ranks + 1);
    EP_HOST_ASSERT(recv_x.dim() == 2 and recv_x.element_size() == 1);
    EP_HOST_ASSERT(recv_sf.dim() == 2 and recv_sf.element_size() == sizeof(sf_pack_t));
    EP_HOST_ASSERT(recv_topk_idx.dim() == 2 and recv_topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
    EP_HOST_ASSERT(recv_topk_weights.dim() == 2 and recv_topk_weights.scalar_type() == torch::kFloat32);
    EP_HOST_ASSERT(recv_src_token_idx.dim() == 1 and recv_src_token_idx.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(num_tokens_per_rank > 0);
    EP_HOST_ASSERT(num_ranks > 0 and num_ranks <= 32);
    EP_HOST_ASSERT(num_sms > 0);

    const int num_recv_tokens = static_cast<int>(recv_x.size(0));
    const int hidden_bytes = static_cast<int>(recv_x.size(1) * recv_x.element_size());
    const int num_sf_packs = static_cast<int>(recv_sf.size(1));
    const int num_topk = static_cast<int>(recv_topk_idx.size(1));
    const int token_bytes = l4_grouped_fp8_dispatch_token_bytes_impl(hidden_bytes, num_sf_packs, num_topk);
    EP_HOST_ASSERT(recv_sf.size(0) == num_recv_tokens);
    EP_HOST_ASSERT(recv_topk_idx.size(0) == num_recv_tokens);
    EP_HOST_ASSERT(recv_topk_weights.size(0) == num_recv_tokens and recv_topk_weights.size(1) == num_topk);
    EP_HOST_ASSERT(recv_src_token_idx.numel() == num_recv_tokens);
    EP_HOST_ASSERT(hidden_bytes % sizeof(int4) == 0);
    EP_HOST_ASSERT(recv_buffer.numel() >= static_cast<int64_t>(num_ranks) * num_tokens_per_rank * token_bytes);
}

static void l4_grouped_unpack_fp8_dispatch_out(const torch::Tensor& recv_buffer,
                                               const torch::Tensor& recv_rank_prefix,
                                               const torch::Tensor& recv_x,
                                               const torch::Tensor& recv_sf,
                                               const torch::Tensor& recv_topk_idx,
                                               const torch::Tensor& recv_topk_weights,
                                               const torch::Tensor& recv_src_token_idx,
                                               const int& num_tokens_per_rank,
                                               const int& num_sms) {
    const int num_ranks = static_cast<int>(recv_rank_prefix.numel()) - 1;
    l4_grouped_check_fp8_dispatch_unpack_args(
        recv_buffer, recv_rank_prefix, recv_x, recv_sf, recv_topk_idx,
        recv_topk_weights, recv_src_token_idx, num_tokens_per_rank, num_ranks, num_sms);

    constexpr int num_threads = 256;
    const int hidden_bytes = static_cast<int>(recv_x.size(1) * recv_x.element_size());
    const int num_sf_packs = static_cast<int>(recv_sf.size(1));
    const int num_topk = static_cast<int>(recv_topk_idx.size(1));
    const int token_bytes = l4_grouped_fp8_dispatch_token_bytes_impl(hidden_bytes, num_sf_packs, num_topk);
    const L4GroupedUnpackFP8DispatchRuntime::Args args = {
        .num_ranks = num_ranks,
        .hidden_bytes = hidden_bytes,
        .num_sf_packs = num_sf_packs,
        .num_topk = num_topk,
        .token_bytes = token_bytes,
        .recv_buffer = recv_buffer.data_ptr<uint8_t>(),
        .recv_rank_prefix = recv_rank_prefix.data_ptr<int>(),
        .recv_x = recv_x.data_ptr(),
        .recv_sf = reinterpret_cast<sf_pack_t*>(recv_sf.data_ptr()),
        .recv_topk_idx = recv_topk_idx.data_ptr<topk_idx_t>(),
        .recv_topk_weights = recv_topk_weights.data_ptr<float>(),
        .recv_src_token_idx = recv_src_token_idx.data_ptr<int>(),
        .num_recv_tokens = static_cast<int>(recv_x.size(0)),
        .num_tokens_per_rank = num_tokens_per_rank,
        .launch_args = jit::LaunchArgs(num_sms, num_threads, 0, 1, false)};
    const auto code = L4GroupedUnpackFP8DispatchRuntime::generate(args);
    const auto runtime = jit::compiler->build("l4_grouped_unpack_fp8_dispatch", code);
    L4GroupedUnpackFP8DispatchRuntime::launch(runtime, args, at::cuda::getCurrentCUDAStream());
}

}  // namespace deep_ep::elastic
