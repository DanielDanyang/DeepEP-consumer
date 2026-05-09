#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <torch/python.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"

namespace deep_ep::elastic {

class HostStagingPackRuntime final : public jit::LaunchRuntime<HostStagingPackRuntime> {
public:
    struct Args {
        int hidden_bytes;
        int num_sf_packs;
        int num_topk;

        const void* x;
        const sf_pack_t* sf;
        const topk_idx_t* topk_idx;
        const float* topk_weights;
        void* packed_x;
        sf_pack_t* packed_sf;
        topk_idx_t* packed_topk_idx;
        float* packed_topk_weights;
        int* packed_src_token_idx;
        int* packed_count;
        int num_tokens;
        int rank_idx;
        int num_ranks;
        int num_scaleup_ranks;
        int num_experts;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/host_staging.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&host_staging_pack_fp8_dispatch_impl<{}, {}, {}, {}>);
}}
)",
                           args.launch_args.num_threads,
                           args.hidden_bytes,
                           args.num_sf_packs,
                           args.num_topk);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.x,
            args.sf,
            args.topk_idx,
            args.topk_weights,
            args.packed_x,
            args.packed_sf,
            args.packed_topk_idx,
            args.packed_topk_weights,
            args.packed_src_token_idx,
            args.packed_count,
            args.num_tokens,
            args.rank_idx,
            args.num_ranks,
            args.num_scaleup_ranks,
            args.num_experts));
    }
};

class HostStagingPackBf16CombineRuntime final : public jit::LaunchRuntime<HostStagingPackBf16CombineRuntime> {
public:
    struct Args {
        int hidden_bytes;
        int num_topk;

        const nv_bfloat16* x;
        const topk_idx_t* topk_idx;
        nv_bfloat16* packed_x;
        int* packed_src_token_idx;
        int* packed_count;
        int num_tokens;
        int rank_idx;
        int num_ranks;
        int num_scaleup_ranks;
        int num_experts;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/host_staging.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&host_staging_pack_bf16_combine_impl<{}, {}, {}>);
}}
)",
                           args.launch_args.num_threads,
                           args.hidden_bytes,
                           args.num_topk);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.x,
            args.topk_idx,
            args.packed_x,
            args.packed_src_token_idx,
            args.packed_count,
            args.num_tokens,
            args.rank_idx,
            args.num_ranks,
            args.num_scaleup_ranks,
            args.num_experts));
    }
};

static void host_staging_check_fp8_dispatch_pack_args(const torch::Tensor& x,
                                                      const torch::Tensor& sf,
                                                      const torch::Tensor& topk_idx,
                                                      const std::optional<torch::Tensor>& topk_weights,
                                                      const torch::Tensor& packed_x,
                                                      const torch::Tensor& packed_sf,
                                                      const torch::Tensor& packed_topk_idx,
                                                      const torch::Tensor& packed_topk_weights,
                                                      const torch::Tensor& packed_src_token_idx,
                                                      const torch::Tensor& packed_count,
                                                      const int& num_ranks,
                                                      const int& num_scaleup_ranks,
                                                      const int& num_experts,
                                                      const int& num_sms) {
    EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous());
    EP_HOST_ASSERT(sf.is_cuda() and sf.is_contiguous());
    EP_HOST_ASSERT(topk_idx.is_cuda() and topk_idx.is_contiguous());
    EP_HOST_ASSERT(packed_x.is_cuda() and packed_x.is_contiguous());
    EP_HOST_ASSERT(packed_sf.is_cuda() and packed_sf.is_contiguous());
    EP_HOST_ASSERT(packed_topk_idx.is_cuda() and packed_topk_idx.is_contiguous());
    EP_HOST_ASSERT(packed_topk_weights.is_cuda() and packed_topk_weights.is_contiguous());
    EP_HOST_ASSERT(packed_src_token_idx.is_cuda() and packed_src_token_idx.is_contiguous());
    EP_HOST_ASSERT(packed_count.is_cuda() and packed_count.is_contiguous());
    EP_HOST_ASSERT(x.dim() == 2 and sf.dim() == 2 and topk_idx.dim() == 2);
    EP_HOST_ASSERT(x.element_size() == 1 and "Host-staging pack currently targets FP8 dispatch payloads");
    EP_HOST_ASSERT(sf.element_size() == sizeof(sf_pack_t));
    EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
    EP_HOST_ASSERT(packed_x.scalar_type() == x.scalar_type() and packed_x.sizes() == x.sizes());
    EP_HOST_ASSERT(packed_sf.scalar_type() == sf.scalar_type() and packed_sf.sizes() == sf.sizes());
    EP_HOST_ASSERT(packed_topk_idx.scalar_type() == topk_idx.scalar_type() and packed_topk_idx.sizes() == topk_idx.sizes());
    EP_HOST_ASSERT(packed_topk_weights.scalar_type() == torch::kFloat32);
    EP_HOST_ASSERT(packed_src_token_idx.scalar_type() == torch::kInt32 and packed_src_token_idx.numel() >= x.size(0));
    EP_HOST_ASSERT(packed_count.scalar_type() == torch::kInt32 and packed_count.numel() == 1);
    EP_HOST_ASSERT(num_ranks > 0 and num_scaleup_ranks > 0 and num_ranks % num_scaleup_ranks == 0);
    EP_HOST_ASSERT(num_experts > 0 and num_experts % (num_ranks / num_scaleup_ranks) == 0);
    EP_HOST_ASSERT(num_sms > 0);

    const int num_tokens = static_cast<int>(x.size(0));
    const int hidden_bytes = static_cast<int>(x.size(1) * x.element_size());
    const int num_sf_packs = static_cast<int>(sf.size(1));
    const int num_topk = static_cast<int>(topk_idx.size(1));
    EP_HOST_ASSERT(sf.size(0) == num_tokens and topk_idx.size(0) == num_tokens);
    EP_HOST_ASSERT(hidden_bytes % sizeof(int4) == 0);

    if (topk_weights.has_value()) {
        EP_HOST_ASSERT(topk_weights->is_cuda() and topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        EP_HOST_ASSERT(topk_weights->size(0) == num_tokens and topk_weights->size(1) == num_topk);
    }
    EP_HOST_ASSERT(packed_topk_weights.dim() == 2 and
                   packed_topk_weights.size(0) == num_tokens and packed_topk_weights.size(1) == num_topk);

}

static void host_staging_pack_fp8_dispatch_out(const torch::Tensor& x,
                                               const torch::Tensor& sf,
                                               const torch::Tensor& topk_idx,
                                               const std::optional<torch::Tensor>& topk_weights,
                                               const torch::Tensor& packed_x,
                                               const torch::Tensor& packed_sf,
                                               const torch::Tensor& packed_topk_idx,
                                               const torch::Tensor& packed_topk_weights,
                                               const torch::Tensor& packed_src_token_idx,
                                               const torch::Tensor& packed_count,
                                               const int& rank_idx,
                                               const int& num_ranks,
                                               const int& num_scaleup_ranks,
                                               const int& num_experts,
                                               const int& num_sms) {
    host_staging_check_fp8_dispatch_pack_args(x, sf, topk_idx, topk_weights,
                                              packed_x, packed_sf, packed_topk_idx, packed_topk_weights,
                                              packed_src_token_idx, packed_count,
                                              num_ranks, num_scaleup_ranks, num_experts, num_sms);

    const int num_tokens = static_cast<int>(x.size(0));
    const int hidden_bytes = static_cast<int>(x.size(1) * x.element_size());
    const int num_sf_packs = static_cast<int>(sf.size(1));
    const int num_topk = static_cast<int>(topk_idx.size(1));
    float* topk_weights_ptr = topk_weights.has_value() ? topk_weights->data_ptr<float>() : nullptr;

    constexpr int num_threads = 256;
    const HostStagingPackRuntime::Args args = {
        .hidden_bytes = hidden_bytes,
        .num_sf_packs = num_sf_packs,
        .num_topk = num_topk,
        .x = x.data_ptr(),
        .sf = reinterpret_cast<sf_pack_t*>(sf.data_ptr()),
        .topk_idx = topk_idx.data_ptr<topk_idx_t>(),
        .topk_weights = topk_weights_ptr,
        .packed_x = packed_x.data_ptr(),
        .packed_sf = reinterpret_cast<sf_pack_t*>(packed_sf.data_ptr()),
        .packed_topk_idx = packed_topk_idx.data_ptr<topk_idx_t>(),
        .packed_topk_weights = packed_topk_weights.data_ptr<float>(),
        .packed_src_token_idx = packed_src_token_idx.data_ptr<int>(),
        .packed_count = packed_count.data_ptr<int>(),
        .num_tokens = num_tokens,
        .rank_idx = rank_idx,
        .num_ranks = num_ranks,
        .num_scaleup_ranks = num_scaleup_ranks,
        .num_experts = num_experts,
        .launch_args = jit::LaunchArgs(num_sms, num_threads, 0, 1, false)};
    const auto code = HostStagingPackRuntime::generate(args);
    const auto runtime = jit::compiler->build("host_staging_pack_fp8_dispatch", code);
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(packed_count.data_ptr<int>(), 0, sizeof(int), at::cuda::getCurrentCUDAStream()));
    HostStagingPackRuntime::launch(runtime, args, at::cuda::getCurrentCUDAStream());
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
host_staging_pack_fp8_dispatch(const torch::Tensor& x,
                               const torch::Tensor& sf,
                               const torch::Tensor& topk_idx,
                               const std::optional<torch::Tensor>& topk_weights,
                               const int& rank_idx,
                               const int& num_ranks,
                               const int& num_scaleup_ranks,
                               const int& num_experts,
                               const int& num_sms) {
    auto packed_x = torch::empty_like(x);
    auto packed_sf = torch::empty_like(sf);
    auto packed_topk_idx = torch::empty_like(topk_idx);
    auto packed_topk_weights = topk_weights.has_value() ?
        torch::empty_like(topk_weights.value()) :
        torch::empty({x.size(0), topk_idx.size(1)}, x.options().dtype(torch::kFloat32));
    auto packed_src_token_idx = torch::empty({x.size(0)}, x.options().dtype(torch::kInt32));
    auto packed_count = torch::empty({1}, x.options().dtype(torch::kInt32));
    host_staging_pack_fp8_dispatch_out(x, sf, topk_idx, topk_weights,
                                       packed_x, packed_sf, packed_topk_idx, packed_topk_weights,
                                       packed_src_token_idx, packed_count,
                                       rank_idx, num_ranks, num_scaleup_ranks, num_experts, num_sms);
    return {packed_x, packed_sf, packed_topk_idx, packed_topk_weights, packed_src_token_idx, packed_count};
}

static void host_staging_check_bf16_combine_pack_args(const torch::Tensor& x,
                                                      const torch::Tensor& topk_idx,
                                                      const torch::Tensor& packed_x,
                                                      const torch::Tensor& packed_src_token_idx,
                                                      const torch::Tensor& packed_count,
                                                      const int& num_ranks,
                                                      const int& num_scaleup_ranks,
                                                      const int& num_experts,
                                                      const int& num_sms) {
    EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous());
    EP_HOST_ASSERT(topk_idx.is_cuda() and topk_idx.is_contiguous());
    EP_HOST_ASSERT(packed_x.is_cuda() and packed_x.is_contiguous());
    EP_HOST_ASSERT(packed_src_token_idx.is_cuda() and packed_src_token_idx.is_contiguous());
    EP_HOST_ASSERT(packed_count.is_cuda() and packed_count.is_contiguous());
    EP_HOST_ASSERT(x.dim() == 2 and topk_idx.dim() == 2);
    EP_HOST_ASSERT(x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
    EP_HOST_ASSERT(packed_x.scalar_type() == x.scalar_type() and packed_x.sizes() == x.sizes());
    EP_HOST_ASSERT(packed_src_token_idx.scalar_type() == torch::kInt32 and packed_src_token_idx.numel() >= x.size(0));
    EP_HOST_ASSERT(packed_count.scalar_type() == torch::kInt32 and packed_count.numel() == 1);
    EP_HOST_ASSERT(topk_idx.size(0) == x.size(0));
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(num_ranks > 0 and num_scaleup_ranks > 0 and num_ranks % num_scaleup_ranks == 0);
    EP_HOST_ASSERT(num_experts > 0 and num_experts % (num_ranks / num_scaleup_ranks) == 0);
    EP_HOST_ASSERT(num_sms > 0);
}

static void host_staging_pack_bf16_combine_out(const torch::Tensor& x,
                                               const torch::Tensor& topk_idx,
                                               const torch::Tensor& packed_x,
                                               const torch::Tensor& packed_src_token_idx,
                                               const torch::Tensor& packed_count,
                                               const int& rank_idx,
                                               const int& num_ranks,
                                               const int& num_scaleup_ranks,
                                               const int& num_experts,
                                               const int& num_sms) {
    host_staging_check_bf16_combine_pack_args(
        x, topk_idx, packed_x, packed_src_token_idx, packed_count,
        num_ranks, num_scaleup_ranks, num_experts, num_sms);

    constexpr int num_threads = 256;
    const HostStagingPackBf16CombineRuntime::Args args = {
        .hidden_bytes = static_cast<int>(x.size(1) * x.element_size()),
        .num_topk = static_cast<int>(topk_idx.size(1)),
        .x = reinterpret_cast<nv_bfloat16*>(x.data_ptr()),
        .topk_idx = topk_idx.data_ptr<topk_idx_t>(),
        .packed_x = reinterpret_cast<nv_bfloat16*>(packed_x.data_ptr()),
        .packed_src_token_idx = packed_src_token_idx.data_ptr<int>(),
        .packed_count = packed_count.data_ptr<int>(),
        .num_tokens = static_cast<int>(x.size(0)),
        .rank_idx = rank_idx,
        .num_ranks = num_ranks,
        .num_scaleup_ranks = num_scaleup_ranks,
        .num_experts = num_experts,
        .launch_args = jit::LaunchArgs(num_sms, num_threads, 0, 1, false)};
    const auto code = HostStagingPackBf16CombineRuntime::generate(args);
    const auto runtime = jit::compiler->build("host_staging_pack_bf16_combine", code);
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(packed_count.data_ptr<int>(), 0, sizeof(int), at::cuda::getCurrentCUDAStream()));
    HostStagingPackBf16CombineRuntime::launch(runtime, args, at::cuda::getCurrentCUDAStream());
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
host_staging_pack_bf16_combine(const torch::Tensor& x,
                               const torch::Tensor& topk_idx,
                               const int& rank_idx,
                               const int& num_ranks,
                               const int& num_scaleup_ranks,
                               const int& num_experts,
                               const int& num_sms) {
    auto packed_x = torch::empty_like(x);
    auto packed_src_token_idx = torch::empty({x.size(0)}, x.options().dtype(torch::kInt32));
    auto packed_count = torch::empty({1}, x.options().dtype(torch::kInt32));
    host_staging_pack_bf16_combine_out(x, topk_idx, packed_x, packed_src_token_idx, packed_count,
                                       rank_idx, num_ranks, num_scaleup_ranks, num_experts, num_sms);
    return {packed_x, packed_src_token_idx, packed_count};
}

}  // namespace deep_ep::elastic
