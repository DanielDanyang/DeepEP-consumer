#pragma once

#include <torch/python.h>
#include <deep_ep/common/exception.cuh>

namespace deep_ep::elastic {

/*
 * C++ host 层小工具。
 *
 *     Python ElasticBuffer
 *             |
 *             v
 *     csrc/elastic/buffer.hpp
 *             |
 *             +-- get_global_comm_stream()  统一通信 stream
 *             +-- get_shape<N>()            张量维度校验 + int 化
 *             +-- get_data_ptr()            optional tensor -> nullable pointer
 *
 * 这些函数都只在 host 侧调度逻辑使用，不参与 device kernel specialization。
 */

static at::cuda::CUDAStream get_global_comm_stream() {
    // 使用 PyTorch stream pool 的高优先级 stream。所有 ElasticBuffer 实例共享一个
    // communication stream，方便 Python 层做 compute/communication overlap。
    static std::optional<at::cuda::CUDAStream> comm_stream = std::nullopt;
    if (not comm_stream.has_value())
        comm_stream = at::cuda::getStreamFromPool(true);
    return comm_stream.value();
}

template <int kNumDims>
static auto get_shape(const torch::Tensor& t) {
    // 统一把 int64_t sizes 转成 int；v2 kernel 的大多数 template/参数都使用 int。
    EP_HOST_ASSERT(t.dim() == kNumDims);
    return [&t] <size_t... Is> (std::index_sequence<Is...>) {
        return std::make_tuple(static_cast<int>(t.sizes()[Is])...);
    }(std::make_index_sequence<kNumDims>());
}

template <typename dtype_t = void>
static dtype_t* get_data_ptr(const std::optional<torch::Tensor>& t) {
    return t.has_value() ? t->data_ptr<dtype_t>() : nullptr;
}

}  // deep_ep::elastic
