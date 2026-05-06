#pragma once

#include <stdexcept>


namespace deep_ep::elastic {

/*
 * combine 规约公共工具。
 *
 * combine main kernel 和 reduce epilogue 都会做类似事情:
 *
 *     top-k slots -> load candidate hidden vectors -> reduce -> shared/output buffer
 *
 * 本文件提供:
 *   - layout 选择: 按 rank 维度还是 top-k 维度放 reduce buffer
 *   - vector 类型: SM100+ 可用 32B longlong4_t，否则 int4
 *   - top-k slot compaction
 *   - BF16/FP32 reduce helper
 *
 * reduce buffer 维度选择:
 *
 *     allow_multiple_reduction=true:
 *       如果 ranks <= topk，用 rank layout，减少无效 top-k slot
 *
 *     allow_multiple_reduction=false:
 *       保留 top-k layout，让最终 epilogue 做一次集中规约，精度更稳定但通信更多。
 */

template <bool kAllowMultipleReduction, int kNumRanks, int kNumTopk>
constexpr bool use_rank_layout() {
    if constexpr (not kAllowMultipleReduction)
        return false;
    return kNumRanks <= kNumTopk;
}

template <bool kAllowMultipleReduction, int kNumRanks, int kNumTopk>
constexpr int get_num_tokens_in_layout() {
    return use_rank_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>() ? kNumRanks : kNumTopk;
}

template <int kLength, int kMaxUnrollFactor, int kWarpSize = 32>
constexpr int get_max_unroll_factor() {
    for (int i = kMaxUnrollFactor; i >= 1; -- i)
        if (kLength % (kWarpSize * i) == 0)
            return i;
    throw std::logic_error("Invalid length, cannot find unrolling factor");
}

// Determine the vector type for combine loads/stores based on arch and hidden size alignment
template <int kHiddenBytes>
struct CombineVecTraits {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    // On SM100+, use longlong4_t (32 bytes) if hidden is aligned, otherwise fall back to int4 (16 bytes)
    static constexpr bool kUseLonglong4 = (kHiddenBytes % sizeof(longlong4_t) == 0) and
                                          ((kHiddenBytes / sizeof(longlong4_t)) % 32 == 0);
    using vec_t = std::conditional_t<kUseLonglong4, longlong4_t, int4>;
#else
    using vec_t = int4;
#endif
};

template <int kNumValidTopk, typename fetch_func_t>
__device__ __forceinline__
void compute_topk_slots(int (&topk_slot_idx)[kNumValidTopk], uint32_t mask,
                        const fetch_func_t& fetch_func) {
    // 将 mask 中有效 lane 对应的 slot index 压到数组前部，无效位置填 -1。
    // 无条件 fetch 是为了减少分支发散。
    #pragma unroll
    for (int k = 0; k < kNumValidTopk; ++ k) {
        const int lowest_idx = __ffs(mask) - 1;
        // Here we perform the exchange unconditionally to avoid `BRA.DIV`
        const auto fetched = fetch_func(lowest_idx);
        mask &= mask - 1;
        topk_slot_idx[k] = lowest_idx >= 0 ? fetched : -1;
    }
}

template <int kHiddenVec, int kUnrollFactor, int kNumExpectedTopk, int kNumValidTopk,
          typename vec_t, typename get_src_buffer_ptr_func_t, typename wait_buffer_func_t>
__device__ __forceinline__
void combine_reduce(const int& lane_idx, int (&topk_slot_idx)[kNumValidTopk],
                    vec_t* dst_buffer_ptr,
                    const get_src_buffer_ptr_func_t& get_src_buffer_ptr_func,
                    const wait_buffer_func_t& wait_buffer_func,
                    vec_t* bias_0 = nullptr, vec_t* bias_1 = nullptr) {
    // 每个 warp 负责一个 token 的 hidden 向量规约:
    //
    //     lane0..31 处理向量条带
    //     kUnrollFactor 提高每轮加载/计算吞吐
    //     wait_buffer_func 在第一次写 shared/output 前释放 TMA staging buffer
    //
    // 两路或一路无 bias 时走 BF16 hadd 快路径；否则用 FP32 accumulate 再 cast BF16。
    constexpr int kNumElemsPerVec = sizeof(vec_t) / sizeof(nv_bfloat16);
    EP_STATIC_ASSERT(kNumElemsPerVec % 2 == 0, "Invalid number of elements");
    EP_STATIC_ASSERT(kHiddenVec % (kUnrollFactor * 32) == 0, "Invalid unrolling");

    // We use BF16 add as much as possible, as casting is slow
    const bool enable_hadd_bypass =
        (bias_0 == nullptr and bias_1 == nullptr) and
        (kNumValidTopk <= 2 or topk_slot_idx[2] < 0);
    EP_STATIC_ASSERT(kNumValidTopk > 0, "Invalid top-k");

    if (enable_hadd_bypass) {
        #pragma unroll 1
        for (int i = 0; i < kHiddenVec / (kUnrollFactor * 32); ++ i) {
            // Read values 0
            const auto slot_0 = topk_slot_idx[0];
            const auto src_base_ptr_0 = get_src_buffer_ptr_func(slot_0);
            vec_t values_0[kUnrollFactor] = {};
            #pragma unroll
            for (int j = 0; j < kUnrollFactor; ++ j) {
                values_0[j] = ptx::ldg_with_gez_pred(
                    src_base_ptr_0 + (i * (kUnrollFactor * 32) + j * 32 + lane_idx), slot_0);
            }

            // Read values 1
            vec_t values_1[kUnrollFactor] = {};
            const auto slot_1 = kNumValidTopk == 1 ? -1 : topk_slot_idx[1];
            const auto src_base_ptr_1 = get_src_buffer_ptr_func(slot_1);
            #pragma unroll
            for (int j = 0; j < kUnrollFactor; ++ j) {
                values_1[j] = ptx::ldg_with_gez_pred(
                    src_base_ptr_1 + (i * (kUnrollFactor * 32) + j * 32 + lane_idx), slot_1);
            }

            // Wait buffer releases for the first write
            if (i == 0)
                wait_buffer_func();

            // Reduce into shared memory
            const auto bf162_view_0 = reinterpret_cast<nv_bfloat162*>(values_0);
            const auto bf162_view_1 = reinterpret_cast<nv_bfloat162*>(values_1);
            #pragma unroll
            for (int j = 0; j < kUnrollFactor; ++ j) {
                #pragma unroll
                for (int l = 0; l < kNumElemsPerVec / 2; ++ l)
                    bf162_view_0[j * (kNumElemsPerVec / 2) + l] += bf162_view_1[j * (kNumElemsPerVec / 2) + l];
                dst_buffer_ptr[i * (kUnrollFactor * 32) + j * 32 + lane_idx] = values_0[j];
            }
        }
    } else {
        #pragma unroll 1
        for (int i = 0; i < kHiddenVec / (kUnrollFactor * 32); ++ i) {
            // Add bias
            float2 reduced[kUnrollFactor * kNumElemsPerVec / 2] = {};
            const auto add_bias = [&](const vec_t* base_ptr) {
                // Read
                vec_t values[kUnrollFactor];
                #pragma unroll
                for (int j = 0; j < kUnrollFactor; ++ j)
                    values[j] = ptx::ldg(base_ptr + i * (kUnrollFactor * 32) + j * 32 + lane_idx);

                // Reduce
                const auto bf162_view = reinterpret_cast<nv_bfloat162*>(values);
                #pragma unroll
                for (int j = 0; j < kUnrollFactor * kNumElemsPerVec / 2; ++ j)
                    ptx::accumulate(reduced[j], bf162_view[j]);
            };
            bias_0 != nullptr ? add_bias(bias_0) : void();
            bias_1 != nullptr ? add_bias(bias_1) : void();

            #pragma unroll
            for (int k = 0; k < kNumValidTopk; ++ k) {
                // We have a limitation on `k` to reduce the branch instruction count
                if (k >= kNumExpectedTopk and topk_slot_idx[k] < 0)
                    break;

                // Read values
                const auto src_base_ptr = get_src_buffer_ptr_func(topk_slot_idx[k]);
                vec_t values[kUnrollFactor] = {};
                #pragma unroll
                for (int j = 0; j < kUnrollFactor; ++ j) {
                    values[j] = ptx::ldg_with_gez_pred(
                        src_base_ptr + (i * (kUnrollFactor * 32) + j * 32 + lane_idx), topk_slot_idx[k]);
                }

                // Reduce
                const auto bf162_view = reinterpret_cast<nv_bfloat162*>(values);
                #pragma unroll
                for (int j = 0; j < kUnrollFactor * kNumElemsPerVec / 2; ++ j)
                    ptx::accumulate(reduced[j], bf162_view[j]);
            }

            // Wait buffer releases for the first write
            if (i == 0)
                wait_buffer_func();

            // Cast into shared memory
            #pragma unroll
            for (int j = 0; j < kUnrollFactor; ++ j) {
                vec_t casted_value;
                auto bf162_view = reinterpret_cast<nv_bfloat162*>(&casted_value);
                #pragma unroll
                for (int l = 0; l < kNumElemsPerVec / 2; ++ l)
                    bf162_view[l] = __float22bfloat162_rn(reduced[j * (kNumElemsPerVec / 2) + l]);
                dst_buffer_ptr[i * (kUnrollFactor * 32) + j * 32 + lane_idx] = casted_value;
            }
        }
    }
}

}  // namespace deep_ep::elastic
