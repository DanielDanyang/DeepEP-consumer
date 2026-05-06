#include <pybind11/pybind11.h>
#include <torch/python.h>

#include <deep_ep/common/compiled.cuh>

#include "jit/api.hpp"
#include "elastic/buffer.hpp"
#include "legacy/buffer.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

/*
 * Python extension 总入口。
 *
 * v2 / elastic 路径在这里注册到 deep_ep._C：
 *
 *     Python import deep_ep._C
 *              |
 *              v
 *     PYBIND11_MODULE
 *          |
 *          +-- jit::register_apis          JIT 编译/缓存辅助 API
 *          +-- legacy::register_apis       V1 兼容路径
 *          +-- elastic::register_apis      V2 ElasticBuffer 主路径
 *
 * 阅读 v2 时只需要顺着 elastic::register_apis 进入 csrc/elastic/buffer.hpp。
 */

bool is_sm90_compiled() {
#ifndef DISABLE_SM90_FEATURES
    return true;
#else
    return false;
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepEP: an efficient expert-parallel communication library";

    // 是否启用了 SM90+ 特性。v2 的 FP8/TMA/mbarrier 路径会依赖这个编译开关。
    m.def("is_sm90_compiled", []() { return deep_ep::kEnableSM90Features; });

    // top-k index 的实际整数类型由 EP_NUM_TOPK_IDX_BITS 编译期开关决定。
    m.attr("topk_idx_t") = py::cast(c10::CppTypeToScalarType<deep_ep::topk_idx_t>::value);

    // JIT API: 运行时生成 CUDA 源码、编译 cubin、缓存并加载。
    deep_ep::jit::register_apis(m);

    // V1 legacy API 保持兼容，不属于本次 v2 主线阅读。
    deep_ep::legacy::register_apis(m);

    // V2 ElasticBuffer API: dispatch/combine/barrier/Engram/PP/AGRS 都从这里暴露。
    deep_ep::elastic::register_apis(m);
}
