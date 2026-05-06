#pragma once

#include "compiler.hpp"
#include "include_parser.hpp"
#include "kernel_runtime.hpp"

namespace deep_ep::jit {

/*
 * JIT Python API 入口。
 *
 * Python 安装/初始化阶段会调用 init_jit，把运行时路径注入 C++:
 *
 *     library_root_path  -> deep_ep/include 下的 impl/common headers
 *     cuda_home          -> nvcc/cuobjdump
 *     nccl_root          -> NCCL headers for device Gin
 *
 * v2 kernel launch wrapper 随后才能生成源码、解析 include hash、编译并加载 cubin。
 */

static void init(const std::string& library_root_path,
                 const std::string& cuda_home_path_by_python, const std::string& nccl_root_path_by_python) {
    Compiler::prepare_init(library_root_path, cuda_home_path_by_python, nccl_root_path_by_python);
    KernelRuntime::prepare_init(cuda_home_path_by_python);
    IncludeParser::prepare_init(library_root_path);
}

static void register_apis(pybind11::module_& m) {
    m.def("init_jit", &init);
}

}  // namespace deep_ep::jit
