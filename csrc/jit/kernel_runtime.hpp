#pragma once

#include <ATen/cuda/CUDAContext.h>

#include <deep_ep/common/exception.cuh>

#include "../utils/format.hpp"
#include "../utils/lazy_init.hpp"
#include "handle.hpp"

namespace deep_ep::jit {

/*
 * 已编译 cubin 的加载与生命周期。
 *
 * cache directory:
 *
 *     kernel.<name>.<hash>/
 *       +-- kernel.cu
 *       +-- kernel.cubin
 *       +-- optional kernel.ptx/kernel.sass
 *
 * KernelRuntime 做两件事:
 *   1. 用 cuobjdump 找到唯一的 entry kernel symbol
 *   2. load_kernel(cubin, symbol) 得到 CUDA function handle
 *
 * JIT 生成源码中的 __instantiate_kernel 只是为了强制模板实例化，不应被当成 entry。
 */

class KernelRuntime final {
public:
    static std::filesystem::path cuda_home;

    LibraryHandle library;
    KernelHandle kernel;

    explicit KernelRuntime(const std::filesystem::path& dir_path) {
        EP_HOST_ASSERT(not cuda_home.empty());

        // NOLINT(*-pro-type-member-init)
        const auto cuobjdump_path = cuda_home / "bin" / "cuobjdump";
        const auto cubin_path = dir_path / "kernel.cubin";
        if (get_env<int>("EP_JIT_DEBUG"))
            printf("Loading CUBIN: %s\n", cubin_path.c_str());

        // 找 cubin 中唯一真正的 __global__ entry symbol。
        // TODO: use kernel enumeration for newer drivers
        const std::vector<std::string> illegal_names = {"vprintf", "__instantiate_kernel", "__internal", "__assertfail"};
        const auto [exit_code, symbols] = call_external_command(fmt::format("{} -symbols {}", cuobjdump_path.c_str(), cubin_path.c_str()));
        EP_HOST_ASSERT(exit_code == 0);
        std::istringstream iss(symbols);
        std::vector<std::string> symbol_names;
        for (std::string line; std::getline(iss, line); ) {
            if (line.find("STT_FUNC") == 0 and line.find("STO_ENTRY") != std::string::npos and
                std::none_of(illegal_names.begin(), illegal_names.end(),
                [&](const auto name) { return line.find(name) != std::string::npos; })) {
                const auto last_space = line.rfind(' ');
                symbol_names.push_back(line.substr(last_space + 1));
            }
        }

        // 如果多于/少于一个 entry，说明 cache 目录或生成代码异常。
        if (symbol_names.size() != 1) {
            printf("Corrupted JIT cache directory (expected 1 kernel symbol, found %zu): %s, "
                   "please run `rm -rf %s` and restart your task.\n",
                   symbol_names.size(), dir_path.c_str(), dir_path.c_str());
            printf("Symbol names: ");
            for (const auto& symbol: symbol_names)
                printf("%s, ", symbol.c_str());
            printf("\n");
            EP_HOST_ASSERT(false and "Corrupted JIT cache directory");
        }

        // Load from the library
        kernel = load_kernel(cubin_path, symbol_names[0], &library);
    }

    static void prepare_init(const std::string& cuda_home_path_by_python) {
        cuda_home = cuda_home_path_by_python;
    }

    static bool check_validity(const std::filesystem::path& dir_path) {
        if (not std::filesystem::exists(dir_path))
            return false;
        // NOTES: if the directory exists, kernel.cu and kernel.cubin must both exist,
        // because the directory is created atomically via rename
        if (not std::filesystem::exists(dir_path / "kernel.cu") or
            not std::filesystem::exists(dir_path / "kernel.cubin")) {
            printf("Corrupted JIT cache directory (missing kernel.cu or kernel.cubin): %s, "
                   "please run `rm -rf %s` and restart your task.\n",
                   dir_path.c_str(), dir_path.c_str());
            EP_HOST_ASSERT(false and "Corrupted JIT cache directory");
        }
        return true;
    }

    ~KernelRuntime() noexcept(false) {
        unload_library(library);
    }
};

EP_DECLARE_STATIC_VAR_IN_CLASS(KernelRuntime, cuda_home);

} // namespace deep_ep::jit
